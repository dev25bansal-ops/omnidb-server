"""Hand-rolled mutation analysis for the pure ranking modules.

mutmut 2.5.1 crashes on Python 3.14 (cp1252 banner encode + "cannot pickle
itertools.count" in its trampoline machinery) and mutmut 3.x requires
[tool.mutmut] config in files that are not ours to edit, so this script is the
fallback: it applies a fixed table of hand-written mutations to fusion.py and
scoring.py one at a time, runs the scoped pytest suite against each mutant,
records KILLED/SURVIVED, and restores the originals in a finally block so no
mutated file is ever left behind.

**The script is a gate.** It exits non-zero when the kill rate falls below
``--min-score`` (default 85) or when any anchor fails to apply. Before this
change it returned 0 unconditionally, so it could not fail and its 33% score
was never enforced. Two further defects it had, both now fixed:

1. **8 of 30 anchors were silently NOT_APPLICABLE** — the entire RRF half of
   the table no longer matched the source, because ``rrf`` had grown a
   ``weights`` argument and the accumulation line now reads
   ``weight / (k + position + 1)`` rather than ``1.0 / (k + position + 1)``.
   A stale anchor that matches nothing looks identical to a mutant nobody
   killed, so the score was computed over a table that was mostly not testing
   anything. ``--fail-on-not-applicable`` (default on) now turns that into a
   red run.
2. **The score counted NOT_APPLICABLE anchors as survivors**, and counted a
   genuine equivalent mutant as a failure. Mutant 25 (quote-doubling) is
   equivalent by construction: the tokenizer cannot emit a quote, so removing
   the doubling is unobservable. It is declared in ``EQUIVALENT`` and excluded
   from the denominator, so the score measures mutants that *could* have been
   killed.

Usage:
    python scripts/mutation_check.py
    python scripts/mutation_check.py --min-score 90
    python scripts/mutation_check.py --paths-to-mutate fusion
    python scripts/mutation_check.py --json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PKG = REPO / "omnidb_server"
FUSION = PKG / "fusion.py"
SCORING = PKG / "scoring.py"
GRAPH_LEG = PKG / "graph_leg.py"

# The scoped suite. Path-filtered: only tests that actually exercise fusion.py
# and scoring.py belong here, or a green score is bought by including tests
# that would pass regardless of the mutation.
TESTS = [
    "tests/test_fusion_values.py",
    "tests/test_scoring_values.py",
    "tests/test_fusion_and_admission_properties.py",
    "tests/test_fuzz_fusion.py",
    "tests/test_scoring_properties.py",
    "tests/test_graph_leg_values.py",
    "tests/test_memory.py",
]
PYTEST_TIMEOUT_S = 60

# Mutants that are *equivalent*: behaviour is identical for every input
# reachable today, so no test can kill them. Declared rather than silently
# counted as survivors — a new entry here needs a justification, and the count
# is reported on every run.
#
# Mutant 28 (finalized-check removal) USED TO BE HERE and is not equivalent.
# Its old justification claimed the removal "only wastes heap pops and never
# changes any emitted key". The conclusion about `best` was right (the max-key
# guard keeps the maximum), but the premise is wrong under this algorithm's
# hard pop cap: a wasted pop is one fewer node finalised, so the output SET
# shrinks. Fuzzing found mismatches on 124 of 300 random multi-path graphs;
# test_the_finalized_check_is_load_bearing_under_a_hard_budget is the minimal
# deterministic reproducer (budget 5: guard emits [1,2,3,4], removal emits
# [1,2,3]). It is now an ordinary mutant and must be killed.
EQUIVALENT: dict[int, str] = {
    33: "quote-doubling is dead defense: \\w+ can never emit a double quote, "
        "so .replace(chr(34), chr(34)*2) is unobservable. Pinned by "
        "test_quote_doubling_is_dead_defense_only_because_the_tokenizer_excludes_quotes "
        "(renumbered from 25 when the graph_leg mutants prepended the table)",
}

# The tokenizing assignment in fusion.fts_match_query, as one exact string, so
# the two anchors that mutate its `[:max_terms]` cap stay byte-accurate against
# a two-line comprehension. A stale anchor is indistinguishable from an unkilled
# mutant to the score, and the gate fails on NOT_APPLICABLE for exactly that
# reason — which is how the F35/F36 refactors surfaced here.
_TOKEN_LINE = (
    "tokens = [t for t in _TOKEN.findall(_fold(text))\n"
    "              if t not in _STOP_WORDS]"
)

# Each mutation: (module, description, list of (old, new) exact-text replacements).
MUTATIONS: list[tuple[str, str, list[tuple[str, str]]]] = [
    # ---------------- scoring.py: recency decay ----------------
    ("scoring", "exp(-age/hl) -> exp(age/hl): sign flip inverts recency decay",
     [("recency = math.exp(-age / half_life)", "recency = math.exp(age / half_life)")]),
    ("scoring", "drop max clamp on age_ms: negative age explodes exponent",
     [("age = max(float(age_ms), 0.0)", "age = float(age_ms)")]),
    ("scoring", "drop max clamp on half_life_ms: zero half-life divides by zero",
     [("half_life = max(float(half_life_ms), 1e-3)", "half_life = float(half_life_ms)")]),
    ("scoring", "drop max clamp on access_count: negative count poisons log1p",
     [("accesses = max(float(access_count), 0.0)", "accesses = float(access_count)")]),
    ("scoring", "log1p -> log: frequency of 0 access becomes -inf",
     [("frequency = math.log1p(accesses)", "frequency = math.log(accesses)")]),
    ("scoring", "recency_weight default 0.15 -> 0.0: recency stops contributing",
     [("recency_weight: float = 0.15", "recency_weight: float = 0.0")]),
    ("scoring", "freq_weight default 0.05 -> 0.0: frequency stops contributing",
     [("freq_weight: float = 0.05", "freq_weight: float = 0.0")]),
    ("scoring", "recency_weight * recency -> + recency: weights ignored",
     [("score = float(rrf) + recency_weight * recency + freq_weight * frequency",
       "score = float(rrf) + recency + frequency")]),
    ("scoring", "return drops recency component (returns rrf as placeholder)",
     [("return score, recency, frequency", "return score, float(rrf), frequency")]),
    ("scoring", "return drops frequency component (returns rrf as placeholder)",
     [("return score, recency, frequency", "return score, recency, float(rrf)")]),
    ("scoring", "score omits recency term entirely",
     [("score = float(rrf) + recency_weight * recency + freq_weight * frequency",
       "score = float(rrf) + freq_weight * frequency")]),
    ("scoring", "score omits frequency term entirely",
     [("score = float(rrf) + recency_weight * recency + freq_weight * frequency",
       "score = float(rrf) + recency_weight * recency")]),
    ("scoring", "score omits rrf term entirely (pure behavioral score)",
     [("score = float(rrf) + recency_weight * recency + freq_weight * frequency",
       "score = recency_weight * recency + freq_weight * frequency")]),
    ("scoring", "half-life floor 1e-3 -> 1e6: decay effectively frozen",
     [("half_life = max(float(half_life_ms), 1e-3)", "half_life = max(float(half_life_ms), 1e6)")]),

    # ---------------- fusion.py: RRF ----------------
    ("fusion", "rrf denominator k+position+1 -> k+position: off-by-one rank weight",
     [("weight / (k + position + 1)", "weight / (k + position)")]),
    ("fusion", "rrf k+k -> k*k: quadratic rank damping",
     [("weight / (k + position + 1)", "weight / ((k + position) * (k + position))")]),
    ("fusion", "rrf position+1 removed AND k dropped: plain reciprocal of raw index",
     [("weight / (k + position + 1)", "weight / (position + 1)")]),
    ("fusion", "rrf accumulation replaced by overwrite: later ranking wins",
     [("scores[mid] = scores.get(mid, 0.0) + weight / (k + position + 1)",
       "scores[mid] = weight / (k + position + 1)")]),
    ("fusion", "rrf scores all set to 0.0: fusion signal destroyed",
     [("scores[mid] = scores.get(mid, 0.0) + weight / (k + position + 1)",
       "scores[mid] = 0.0")]),
    ("fusion", "rrf enumerate position offset dropped (+1 removal)",
     [("for position, mid in enumerate(ranked):",
       "for position, mid in enumerate(ranked):\n            pass"),
      ("scores[mid] = scores.get(mid, 0.0) + weight / (k + position + 1)",
       "scores[mid] = scores.get(mid, 0.0) + weight / (k + position + 2)")]),
    ("fusion", "rrf default k=60 -> k=1: much steeper rank decay",
     [("def rrf(rankings: list[list[int]], k: int = 60,",
       "def rrf(rankings: list[list[int]], k: int = 1,")]),
    ("fusion", "rrf ignores the second ranking list entirely",
     [("for ranked, weight in zip(rankings, weights):",
       "for ranked, weight in zip(rankings[:1], weights):")]),
    ("graph_leg", "decay multiplication removed: walk does not discount by distance",
     [("heapq.heappush(heap, (-(key * decay), int(neighbour)))",
       "heapq.heappush(heap, (-key, int(neighbour)))")]),
    ("graph_leg", "decay inverted: 1/decay makes deeper nodes STRONGER",
     [("heapq.heappush(heap, (-(key * decay), int(neighbour)))",
       "heapq.heappush(heap, (-(key / decay), int(neighbour)))")]),
    ("graph_leg", "seed exclusion removed: leg re-emits its own seeds",
     [("((node, key) for node, key in best.items() if node not in masses),",
       "((node, key) for node, key in best.items()),")]),
    ("graph_leg", "mass kernel constant: every seed carries identical restart mass",
     [("masses[int(mid)] = 1.0 / (k + position + 1)",
       "masses[int(mid)] = 1.0")]),
    ("graph_leg", "per-seed quota destroyed: one global budget reintroduces hub starvation",
     [("per_seed = max(1, int(budget) // len(masses))",
       "per_seed = max(1, int(budget))")]),
    ("graph_leg", "finalized check removed: higher-key proposal overwrites the true max",
     [("if seen.get(node, -1.0) >= key:",
       "if False:")]),
    ("graph_leg", "id tie-break removed: ties resolved by dict/set order",
     [('key=lambda kv: (-kv[1], kv[0]),',
       'key=lambda kv: -kv[1],')]),
    ("graph_leg", "budget ignored: expansion runs to graph exhaustion",
     [("while heap and pops < per_seed:",
       "while heap:")]),
    ("fusion", "rrf returns empty dict: no fused candidates at all",
     [("scores: dict[int, float] = {}", "return {}")]),

    # ---------------- fusion.py: FTS query builder ----------------
    ("fusion", "regex \\w+ -> \\w: single-char tokens gut recall",
     [('_TOKEN = re.compile(r"\\w+")', '_TOKEN = re.compile(r"\\w")')]),
    ("fusion", "quote-doubling removal: embedded quotes break out of FTS quoting",
     [('\'"%s"\' % token.replace(\'"\', \'""\')', '\'"%s"\' % token')]),
    ("fusion", 'OR join -> AND join: recall collapses to conjunctive match',
     [('" OR ".join(', '" AND ".join(')]),
    ("fusion", "max_terms slice removed: unbounded term count",
     [(_TOKEN_LINE + "[:max_terms]", _TOKEN_LINE)]),
    ("fusion", ".lower() removed: case sensitivity restored",
     [("_TOKEN.findall(_fold(text))", "_TOKEN.findall(text)")]),
    ("fusion", "max_terms default 24 -> 1: only first token searched",
     [("def fts_match_query(text: str, max_terms: int = 24)",
       "def fts_match_query(text: str, max_terms: int = 1)")]),
    ("fusion", "empty text yields [''] token instead of empty expression",
     [(_TOKEN_LINE + "[:max_terms]", _TOKEN_LINE + "[:max_terms] or ['']")]),
]


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _apply(source: str, replacements: list[tuple[str, str]]) -> str | None:
    mutated = source
    for old, new in replacements:
        if old not in mutated:
            return None
        mutated = mutated.replace(old, new, 1)
    return mutated


def _write_exact(p: Path, text: str) -> None:
    # write_text would translate \n -> os.linesep on Windows; the sources are
    # LF-terminated and must stay byte-identical after restore
    p.write_bytes(text.encode("utf-8"))


def _run_suite() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "pytest", *TESTS, "-x", "-q"],
        cwd=REPO, capture_output=True, text=True, timeout=PYTEST_TIMEOUT_S,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="emit JSON to stdout")
    parser.add_argument(
        "--min-score", type=float, default=85.0,
        help="minimum mutation score (percent) required to pass; default 85",
    )
    parser.add_argument(
        "--paths-to-mutate", action="append", choices=("fusion", "scoring"),
        help="restrict to one module; repeatable. default: both",
    )
    parser.add_argument(
        "--fail-on-not-applicable", dest="fail_na", action="store_true",
        default=True,
        help="exit non-zero when an anchor no longer matches the source (default)",
    )
    parser.add_argument(
        "--allow-not-applicable", dest="fail_na", action="store_false",
        help="tolerate stale anchors instead of failing (for triage)",
    )
    args = parser.parse_args()

    selected = args.paths_to_mutate or ["fusion", "scoring", "graph_leg"]
    table = [
        (i, mod, desc, reps)
        for i, (mod, desc, reps) in enumerate(MUTATIONS, start=1)
        if mod in selected
    ]
    targets = {m: p for m, p in (("fusion", FUSION), ("scoring", SCORING),
                       ("graph_leg", GRAPH_LEG))
               if m in selected}

    pristine = {p: p.read_text(encoding="utf-8") for p in targets.values()}
    pristine_sums = {p.name: _sha256(p) for p in pristine}

    baseline = subprocess.run(
        [sys.executable, "-m", "pytest", *TESTS, "-x", "-q"],
        cwd=REPO, capture_output=True, text=True, timeout=300,
    )
    if baseline.returncode != 0:
        print("BASELINE FAILED — aborting mutation run", file=sys.stderr)
        print(baseline.stdout[-2000:], file=sys.stderr)
        return 1
    print(f"baseline: {baseline.stdout.strip().splitlines()[-1]}")

    results: list[dict] = []
    try:
        for mutant_id, module_name, description, replacements in table:
            target = targets[module_name]
            mutated = _apply(pristine[target], replacements)
            if mutated is None:
                results.append({"id": mutant_id, "module": module_name,
                                "description": description,
                                "status": "NOT_APPLICABLE"})
                print(f"[{mutant_id:02d}] NOT_APPLICABLE  {description}")
                continue
            _write_exact(target, mutated)
            start = time.monotonic()
            try:
                proc = _run_suite()
                status = "KILLED" if proc.returncode != 0 else "SURVIVED"
            except subprocess.TimeoutExpired:
                status = "KILLED_TIMEOUT"
            elapsed = time.monotonic() - start
            results.append({"id": mutant_id, "module": module_name,
                            "description": description, "status": status,
                            "seconds": round(elapsed, 1)})
            print(f"[{mutant_id:02d}] {status:<14} {description}")

            # restore immediately so an interrupted run never leaves mutants behind
            for p, text in pristine.items():
                _write_exact(p, text)
    finally:
        for p, text in pristine.items():
            _write_exact(p, text)

    killed = sum(1 for r in results if r["status"].startswith("KILLED"))
    survived = sum(1 for r in results if r["status"] == "SURVIVED")
    na = sum(1 for r in results if r["status"] == "NOT_APPLICABLE")
    equivalent = [r["id"] for r in results if r["id"] in EQUIVALENT]

    # The denominator excludes declared-equivalent mutants: they cannot be
    # killed by any test, so counting them as survivors understates the score
    # and makes the gate unmeetable.
    countable = [r for r in results if r["id"] not in EQUIVALENT]
    countable_killed = sum(1 for r in countable if r["status"].startswith("KILLED"))
    score = 100.0 * countable_killed / len(countable) if countable else 0.0

    print(f"\n{len(results)} mutants: {killed} killed, {survived} survived, "
          f"{na} n/a, {len(equivalent)} declared equivalent")
    print(f"mutation score (equivalent excluded): {score:.1f}% "
          f"({countable_killed}/{len(countable)})")

    after_sums = {p.name: _sha256(p) for p in pristine}
    checksums_ok = after_sums == pristine_sums
    print("checksum verification:",
          "OK - sources byte-identical to originals" if checksums_ok
          else f"MISMATCH {after_sums} vs {pristine_sums}")
    if not checksums_ok:
        return 1

    if args.json:
        print(json.dumps({"results": results,
                          "score": score,
                          "equivalent": {str(k): v for k, v in EQUIVALENT.items()},
                          "pristine_sha256": pristine_sums}, indent=2))

    # --- the gate ---------------------------------------------------------
    failures: list[str] = []
    if na and args.fail_na:
        stale = [r["id"] for r in results if r["status"] == "NOT_APPLICABLE"]
        failures.append(
            f"{na} anchor(s) no longer match the source: {stale}. "
            "The mutation table is stale; update it or the score is not measuring "
            "what it claims to measure."
        )
    if score < args.min_score:
        failures.append(
            f"mutation score {score:.1f}% is below --min-score {args.min_score:.1f}%. "
            f"Unkillable: "
            f"{[r['id'] for r in countable if r['status'] == 'SURVIVED']}"
        )
    if failures:
        print("\nMUTATION GATE FAILED", file=sys.stderr)
        for line in failures:
            print(f"  - {line}", file=sys.stderr)
        return 1
    print("MUTATION GATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
