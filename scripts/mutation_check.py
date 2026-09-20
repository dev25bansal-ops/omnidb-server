"""Hand-rolled mutation analysis for the pure ranking modules.

mutmut 2.5.1 crashes on Python 3.14 (cp1252 banner encode + "cannot pickle
itertools.count" in its trampoline machinery) and mutmut 3.x requires
[tool.mutmut] config in files that are not ours to edit, so this script is the
fallback: it applies a fixed table of hand-written mutations to fusion.py and
scoring.py one at a time, runs the scoped pytest suite against each mutant,
records KILLED/SURVIVED, and restores the originals in a finally block so no
mutated file is ever left behind.

Usage: python scripts/mutation_check.py [--json]
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

TESTS = ["tests/test_scoring_properties.py", "tests/test_memory.py"]
PYTEST_TIMEOUT_S = 60

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
     [("+ 1.0 / (k + position + 1)", "+ 1.0 / (k + position)")]),
    ("fusion", "rrf k+k -> k*k: quadratic rank damping",
     [("+ 1.0 / (k + position + 1)", "+ 1.0 / ((k + position) * (k + position))")]),
    ("fusion", "rrf position+1 removed AND k dropped: plain reciprocal of raw index",
     [("+ 1.0 / (k + position + 1)", "+ 1.0 / (position + 1)")]),
    ("fusion", "rrf accumulation replaced by overwrite: later ranking wins",
     [("scores[mid] = scores.get(mid, 0.0) + 1.0 / (k + position + 1)",
       "scores[mid] = 1.0 / (k + position + 1)")]),
    ("fusion", "rrf scores all set to 0.0: fusion signal destroyed",
     [("scores[mid] = scores.get(mid, 0.0) + 1.0 / (k + position + 1)",
       "scores[mid] = 0.0")]),
    ("fusion", "enumerate position offset dropped (+1 removal)",
     [("for position, mid in enumerate(ranked):",
       "for position, mid in enumerate(ranked):\n            pass"),
      ("scores[mid] = scores.get(mid, 0.0) + 1.0 / (k + position + 1)",
       "scores[mid] = scores.get(mid, 0.0) + 1.0 / (k + position + 2)")]),
    ("fusion", "rrf default k=60 -> k=1: much steeper rank decay",
     [("def rrf(rankings: list[list[int]], k: int = 60)", "def rrf(rankings: list[list[int]], k: int = 1)")]),
    ("fusion", "rrf ignores the second ranking list entirely",
     [("for ranked in rankings:", "for ranked in rankings[:1]:")]),
    ("fusion", "rrf returns empty dict: no fused candidates at all",
     [("scores: dict[int, float] = {}", "pass\n    return {}")]),

    # ---------------- fusion.py: FTS query builder ----------------
    ("fusion", "regex \\w+ -> \\w: single-char tokens gut recall",
     [('_TOKEN = re.compile(r"\\w+")', '_TOKEN = re.compile(r"\\w")')]),
    ("fusion", "quote-doubling removal: embedded quotes break out of FTS quoting",
     [('\'"%s"\' % token.replace(\'"\', \'""\')', '\'"%s"\' % token')]),
    ("fusion", 'OR join -> AND join: recall collapses to conjunctive match',
     [('" OR ".join(', '" AND ".join(')]),
    ("fusion", "max_terms slice removed: unbounded term count",
     [("tokens = _TOKEN.findall(text.lower())[:max_terms]",
       "tokens = _TOKEN.findall(text.lower())")]),
    ("fusion", ".lower() removed: case sensitivity restored",
     [("_TOKEN.findall(text.lower())", "_TOKEN.findall(text)")]),
    ("fusion", "max_terms default 24 -> 1: only first token searched",
     [("def fts_match_query(text: str, max_terms: int = 24)",
       "def fts_match_query(text: str, max_terms: int = 1)")]),
    ("fusion", "empty text yields [''] token instead of empty expression",
     [("tokens = _TOKEN.findall(text.lower())[:max_terms]",
       "tokens = _TOKEN.findall(text.lower())[:max_terms] or ['']")]),
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="emit JSON to stdout")
    args = parser.parse_args()

    pristine = {p: p.read_text(encoding="utf-8") for p in (FUSION, SCORING)}
    pristine_sums = {p.name: _sha256(p) for p in pristine}

    baseline = subprocess.run(
        [sys.executable, "-m", "pytest", *TESTS, "-x", "-q"],
        cwd=REPO, capture_output=True, text=True, timeout=120,
    )
    if baseline.returncode != 0:
        print("BASELINE FAILED — aborting mutation run", file=sys.stderr)
        print(baseline.stdout[-2000:], file=sys.stderr)
        return 1
    print(f"baseline: {baseline.stdout.strip().splitlines()[-1]}")

    results: list[dict] = []
    try:
        for module_name, description, replacements in MUTATIONS:
            target = FUSION if module_name == "fusion" else SCORING
            mutated = _apply(pristine[target], replacements)
            if mutated is None:
                results.append({"id": len(results) + 1, "module": module_name,
                                "description": description,
                                "status": "NOT_APPLICABLE"})
                print(f"[{len(results):02d}] NOT_APPLICABLE  {description}")
                continue
            _write_exact(target, mutated)
            start = time.monotonic()
            try:
                proc = subprocess.run(
                    [sys.executable, "-m", "pytest", *TESTS, "-x", "-q"],
                    cwd=REPO, capture_output=True, text=True, timeout=PYTEST_TIMEOUT_S,
                )
                elapsed = time.monotonic() - start
                status = "KILLED" if proc.returncode != 0 else "SURVIVED"
            except subprocess.TimeoutExpired:
                status = "KILLED_TIMEOUT"
                elapsed = time.monotonic() - start
            results.append({"id": len(results) + 1, "module": module_name,
                            "description": description, "status": status,
                            "seconds": round(elapsed, 1)})
            print(f"[{len(results):02d}] {status:<14} {description}")

            # restore immediately so an interrupted run never leaves mutants behind
            for p, text in pristine.items():
                _write_exact(p, text)
    finally:
        for p, text in pristine.items():
            _write_exact(p, text)

    killed = sum(1 for r in results if r["status"].startswith("KILLED"))
    survived = sum(1 for r in results if r["status"] == "SURVIVED")
    na = sum(1 for r in results if r["status"] == "NOT_APPLICABLE")
    print(f"\n{len(results)} mutants: {killed} killed, {survived} survived, {na} n/a")

    after_sums = {p.name: _sha256(p) for p in pristine}
    checksums_ok = after_sums == pristine_sums
    print("checksum verification:",
          "OK — sources byte-identical to originals" if checksums_ok
          else f"MISMATCH {after_sums} vs {pristine_sums}")
    if not checksums_ok:
        return 1

    if args.json:
        print(json.dumps({"results": results,
                          "pristine_sha256": pristine_sums}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
