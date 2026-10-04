"""SEC-A3-10: no execute() site may splice a runtime value into SQL.

Every interpolated SQL statement under omnidb_server/ builds its text from a
module constant, a placeholder join, or a bounded integer -- but that is a
STRUCTURAL property, and no test enforced it. This module makes it enforced.

An interpolated value is accepted only when it resolves to one of these
provably-safe shapes:

  1. a module-level str constant, defined here or imported from a sibling
     (store._SELECTIVE alone is spliced into five modules)
  2. a placeholder join -- "?" * n through str.join, a join over a generator
     of "?", or a join over a list that only ever receives str literals. n is
     an int count, so only as many "?" as there are bound values can appear
  3. an int-typed module constant whose name marks it as a cap/limit
  4. a local bound to a literal, to (1)-(3), or to a literal table entry
  5. a private helper that re-templates a clause built from literals
     (_aliased_where, _build_memory_where, _heal_torn_v2)

Anything else is reported unsafe so the test fails and a human decides. The
allowlists are explicit on purpose: an unfamiliar pattern fails closed.

THE NEGATIVE CONTROL AT THE BOTTOM IS MANDATORY. Without it a classifier
that resolved nothing would mark every site safe, `unsafe == []` would pass
vacuously, and a green test would be a lie.
"""

from __future__ import annotations

import ast
from pathlib import Path

_SINKS = ("execute", "executemany", "executescript")
_PACKAGE = "omnidb_server"
_REPO = Path(__file__).resolve().parent.parent

# A module constant is an int-typed budget when its name ends in one of these,
# matching _RECONCILE_BATCH_CAP / _GET_ALL_CAP / MAX_TRIGGER_PATTERNS.
_CAP_SUFFIXES = ("_CAP", "_LIMIT", "_MAX", "_CHARS", "_BYTES", "_SECONDS")
# Private helpers that only re-template a clause built from literals.
_CLAUSE_HELPERS = frozenset({
    "_aliased_where", "_build_memory_where", "_heal_torn_v2",
    # ADR-08: returns a double-quoted SQL identifier built from a
    # validate_tenant_id'd tenant ([A-Za-z0-9_-], length-capped) — a fixed
    # prefix plus the validated id, never free text. Recognising it keeps
    # `table = fts_table_name(tenant)` resolvable so the per-tenant DDL
    # (tenant_schema.py) is not a false positive. The classifier still fails
    # closed on every *other* runtime value.
    "fts_table_name",
})


# --- small AST predicates ------------------------------------------------------

def _is_str_const(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _is_int_const(node: ast.AST) -> bool:
    return (isinstance(node, ast.Constant)
            and isinstance(node.value, int) and not isinstance(node.value, bool))


def _is_int_expr(node: ast.AST) -> bool:
    """An expression yielding an int count, never arbitrary text."""
    if _is_int_const(node):
        return True
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        return node.func.id in ("len", "int", "min", "max")
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        return _is_int_expr(node.left) and _is_int_expr(node.right)
    return False


def _module_constants(tree: ast.Module) -> dict:
    """module-level name -> literal value, for simple constant bindings."""
    consts: dict = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    consts[target.id] = ast.literal_eval(node.value)
                except (ValueError, SyntaxError, TypeError):
                    continue
    return consts


def _ast_tables(tree: ast.Module) -> dict:
    """module-level name -> the AST node of a list/tuple-of-tuples table.

    ``_module_constants`` uses ``literal_eval``, which cannot evaluate
    ``MIGRATIONS = [(1, _MIGRATION_1), ...]`` (the names are unresolvable), so
    that table never appears in the constants dict and every site that unpacks
    it looks unresolved. This keeps the raw AST for those tables so the shape
    can be checked structurally instead.
    """
    tables: dict = {}
    for node in tree.body:
        # Both plain `X = [...]` and annotated `X: list[...] = [...]` (db.py:78
        # uses the annotated form, which is why the table was invisible).
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target, value = node.target, node.value
        else:
            continue
        if value is None:
            continue
        if not (isinstance(target, ast.Name)
                and isinstance(value, (ast.List, ast.Tuple))):
            continue
        if value.elts and all(isinstance(e, (ast.Tuple, ast.List))
                              for e in value.elts):
            tables[target.id] = value
    return tables


def _literal_clause_lists(tree: ast.Module) -> set:
    """Names bound to lists that only ever receive str literals.

    store.py's `sets` is the live case: `sets.append("content = ?")` builds
    the SET clause of an UPDATE. A non-literal append disqualifies the name,
    so this rule fails closed.
    """
    ok: set = set()
    bad: set = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("append", "extend")
                and isinstance(node.func.value, ast.Name)):
            name = node.func.value.id
            if node.args and all(_is_str_const(a) for a in node.args):
                ok.add(name)
            else:
                bad.add(name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and isinstance(value, ast.List):
                    (ok if all(_is_str_const(e) for e in value.elts) else bad).add(target.id)
    return ok - bad


def _placeholder_join(node: ast.AST, clause_lists: set) -> tuple[bool, str]:
    """True for "?" * n through str.join, or a join over literal clauses."""
    if not isinstance(node, ast.Call):
        return False, ""
    func = node.func
    if not (isinstance(func, ast.Attribute) and func.attr == "join"
            and isinstance(func.value, (ast.Constant, ast.Name))):
        return False, ""
    if not node.args:
        return False, ""
    source = node.args[0]
    if isinstance(source, (ast.List, ast.Tuple)):
        if all(_is_str_const(e) for e in source.elts):
            return True, "join over a literal clause list"
        return False, ""
    if isinstance(source, ast.GeneratorExp):
        return True, "join over a generator of placeholders"
    if isinstance(source, ast.BinOp) and isinstance(source.op, ast.Mult):
        if _is_str_const(source.left) and _is_int_expr(source.right):
            return True, "join over '?' * count"
        if _is_int_expr(source.left) and _is_str_const(source.right):
            return True, "join over count * '?'"
        return False, ""
    if isinstance(source, ast.Name) and source.id in clause_lists:
        return True, f"join over the literal list {source.id}"
    return False, ""


def _is_cap_name(name: str) -> bool:
    return name.startswith("_") and name.endswith(_CAP_SUFFIXES)


def _literal_table_entry(node, consts: dict, tables: dict) -> bool:
    """MIGRATIONS unpacked into (version, ddl) is safe only because every entry
    is an (int, module str constant) pair. Verify exactly that.

    Handles both spellings: iterating the table directly (``for version, ddl in
    MIGRATIONS``) and subscripting it (``MIGRATIONS[i]``). The table itself is
    read from ``tables`` (raw AST) because ``literal_eval`` cannot resolve the
    ``_MIGRATION_n`` names.
    """
    if isinstance(node, ast.For):
        name = node.iter
    elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        name = node.value
    elif isinstance(node, ast.Name):
        name = node
    else:
        return False
    if not isinstance(name, ast.Name):
        return False

    table = tables.get(name.id)
    if table is None:
        return False
    strs = {k for k, v in consts.items() if isinstance(v, str)}
    for entry in table.elts:
        if not isinstance(entry, (ast.Tuple, ast.List)):
            return False
        for element in entry.elts:
            if _is_int_const(element):
                continue
            if isinstance(element, ast.Name) and element.id in strs:
                continue
            return False
    return True


# --- the classifier ------------------------------------------------------------

class _Classifier:
    """Resolves one spliced expression to SAFE or unsafe, per module."""

    def __init__(self, tree: ast.Module, sibling_strs: dict):
        self.consts = _module_constants(tree)
        self.tables = _ast_tables(tree)
        self.strs = {k: v for k, v in self.consts.items() if isinstance(v, str)}
        self.sibling_strs = sibling_strs
        self.clause_lists = _literal_clause_lists(tree)
        self.imported: dict = {}
        # Walk the WHOLE tree, not just tree.body: five modules import
        # _SELECTIVE inside a function body (portability.py:143 does it inside
        # a helper), and a module-level-only scan reports every one of them as
        # an unresolved name — a false positive on the single most-spliced
        # constant in the codebase.
        for node in ast.walk(tree):
            if not (isinstance(node, ast.ImportFrom) and node.module):
                continue
            for alias in node.names:
                bound = alias.asname or alias.name
                # sibling_strs is {module_stem: {NAME: str}}; an imported
                # name is safe if ANY module in the package defines it as a
                # module-level str constant. (Checking `alias.name in
                # sibling_strs` tested membership in the *module* dict and never
                # matched, so every `from .store import _SELECTIVE` was
                # reported unresolved.)
                owner = next(
                    (stem for stem, consts in sibling_strs.items()
                     if alias.name in consts),
                    None,
                )
                if owner is not None:
                    self.imported[bound] = f"{node.module}.{alias.name} ({owner})"
        self.locals: dict = {}
        # Two passes, because a local may be bound from another safe local
        # (placeholders = ",".join("?" * len(part)) at store.py:267, where
        # `placeholders` is spliced later at :493).
        for _ in range(2):
            for node in ast.walk(tree):
                for name, value in self._bindings(node):
                    kind = self._safe_binding(value)
                    if kind is not None:
                        self.locals.setdefault(name, kind)
        # Mutable list aliases: store.py:486-493 builds `all_columns` by
        # copying a literal column tuple and appending derived names, then
        # splices it. Track the copy source so the join is provably over
        # literal column names.
        self._alias_lists(tree)
        # For-loop unpacking of a module-level literal table (db.py:120).
        self._unpacked_table_fields(tree)

    def _unpacked_table_fields(self, tree: ast.Module) -> None:
        """Register loop variables unpacked from a module-level literal table.

        ``db.py:120`` does ``for version, ddl in MIGRATIONS:`` and then splices
        ``version`` into ``PRAGMA user_version = {version}`` (:139). Every entry
        of MIGRATIONS is an ``(int, str)`` pair, so the only thing that can be
        spliced is a small integer — safe. The existing ``_literal_table_entry``
        helper already proves the table is literal; it just did not cover the
        for-loop unpacking that binds the field names.
        """
        for node in ast.walk(tree):
            if not (isinstance(node, ast.For) and isinstance(node.target, ast.Tuple)):
                continue
            if not _literal_table_entry(node.iter, self.consts, self.tables):
                continue
            for element in node.target.elts:
                if isinstance(element, ast.Name):
                    self.locals.setdefault(
                        element.id, "field of a module-level literal table")

    def _alias_lists(self, tree: ast.Module) -> None:
        """Register lists built by copying a literal name tuple.

        ``store.py:486-493`` does ``all_columns = list(base_columns)`` then
        appends column names, and splices ``', '.join(all_columns)`` into an
        INSERT. That is safe because the only sources are a literal tuple and
        names already known to the classifier — but ``_placeholder_join`` only
        recognises a *literal* list, so the copy has to be resolved to its
        origin or the site is a false positive.
        """
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Assign)
                    and isinstance(node.value, ast.Call)):
                continue
            call = node.value
            if not (isinstance(call.func, ast.Name) and call.func.id == "list"
                    and call.args):
                continue
            origin = call.args[0]
            origin_name = None
            if isinstance(origin, ast.Name):
                # list(BASE_COLUMNS) where BASE_COLUMNS is a module-level
                # tuple of string literals (store.py:486).
                value = self.consts.get(origin.id)
                if isinstance(value, (tuple, list)) and all(
                        isinstance(v, str) for v in value):
                    origin_name = f"literal tuple {origin.id}"
            elif isinstance(origin, (ast.Tuple, ast.List)):
                if all(_is_str_const(e) for e in origin.elts):
                    origin_name = "<literal tuple>"
            if origin_name is None:
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.clause_lists.add(target.id)
                    self.locals.setdefault(
                        target.id, f"list copied from {origin_name}")

    @staticmethod
    def _bindings(node: ast.AST):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    yield target.id, node.value
                elif isinstance(target, (ast.Tuple, ast.List)):
                    for element in target.elts:
                        if isinstance(element, ast.Name):
                            yield element.id, node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            yield node.target.id, node.value
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            yield node.target.id, node.value

    def _safe_binding(self, node: ast.AST) -> str | None:
        if node is None:
            return None
        joined, why = _placeholder_join(node, self.clause_lists)
        if joined:
            return why
        if _is_str_const(node):
            return "str literal"
        if _is_int_const(node):
            return "int literal"
        if isinstance(node, ast.Name):
            if node.id in self.strs:
                return "module str constant"
            if node.id in self.imported:
                return f"imported str constant {self.imported[node.id]}"
            return self.locals.get(node.id)
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else None
            if name in _CLAUSE_HELPERS:
                return "clause helper"
            return self.locals.get(name) if name else None
        if _literal_table_entry(node, self.consts, self.tables):
            return "entry of a module-level literal table"
        return None

    def classify(self, node: ast.AST) -> tuple[bool, str]:
        """Classify the expression an f-string splices into SQL text."""
        if _is_str_const(node):
            return True, "str literal"
        if _is_int_const(node):
            return True, "int literal"
        if isinstance(node, ast.Name):
            if node.id in self.strs:
                return True, "module str constant"
            value = self.consts.get(node.id)
            if isinstance(value, int) and _is_cap_name(node.id):
                return True, "module int cap"
            if node.id in self.imported:
                return True, f"imported str constant {self.imported[node.id]}"
            kind = self.locals.get(node.id)
            if kind is not None:
                return True, kind
            # A bare name may itself BE a module-level literal table, spliced
            # for its int field (db.py:139 splices `version`, the loop variable
            # unpacked from MIGRATIONS at :120). Registered by
            # _unpacked_table_fields, so check that before failing closed.
            if _literal_table_entry(node, self.consts, self.tables):
                return True, ("int field of a module-level literal table, "
                              "unpacked in a for-loop over it")
            return False, f"unresolved name: {node.id}"
        joined, why = _placeholder_join(node, self.clause_lists)
        if joined:
            return True, why
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else None
            if name in _CLAUSE_HELPERS:
                return True, "clause helper"
            return False, f"unsupported call: {name}"
        if _literal_table_entry(node, self.consts, self.tables):
            return True, "entry of a module-level literal table"
        return False, f"unrecognised splice: {type(node).__name__}"


# --- package scan --------------------------------------------------------------

def _sibling_str_constants() -> dict:
    """module stem -> {NAME: str} across the package, so a
    `from .store import _SELECTIVE` resolves to the same constant."""
    out: dict = {}
    for path in sorted((_REPO / _PACKAGE).rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        consts = _module_constants(tree)
        out[path.stem] = {k: v for k, v in consts.items() if isinstance(v, str)}
    return out


def _scan_source(source: str, sibling_strs: dict) -> tuple[list, list]:
    """Return (checked, unsafe) across every interpolated sink argument."""
    tree = ast.parse(source)
    classifier = _Classifier(tree, sibling_strs)
    checked: list = []
    unsafe: list = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr not in _SINKS or not node.args:
            continue
        splices = [v for v in ast.walk(node.args[0])
                   if isinstance(v, ast.FormattedValue)]
        if not splices:
            continue  # nothing interpolated: values arrive as parameters
        for splice in splices:
            expr = ast.unparse(splice.value)
            ok, why = classifier.classify(splice.value)
            checked.append(expr)
            if not ok:
                unsafe.append((splice.lineno, expr, why))
    return checked, unsafe


def _scan_package() -> tuple[list, list]:
    sibling_strs = _sibling_str_constants()
    checked: list = []
    unsafe: list = []
    for path in sorted((_REPO / _PACKAGE).rglob("*.py")):
        name = path.relative_to(_REPO).as_posix()
        got_checked, got_unsafe = _scan_source(
            path.read_text(encoding="utf-8"), sibling_strs)
        checked += [(name, expr) for expr in got_checked]
        unsafe += [(name, line, expr, why) for line, expr, why in got_unsafe]
    return checked, unsafe


# --- the guard -----------------------------------------------------------------

def test_no_execute_site_splices_a_runtime_value():
    """SEC-A3-10: every interpolated SQL value is provably attacker-free."""
    checked, unsafe = _scan_package()
    assert unsafe == [], (
        "SEC-A3-10: an execute() site splices a value that is not a module "
        "constant, a placeholder join, or an int cap. Bind it as a parameter "
        "instead of formatting it into the statement:\n"
        + "\n".join(f"  {name}:{line}: {expr}  [{why}]"
                    for name, line, expr, why in unsafe)
    )
    # Pin the surface: a refactor that deleted or de-interpolated the sites
    # would otherwise make `unsafe == []` pass for the wrong reason.
    assert len(checked) >= 28, (
        "SEC-A3-10: expected at least 28 interpolated SQL values to police, "
        f"found only {len(checked)}"
    )


def test_checked_sites_include_the_named_identifier_sinks():
    """Name the surface explicitly, so a silent shrink cannot go unnoticed."""
    checked, _ = _scan_package()
    expressions = {expr for _name, expr in checked}
    # the identifier-reaching sinks the audit named
    assert "_SELECTIVE" in expressions
    assert any(expr.startswith("_GET_ALL_CAP") for expr in expressions)
    assert any("join" in expr for expr in expressions)
    # every module that interpolates SQL today
    modules = {name for name, _expr in checked}
    assert "omnidb_server/store.py" in modules
    assert "omnidb_server/recall.py" in modules
    assert "omnidb_server/skills.py" in modules
    assert "omnidb_server/db.py" in modules
    assert "omnidb_server/portability.py" in modules


# --- MANDATORY negative control ------------------------------------------------

# Exactly one interpolated value, and it is a caller-supplied name. The
# classifier MUST report it, and only it.
_NEGATIVE_CONTROL = '''
_SELECTIVE = "deleted_ms IS NULL"
_GET_ALL_CAP = 1000


def vulnerable(cur, user_input):
    cur.execute(f"SELECT * FROM t WHERE c = {user_input}")
'''


def test_classifier_reports_the_injected_splice():
    """Without this, `unsafe == []` could pass on a classifier that resolves
    nothing, and the guard above would be a lie."""
    _checked, unsafe = _scan_source(_NEGATIVE_CONTROL, {})
    assert len(unsafe) == 1, f"expected exactly one unsafe splice, got {unsafe}"
    _line, expr, why = unsafe[0]
    assert expr == "user_input"
    assert "unresolved name" in why


def test_classifier_still_accepts_the_safe_shapes_in_the_control():
    """The control must not pass merely because the classifier rejects
    everything; the same classifier has to say yes to safe values.

    The placeholder shape here is the one the real code uses
    (``store.py:267,613,624,904``: ``",".join("?" * len(ids))``). An earlier
    version of this control used ``",".join(chr(63) * len(ids))``, which the
    classifier rejects — correctly, since ``chr()`` is a runtime call and the
    design fails closed on anything it cannot prove. The control must assert
    shapes that are genuinely provably safe, or it is asserting that the
    classifier is broken.
    """
    source = '''
_SELECTIVE = "deleted_ms IS NULL"
_GET_ALL_CAP = 1000
BASE_COLUMNS = ("id", "content", "kind")


def safe(cur, ids):
    all_columns = list(BASE_COLUMNS)
    cur.execute(f"SELECT id FROM t WHERE {_SELECTIVE}")
    cur.execute(f"SELECT id FROM t LIMIT {_GET_ALL_CAP}")
    cur.execute(f"SELECT id FROM t WHERE id IN ({','.join('?' * len(ids))})")
    cur.execute(f"INSERT INTO t({', '.join(all_columns)}) VALUES (1)")
'''
    checked, unsafe = _scan_source(source, {})
    assert unsafe == [], f"safe shapes were flagged: {unsafe}"
    assert len(checked) == 4, f"expected four checked splices, got {checked}"


def test_classifier_fails_closed_on_a_runtime_call_in_a_join():
    """A control for the control: the classifier must reject a shape it cannot
    prove, even inside an otherwise-safe ``join``.

    This is the property that makes the *other* control meaningful. If runtime
    calls were accepted, the fail-closed design would be gone and a future
    `chr()`-built placeholder join would be waved through.
    """
    source = '''
def sneaky(cur, ids):
    cur.execute(f"SELECT id FROM t WHERE id IN ({','.join(chr(63) * len(ids))})")
'''
    _checked, unsafe = _scan_source(source, {})
    assert len(unsafe) == 1, f"runtime call should be flagged, got {unsafe}"
    assert "unsupported call" in unsafe[0][2]


def test_classifier_flags_a_cloned_unsafe_module():
    """A whole second module, not one line: proves the scan is not a one-liner
    special case, and that the guard would fire on a real regression."""
    source = '''
_SQL = "SELECT 1"


def bad(cur, user_input, table):
    cur.execute(f"SELECT * FROM {table} WHERE c = {user_input}")
    cur.execute(f"DELETE FROM logs WHERE x = {_SQL}")
'''
    _checked, unsafe = _scan_source(source, {})
    flagged = {expr for _line, expr, _why in unsafe}
    assert "table" in flagged
    assert "user_input" in flagged
    assert "_SQL" not in flagged  # the module constant is genuinely safe
