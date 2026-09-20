


def _chain(store):
    n3 = store.remember("node three")["id"]
    n2 = store.remember("node two", links=[{"to": n3, "label": "calls"}])["id"]
    n1 = store.remember("node one", links=[{"to": n2, "label": "calls"}])["id"]
    return n1, n2, n3


def test_related_two_hops(store):
    n1, n2, n3 = _chain(store)
    got = {r["id"]: r["depth"] for r in store.related(n1, max_hops=2)}
    assert got[n2] == 1
    assert got[n3] == 2


def test_related_hop_limit(store):
    n1, n2, n3 = _chain(store)
    got = {r["id"] for r in store.related(n1, max_hops=1)}
    assert n2 in got
    assert n3 not in got


def test_related_label_filter(store):
    n1, n2, _ = _chain(store)
    assert store.related(n1, max_hops=2, label="imports") == []
    got = {r["id"] for r in store.related(n1, max_hops=2, label="calls")}
    assert n2 in got


def test_related_undirected(store):
    n1, _, n3 = _chain(store)
    ids = {r["id"] for r in store.related(n3, max_hops=2)}
    assert n1 in ids  # edges traverse both directions
