import pytest


@pytest.mark.parametrize(
    "items, expected_ids",
    [
        ([{"content": "automatic first"}, {"content": "explicit last", "_explicit_id": 100}], [1, 100]),
        ([{"content": "explicit first", "_explicit_id": 100}, {"content": "automatic last"}], [100, 101]),
        ([{"content": "automatic first"}, {"content": "explicit middle", "_explicit_id": 100}, {"content": "automatic last"}], [1, 100, 101]),
    ],
)
def test_batch_preserves_actual_insert_identity(store, items, expected_ids):
    results = store.remember_many(items)

    assert [result["id"] for result in results] == expected_ids
    for result, item in zip(results, items):
        assert store.get(result["id"])["content"] == item["content"]
    assert store.index.indexed_ids() == set(expected_ids)
