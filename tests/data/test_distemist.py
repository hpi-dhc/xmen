from xmen.data import load_distemist_linking, get_cuis

ds = load_distemist_linking()


def test_stats():
    assert len(ds["train"]) == 583
    cuis = get_cuis(ds["train"])
    assert len(cuis) == 5374
