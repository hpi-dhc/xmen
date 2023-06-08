from benchmarks.dataloaders import load_distemist_linking
from xmen.data import get_cuis

ds = load_distemist_linking()


def test_stats():
    assert len(ds["train"]) == 583
    cuis = get_cuis(ds["train"])
    assert len(cuis) == 5374
