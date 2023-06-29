from benchmarks.dataloaders import load_distemist
from xmen.data import get_cuis

ds = load_distemist()[0]


def test_stats():
    assert len(ds["train"]) + len(ds["validation"]) == 583
    cuis = get_cuis(ds["train"]) + get_cuis(ds["validation"])
    assert len(cuis) == 5374

    assert len(ds["test"]) == 250
