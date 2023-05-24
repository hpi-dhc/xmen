from xmen.data import load_medmentions_full, load_medmentions_st21pv, get_cuis, ConceptMerger


def test_medmentions_full_stats():
    ds = load_medmentions_full()
    ds = ConceptMerger().transform_batch(ds)

    assert len(ds["train"]) == 2635
    assert len(ds["validation"]) == 878
    assert len(ds["test"]) == 879

    all_cuis = get_cuis(ds["train"]) + get_cuis(ds["validation"]) + get_cuis(ds["test"])

    assert len(all_cuis) == 352321  # Paper: 352496
    assert len(set(all_cuis)) == 34724


def test_medmentions_st21pv_stats():
    ds = load_medmentions_st21pv()
    ds = ConceptMerger().transform_batch(ds)

    assert len(ds["train"]) == 2635
    assert len(ds["validation"]) == 878
    assert len(ds["test"]) == 879

    assert len(get_cuis(ds["train"])) == 122184  # Paper: 122241
    assert len(get_cuis(ds["validation"])) == 40864  # Paper: 40884
    assert len(get_cuis(ds["test"])) == 40144  # Paper: 40157

    all_cuis = get_cuis(ds["train"]) + get_cuis(ds["validation"]) + get_cuis(ds["test"])

    assert len(all_cuis) == 203192  # 203282
    assert len(set(all_cuis)) == 25419


def test_overlap():
    ds = load_medmentions_st21pv()
    ds = ConceptMerger().transform_batch(ds)

    train_cuis_set = set(get_cuis(ds["train"]))
    valid_cuis_set = set(get_cuis(ds["validation"]))
    test_cuis_set = set(get_cuis(ds["test"]))

    assert len(train_cuis_set) == 18520
    assert len(valid_cuis_set) == 8643
    assert len(test_cuis_set) == 8457

    assert len(valid_cuis_set.intersection(train_cuis_set)) == 4984
    assert len(test_cuis_set.intersection(train_cuis_set)) == 4867
    assert len(test_cuis_set.intersection(train_cuis_set.union(valid_cuis_set))) == 5217
