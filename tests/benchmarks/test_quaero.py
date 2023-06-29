from xmen.data import get_cuis
from benchmarks.dataloaders import load_quaero

quaero_ds = load_quaero()[0]


def test_stats():
    def get_doc_ids(subset):
        return set([d.split("_")[0] for d in subset["document_id"]])

    def count_entities(subset):
        return sum([len(s["entities"]) for s in subset])

    def count_unique_concepts(subset):
        return len(set(get_cuis(subset)))

    emea = quaero_ds.filter(lambda d: d["corpus_id"] == "quaero_emea_bigbio_kb")
    medline = quaero_ds.filter(lambda d: d["corpus_id"] == "quaero_medline_bigbio_kb")

    assert len(get_doc_ids(emea["train"])) == 3
    assert count_entities(emea["train"]) == 2695
    # assert count_unique_concepts(emea["train"]) == 648

    assert len(get_doc_ids(emea["validation"])) == 3
    assert count_entities(emea["validation"]) == 2260
    # assert count_unique_concepts(emea["validation"]) == 523

    assert len(get_doc_ids(emea["test"])) == 4
    assert count_entities(emea["test"]) == 2204
    # assert count_unique_concepts(emea["validation"]) == 474

    assert len(get_doc_ids(medline["train"])) == 833
    assert count_entities(medline["train"]) == 2994
    # assert count_unique_concepts(medline["train"]) == 1860

    assert len(get_doc_ids(medline["validation"])) == 832
    assert count_entities(medline["validation"]) == 2977
    # assert count_unique_concepts(medline["validation"]) == 1848

    assert len(get_doc_ids(medline["test"])) == 833
    assert count_entities(medline["test"]) == 3103
    # assert count_unique_concepts(medline["test"]) == 1909
