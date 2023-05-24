from xmen.data import load_mantra_gsc, get_cuis, ConceptMerger
from xmen import evaluation
from dummy_linker import CopyLinker, NullLinker

import pytest

mantra_ds_raw = None  # load_mantra_gsc()
mantra_ds = None  # ConceptMerger().transform_batch(mantra_ds_raw)

# "The number of final annotations was 5530" (but there are two duplicates)
NUM_CONCEPTS_MANTRA_GSC = 5530 - 2

ALL_METRICS = ["strict", "partial", "loose"]


@pytest.mark.skip(
    reason="Mantra currently not on Huggingface? see: https://github.com/bigscience-workshop/biomedical/issues/891"
)
def test_stats():
    assert len(mantra_ds_raw["train"]) == 1450
    assert len(get_cuis(mantra_ds_raw["train"])) == 5530
    assert len(get_cuis(mantra_ds["train"])) == NUM_CONCEPTS_MANTRA_GSC


@pytest.mark.skip(
    reason="Mantra currently not on Huggingface? see: https://github.com/bigscience-workshop/biomedical/issues/891"
)
def test_evaluation_identity():
    pred = CopyLinker().predict_batch(mantra_ds["train"])

    metrics = evaluation.evaluate(mantra_ds["train"], pred, allow_multiple_gold_candidates=False, metrics=ALL_METRICS)

    for m in ["strict", "partial", "loose"]:
        n_annotations = metrics[m]["n_annos_gold"]

        assert n_annotations == NUM_CONCEPTS_MANTRA_GSC, m
        assert metrics[m]["precision"] == 1.0, m
        assert metrics[m]["recall"] == 1.0, m
        assert metrics[m]["fscore"] == 1.0, m
        assert metrics[m]["fp"] == 0, m
        assert metrics[m]["fn"] == 0, m

    assert metrics["strict"]["ptp"] == n_annotations
    assert metrics["strict"]["rtp"] == n_annotations

    assert metrics["partial"]["ptp"] == n_annotations
    assert metrics["partial"]["rtp"] == n_annotations

    assert metrics["loose"]["ptp"] < n_annotations
    assert metrics["loose"]["rtp"] < n_annotations


@pytest.mark.skip(
    reason="Mantra currently not on Huggingface? see: https://github.com/bigscience-workshop/biomedical/issues/891"
)
def test_evaluation_null_allow_multiple_gold_candidates():
    pred = NullLinker().predict_batch(mantra_ds["train"])

    metrics = evaluation.evaluate(mantra_ds["train"], pred, allow_multiple_gold_candidates=True, metrics=ALL_METRICS)

    n_annotations_system = metrics["strict"]["n_annos_system"]
    n_annotations_gold = metrics["strict"]["n_annos_gold"]

    assert n_annotations_system == 0
    assert n_annotations_gold == NUM_CONCEPTS_MANTRA_GSC

    for m in ALL_METRICS:
        assert metrics[m]["precision"] == 0.0
        assert metrics[m]["recall"] == 0.0
        assert metrics[m]["fscore"] == 0.0
        assert metrics[m]["ptp"] == 0
        assert metrics[m]["fp"] == 0
        assert metrics[m]["rtp"] == 0

    assert metrics["strict"]["fn"] < n_annotations_gold
    assert metrics["partial"]["fn"] < n_annotations_gold
    assert metrics["loose"]["fn"] < n_annotations_gold


@pytest.mark.skip(
    reason="Mantra currently not on Huggingface? see: https://github.com/bigscience-workshop/biomedical/issues/891"
)
def test_evaluation_null_dont_allow_multiple_gold_candidates():
    pred = NullLinker().predict_batch(mantra_ds["train"])

    metrics = evaluation.evaluate(mantra_ds["train"], pred, allow_multiple_gold_candidates=False, metrics=ALL_METRICS)

    n_annotations_system = metrics["strict"]["n_annos_system"]
    n_annotations_gold = metrics["strict"]["n_annos_gold"]

    assert n_annotations_system == 0
    assert n_annotations_gold == NUM_CONCEPTS_MANTRA_GSC

    for m in ["strict", "partial", "loose"]:
        assert metrics[m]["precision"] == 0.0
        assert metrics[m]["recall"] == 0.0
        assert metrics[m]["fscore"] == 0.0
        assert metrics[m]["ptp"] == 0
        assert metrics[m]["fp"] == 0
        assert metrics[m]["rtp"] == 0

    assert metrics["strict"]["fn"] == NUM_CONCEPTS_MANTRA_GSC
    assert metrics["partial"]["fn"] == NUM_CONCEPTS_MANTRA_GSC
    assert metrics["loose"]["fn"] < NUM_CONCEPTS_MANTRA_GSC
