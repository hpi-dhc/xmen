from xmen.data import get_cuis, ConceptMerger
from xmen import evaluation
from dummy_linker import CopyLinker, NullLinker

from benchmarks.dataloaders import load_mantra_gsc

import pytest

# "The number of final annotations was 5530" (but there are two duplicates)
NUM_CONCEPTS_MANTRA_GSC = 5530 - 2

ALL_METRICS = ["strict", "partial", "loose"]


@pytest.mark.skip(
    reason="Mantra GSC not available at biosemantics.org at the moment, need to wait until hosted elsewhere"
)
class TestMantraGSC:
    @pytest.fixture(scope="class")
    def mantra_data(self):
        mantra_ds_raw = load_mantra_gsc()[0]
        mantra_ds = ConceptMerger().transform_batch(mantra_ds_raw)
        return mantra_ds_raw, mantra_ds

    def test_stats(self, mantra_data):
        mantra_ds_raw, mantra_ds = mantra_data
        assert len(mantra_ds_raw["test"]) == 1450
        assert len(get_cuis(mantra_ds_raw["test"])) == 5530
        assert len(get_cuis(mantra_ds["test"])) == NUM_CONCEPTS_MANTRA_GSC

    def test_evaluation_identity(self, mantra_data):
        _, mantra_ds = mantra_data
        pred = CopyLinker().predict_batch(mantra_ds["test"])

        metrics = evaluation.evaluate(
            mantra_ds["test"],
            pred,
            allow_multiple_gold_candidates=False,
            metrics=ALL_METRICS,
            top_k_predictions=None,
        )

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

    def test_evaluation_null_allow_multiple_gold_candidates(self, mantra_data):
        _, mantra_ds = mantra_data
        pred = NullLinker().predict_batch(mantra_ds["test"])

        metrics = evaluation.evaluate(mantra_ds["test"], pred, allow_multiple_gold_candidates=True, metrics=ALL_METRICS)

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

    def test_evaluation_null_dont_allow_multiple_gold_candidates(self, mantra_data):
        _, mantra_ds = mantra_data
        pred = NullLinker().predict_batch(mantra_ds["test"])

        metrics = evaluation.evaluate(
            mantra_ds["test"], pred, allow_multiple_gold_candidates=False, metrics=ALL_METRICS
        )

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
