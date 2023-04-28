from xmen import evaluation
from dummy_linker import CopyLinker, NullLinker
from datasets import concatenate_datasets

import pytest
from hydra import initialize, compose

#with initialize(config_path="../../conf", version_base="1.1"):
#    # config is relative to a module
#    cfg = compose(config_name="config")
#    # dl = DataLoader()
#
#    if "path" in cfg.benchmarks.bronco150:
#        # TODO read BRONCO once integated into HF
#        bronco_ds = None
#        # bronco_ds = dl.load_bronco(cfg.benchmarks.bronco150.path)
#        # bronco_concat = concatenate_datasets(bronco_ds.values())
#    else:
#        bronco_ds = None
#
#NUM_CONCEPTS_BRONCO_150 = 8760
#NUM_SENTENCES_BRONCO_150 = 8976
#
#
#@pytest.mark.skipif(not bronco_ds, reason="Need to provide path for BRONCO in config to run test")
#def test_huggingface_loader():
#    assert len(bronco_concat) == NUM_SENTENCES_BRONCO_150
#
#    concepts = [cui for e in bronco_concat["entities"] for c in e["concepts"] for cui in c["concept_id"]]
#    assert len(concepts) == NUM_CONCEPTS_BRONCO_150
#
#
#@pytest.mark.skipif(not bronco_ds, reason="Need to provide path for BRONCO in config to run test")
#def test_evaluation_identity():
#    pred = CopyLinker().predict_batch(bronco_concat)
#
#    metrics = evaluation.evaluate(bronco_concat, pred, False, top_k_predictions=None)
#
#    for m in ["strict", "loose"]:
#        n_annotations = metrics[m]["n_annos_gold"]
#        n_annotations_system = metrics[m]["n_annos_system"]
#
#        assert n_annotations == NUM_CONCEPTS_BRONCO_150, m
#        assert n_annotations == n_annotations_system, m
#        assert metrics[m]["precision"] == 1.0, m
#        assert metrics[m]["recall"] == 1.0, m
#        assert metrics[m]["fscore"] == 1.0, m
#        assert metrics[m]["fp"] == 0, m
#        assert metrics[m]["fn"] == 0, m
#