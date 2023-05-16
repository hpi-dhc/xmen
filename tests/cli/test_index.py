from pathlib import Path
import os
from omegaconf import OmegaConf
import pickle
from xmen.knowledge_base import create_flat_term_dict
from xmen.cli.index import *
from xmen.linkers.faiss_indexer import *

file_path = Path(os.path.dirname(os.path.realpath(__file__)))
cfg = OmegaConf.load(file_path / "dummy_index" / "index.yaml")
dict_path = file_path / "dummy_index" / "test.jsonl"


def test_term_dict():
    term_dict = create_flat_term_dict([dict_path])
    assert len(term_dict) == 22
    assert len(term_dict.cui.unique()) == 11
    pass


def test_ngram_indices(tmp_path):
    if not os.path.exists(tmp_path):
        tmp_path.mkdir()
    build_ngram(cfg, tmp_path, dict_path)
    assert len(os.listdir(tmp_path / "index" / "ngrams")) == 5

    candiate_gen = TFIDFNGramLinker.load_candidate_generator(tmp_path / "index" / "ngrams")
    assert len(candiate_gen.kb.alias_to_cuis) == 22
    assert len(candiate_gen.kb.cui_to_entity) == 11
    pass


def test_sapbert_indices(tmp_path):
    sapbert_folder = "sapbert"

    if not os.path.exists(tmp_path):
        print(tmp_path)
        tmp_path.mkdir()
    build_sapbert(cfg, tmp_path, dict_path, -1)
    assert len(os.listdir(tmp_path / "index" / sapbert_folder)) == 2

    with open(tmp_path / "index" / f"{sapbert_folder}" / "dict.pickle", "rb") as f:
        df = pickle.load(f)
        assert len(df) == 22
        assert len(df["cui"].unique()) == 11

    indexer = DenseHNSWFlatIndexer(768)
    index_file = str(tmp_path / "index" / f"{sapbert_folder}" / "embed_faiss_hier.pickle")
    indexer.deserialize_from(index_file)
    assert indexer.index.ntotal == 22
