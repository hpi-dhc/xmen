from omegaconf import OmegaConf
from pathlib import Path
import os
import sys

sys.path.append("..")
from xmen.cli.dict import get_concept_details

file_path = Path(os.path.dirname(os.path.realpath(__file__)))
data_dir = file_path / "dummy_dict"


# CUSTOM DICT TEST (1)
def test_custom_dict():
    # read the config
    cfg_path = data_dir / "test_custom.yaml"
    cfg = OmegaConf.load(cfg_path)

    # load get_concept_details from a custom code script
    script_path = data_dir / "custom_code.py"

    cfg.dict.custom.dic_path = str(file_path / cfg.dict.custom.dic_path)

    # get concept_details from dummy_data for given config and assert numbers
    concept_details = get_concept_details(cfg, script_path)

    assert len(concept_details) == 2

    assert len(concept_details["C0001"]["aliases"]) == 0
    assert len(concept_details["C0001"]["types"]) == 1

    assert len(concept_details["C0002"]["aliases"]) == 0
    assert len(concept_details["C0002"]["types"]) == 1

    # modify SEMANTIC GROUPS and LANGUAGE and assert changes
    cfg.dict.custom.lang.append("ger")
    cfg.dict.custom.sem.append("diso")

    concept_details = get_concept_details(cfg, script_path)

    assert len(concept_details) == 3

    assert len(concept_details["C0001"]["aliases"]) == 0
    assert len(concept_details["C0001"]["types"]) == 1

    assert len(concept_details["C0002"]["aliases"]) == 1
    assert len(concept_details["C0002"]["types"]) == 1

    assert len(concept_details["C0001"]["aliases"]) == 0
    assert len(concept_details["C0001"]["types"]) == 1


# UMLS DICT TESTs (5)
def test_lang():
    yaml = data_dir / "test_lang.yaml"
    cfg = OmegaConf.load(yaml)
    cfg.dict.umls.meta_path = str(file_path / cfg.dict.umls.meta_path)

    concept_details = get_concept_details(cfg, custom_path=None)

    assert len(concept_details) == 1
    assert len(concept_details["C0001361"]["aliases"]) == 0
    assert len(concept_details["C0001361"]["types"]) == 1

    cfg.dict.lang = "FRE"  # change language and assert changes

    concept_details = get_concept_details(cfg, custom_path=None)

    assert len(concept_details) == 1
    assert len(concept_details["C0001361"]["aliases"]) == 0
    assert len(concept_details["C0001361"]["types"]) == 1


def test_sem_groups():
    yaml = data_dir / "test_sem_groups.yaml"
    cfg = OmegaConf.load(yaml)
    cfg.dict.umls.meta_path = str(file_path / cfg.dict.umls.meta_path)

    concept_details = get_concept_details(cfg, custom_path=None)

    assert len(concept_details) == 1
    assert len(concept_details["C0000005"]["aliases"]) == 1
    assert len(concept_details["C0000005"]["types"]) == 3

    cfg.dict.umls.semantic_groups[0] = "DISO"  # substitute sem group and assert changes

    concept_details = get_concept_details(cfg, custom_path=None)

    assert len(concept_details) == 1
    assert len(concept_details["C0001361"]["aliases"]) == 4
    assert len(concept_details["C0001361"]["types"]) == 1


def test_sabs():
    yaml = data_dir / "test_sabs.yaml"
    cfg = OmegaConf.load(yaml)
    cfg.dict.umls.meta_path = str(file_path / cfg.dict.umls.meta_path)

    concept_details = get_concept_details(cfg, custom_path=None)

    assert len(concept_details) == 1
    assert len(concept_details["C0000005"]["aliases"]) == 1
    assert len(concept_details["C0000005"]["types"]) == 3

    cfg.dict.umls.sabs.append("ICPCGER")  # append new SAB and assert changes

    concept_details = get_concept_details(cfg, custom_path=None)

    assert len(concept_details) == 2

    assert len(concept_details["C0001361"]["aliases"]) == 0
    assert len(concept_details["C0001361"]["types"]) == 1

    assert len(concept_details["C0000005"]["aliases"]) == 1
    assert len(concept_details["C0000005"]["types"]) == 3


def test_non_supressed_only():
    yaml = data_dir / "test_non_supressed_only.yaml"
    cfg = OmegaConf.load(yaml)
    cfg.dict.umls.meta_path = str(file_path / cfg.dict.umls.meta_path)

    concept_details = get_concept_details(cfg, custom_path=None)

    assert len(concept_details) == 2
    assert len(concept_details["C0001361"]["aliases"]) == 4
    assert len(concept_details["C0001361"]["types"]) == 1

    assert len(concept_details["C0000005"]["aliases"]) == 1
    assert len(concept_details["C0000005"]["types"]) == 3

    cfg.dict.umls.non_suppressed_only = False

    concept_details = get_concept_details(cfg, custom_path=None)

    assert len(concept_details) == 2

    assert len(concept_details["C0001361"]["aliases"]) == 4
    assert len(concept_details["C0001361"]["types"]) == 1

    assert len(concept_details["C0000005"]["aliases"]) == 1
    assert len(concept_details["C0000005"]["types"]) == 3


def test_subconfig():
    yaml = data_dir / "test_subconfig.yaml"

    # first subconfig
    cfg = OmegaConf.load(yaml)
    cfg.dict = cfg.dict["key1"]
    cfg.name = "test"
    cfg.dict.umls.meta_path = str(file_path / cfg.dict.umls.meta_path)
    concept_details = get_concept_details(cfg, custom_path=None)

    assert len(concept_details) == 1
    assert len(concept_details["C0001361"]["aliases"]) == 0
    assert len(concept_details["C0001361"]["types"]) == 1

    # first subconfig
    cfg = OmegaConf.load(yaml)
    cfg.dict = cfg.dict["key2"]
    cfg.name = "test"
    cfg.dict.umls.meta_path = str(file_path / cfg.dict.umls.meta_path)
    concept_details = get_concept_details(cfg, custom_path=None)

    assert len(concept_details) == 1
    assert len(concept_details["C0001361"]["aliases"]) == 0
    assert len(concept_details["C0001361"]["types"]) == 1
