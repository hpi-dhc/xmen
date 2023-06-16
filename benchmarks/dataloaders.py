from typing import List, Union
from pathlib import Path
import datasets


def load_dataset(dataset: str):
    """
    Loads a dataset using the appropriate data loader function from the xmen package.

    Args:
    - dataset (str): Name of the dataset to be loaded.

    Returns:
    - The loaded dataset.
    """
    import sys

    loader_fn = getattr(sys.modules[__name__], f"load_{dataset}")
    return loader_fn()


def load_mantra_gsc():
    """
    Loads all subsets of Mantra GSC into one dataset

    TODO: Simplify when Mantra is eventually on the Hugging Face Hub

    Returns:
    - The loaded Mantra-GSC dataset.
    """
    import bigbio

    mantra_path = str(Path(bigbio.__file__).parent / "biodatasets" / "mantra_gsc" / "mantra_gsc.py")
    configs = [c for c in datasets.get_dataset_infos(mantra_path).keys() if "bigbio" in c]

    ds_map = {c: datasets.load_dataset(mantra_path, c) for c in configs}
    ds = []
    for conf, ds_dict in ds_map.items():
        for k in ds_dict.keys():
            ds_dict[k] = ds_dict[k].add_column("corpus_id", [conf] * len(ds_dict[k]))
            ds_dict[k] = ds_dict[k].add_column("lang", [conf.split("_")[2]] * len(ds_dict[k]))
        ds.append(ds_dict)
    output = datasets.dataset_dict.DatasetDict()
    for s in ["train"]:
        output[s] = datasets.concatenate_datasets([d[s] for d in ds])
    return output


def _load_medmentions(config_name):
    """
    Loads the MedMentions dataset using the _load_bigbio_dataset function with the appropriate parameters and applies a
    transformation to drop the 'UMLS:' prefix in the 'db_id' field of the 'normalized' entities.

    Args:
    - config_name (str): The name of the MedMentions configuration file.

    Returns:
    - The loaded and transformed MedMentions dataset.
    """

    def drop_prefix(entities):
        for e in entities:
            for n in e["normalized"]:
                n["db_id"] = n["db_id"].replace("UMLS:", "")
        return entities

    return [ _load_bigbio_dataset(
        [config_name],
        "medmentions",
        lambda _: "en",
        splits=["train", "validation", "test"],
    ).map(lambda d: {"entities": drop_prefix(d["entities"])}) ]


def load_medmentions_full():
    """
    Loads the full MedMentions dataset.

    Returns:
    - A dataset loaded from the MedMentions dataset with full bigbio knowledge base.
    """
    return _load_medmentions("medmentions_full_bigbio_kb")


def load_medmentions_st21pv():
    """
    Loads the MedMentions dataset with ST21PV subset.

    Returns:
    - A dataset loaded from the MedMentions dataset with ST21PV subset and bigbio knowledge base.
    """
    return _load_medmentions("medmentions_st21pv_bigbio_kb")


def load_quaero():
    """
    Loads the Quaero dataset.

    Returns:
    - A dataset loaded from the Quaero dataset with bigbio knowledge base.
    """
    return [ _load_bigbio_dataset(
        ["quaero_emea_bigbio_kb", "quaero_medline_bigbio_kb"],
        "quaero",
        lambda _: "fr",
        splits=["train", "validation", "test"],
    ) ]


def load_distemist():
    """
    Loads the DisTEMIST (EL track) dataset.

    Returns:
    - A dataset loaded from the DisTEMIST Linking dataset with bigbio knowledge base.

    Raises:
    - AssertionError: If the loaded dataset has an unexpected format.
    """
    ds = _load_bigbio_dataset(
        ["distemist_linking_bigbio_kb"],
        "distemist",
        lambda _: "es",
        splits=["train", "test"],
    )

    # Own validation set (20% of distemist training set / EL sub-track)
    with open(Path(__file__).parent / 'benchmark' / 'distemist_validation_docs.txt', 'r') as fh:
        valid_ids = [l.strip() for l in fh.readlines()]
        
    ds_train = ds['train'].filter(lambda d: d['document_id'] not in valid_ids)
    ds_valid = ds['train'].filter(lambda d: d['document_id'] in valid_ids)

    ds['train'] = ds_train
    ds['validation'] = ds_valid

    return [ ds ]



def _load_bigbio_dataset(config_names: List[str], dataset_name: str, lang_mapper, splits):
    """
    Loads a biomedical dataset and returns a concatenated dataset for the specified splits.

    Args:
    - config_names (List[str]): A list of configuration names to load the dataset for.
    - dataset_name (str): The name of the dataset to load.
    - lang_mapper (function): A function that maps configuration names to language codes.
    - splits (List[str]): A list of splits to concatenate the dataset for.

    Returns:
    - output (datasets.DatasetDict): A concatenated dataset for the specified splits.
    """
    # TODO: implement loading all available configs for a dataset
    assert config_names is not None, "Not implemented"

    ds_map = {c: datasets.load_dataset(f"bigscience-biomedical/{dataset_name}", c) for c in config_names}
    ds = []
    for conf, ds_dict in ds_map.items():
        for k in ds_dict.keys():
            ds_dict[k] = ds_dict[k].add_column("corpus_id", [conf] * len(ds_dict[k]))
            ds_dict[k] = ds_dict[k].add_column("lang", [lang_mapper(conf)] * len(ds_dict[k]))
        ds.append(ds_dict)
    output = datasets.dataset_dict.DatasetDict()
    for s in splits:
        output[s] = datasets.concatenate_datasets([d[s] for d in ds])
    return output
