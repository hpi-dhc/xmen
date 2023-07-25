from typing import List, Union
from pathlib import Path
import datasets

RANDOM_SEED = 42


def load_dataset(dataset: Union[str, Path], **kwargs):
    """
    Loads a dataset using the appropriate data loader function from the xmen package.

    Args:
    - dataset (str | Path): Name or path of the dataset to be loaded.

    Returns:
    - The loaded dataset.
    """
    import sys

    if Path(dataset).exists():
        return [datasets.load_from_disk(dataset)]

    loader_fn = getattr(sys.modules[__name__], f"load_{dataset}")
    return loader_fn(**kwargs)


def load_bronco_diagnosis(data_dir):
    return _load_bronco("DIAGNOSIS", data_dir)

def load_bronco_treatment(data_dir):
    return _load_bronco("TREATMENT", data_dir)

def load_bronco_medication(data_dir):
    return _load_bronco("MEDICATION", data_dir)


def _load_bronco(label: str, data_dir):
    def filter_entities(bigbio_entities, valid_entities):
        filtered_entities = []
        for ent in bigbio_entities:
            if ent["type"] in valid_entities:
                filtered_entities.append(ent)
        return filtered_entities

    bronco = _load_bigbio_dataset(
        ["bronco_bigbio_kb"], "bronco", lambda _: "de", splits=["train"], data_dir=data_dir
    ).map(lambda row: {"entities": filter_entities(row["entities"], [label])})["train"]

    res = []
    for k_test in range(4, -1, -1):
        k_valid = k_test - 1 if k_test > 0 else 4
        test_split = bronco.select([k_test])
        validation_split = bronco.select([k_valid])
        train_split = bronco.select([i for i in range(0, 5) if i not in [k_valid, k_test]])
        res.append(datasets.DatasetDict({"train": train_split, "validation": validation_split, "test": test_split}))

    return res


def load_mantra_gsc(subsets=None):
    """
    Loads all subsets of Mantra GSC into one dataset

    TODO: Simplify when Mantra is eventually on the Hugging Face Hub

    Returns:
    - The loaded Mantra-GSC dataset.
    """
    import bigbio

    mantra_path = str(Path(bigbio.__file__).parent / "biodatasets" / "mantra_gsc" / "mantra_gsc.py")
    if not subsets:
        subsets = [c for c in datasets.get_dataset_infos(mantra_path).keys() if "bigbio" in c]

    ds_map = {c: datasets.load_dataset(mantra_path, c) for c in subsets}
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

    return [
        _load_bigbio_dataset(
            [config_name],
            "medmentions",
            lambda _: "en",
            splits=["train", "validation", "test"],
        ).map(lambda d: {"entities": drop_prefix(d["entities"])})
    ]


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


def load_quaero(subsets = None):
    """
    Loads the Quaero dataset.

    Returns:
    - A dataset loaded from the Quaero dataset with bigbio knowledge base.
    """
    if not subsets:
        subsets = ["quaero_emea_bigbio_kb", "quaero_medline_bigbio_kb"]
    return [
        _load_bigbio_dataset(
            subsets,
            "quaero",
            lambda _: "fr",
            splits=["train", "validation", "test"],
        )
    ]


def load_distemist(subsets = None):
    """
    Loads the DisTEMIST (EL track) dataset.

    Returns:
    - A dataset loaded from the DisTEMIST Linking dataset with bigbio knowledge base.

    Raises:
    - AssertionError: If the loaded dataset has an unexpected format.
    """
    if not subsets:
        subsets = ["distemist_linking_bigbio_kb"] 
    ds = _load_bigbio_dataset(
        subsets,
        "distemist",
        lambda _: "es",
        splits=["train", "test"],
    )

    # Own validation set (20% of distemist training set / EL sub-track)
    with open(Path(__file__).parent / "benchmark" / "distemist_validation_docs.txt", "r") as fh:
        valid_ids = [l.strip() for l in fh.readlines()]

    ds_train = ds["train"].filter(lambda d: d["document_id"] not in valid_ids)
    ds_valid = ds["train"].filter(lambda d: d["document_id"] in valid_ids)

    ds["train"] = ds_train
    ds["validation"] = ds_valid

    def unmerge_multi_annotations(document):
        entities = []
        for e in document["entities"]:
            for n in e["normalized"]:
                en = e.copy()
                en["normalized"] = [n]
                entities.append(en)
        return {"entities": entities}

    return [ds.map(unmerge_multi_annotations)]


def _load_bigbio_dataset(config_names: List[str], dataset_name: str, lang_mapper, splits, data_dir=None):
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

    ds_map = {c: datasets.load_dataset(f"bigbio/{dataset_name}", c, data_dir) for c in config_names}
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
