from typing import List, Union
from pathlib import Path
from scispacy import umls_utils
import datasets
from bigbio.dataloader import BigBioConfigHelpers


def load_dataset(dataset: str):
    """
    Loads a dataset using the appropriate data loader function from the xmen package.

    Args:
    - dataset (str): Name of the dataset to be loaded.

    Returns:
    - The loaded dataset.
    """
    from xmen.data import dataloaders as own_module

    loader_fn = getattr(own_module, f"load_{dataset}")
    return loader_fn()


def load_mantra_gsc():
    """
    Loads the Mantra-GSC dataset using the _load_bigbio_dataset function with the appropriate parameters.

    Returns:
    - The loaded Mantra-GSC dataset.
    """
    return _load_bigbio_dataset(
        None,
        "mantra_gsc",
        lambda conf_name: conf_name.split("_")[2],
        splits=["train"],
        use_bigbio=True,
    )


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

    return _load_bigbio_dataset(
        [config_name],
        "medmentions",
        lambda _: "en",
        splits=["train", "validation", "test"],
    ).map(lambda d: {"entities": drop_prefix(d["entities"])})


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
    return _load_bigbio_dataset(
        ["quaero_emea_bigbio_kb", "quaero_medline_bigbio_kb"],
        "quaero",
        lambda _: "fr",
        splits=["train", "validation", "test"],
    )


def load_distemist_linking():
    """
    Loads the DistemIST Linking dataset.

    Returns:
    - A dataset loaded from the DistemIST Linking dataset with bigbio knowledge base.

    Raises:
    - AssertionError: If the loaded dataset has an unexpected format.
    """
    return _load_bigbio_dataset(
        ["distemist_linking_bigbio_kb"],
        "distemist",
        lambda _: "es",
        splits=["train"],
    )


def load_bronco(bronco_150_xml_path: Union[Path, str]):
    pass  # return load_dataset(str(data_dir / 'bronco'), bronco_150_xml_path = bronco_150_xml_path)


def _load_bigbio_dataset(
    config_names: List[str],
    dataset_name: str,
    lang_mapper,
    splits,
    use_bigbio=False,
):
    """
    Loads a biomedical dataset and returns a concatenated dataset for the specified splits.

    Args:
    - config_names (List[str]): A list of configuration names to load the dataset for.
    - dataset_name (str): The name of the dataset to load.
    - lang_mapper (function): A function that maps configuration names to language codes.
    - splits (List[str]): A list of splits to concatenate the dataset for.
    - use_bigbio (bool): A flag that indicates whether to use BigBioConfigHelpers to load the dataset or not. Defaults to False.

    Returns:
    - output (datasets.DatasetDict): A concatenated dataset for the specified splits.
    """
    if use_bigbio:
        conhelps = BigBioConfigHelpers()
        configs = conhelps.for_dataset(dataset_name).filtered(lambda conf: conf.is_bigbio_schema)
        ds_map = {c.config.name: c.load_dataset() for c in configs}
    else:
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
