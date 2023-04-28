from xmen.umls import get_umls_concepts
import importlib.util
import sys
import json
from omegaconf import DictConfig
from .utils import get_alias_count
from ..log import logger


def assemble(cfg: DictConfig, output: str, custom_path=None):
    """
    Assembles a list of concept details into a JSONL file at the given output path.

    Args:
    - cfg: The configuration dictionary.
    - output: The output path for the JSONL file.
    - custom_path: The path to custom code, if applicable.
    """
    concept_details = get_concept_details(cfg, custom_path)

    n_concepts = len(concept_details)
    if n_concepts == 0:
        logger.warn("List of concepts is empty")

    logger.info(f"Number of concepts: {n_concepts}")
    logger.info(f"Number of aliases: {get_alias_count(concept_details)}")

    output.parent.mkdir(exist_ok=True, parents=True)
    with open(output, "w") as fout:
        for value in concept_details.values():
            fout.write(json.dumps(value) + "\n")


def get_concept_details(cfg, custom_path):
    """
    Returns the concept details from UMLS or custom code.

    Args:
    - cfg: The OmegaConf config object.
    - custom_path (str): The path to the custom code file.

    Returns:
    - dict: A dictionary containing the concept details.
    """
    if "umls" in cfg.dict:
        return get_umls_concepts(
            cfg.dict.umls.meta_path,
            cfg.dict.umls.get("lang", []),
            sabs=cfg.dict.umls.get("sabs", []),
            sources=cfg.dict.umls.get("sources", []),
            semantic_groups=cfg.dict.umls.get("semantic_groups", None),
            semantic_types=cfg.dict.umls.get("semantic_types", None),
            non_suppressed_only=cfg.dict.umls.get("non_suppressed_only", False),
            semantic_group_file_version=cfg.dict.umls.get("semantic_group_file_version", False),
        )
    elif "custom" in cfg.dict:
        return get_custom_concepts(cfg, custom_path)


def get_custom_concepts(cfg, path) -> dict:
    """
    Load and execute a Python module located in the given path that implements a function `get_concept_details(cfg)`.
    This function should return a dictionary containing details about custom concepts.

    Args:
    - cfg: Configuration object to pass to the `get_concept_details` function.
    - path: Path to the Python module containing the `get_concept_details` function.

    Returns:
    - Dictionary containing details about custom concepts.
    """
    spec = importlib.util.spec_from_file_location("custom_code", path)
    custom_module = importlib.util.module_from_spec(spec)
    sys.modules["custom_code"] = custom_module
    spec.loader.exec_module(custom_module)
    return custom_module.get_concept_details(cfg)
