from xmen.ext.scispacy.umls_utils import (
    read_umls_file_headers,
    read_umls_concepts,
    read_umls_types,
    read_umls_definitions,
)
from scispacy import umls_semantic_type_tree
from scispacy.linking_utils import DEFAULT_UMLS_TYPES_PATH
from langcodes import Language
import pandas as pd
from pathlib import Path
import os
from .log import logger


def read_umls_sabs(meta_path):
    """
    Reads the Metathesaurus source abbreviation (SAB) information from the UMLS data file "MRSAB.RRF"
    located at `meta_path` directory and returns it as a pandas DataFrame.

    Args:
    - meta_path (str): Path to the directory containing the UMLS data files.

    Returns:
    - pandas.DataFrame: A DataFrame with the following columns: 'RSAB', 'VSAB', 'SON', 'SF', 'TTY', 'ATN',
    'CVF', 'TS', 'LUI', 'STY', 'SRL', 'SUPPRESS', 'CVF_CUI'.

    Raises:
    - AssertionError: If the number of headers in the data file is not equal to the number of splits in a line.
    """
    res = []
    sab_filename = "MRSAB.RRF"
    headers = read_umls_file_headers(meta_path, sab_filename)
    with open(f"{meta_path}/{sab_filename}") as fin:
        for line in fin:
            splits = line.strip().split("|")
            assert len(headers) == len(splits)
            sabs = dict(zip(headers, splits))
            res.append(sabs)
    return pd.DataFrame(res)


def get_semantic_groups(version: str = None):
    """
    Retrieves the semantic groups and their corresponding TUIs (Type Unique Identifiers) from the
    UMLS semantic groups file "SemGroups.txt" located in the `resources` directory.

    Args:
    - version (str, optional): Version of the UMLS semantic groups to retrieve. Defaults to None.

    Returns:
    - pandas.DataFrame: A DataFrame with the following columns: 'GRP', 'GRP_NAME', 'TUI', 'TUI_NAME'.
    """
    sem_group_path = (
        Path(os.path.dirname(os.path.abspath(__file__)))
        / "resources"
        / f'SemGroups{"-" + version if version else ""}.txt'
    )
    sem_groups = pd.read_csv(sem_group_path, sep="|")
    sem_groups.columns = ["GRP", "GRP_NAME", "TUI", "TUI_NAME"]
    return sem_groups


def filter_semantic_groups(semantic_groups, concept_details, version=None):
    """
    Filters the `concept_details` dictionary by keeping only the concepts whose TUIs (Type Unique Identifiers)
    belong to the specified semantic groups.

    Args:
    - semantic_groups (list of str): List of semantic group names to filter by.
    - concept_details (dict): Dictionary with concept IDs as keys and their details as values.
    - version (str, optional): Version of the UMLS semantic groups to use for filtering. Defaults to None.

    Returns:
    - dict: A filtered version of the `concept_details` dictionary containing only concepts whose TUIs
    belong to the specified semantic groups.
    """
    sem_groups = get_semantic_groups(version)
    logger.info(f"> Filtering by semantic groups with version {version} and {len(sem_groups)} types.")
    valid_tuis = sem_groups[sem_groups.GRP.isin(semantic_groups)].TUI.unique()
    return {k: v for k, v in concept_details.items() if any([t in valid_tuis for t in v["types"]])}


def get_sem_type_tree():
    """
    Constructs a UMLS semantic type tree from the TSV file located at the default UMLS types path
    and returns it as a UmlsSemanticTypeTree object.

    Returns:
    - UmlsSemanticTypeTree: A UmlsSemanticTypeTree object representing the UMLS semantic type tree.
    """
    return umls_semantic_type_tree.construct_umls_tree_from_tsv(DEFAULT_UMLS_TYPES_PATH)


def filter_semantic_types(tuis, expand_semantic_types, concept_details):
    """
    Filters the `concept_details` dictionary by keeping only the concepts whose TUIs (Type Unique Identifiers)
    belong to the specified semantic types.

    Args:
    - tuis (list of str): List of semantic type abbreviations to filter by.
    - expand_semantic_types (bool): Whether or not to include child semantic types of the specified ones.
    - concept_details (dict): Dictionary with concept IDs as keys and their details as values.

    Returns:
    - dict: A filtered version of the `concept_details` dictionary containing only concepts whose TUIs
    belong to the specified semantic types.
    """
    if expand_semantic_types:
        tuis = expand_tuis(tuis, get_sem_type_tree())
    return {k: v for k, v in concept_details.items() if any([t in tuis for t in v["types"]])}


def get_alias_count(concept_details):
    """
    Calculates and returns the total number of aliases for all the concepts in the `concept_details` dictionary.

    Args:
    - concept_details (dict): Dictionary with concept IDs as keys and their details as values.

    Returns:
    - int: Total number of aliases for all the concepts in the `concept_details` dictionary.
    """
    return sum([len(c["aliases"]) + 1 for c in concept_details.values()])


def expand_tuis(tuis, sem_type_tree):
    """
    Recursively expands a list of UMLS semantic type abbreviations to include their child semantic types,
    using the specified semantic type tree.

    Args:
    - tuis (list of str): List of semantic type abbreviations to expand.
    - sem_type_tree (UMLSSemanticTypeTree): A UMLSSemanticTypeTree object representing the UMLS semantic type tree.

    Returns:
    - list of str: A list of semantic type abbreviations that includes the specified types and all their child types.
    """
    result = tuis
    for t in tuis:
        children = [c.type_id for c in sem_type_tree.type_id_to_node[t].children]
        if len(children) > 0:
            result += children
            result += expand_tuis(children, sem_type_tree)
    return list(set(result))


def _get_lang_code(lang):
    """
    Converts a language name or ISO code to its ISO 639-2/B code.

    Args:
    - lang (str): Language name or ISO code.

    Returns:
    - str: ISO 639-2/B code for the specified language.

    Raises:
    - AssertionError: If the specified language is not valid.
    """
    lang_obj = Language.get(lang)
    assert lang_obj.is_valid()
    return lang_obj.to_alpha3(variant="B").upper()


def get_umls_concepts(
    meta_path: str,
    langs: str,
    sabs: list,
    sources: list,
    semantic_groups: list,
    semantic_types: list,
    expand_semantic_types=True,
    non_suppressed_only=False,
    semantic_group_file_version=None,
) -> dict:
    """
    Reads UMLS concepts and related metadata from the specified sources and languages, and filters them by semantic
    groups and semantic types.

    Args:
    - meta_path (str): Path to the UMLS metathesaurus directory.
    - langs (str): A comma-separated string of language codes to include. If `None` or `'all'`, all available languages
                   will be included.
    - sabs (list): A list of source abbreviations to include. If empty, all available sources will be included.
    - sources (list): A list of source full names to include. If empty, all available sources will be included.
    - semantic_groups (list): A list of semantic group abbreviations to include. If empty, no semantic group filtering
                              will be performed.
    - semantic_types (list): A list of semantic type abbreviations to include. If empty, no semantic type filtering
                             will be performed.
    - expand_semantic_types (bool): Whether to expand the specified semantic types to include all child types.
    - non_suppressed_only (bool): Whether to include only non-suppressed concepts.
    - semantic_group_file_version (str): The version of the UMLS Semantic Groups file to use. If `None`, the latest
                                         version will be used.

    Returns:
    - dict: A dictionary containing the UMLS concepts and related metadata, filtered according to the specified
            criteria.
    """

    meta_path = Path(meta_path)
    if not langs or "all" in langs:
        langs = None
    else:
        langs = [_get_lang_code(lang) for lang in langs]

    logger.info(f"Using UMLS metathesaurus: {meta_path}")

    if not sabs:
        sabs = []

    if sources:
        sab_df = read_umls_sabs(meta_path)
        sabs += list(sab_df[sab_df.SF.isin(sources)].RSAB.unique())

    logger.info(f"Using sources: {sabs}")

    concept_details = {}
    if not sabs:
        sabs = [None]
    for source in sabs:
        logger.info(
            f'>> Reading concepts from {"all sources" if not source else source} and {"all languages" if not langs else f"languages: {langs}"}'
        )
        for lang in langs if langs else [None]:
            read_umls_concepts(meta_path, concept_details, source=source, lang=lang, non_suppressed=non_suppressed_only)

    logger.info(">> Reading types ... ")
    read_umls_types(meta_path, concept_details)

    if semantic_groups:
        logger.info(f"> Number of concepts before semantic group filtering: {len(concept_details)}")
        concept_details = filter_semantic_groups(semantic_groups, concept_details, semantic_group_file_version)

    if semantic_types:
        logger.info(f"> Number of concepts before semantic type filtering: {len(concept_details)}")
        concept_details = filter_semantic_types(semantic_types, expand_semantic_types, concept_details)

    logger.info(">> Reading definitions ... ")
    read_umls_definitions(meta_path, concept_details)

    logger.info(f"> Number of concepts before de-duplication: {len(concept_details)}")
    logger.info(f"> Number of aliases before de-duplication: {get_alias_count(concept_details)}")

    for concept in concept_details.values():
        # Some concepts have many duplicate aliases. Here we remove them.
        concept["aliases"] = list(set(concept["aliases"]))

        # if a concept doesn't have a canonical name, use the first alias instead
        if "canonical_name" not in concept:
            aliases = concept["aliases"]
            concept["canonical_name"] = aliases[0]
            del aliases[0]

        # deleting `is_from_preferred_source`
        if "is_from_preferred_source" in concept:
            del concept["is_from_preferred_source"]

    return concept_details
