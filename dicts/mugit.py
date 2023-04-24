from pathlib import Path
from scispacy import umls_utils
import logging
import pandas as pd


log = logging.getLogger(__name__)


def get_concept_details(umls_meta_path: str, mugit_path: str, mugit_file: str) -> dict:
    concept_details = {}

    log.info(">> Reading concepts ...")

    mugit_interface = pd.read_csv(Path(mugit_path) / mugit_file, sep="\t", header=None)
    mugit_interface.columns = ["SNOMED_ID", "TERM_ID", "English", "German"]
    mugit_interface.set_index("SNOMED_ID", inplace=True)
    mugit_interface.sort_index(inplace=True)
    mugit_interface.dropna(inplace=True)

    concepts_filename = "MRCONSO.RRF"
    headers = umls_utils.read_umls_file_headers(umls_meta_path, concepts_filename)

    snomed_ids = set()

    with open(f"{umls_meta_path}/{concepts_filename}") as fin:
        for line in fin:
            splits = line.strip().split("|")
            assert len(headers) == len(splits), (headers, splits)
            concept = dict(zip(headers, splits))
            if concept["SAB"] == "SNOMEDCT_US":
                snomed_id = int(concept["CODE"])
                if snomed_id in snomed_ids:
                    continue
                snomed_ids.add(snomed_id)
                if snomed_id in mugit_interface.index:
                    interface_terms = mugit_interface.loc[[snomed_id]]
                    concept_id = concept["CUI"]
                    if concept_id not in concept_details:
                        concept_details[concept_id] = {
                            "concept_id": concept_id,
                            "canonical_name": None,
                            "types": [],
                            "aliases": interface_terms.German.tolist(),
                        }
                    else:
                        concept_details[concept_id]["aliases"] += interface_terms.German.tolist()

    # Remove duplicates
    for v in concept_details.values():
        v["aliases"] = list(set(v["aliases"]))

    log.info(">> Reading types ... ")
    umls_utils.read_umls_types(umls_meta_path, concept_details)

    log.info(">> Reading definitions ... ")
    umls_utils.read_umls_definitions(umls_meta_path, concept_details)

    for concept in concept_details.values():
        # deleting `is_from_preferred_source`
        if "is_from_preferred_source" in concept:
            del concept["is_from_preferred_source"]

    return concept_details
