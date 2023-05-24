from xmen.ext.scispacy.umls_utils import read_umls_file_headers


class CUIReplacer:
    """
    A class that replaces retired CUIs in normalized entities with their replacements.

    Args:
    - umls_meta_path (str): The path to the UMLS metadata directory

    Attributes:
    - retired_cuis (dict): A dictionary containing retired CUIs as keys and their replacements as values
    """

    def __init__(self, umls_meta_path):
        self.retired_cuis = self._load_retired_cuis(umls_meta_path)

    def transform_batch(self, dataset):
        """
        Replaces retired CUIs in the normalized entities of a dataset with their replacements.

        Args:
        - dataset (MapDataset): A dataset containing entities with normalized forms

        Returns:
        - MapDataset: The transformed dataset with replaced CUIs in the normalized entities
        """

        def replace_cuis(entities):
            """
            Helper function that replaces retired CUIs in normalized forms of entities with their replacements.

            Args:
            - entities (list): A list of entities containing normalized forms with CUIs to replace

            Returns:
            - list: A list of entities with replaced CUIs in the normalized forms
            """
            for e in entities:
                for n in e["normalized"]:
                    c = n["db_id"]
                    n["db_id"] = self.retired_cuis.get(c, c)
            return entities

        return dataset.map(lambda i: {"entities": replace_cuis(i["entities"])})

    def _load_retired_cuis(self, meta_path):
        """
        Loads the retired CUIs mapping from the MRCUI.RRF file in the metadata directory.

        Args:
        - meta_path (str): The path to the UMLS metadata directory

        Returns:
        - dict: A dictionary containing retired CUIs as keys and their replacements as values
        """
        res = []
        cui_filename = "MRCUI.RRF"
        headers = read_umls_file_headers(meta_path, cui_filename)
        mapping = {}
        with open(f"{meta_path}/{cui_filename}") as fin:
            for line in fin:
                splits = line.strip().split("|")
                assert len(headers) == len(splits)
                res = dict(zip(headers, splits))
                if res["CUI2"]:
                    mapping[res["CUI1"]] = res["CUI2"]
        return mapping
