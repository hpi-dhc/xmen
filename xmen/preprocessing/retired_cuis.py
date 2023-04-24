from scispacy import umls_utils


class CUIReplacer:
    def __init__(self, umls_meta_path):
        self.retired_cuis = self._load_retired_cuis(umls_meta_path)

    def transform_batch(self, dataset):
        def replace_cuis(entities):
            for e in entities:
                for n in e["normalized"]:
                    c = n["db_id"]
                    n["db_id"] = self.retired_cuis.get(c, c)
            return entities

        return dataset.map(lambda i: {"entities": replace_cuis(i["entities"])})

    def _load_retired_cuis(self, meta_path):
        res = []
        cui_filename = "MRCUI.RRF"
        headers = umls_utils.read_umls_file_headers(meta_path, cui_filename)
        mapping = {}
        with open(f"{meta_path}/{cui_filename}") as fin:
            for line in fin:
                splits = line.strip().split("|")
                assert len(headers) == len(splits)
                res = dict(zip(headers, splits))
                if res["CUI2"]:
                    mapping[res["CUI1"]] = res["CUI2"]
        return mapping
