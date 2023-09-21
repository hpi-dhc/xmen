from itertools import groupby


class Deduplicator:
    def _deduplicate(self, document):
        result = []
        grp_key = lambda e: (e["offsets"], e["type"], e["text"])
        # sorted_ents = sorted(document["entities"], key=grp_key)
        for _, grp in groupby(document["entities"], grp_key):
            for i, g in enumerate(grp):
                g["normalized"] = g["normalized"][i:]
                result.append(g)
        return {"entities": result}

    def transform_batch(self, ds):
        return ds.map(lambda e: self._deduplicate(e), load_from_cache_file=False)
