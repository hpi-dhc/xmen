from itertools import groupby


class ConceptMerger:
    def transform_batch(self, dataset):
        def _merge_entities(d):
            ents = d["entities"]
            merged = []
            for k, grp in groupby(
                sorted(ents, key=lambda e: e["offsets"][0][0]),
                lambda e: (e["text"], e["offsets"]),
            ):
                grp = list(grp)
                normalized = []
                for e in grp:
                    for n in e["normalized"]:
                        if not n in normalized:
                            normalized.append(n)
                merged.append(
                    {
                        "id": "+".join([e["id"] for e in grp]),
                        "normalized": normalized,
                        "type": [e["type"] for e in grp],
                        "text": k[0],
                        "offsets": k[1],
                    }
                )
            return {"entities": merged}

        return dataset.map(_merge_entities)
