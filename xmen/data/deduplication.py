from itertools import groupby

class Deduplicator():
    
    def _deduplicate(self, document):
        result = []
        sorted_ents = sorted(document['entities'], key=lambda e: e['offsets'])
        for o, grp in groupby(sorted_ents, lambda e: e['offsets']):
            for i, g in enumerate(grp):
                g['normalized'] = g['normalized'][i:]
                result.append(g)
        return {'entities' : result}
    
    def transform_batch(self, ds):
        return ds.map(lambda e: self._deduplicate(e), load_from_cache_file=False)