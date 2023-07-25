from .indexed_dataset import IndexedDatasetDict, IndexedDataset
from .util import *
from .integrations import *

from .abbrevations import AbbreviationExpander
from .sampling import Sampler
from .merge_concepts import ConceptMerger
from .retired_cuis import CUIReplacer
from .filter import EmptyNormalizationFilter, MissingCUIFilter
from .semantic_types import SemanticTypeFilter
from .semantic_groups import SemanticGroupFilter
from .deduplication import Deduplicator

from datasets import Features, Value, Sequence

features = Features(
    {
        "id": Value(dtype="string", id=None),
        "document_id": Value(dtype="string", id=None),
        "passages": [
            {
                "id": Value(dtype="string", id=None),
                "type": Value(dtype="string", id=None),
                "text": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
                "offsets": Sequence(feature=[Value(dtype="int32", id=None)], length=-1, id=None),
            }
        ],
        "entities": [
            {
                "id": Value(dtype="string", id=None),
                "offsets": Sequence(
                    feature=Sequence(feature=Value(dtype="int32", id=None), length=-1, id=None), length=-1, id=None
                ),
                "text": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
                "type": Value(dtype="string", id=None),
                "normalized": [
                    {
                        "db_id": Value(dtype="string", id=None),
                        "db_name": Value(dtype="string", id=None),
                        "score": Value(dtype="float", id=None),
                    }
                ],
                "long_form": Value(dtype="string", id=None),
            }
        ],
        "events": [
            {
                "id": Value(dtype="string", id=None),
                "type": Value(dtype="string", id=None),
                "trigger": {
                    "text": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
                    "offsets": Sequence(feature=[Value(dtype="int32", id=None)], length=-1, id=None),
                },
                "arguments": [{"role": Value(dtype="string", id=None), "ref_id": Value(dtype="string", id=None)}],
            }
        ],
        "coreferences": [
            {
                "id": Value(dtype="string", id=None),
                "entity_ids": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
            }
        ],
        "relations": [
            {
                "id": Value(dtype="string", id=None),
                "type": Value(dtype="string", id=None),
                "arg1_id": Value(dtype="string", id=None),
                "arg2_id": Value(dtype="string", id=None),
                "normalized": [{"db_name": Value(dtype="string", id=None), "db_id": Value(dtype="string", id=None)}],
            }
        ],
        "corpus_id": Value(dtype="string", id=None),
        "lang": Value(dtype="string", id=None),
    }
)
