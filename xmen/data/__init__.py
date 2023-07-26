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
        "id": Value(dtype="string"),
        "document_id": Value(dtype="string"),
        "passages": [
            {
                "id": Value(dtype="string"),
                "type": Value(dtype="string"),
                "text": Sequence(feature=Value(dtype="string")),
                "offsets": Sequence(feature=[Value(dtype="int32")]),
            }
        ],
        "entities": [
            {
                "id": Value(dtype="string"),
                "offsets": Sequence(feature=Sequence(feature=Value(dtype="int32"))),
                "text": Sequence(feature=Value(dtype="string")),
                "type": Value(dtype="string"),
                "normalized": [
                    {
                        "db_id": Value(dtype="string"),
                        "db_name": Value(dtype="string"),
                        "score": Value(dtype="float"),
                        "predicted_by": Sequence(Value(dtype="string")),
                    }
                ],
                "long_form": Value(dtype="string"),
            }
        ],
        "events": [
            {
                "id": Value(dtype="string"),
                "type": Value(dtype="string"),
                "trigger": {
                    "text": Sequence(feature=Value(dtype="string")),
                    "offsets": Sequence(feature=[Value(dtype="int32")]),
                },
                "arguments": [{"role": Value(dtype="string"), "ref_id": Value(dtype="string")}],
            }
        ],
        "coreferences": [
            {
                "id": Value(dtype="string"),
                "entity_ids": Sequence(feature=Value(dtype="string")),
            }
        ],
        "relations": [
            {
                "id": Value(dtype="string"),
                "type": Value(dtype="string"),
                "arg1_id": Value(dtype="string"),
                "arg2_id": Value(dtype="string"),
                "normalized": [{"db_name": Value(dtype="string"), "db_id": Value(dtype="string")}],
            }
        ],
        "corpus_id": Value(dtype="string"),
        "lang": Value(dtype="string"),
    }
)
