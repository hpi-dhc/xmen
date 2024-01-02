import datasets
from typing import List, Dict
from . import init_schema


def from_spacy(docs, span_key=None, doc_id_key=None):
    """
    Converts a list of NER-tagged spaCy documents into a Hugging Face Datasets dataset with the BigBIO schema.

    Args:
    - docs: a list of spaCy documents.
    - span_key: a string specifying the span attribute to use for entity spans. If None, the default 'ents' attribute is used.
    - doc_id_key: a string specifying the key for the document ID to use. If None, document ID is assigned based on position in the input list.

    Returns:
    - a Hugging Face Datasets dataset.
    """
    id_range = list(range(0, len(docs)))

    ds_dict = {
        "id": [],
        "document_id": [],
        "passages": [],
        "entities": [],
    }

    for id, doc in zip(id_range, docs):
        if doc_id_key:
            document_id = doc.user_data[doc_id_key]
        else:
            document_id = str(id)
        ds_dict["id"].append(str(id))
        ds_dict["document_id"].append(document_id)
        passages = []
        for sid, sent in enumerate(doc.sents):
            passages.append(
                {"id": sid, "type": "sentence", "text": [sent.text], "offsets": [[sent.start_char, sent.end_char]]}
            )
        ds_dict["passages"].append(passages)
        entities = []
        if span_key:
            spans = doc.spans[span_key]
        else:
            spans = doc.ents
        for span_id, span in enumerate(sorted(spans, key=lambda s: s.start_char)):
            entities.append(
                {
                    "id": span_id,
                    "type": span.label_,
                    "text": [span.text],
                    "offsets": [[span.start_char, span.end_char]],
                    "normalized": [],
                }
            )
        ds_dict["entities"].append(entities)

    n_docs = len(ds_dict["document_id"])
    ds_dict["coreferences"] = [[]] * n_docs
    ds_dict["relations"] = [[]] * n_docs
    ds_dict["events"] = [[]] * n_docs
    return datasets.Dataset.from_dict(init_schema(ds_dict))


def from_spans(
    entities: List[List[Dict]], sentences: List[str], document_ids: List[str] = None, sentence_offsets: List[int] = None
):
    """
    Converts a list of spans into a Hugging Face Datasets dataset with the BigBIO schema.

    Args:
        - entities: a list of lists of entities. Each entity is a dictionary with the following keys:
            - char_start_index: the start character index of the entity span.
            - char_end_index: the end character index of the entity span.
            - label: the entity type.
            - span: the entity text span.
        - sentences: a list of sentences.
        - document_ids: a list of document IDs. If None, document IDs are assigned consecutively.
        - sentence_offsets: a list of sentence offsets. If None, sentence offsets are assigned based on sentence length.
    """
    if not document_ids:
        document_ids = [i for i in range(0, len(sentences))]

    # Convert to str
    document_ids = [str(i) for i in document_ids]

    if not sentence_offsets:
        sentence_offsets = []
        prev_doc_id = None
        for doc_id, sent in zip(document_ids, sentences):
            if prev_doc_id is None or prev_doc_id != doc_id:
                offset = 0
            sentence_offsets.append(offset)
            offset += len(sent) + 1
            prev_doc_id = doc_id

    ents = {d: [] for d in document_ids}
    passages = {d: [] for d in document_ids}

    for sentence_entities, sentence, doc_id, sentence_start in zip(entities, sentences, document_ids, sentence_offsets):
        for entity in sentence_entities:
            off0 = entity["char_start_index"] + sentence_start
            off1 = entity["char_end_index"] + sentence_start
            ent = {
                "id": f"{doc_id}_{len(ents[doc_id])}",
                "type": entity["label"],
                "text": [entity["span"]],
                "offsets": [[off0, off1]],
                "normalized": [],
            }
            ents[doc_id].append(ent)
        passage = {
            "text": [sentence],
            "type": "sentence",
            "offsets": [[sentence_start, sentence_start + len(sentence)]],
        }
        passages[doc_id].append(passage)

    out = {"id": [], "document_id": [], "passages": [], "entities": []}

    for doc_id, doc_passages in passages.items():
        out["id"].append(doc_id)
        out["document_id"].append(doc_id)

        for i, passage in enumerate(doc_passages):
            passage["id"] = f"{doc_id}_{i}"
        out["passages"].append(doc_passages)

        doc_ents = ents[doc_id]
        out["entities"].append(doc_ents)

    n_docs = len(out["document_id"])
    out["coreferences"] = [[]] * n_docs
    out["relations"] = [[]] * n_docs
    out["events"] = [[]] * n_docs
    return datasets.Dataset.from_dict(init_schema(out))
