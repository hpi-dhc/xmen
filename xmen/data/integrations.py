import datasets


def from_spacy(docs, span_key=None, doc_id_key=None):
    """
    Converts a list of spaCy documents into a Hugging Face Datasets dataset.

    Args:
    - docs: a list of spaCy documents.
    - span_key: a string specifying the span attribute to use for entity spans. If None, the default 'ents' attribute is used.
    - doc_id_key: a string specifying the key for the document ID to use. If None, document ID is assigned based on position in the input list.

    Returns:
    - a Hugging Face Datasets dataset.
    """
    ds = []
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
            document_id = id
        ds_dict["id"].append(id)
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
    return datasets.Dataset.from_dict(ds_dict)
