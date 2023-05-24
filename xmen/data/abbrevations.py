import spacy


class AbbreviationExpander:
    """
    A class to expand abbreviations in text data.

    Args:
    - spacy_model (str): the name of the spaCy model to use. Default is "en_core_sci_sm".

    Attributes:
    - nlp (Language): the spaCy pipeline object.
    - ab (AbbreviationDetector): the abbreviation detector object.
    """

    def __init__(self, spacy_model: str = "en_core_sci_sm"):
        from scispacy.abbreviation import AbbreviationDetector

        self.nlp = spacy.load(spacy_model)
        ab = self.nlp.add_pipe("abbreviation_detector")

    def _expand_abbreviations(self, examples):
        """
        Expands abbreviations in a given example.

        Args:
        - examples (dict): A dictionary with "passages" and "entities" keys containing a list of dictionaries.

        Returns:
        - dict: A dictionary with "entities" key containing a list of dictionaries.
        """

        def get_iter(examples):
            for passages, ents in zip(examples["passages"], examples["entities"]):
                for p in passages:
                    for t, off in zip(p["text"], p["offsets"]):
                        ents_in_text = [
                            e for e in ents if e["offsets"][0][0] >= off[0] and e["offsets"][-1][1] <= off[1]
                        ]
                        yield t, (off, ents_in_text)

        for doc, ctx in self.nlp.pipe(get_iter(examples), as_tuples=True, disable=["ner"]):
            off, ents_in_text = ctx
            abbrv = doc._.abbreviations
            for e in ents_in_text:
                e["long_form"] = None
                e_start, e_end = e["offsets"][0][0], e["offsets"][-1][1]
                for a in abbrv:
                    if e_start - off[0] <= a.start_char and e_end - off[0] >= a.end_char:
                        e["long_form"] = a._.long_form.text
                        abbrv.remove(a)

        return {"entities": examples["entities"]}

    def transform_batch(self, dataset):
        """
        Transforms a given dataset by expanding abbreviations in it.

        Args:
        - dataset (Dataset): A dataset object from the Hugging Face datasets library.

        Returns:
        - Dataset: A transformed dataset object with the same number of examples.
        """
        return dataset.map(self._expand_abbreviations, batched=True)
