from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import spacy
import re
from xmen.log import logger
from datasets import DatasetDict
from typing import List, Iterator


class Translator:

    ENGLISH = "eng_Latn"
    GERMAN = "deu_Latn"
    FRENCH = "fra_Latn"
    SPANISH = "spa_Latn"
    DUTCH = "nld_Latn"

    def __init__(self, src_lang: str, target_lang: str, cuda: bool = True, batch_size: int = 4) -> None:
        self.src_lang = src_lang
        self.target_lang = target_lang
        self.batch_size = batch_size

        logger.info("Initializing Translation Model")

        self.tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang=src_lang)

        self.model = AutoModelForSeq2SeqLM.from_pretrained("ychenNLP/nllb-200-3.3b-easyproject")
        if cuda:
            self.model.cuda()

        self.nlp = spacy.load("en_core_sci_sm")

    def __getstate__(self):
        return {}

    def _annos_to_tagged(self, document):
        tagged_passages = []
        e_id = 0
        for p in document["passages"]:
            tagged_passage = {"id": p["id"], "type": p["type"], "text": []}
            for text, passage_offset in zip(p["text"], p["offsets"]):
                for sent in self.nlp(text).sents:
                    o = (passage_offset[0] + sent.start_char, passage_offset[0] + sent.end_char)
                    entities = [
                        e for e in document["entities"] if e["offsets"][0][0] >= o[0] and e["offsets"][-1][1] <= o[1]
                    ]
                    transformed_text = sent.text
                    offset_diff = -o[0]

                    for entity in entities:
                        e_id += 1
                        start = entity["offsets"][0][0] + offset_diff
                        end = entity["offsets"][-1][1] + offset_diff
                        s_start = f"[{e_id}]"
                        s_end = f"[/{e_id}]"
                        transformed_text = (
                            transformed_text[:start]
                            + s_start
                            + transformed_text[start:end]
                            + s_end
                            + transformed_text[end:]
                        )
                        offset_diff += len(str(e_id)) * 2 + 5  # Length of entity_type + 5 (for the tags)
                    tagged_passage["text"].append(transformed_text)
            tagged_passages.append(tagged_passage)
        return {"passages": tagged_passages}

    def _translate_tagged_document(self, document):
        output_passages = []

        for p in document["passages"]:
            translated_passage = p.copy()
            tagged_sentences = p["text"]
            translated_sentences = []
            for idx in range(0, len(tagged_sentences), self.batch_size):
                start_idx = idx
                end_idx = idx + self.batch_size
                inputs = self.tokenizer(
                    tagged_sentences[start_idx:end_idx],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to("cuda")
                max_input_length = max([len(i) for i in inputs["input_ids"]])

                with torch.no_grad():
                    translated_tokens = self.model.generate(
                        **inputs,
                        forced_bos_token_id=self.tokenizer.lang_code_to_id[self.target_lang],
                        max_length=max(128, int(max_input_length * 1.5)),
                        num_beams=5,
                        num_return_sequences=1,
                        early_stopping=True,
                    )

                output = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
                translated_sentences.extend(output)
            translated_passage["text"] = translated_sentences
            output_passages.append(translated_passage)
        return {"passages": output_passages}

    def _remove_nested_tags_text(self, t, document_id):
        pattern = "\[(\/)?(\d+)\]"
        clean_text = t
        opened = []
        index = 0
        while m := re.search(pattern, clean_text[index:]):
            index += m.span()[1]
            if not m.group(1):  # is start tag
                m_next = re.search(pattern, clean_text[index:])
                if not m.group(2) in opened and m_next and m_next.group(1) == "/" and m.group(2) == m_next.group(2):
                    opened.append(m.group(2))
                    continue  # alright
                else:  # Invalid tag
                    clean_text = clean_text.replace(m.group(0), "", 1)
                    index -= len(m.group(0))
                    if not m.group(0) in clean_text:  # again
                        logger.debug(f"Document {document_id} - Replacing nested tag {m.group(0)} in:\n{t}")
                        clean_text = clean_text.replace(f"[/{m.group(2)}]", "")
                    else:
                        logger.debug(f"Document {document_id} - Replacing single tag {m.group(0)} in:\n{t}")
            else:  # is end tag
                if not m.group(2) in opened:
                    logger.debug(f"Document {document_id} - Closing unopened tag {m.group(0)} in: \n{t}")
                    clean_text = clean_text.replace(m.group(0), "", 1)
                    index -= len(m.group(0))
        return clean_text

    def _remove_nested_entities(self, document):
        passages = document["passages"].copy()
        for p in passages:
            clean_texts = []
            for t in p["text"]:
                errors = True
                while errors:
                    clean_text = self._remove_nested_tags_text(t, document["id"])
                    if t == clean_text:
                        logger.debug("Cleaned document:\n" + clean_text)
                        clean_texts.append(clean_text)
                        errors = False
                    t = clean_text
            p["text"] = clean_texts
        return {"passages": passages}

    def _realign_entities(self, document):
        eid2entity = {str(i + 1): e for i, e in enumerate(document["entities"])}
        passages = []
        entities = []
        start_offset = 0
        for p in document["passages"]:
            new_passage = p.copy()
            new_passage["text"] = []
            new_passage["offsets"] = []
            for t in p["text"]:
                orig_text = t
                sentence_entities = []
                while start_match := re.search("\[(\d+)\]", t):
                    start_span = start_match.span()
                    e_id = start_match.group(1)
                    other_start_match = re.findall(f"\[({e_id})\]", t)
                    end_match = re.search(f"\[\/({e_id})\]", t)
                    if len(other_start_match) > 1 or not end_match:
                        logger.debug(f"Unclosed tag {start_match} in {orig_text} - Skipping")
                        t = t[: start_span[0]] + t[start_span[1] :]
                        continue

                    end_span = end_match.span()
                    ent_text = t[start_span[1] : end_span[0]]
                    t = t[: start_span[0]] + ent_text + t[end_span[1] :]
                    if not e_id in eid2entity:
                        logger.debug(f"{e_id} does not match original entities - Skipping")
                    elif not ent_text.strip():
                        logger.debug(f"Empty entity {e_id} - Skipping")
                    else:
                        entity = eid2entity[e_id].copy()
                        entity["text"] = [ent_text]
                        entity["offsets"] = [
                            [start_offset + start_span[0], start_offset + end_span[0] - len(start_match.group(0))]
                        ]
                        entities.append(entity)
                for e in sentence_entities:
                    assert t[e["offset"][0] : e["offset"][1]] == e["text"], (e, t)
                new_passage["text"].append(t)
                new_passage["offsets"].append([start_offset, start_offset + len(t)])
                start_offset += len(t) + 1
            passages.append(new_passage)
        found_ids = {e["id"] for e in entities}
        missing_ids = []
        for e in eid2entity.values():
            if not e["id"] in found_ids:
                missing_ids.append(e["id"])
        if missing_ids:
            logger.debug(f"Entities {missing_ids} missing from translation in document {document['id']}")
        return {"passages": passages, "entities": entities}

    def _translate_batch(self, dataset):
        tagged_dataset = dataset.map(self._annos_to_tagged, load_from_cache_file=False)
        logger.info("Translating documents")
        return tagged_dataset.map(self._translate_tagged_document, load_from_cache_file=False)

    def _tags_to_entities(self, dataset):
        clean_tagged_dataset = dataset.map(self._remove_nested_entities, load_from_cache_file=False)
        realigned_dataset = clean_tagged_dataset.map(self._realign_entities, load_from_cache_file=False)
        check_passages_offsets(realigned_dataset)
        check_entities_offsets(realigned_dataset)
        return realigned_dataset

    def transform_batch(self, dataset):
        translated_tagged_dataset = self._translate_batch(dataset)
        return self._tags_to_entities(translated_tagged_dataset)


# Offset checks adapted from https://github.com/bigscience-workshop/biomedical/blob/main/tests/test_bigbio.py


def _get_example_text(example: dict) -> str:
    """
    Concatenate all text from passages in an example of a KB schema
    :param example: An instance of the KB schema
    """
    return " ".join([t for p in example["passages"] for t in p["text"]])


def check_passages_offsets(dataset_bigbio: DatasetDict):
    for split in dataset_bigbio:
        if "passages" in dataset_bigbio[split].features:
            for example in dataset_bigbio[split]:
                example_text = _get_example_text(example)

                for passage in example["passages"]:
                    example_id = example["id"]
                    text = passage["text"]
                    offsets = passage["offsets"]
                    for idx, (start, end) in enumerate(offsets):
                        msg = (
                            f"Split:{split} - Example:{example_id} - "
                            f"text:`{example_text[start:end]}` != text_by_offset:`{text[idx]}`"
                        )
                        if not example_text[start:end] == text[idx]:
                            logger.error(msg)


def _check_offsets(
    example_id: int,
    split: str,
    example_text: str,
    offsets: List[List[int]],
    texts: List[str],
) -> Iterator:
    if len(texts) != len(offsets):
        logger.warning(
            f"Split:{split} - Example:{example_id} - "
            f"Number of texts {len(texts)} != number of offsets {len(offsets)}. "
            f"Please make sure that this error already exists in the original "
            f"data and was not introduced in the data loader."
        )
    # offsets are always list of lists
    for idx, (start, end) in enumerate(offsets):
        by_offset_text = example_text[start:end]
        try:
            text = texts[idx]
        except IndexError:
            text = ""
        if by_offset_text != text:
            yield f" text:`{text}` != text_by_offset:`{by_offset_text}`"


def check_entities_offsets(dataset_bigbio: DatasetDict):
    errors = []
    for split in dataset_bigbio:
        if "entities" in dataset_bigbio[split].features:
            for example in dataset_bigbio[split]:
                example_id = example["id"]
                example_text = _get_example_text(example)

                for entity in example["entities"]:
                    for msg in _check_offsets(
                        example_id=example_id,
                        split=split,
                        example_text=example_text,
                        offsets=entity["offsets"],
                        texts=entity["text"],
                    ):
                        entity_id = entity["id"]
                        errors.append(f"Split: {split}, Example:{example_id} - entity:{entity_id} " + msg)
    if len(errors) > 0:
        logger.warning(msg="\n".join(errors) + " Wrong offsets")
