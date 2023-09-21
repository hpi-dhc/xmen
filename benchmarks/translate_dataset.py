import argparse
from pathlib import Path
import logging

from xmen.data.translation import Translator
from xmen.log import logger

from dataloaders import load_dataset

lang_to_code = {
    "en": Translator.ENGLISH,
    "de": Translator.GERMAN,
    "fr": Translator.FRENCH,
    "es": Translator.SPANISH,
    "nl": Translator.DUTCH,
}


def translate_dataset(dataset, src_lang, target_lang, output_dir):
    t = Translator(src_lang, target_lang)
    translated_dataset = t.transform_batch(dataset)

    translated_dataset.save_to_disk(output_dir)

    num_ents_before = len([e for v in dataset.values() for d in v for e in d["entities"]])
    logger.info(f"Number of entities before translation: {num_ents_before}")

    num_ents_after = len([e for v in translated_dataset.values() for d in v for e in d["entities"]])
    logger.info(
        f"Number of entities after translation: {num_ents_after} ({(1 - num_ents_after / num_ents_before)*100:.2f}% loss)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("src_lang", type=str)
    parser.add_argument("target_lang", type=str)
    parser.add_argument("output_dir", type=str)

    args = parser.parse_args()
    dataset_name = args.dataset
    dataset = load_dataset(args.dataset)
    assert len(dataset) == 1
    dataset = dataset[0]
    src_lang = lang_to_code[args.src_lang]
    target_lang = lang_to_code[args.target_lang]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fname = f"{dataset_name}_{args.src_lang}_{args.target_lang}"

    fh = logging.FileHandler(output_dir / f"{fname}.log")
    fh.setLevel(logging.DEBUG)
    fmt_file = "%(asctime)s - %(levelname)s - %(message)s"
    file_formatter = logging.Formatter(fmt_file)
    fh.setFormatter(file_formatter)
    logger.addHandler(fh)

    logger.info(f"Translating dataset to folder { output_dir / fname}")

    translate_dataset(dataset, src_lang, target_lang, output_dir / fname)
