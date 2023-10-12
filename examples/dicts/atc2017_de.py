import pandas as pd
from pathlib import Path
from collections import defaultdict
import openpyxl
import xml.etree.ElementTree as ET
from xmen.log import logger


def get_concept_details(cfg) -> dict:
    path = cfg.dict.custom.atc_path
    wb = openpyxl.load_workbook(f"{path}/ATC GKV AI 2017.xlsx")
    sheet = wb["WIdO-Index sortiert_2017"]
    sort = pd.DataFrame(sheet.values)
    sort = sort.rename(columns={0: "code", 2: "text", 4: "DDD_Info"})
    sort.drop(0, axis=0, inplace=True)
    sort.dropna(axis=0, how="all", inplace=True)
    sort.drop(sort[sort.text.isnull()].index, inplace=True)
    sort.drop(sort[sort.code.isnull()].index, inplace=True)

    sort["code"] = sort["code"].str.rstrip()
    sort = sort[sort["code"].str.len() > 3]

    if drug_bank_xml := cfg.dict.custom.get("drug_bank_xml", None):
        logger.info("Extending ATC by DrugBank synonyms")

        ns = {"": "http://www.drugbank.ca"}
        tree = ET.parse(drug_bank_xml)

        atc2name = defaultdict(set)

        for drug in tree.getroot():
            aliases = set()

            syns = [s.text for s in drug.findall("synonyms/synonym", ns)]
            aliases.update(syns)

            products = [p.text for p in drug.findall("products/product/name", ns)]
            aliases.update(products)

            mixtures = [p.text for p in drug.findall("mixtures/mixture/name", ns)]
            aliases.update(mixtures)

            international = [i.text for i in drug.findall("international-brands/international-brand/name", ns)]
            aliases.update(international)

            atc_codes = drug.find("atc-codes", ns)
            for atc_code in atc_codes:
                atc2name[atc_code.get("code")].update(aliases)

    logger.info("Loading mapping of replaced ATC codes")
    mapping = {}
    for prev, new in pd.read_csv(Path(__file__).parent / "atc_replacement.csv")[
        ["Previous ATC code", "New ATC code"]
    ].itertuples(index=False):
        mapping[prev] = new
        mapping[new] = prev

    logger.info("Building concept dictionary")
    concept_details = {}
    for _, entry in sort.iterrows():
        sid = entry.code
        if not sid in concept_details:
            concept_details[sid] = {"concept_id": sid, "canonical_name": entry.text, "types": [], "aliases": []}
        elif sid in concept_details:
            concept_details[sid]["aliases"].append(entry.text)
        if drug_bank_xml:
            for alias in atc2name[sid]:
                if not alias in concept_details[sid]["aliases"]:
                    concept_details[sid]["aliases"].append(alias)

    result = concept_details.copy()

    for sid, concept in concept_details.items():
        if sid in mapping:
            new_sid = mapping[sid]
            if new_sid in result:
                for a in concept["aliases"]:
                    if not a in result[new_sid]["aliases"]:
                        result[new_sid]["aliases"].append(a)
            else:
                new_concept = concept.copy()
                new_concept["concept_id"] = new_sid
                result[new_sid] = new_concept
    return result
