import pandas as pd


def get_concept_details(cfg) -> dict:
    # load snomed terms
    mugit_snomed_path = cfg.dict.custom.mugit_path
    mugit = pd.read_csv(mugit_snomed_path, sep="\t", header=None, encoding="utf8")
    mugit.columns = ["snomed_id", "term_id", "en", "de"]
    mugit.sort_index(inplace=True)
    mugit.dropna(inplace=True)

    # get lang from cfg. If no lang specified, use both
    lang = cfg.dict.custom.lang if "lang" in cfg.dict.custom else ["de", "en"]

    concept_details = {}
    # both languages selected, english ones go to canonical and all different terms to alias
    if lang == ["de", "en"]:
        for _, entry in mugit.iterrows():
            sid = entry.snomed_id
            if not sid in concept_details:
                concept_details[sid] = {
                    "concept_id": sid,
                    "canonical_name": entry.en,
                    "types": [],
                    "aliases": [entry.de],
                }
            elif sid in concept_details:
                if (
                    entry.en not in concept_details[sid]["aliases"]
                    and entry.en != concept_details[sid]["canonical_name"]
                ):
                    concept_details[sid]["aliases"].append(entry.en)
                if entry.de not in concept_details[sid]["aliases"]:
                    concept_details[sid]["aliases"].append(entry.de)

    # just one language
    elif lang == ["en"] or lang == ["de"]:
        l = lang[0]
        for _, entry in mugit.iterrows():
            sid = entry.snomed_id
            if not sid in concept_details:
                concept_details[sid] = {"concept_id": sid, "canonical_name": entry[l], "types": [], "aliases": []}
            elif sid in concept_details:
                if (
                    entry[l] not in concept_details[sid]["aliases"]
                    and entry[l] != concept_details[sid]["canonical_name"]
                ):
                    concept_details[sid]["aliases"].append(entry[l])

    else:
        print("Languages not supported by MUGIT")

    return concept_details
