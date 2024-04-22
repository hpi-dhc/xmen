from typing import Iterable
from itertools import groupby
import pandas as pd
import warnings
import regex as re

from .ext.neleval.prepare import SelectAlternatives
from .ext.neleval.document import Document
from .ext.neleval.evaluate import Evaluate, StrictMetricWarning
from .ext.neleval.annotation import Annotation, Candidate
from .ext.neleval.configs import get_measure

# constants for interacting with different neleval measures
_PARTIAL_EVAL_MEASURE_FMT_STRING = "overlap-maxmax:None:span+type+kbid"

# equivalent to strong_all_match
_STRICT_EVAL_MEASURE_FMT_STRING = "sets:None:span+type+kbid"

_NER_STRICT_EVAL_MEASURE_FMT_STRING = "sets:None:span+type"

# constants for interacting with different neleval measures
_NER_PARTIAL_EVAL_MEASURE_FMT_STRING = "overlap-maxmax:None:span+type"

# Any match within the unit / sentence counts
_LOOSE_EVAL_MEASURE_FMT_STRING = "sets:None:docid+kbid"

_KEY_TO_METRIC = {
    "strict": _STRICT_EVAL_MEASURE_FMT_STRING,
    "partial": _PARTIAL_EVAL_MEASURE_FMT_STRING,
    "loose": _LOOSE_EVAL_MEASURE_FMT_STRING,
    "ner_strict": _NER_STRICT_EVAL_MEASURE_FMT_STRING,
    "ner_partial": _NER_PARTIAL_EVAL_MEASURE_FMT_STRING,
}


def _get_word_len(row):
    if row.gt_text:
        return len(" ".join(row.gt_text).split(" "))
    else:
        return None


def _get_entity_info(row):
    res = {}
    res["_word_len"] = _get_word_len(row)
    res["_abbrev"] = (len(row.gt_text) == 1) and bool(re.match("[A-Z]{2,3}", row.gt_text[0])) if row.gt_text else None
    return res


def error_analysis(
    ground_truth: Iterable, prediction: Iterable, tasks: list = ["ner", "nen"], allow_multiple_gold_candidates=False
) -> pd.DataFrame:
    """
    Computes error analysis of entity linking predictions by comparing against the ground truth entities, assuming that the entities are aligned.

    Args:
    - ground_truth: An iterable of dictionaries representing the ground truth entities for each document.
    - prediction: An iterable of dictionaries representing the predicted entities for each document.
    - tasks: list of either 'ner' (for named entity recogition), 'nen' (for entity linking), or both
    - allow_multiple_gold_candidates: A boolean indicating whether multiple ground truth entities per mention are allowed. Defaults to False.

    Returns:
    - A pandas DataFrame containing error analysis of entity linking predictions.
    """

    res = []
    for gt, pred in zip(ground_truth, prediction):
        error_df = _get_error_df(gt["entities"], pred["entities"], tasks, allow_multiple_gold_candidates)
        if "corpus_id" in gt:
            error_df["corpus_id"] = gt["corpus_id"]
        error_df["document_id"] = gt["document_id"]
        res.append(error_df)
    ea_df = pd.concat(res).reset_index().drop(columns="index")
    if len(ea_df) == 0:
        return ea_df
    entity_info = pd.DataFrame(list(ea_df.apply(_get_entity_info, axis=1)))
    return pd.concat([entity_info, ea_df], axis=1)


class EntityMatch:
    """
    Utility class for representing an entity match.
    """

    def __init__(self, offset_start, offset_end, normalized, text, type):
        self.offset_start = offset_start
        self.offset_end = offset_end
        self.normalized = normalized
        self.text = text
        self.type = type

    def __repr__(self) -> str:
        return f"EntityMatch({self.offset_start}, {self.offset_end}, {self.normalized}, {self.text}, {self.type})"


NULL_MATCH = EntityMatch(None, None, None, None, None)


def _get_items(entities):
    """
    Helper function for turning BigBIO dictionary entities into EntityMatch objects.
    """
    return [
        EntityMatch(
            e["offsets"][0][0],
            e["offsets"][-1][1],
            e["normalized"] if len(e["normalized"]) > 0 else [{"db_id": "NIL"}],
            e["text"],
            e["type"],
        )
        for e in sorted(entities, key=lambda e: (e["offsets"], e["type"], e["text"]))
    ]


def _get_match_type(pred, gt) -> str:
    """
    Determines the type of match between a predicted entity and a ground truth entity.
    """
    if pred.offset_start == gt.offset_start and pred.offset_end == gt.offset_end:
        return "tp" if pred.type == gt.type else "le"
    elif (pred.offset_start >= gt.offset_start and pred.offset_start < gt.offset_end) or (
        pred.offset_end > gt.offset_start and pred.offset_end <= gt.offset_end
    ):
        return "be" if pred.type == gt.type else "lbe"
    else:
        return "fp"


def _find_concept_index(pred, gt):
    """
    Finds the index of the best matching concept in the predicted entity for the ground truth entity.
    """
    best_index = None
    if gt.normalized and pred.normalized:
        for g in gt.normalized:
            for i, p in enumerate(pred.normalized):
                if g["db_id"] == p["db_id"]:
                    if not best_index or i < best_index:
                        best_index = i
    return best_index


def _get_best_match(pred, gt_items) -> (str, EntityMatch, int):
    """
    Finds the best matching ground truth entity for a predicted entity.
    """
    best_match = NULL_MATCH
    best_match_type = "fp"
    best_match_index = -1
    best_concept_index = None

    order = ["tp", "be", "le", "lbe", "fp"]

    def cmp_match_type(m1, m2):
        return order.index(m1) - order.index(m2)

    for i, gt in enumerate(gt_items):
        match_type = _get_match_type(pred, gt)
        if match_type == "fp":
            continue
        concept_index = _find_concept_index(pred, gt)
        if (cmp_match_type(match_type, best_match_type) < 0) or (
            cmp_match_type(match_type, best_match_type) == 0
            and (best_concept_index is None or concept_index is not None and concept_index < best_concept_index)
        ):
            best_match = gt
            best_match_type = match_type
            best_match_index = i
            best_concept_index = concept_index

    return best_match, best_match_type, best_match_index


def _record_match(ent_res: list, allow_multiple_gold_candidates: bool, pred, gt, e_match_type):
    """
    Records a match between a predicted entity and a ground truth entity.
    """
    pred_s = pred.offset_start
    pred_e = pred.offset_end
    pred_c = pred.normalized
    pred_text = pred.text
    pred_type = pred.type
    gt_s = gt.offset_start
    gt_e = gt.offset_end
    gt_c = gt.normalized
    gt_text = gt.text
    gt_type = gt.type

    if (not gt_c and pred_c) or e_match_type == "fp":  # false positive
        ent_res.append(
            {
                "pred_start": pred_s,
                "pred_end": pred_e,
                "pred_text": pred_text,
                "pred_type": pred_type,
                "gt_start": None,
                "gt_end": None,
                "gt_text": None,
                "gold_type": None,
                "ner_match_type": e_match_type,
                "gold_concept": None,
                "pred_index": -1,
                "pred_index_score": None,
                "pred_top": None,
                "pred_top_score": None,
            }
        )
        return

    def get_match_result(gt_concept):
        is_not_fn = e_match_type != "fn"
        pred_ids = [c["db_id"] for c in pred_c]
        idx = pred_ids.index(gt_concept["db_id"]) if gt_concept["db_id"] in pred_ids else -1
        return {
            "pred_start": pred_s if is_not_fn else None,
            "pred_end": pred_e if is_not_fn else None,
            "pred_text": pred_text if is_not_fn else None,
            "pred_type": pred_type if is_not_fn else None,
            "gt_start": gt_s,
            "gt_end": gt_e,
            "gt_text": gt_text,
            "ner_match_type": e_match_type,
            "gold_concept": gt_concept,
            "gold_type": gt_type,
            "pred_index": int(idx),
            "pred_index_score": pred_c[idx].get("score", None) if idx >= 0 else None,
            "pred_top": pred_c[0]["db_id"] if len(pred_c) > 0 else None,
            "pred_top_score": pred_c[0].get("score", None) if len(pred_c) > 0 else None,
        }

    if not pred_c and gt_c:
        assert len(gt_c) == 1
        ent_res.append(
            {
                "pred_start": pred_s,
                "pred_end": pred_e,
                "pred_text": pred_text,
                "pred_type": pred_type,
                "gt_start": gt_s,
                "gt_end": gt_e,
                "gt_text": gt_text,
                "ner_match_type": e_match_type,
                "gold_concept": gt_c[0],
                "gold_type": gt_type,
                "pred_index": -1,
                "pred_index_score": None,
                "pred_top": None,
                "pred_top_score": None,
            }
        )
        return

    if allow_multiple_gold_candidates:
        matches = [get_match_result(gt_concept) for gt_concept in gt_c]
        true_matches = [m["pred_index"] for m in matches if m["pred_index"] != -1]
        if true_matches:
            best_idx = min(true_matches)
            best_matches = [m for m in matches if m["pred_index"] == best_idx]
            assert len(best_matches) == 1
            ent_res.append(best_matches[0])
        else:  # No matches for any of the concepts
            ent_res.append(matches[0])

    else:
        for gt_concept in gt_c:
            ent_res.append(get_match_result(gt_concept))


def _get_error_df(gt_ents, pred_ents, tasks, allow_multiple_gold_candidates) -> pd.DataFrame:
    """
    Construct a Pandas DataFrame with the error analysis results from the comparison of two lists of named entities.

    Args:
    - gt_ents (list): A list of dictionaries, each representing a ground truth named entity.
    - pred_ents (list): A list of dictionaries, each representing a predicted named entity. Each dictionary must have the same keys as in 'gt_ents'.
    - tasks: list of either 'ner' (for named entity recogition), 'nen' (for entity linking), or both
    - allow_multiple_gold_candidates (bool): A boolean flag indicating whether to allow multiple ground truth entities to match a single predicted entity. Defaults to False.

    Returns:
    A Pandas DataFrame with the errors.
    """
    gt_items = _get_items(gt_ents)
    gt_matches = [None] * len(gt_items)
    pred_items = _get_items(pred_ents)

    result = []

    for pred in pred_items:
        gt, ner_match_type, gt_index = _get_best_match(pred, gt_items)
        if ner_match_type != "fp":
            gt_matches[gt_index] = ner_match_type
        _record_match(result, allow_multiple_gold_candidates, pred, gt, ner_match_type)

    for i, (gt, gt_match) in enumerate(zip(gt_items, gt_matches)):
        if gt_match is None:
            pred, ner_match_type, idx = _get_best_match(gt, pred_items)
            gt_matches[i] = ner_match_type if idx != -1 else "fn"
            _record_match(result, allow_multiple_gold_candidates, pred, gt, gt_matches[i])

    assert all([e is not None for e in gt_matches])

    res_cols = ["gt_start", "gt_end", "gt_text", "gold_type"]
    if "ner" in tasks:
        res_cols += ["pred_start", "pred_end", "pred_text", "pred_type", "ner_match_type"]
    if "nen" in tasks:
        res_cols += ["gold_concept", "pred_index", "pred_index_score", "pred_top", "pred_top_score"]

    if result:
        df = pd.DataFrame(result)
        df["gold_id"] = df["gold_concept"].map(lambda d: d["db_id"] if d else None)
        df = df.drop_duplicates(
            subset=[
                "gt_start",
                "gt_end",
                "gold_type",
                "pred_start",
                "pred_end",
                "pred_type",
                "gold_id",
                "pred_index",
                "pred_index_score",
                "pred_top",
                "pred_top_score",
            ]
        )
        return df[res_cols].sort_values(["gt_start", "gt_end"])
    else:
        return pd.DataFrame(columns=res_cols)


def get_synset(kb, scui):
    kb_ix = kb.cui_to_entity[str(scui)]
    return set([kb_ix.canonical_name] + kb_ix.aliases)


def evaluate(
    ground_truth: Iterable,
    prediction: Iterable,
    allow_multiple_gold_candidates=False,
    top_k_predictions=1,
    threshold=None,
    ner_only=False,
    metrics=["strict"],
) -> dict:
    """
    Evaluate the performance of the system.

    Args:
    - ground_truth: An iterable of ground truth annotations.
    - prediction: An iterable of predicted annotations.
    - allow_multiple_gold_candidates: A boolean indicating whether multiple
            gold standard candidates are allowed for each annotation. Defaults to False.
    - top_k_predictions: An integer indicating the maximum number of predicted
            candidates to consider for each annotation. Defaults to 1.
    - threshold: A float indicating the minimum confidence threshold to consider
            for each predicted candidate. Defaults to None.
    - ner_only: A boolean indicating whether to only consider NER tags. Defaults to False.
    - metrics: A list of strings indicating which metrics to use for evaluation. Defaults to ["strict"].

    Returns:
    - A dictionary containing the evaluation results for each specified metric.
    """
    with warnings.catch_warnings():
        # Ignore division by zero problems raised by neleval
        warnings.filterwarnings("ignore", category=StrictMetricWarning)
        if ner_only:
            allow_multiple_gold_candidates = False
        system_docs = list(
            _to_nel_eval(
                prediction,
                allow_multiple_candidates=False,
                top_k=top_k_predictions,
                threshold=threshold,
                ner_only=ner_only,
            )
        )
        gold_docs = list(
            _to_nel_eval(
                ground_truth,
                allow_multiple_candidates=allow_multiple_gold_candidates,
                top_k=None,
                threshold=None,
                ner_only=ner_only,
            )
        )

        num_annos_system = sum([len(a.candidates) for d in system_docs for a in d.annotations])
        num_annos_gold = sum([len(a.candidates) for d in gold_docs for a in d.annotations])

        if allow_multiple_gold_candidates:
            SelectAlternatives(system_docs, gold_docs)()

        if metrics == "all":
            metrics = list(_KEY_TO_METRIC.keys())

        eval_fn = Evaluate(
            system=system_docs,
            gold=gold_docs,
            measures=[get_measure(_KEY_TO_METRIC[m]) for m in metrics],
        )
        res = eval_fn()
        for v in res.values():
            v["n_docs_system"] = len(system_docs)
            v["n_annos_system"] = num_annos_system
            v["n_docs_gold"] = len(gold_docs)
            v["n_annos_gold"] = num_annos_gold

        return {k: res[_KEY_TO_METRIC[k]] for k in metrics}


def _to_nel_eval(
    units: Iterable,
    allow_multiple_candidates: bool = False,
    top_k: int = None,
    threshold: float = None,
    ner_only=False,
) -> Iterable:
    """
    Converts a list of units to a list of Document objects in neleval format.

    Args:
    - units: An iterable of units containing entity annotations.
    - allow_multiple_candidates: A boolean indicating whether to allow multiple candidate entities.
    - top_k: An integer indicating the number of top candidates to keep.
    - threshold: A float indicating the score threshold for keeping candidates.
    - ner_only: A boolean indicating whether to ignore entity types.

    Returns:
    - An iterable of Document objects in neleval format.
    """
    for u in units:
        entities = sorted(u["entities"], key=lambda e: e["offsets"])
        cid = u["corpus_id"] if "corpus_id" in u and u["corpus_id"] else ""
        unit_id = cid + u["document_id"]
        annotations = []
        for e in entities:
            if "normalized" in e:
                for c in e["normalized"]:
                    if not "type" in c:
                        c["type"] = None
                    if not "score" in c:
                        c["score"] = None
            else:
                e["normalized"] = []
            start, end = min([o[0] for o in e["offsets"]]), max([o[1] for o in e["offsets"]])
            if ner_only:
                anno = Annotation(unit_id, start, end, [Candidate(None, None, e["type"])])
                annotations.append(anno)
            else:
                candidates = []
                for c in e["normalized"]:
                    if not threshold or not c["score"] or c["score"] >= threshold:
                        candidates.append(Candidate(c["db_id"], c["score"], c["type"]))
                candidates.sort(key=lambda c: -c.score if c.score else 0)
                if top_k:
                    candidates = candidates[:top_k]
                if allow_multiple_candidates:
                    anno = Annotation(unit_id, start, end, candidates)
                    annotations.append(anno)
                else:
                    for cand in candidates:
                        anno = Annotation(unit_id, start, end, [cand])
                        annotations.append(anno)
        yield Document(unit_id, annotations)


def evaluate_at_k(ground_truth, pred, eval_k=[1, 2, 4, 8, 16, 32, 64], silent=False):
    """
    Runs evaluation for different values of k (number of candidates)

    Args:
    - ground_truth: ground truth annotations
    - pred: predicted candidates
    - eval_k: different values of k to evaluate
    - silent: whether to print results to standard output

    Returns:
    - a dictionary of evaluation results for each k
    """
    res = {}
    for ki in eval_k:
        eval_res = evaluate(ground_truth, pred, top_k_predictions=ki)
        if not silent:
            print(f"Recall@{ki}", eval_res["strict"]["recall"])
        res[ki] = eval_res
    return res
