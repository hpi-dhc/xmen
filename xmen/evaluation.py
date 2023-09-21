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
_PARTIAL_EVAL_MEASURE_FMT_STRING = "overlap-maxmax:None:span+kbid"

# equivalent to strong_all_match
_STRICT_EVAL_MEASURE_FMT_STRING = "sets:None:span+kbid"

_NER_STRICT_EVAL_MEASURE_FMT_STRING = "sets:None:span"

# constants for interacting with different neleval measures
_NER_PARTIAL_EVAL_MEASURE_FMT_STRING = "overlap-maxmax:None:span"

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
    return len(" ".join(row.gt_text).split(" "))


def _get_entity_info(row):
    res = {}
    res["_word_len"] = _get_word_len(row)
    res["_abbrev"] = (len(row.gt_text) == 1) and bool(re.match("[A-Z]{2,3}", row.gt_text[0]))
    return res


def error_analysis(ground_truth: Iterable, prediction: Iterable, allow_multiple_gold_candidates=False) -> pd.DataFrame:
    """
    Computes error analysis of entity linking predictions by comparing against the ground truth entities, assuming that the entities are aligned.

    Args:
    - ground_truth: An iterable of dictionaries representing the ground truth entities for each document.
    - prediction: An iterable of dictionaries representing the predicted entities for each document.
    - allow_multiple_gold_candidates: A boolean indicating whether multiple ground truth entities per mention are allowed. Defaults to False.

    Returns:
    - A pandas DataFrame containing error analysis of entity linking predictions.
    """

    res = []
    for gt, pred in zip(ground_truth, prediction):
        error_df = _get_error_df(gt["entities"], pred["entities"], allow_multiple_gold_candidates)
        if "corpus_id" in gt:
            error_df["corpus_id"] = gt["corpus_id"]
        error_df["document_id"] = gt["document_id"]
        res.append(error_df)
    ea_df = pd.concat(res).reset_index().drop(columns="index")
    entity_info = pd.DataFrame(list(ea_df.apply(_get_entity_info, axis=1)))
    return pd.concat([entity_info, ea_df], axis=1)


def _get_error_df(gt_ents, pred_ents, allow_multiple_gold_candidates=False) -> pd.DataFrame:
    """
    Construct a Pandas DataFrame with the error analysis results from the comparison of two lists of named entities.

    Args:
    - gt_ents (list): A list of dictionaries, each representing a ground truth named entity.
    - pred_ents (list): A list of dictionaries, each representing a predicted named entity. Each dictionary must have the same keys as in 'gt_ents'.
    - allow_multiple_gold_candidates (bool): A boolean flag indicating whether to allow multiple ground truth entities to match a single predicted entity. Defaults to False.

    Returns:
    A Pandas DataFrame with the errors.
    """

    def get_items(entities):
        return [
            (
                e["offsets"][0][0],
                e["offsets"][-1][1],
                e["normalized"] if len(e["normalized"]) > 0 else [{"db_id": "NIL"}],
                e["text"],
                e["type"],
            )
            for e in sorted(entities, key=lambda e: (e["offsets"], e["type"], e["text"]))
        ]

    gt_items = get_items(gt_ents)
    pred_items_unmatched = get_items(pred_ents)

    # If we have entities with multiple normalizations, we want to order by best match order
    pred_items = []
    for key, group in groupby(pred_items_unmatched, lambda x: (x[0], x[1], x[3], x[4])):
        group = list(group)
        if len(group) == 1:
            pred_items.append(group[0])
        else:

            def best_match_index(pred_normalized):
                matches = [g for g in gt_items if (g[0], g[1], g[3], g[4]) == key]
                best_match = len(pred_normalized)
                best_index = len(pred_normalized)
                for i, m in enumerate(matches):
                    for j, p in enumerate(pred_normalized):
                        if p["db_id"] == m[2][0]["db_id"]:
                            if j < best_index:
                                best_index = j
                                best_match = i
                return best_match

            best_match_indices = sorted([(pred, best_match_index(pred[2])) for pred in group], key=lambda t: t[1])
            matched = [None] * len(best_match_indices)
            for p, i in best_match_indices:
                if i < len(matched) and matched[i] == None:
                    matched[i] = p
                else:
                    found = False
                    for j in range(len(matched)):
                        if not found and matched[j] == None:
                            matched[j] = p
                            found = True
                    assert found == True

            pred_items.extend(matched)
    assert len(pred_items) == len(pred_items_unmatched)

    ent_res = []

    def record_match(
        pred_s: int, pred_e: int, pred_c, pred_text, pred_type, gt_s, gt_e, gt_c, gt_text, gt_type, e_match_type
    ):
        if not gt_c:  # false positive
            ent_res.append(
                {
                    "pred_start": pred_s,
                    "pred_end": pred_e,
                    "pred_text": pred_text,
                    "gt_start": None,
                    "gt_end": None,
                    "gt_text": None,
                    "entity_match_type": e_match_type,
                    "gold_concept": None,
                    "gold_type": None,
                    "pred_index": -1,
                    "pred_index_score": None,
                    "pred_top": None,
                    "pred_top_score": None,
                }
            )
            return

        def get_match_result(gt_concept):
            pred_ids = [c["db_id"] for c in pred_c]
            idx = pred_ids.index(gt_concept["db_id"]) if gt_concept["db_id"] in pred_ids else -1
            return {
                "pred_start": pred_s,
                "pred_end": pred_e,
                "pred_text": pred_text,
                "gt_start": gt_s,
                "gt_end": gt_e,
                "gt_text": gt_text,
                "entity_match_type": e_match_type,
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
                    "gt_start": gt_s,
                    "gt_end": gt_e,
                    "gt_text": gt_text,
                    "entity_match_type": e_match_type,
                    "gold_concept": gt_c[0],
                    "gold_type": "",
                    "pred_index": -1,
                    "pred_index_score": None,
                    "pred_top": None,
                    "pred_top_score": None,
                }
            )

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

    # Initialize variables
    gt_s = gt_c = gt_e = gt_text = None

    while gt_items or pred_items:
        if not gt_items:
            e_match_type = "fp"
            pred_s, pred_e, pred_c, pred_text, pred_type = pred_items.pop()
            record_match(
                pred_s,
                pred_e,
                pred_c,
                pred_text,
                pred_type,
                -1,
                -1,
                None,
                None,
                None,
                e_match_type,
            )
        elif not pred_items:
            e_match_type = "fn"
            gt_s, gt_e, gt_c, gt_text, gt_type = gt_items.pop()
            record_match(
                pred_s,
                pred_e,
                pred_c,
                pred_text,
                pred_type,
                gt_s,
                gt_e,
                gt_c,
                gt_text,
                gt_type,
                e_match_type,
            )
        else:
            pred_s, pred_e, pred_c, pred_text, pred_type = pred_items[0]
            gt_s, gt_e, gt_c, gt_text, gt_type = gt_items[0]

            if pred_s == gt_s and pred_e == gt_e:
                e_match_type = "tp"
                record_match(
                    pred_s,
                    pred_e,
                    pred_c,
                    pred_text,
                    pred_type,
                    gt_s,
                    gt_e,
                    gt_c,
                    gt_text,
                    gt_type,
                    e_match_type,
                )
                gt_items.pop(0)
                pred_items.pop(0)
            elif pred_s >= gt_e:
                e_match_type = "fn"
                record_match(
                    pred_s,
                    pred_e,
                    pred_c,
                    pred_text,
                    pred_type,
                    gt_s,
                    gt_e,
                    gt_c,
                    gt_text,
                    gt_type,
                    e_match_type,
                )
                gt_items.pop(0)
            elif pred_e <= gt_s:
                e_match_type = "fp"
                record_match(
                    pred_s,
                    pred_e,
                    pred_c,
                    pred_text,
                    pred_type,
                    gt_s,
                    gt_e,
                    gt_c,
                    gt_text,
                    gt_type,
                    e_match_type,
                )
                pred_items.pop(0)
            else:
                e_match_type = "be"
                pred_s, pred_e, pred_c, pred_text, pred_type = pred_items.pop(0)
                gt_s, gt_e, gt_c, gt_text, gt_type = gt_items.pop(0)
                while True:
                    record_match(
                        pred_s,
                        pred_e,
                        pred_c,
                        pred_text,
                        pred_type,
                        gt_s,
                        gt_e,
                        gt_c,
                        gt_text,
                        gt_type,
                        e_match_type,
                    )
                    if pred_e < gt_e:
                        if pred_items and pred_items[0][0] <= gt_e:
                            pred_s, pred_e, pred_c, pred_text, pred_type = pred_items.pop(0)
                        else:
                            if gt_items:
                                gt_items.pop(0)
                            break
                    elif gt_e < pred_e:
                        if gt_items and gt_items[0][0] <= pred_e:
                            gt_s, gt_e, gt_c, gt_text, gt_type = gt_items.pop(0)
                        else:
                            if pred_items:
                                pred_items.pop(0)
                            break
                    else:
                        break

    return pd.DataFrame(ent_res)


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
                anno = Annotation(unit_id, start, end, [])
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
