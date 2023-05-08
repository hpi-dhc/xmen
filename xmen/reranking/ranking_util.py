from tqdm.autonotebook import tqdm
import datasets


def find_context(passages, offsets):
    """
    Finds the left and right context of a target span within a list of passages.

    - passages: a list of passages containing text and offsets
    - offsets: a list of start and end offsets representing the target span

    Returns:
    - The left context target span, and the right context of the target span within the passages.

    Raises:
    - AssertionError if the function fails to find the target span within the passages.
    """
    m_start, m_end = min([o[0] for o in offsets]), max([o[1] for o in offsets])
    for p in passages:
        for t, o in zip(p["text"], p["offsets"]):
            t_start, t_end = o[0], o[1]
            if m_start >= t_start and m_end <= t_end:
                l_context = t[0 : (m_start - t_start)]
                m = t[(m_start - t_start) : (m_end - t_start + 1)]
                r_context = t[(m_end - t_start + 1) :]
                return l_context, m, r_context
    assert False


def get_flat_candidate_ds(candidate_ds, ground_truth, expand_abbreviations, kb):
    """
    Expands the given candidate dataset with additional features, and removes the irrelevant columns.

    Args:
    - candidate_ds: the original dataset containing candidates
    - ground_truth: the ground truth dataset
    - expand_abbreviations: a boolean flag indicating whether or not to expand abbreviations
    - kb: the knowledge base

    Returns:
    - A flat candidate dataset with additional columns: "synonyms" and "label".
    - A document index.
    """
    try:
        datasets.disable_progress_bar()
        flat_candidate_ds = candidate_ds.map(
            lambda e, i: get_candidates(e, i, expand_abbreviations),
            batched=True,
            remove_columns=candidate_ds.column_names,
            with_indices=True,
            load_from_cache_file=False,
        )

        doc_index = flat_candidate_ds["doc_index"]
        flat_candidate_ds = flat_candidate_ds.remove_columns(["doc_index"])

        flat_ground_truth = ground_truth.map(
            lambda e, i: get_candidates(e, i, False),
            batched=True,
            remove_columns=candidate_ds.column_names,
            with_indices=True,
            load_from_cache_file=False,
        )
        flat_ground_truth = flat_ground_truth.rename_column("candidates", "label")

        synonyms = [
            [[kb.cui_to_entity[cui].canonical_name] + kb.cui_to_entity[cui].aliases for cui in entry]
            for entry in tqdm(flat_candidate_ds["candidates"])
        ]
        semantic_types = [
            [kb.cui_to_entity[cui].types for cui in entry] for entry in tqdm(flat_candidate_ds["candidates"])
        ]
        flat_candidate_ds = flat_candidate_ds.add_column("synonyms", synonyms)
        flat_candidate_ds = flat_candidate_ds.add_column("types", semantic_types)

        flat_candidate_ds = flat_candidate_ds.add_column("label", flat_ground_truth["label"])
    finally:
        datasets.enable_progress_bar()

    return flat_candidate_ds, doc_index


def get_candidates(examples, doc_indices, expand_abbreviations: bool, check_spans=False):
    """
    Retrieves all candidate entities for each mention in the given examples.

    Args:
    - examples: a dataset containing mentions and their corresponding passages
    - doc_indices: a list of indices that correspond to the original documents in the dataset
    - expand_abbreviations: a boolean flag indicating whether or not to expand abbreviations

    Returns:
    - A dictionary containing information about each mention, including its candidates, scores, and context.
    """
    candidates = []
    scores = []
    mentions = []
    doc_ixs = []
    left_contexts = []
    right_contexts = []
    for doc_passages, l, doc_ix in zip(examples["passages"], examples["entities"], doc_indices):
        for e_id, e in enumerate(l):
            doc_ixs.append((doc_ix, e_id))
            context_left, mention, context_right = find_context(doc_passages, e["offsets"])
            if check_spans:
                for mention_text in e["text"]:
                    assert mention_text in mention, (mention_text, mention)
            e_candidates = [n["db_id"] for n in e["normalized"]]
            candidates.append(e_candidates)
            e_scores = [n.get("score", None) for n in e["normalized"]]
            scores.append(e_scores)
            m_text = " ".join(e["text"])
            if expand_abbreviations and e["long_form"]:
                m_text += f' ({e["long_form"]})'
            mentions.append(m_text)
            left_contexts.append(context_left)
            right_contexts.append(context_right)
    return {
        "mention": mentions,
        "candidates": candidates,
        "scores": scores,
        "doc_index": doc_ixs,
        "context_left": left_contexts,
        "context_right": right_contexts,
    }
