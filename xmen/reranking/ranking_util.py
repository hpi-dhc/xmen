from tqdm.autonotebook import tqdm


def find_context(passages, offsets):
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
    flat_candidate_ds = flat_candidate_ds.add_column("synonyms", synonyms)

    flat_candidate_ds = flat_candidate_ds.add_column("label", flat_ground_truth["label"])

    return flat_candidate_ds, doc_index


def get_candidates(examples, doc_indices, expand_abbreviations: bool):
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
