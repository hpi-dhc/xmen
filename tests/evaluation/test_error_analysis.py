from xmen import evaluation
from xmen.data import make_document, Entity, Concept


def test_error_analysis():
    gt = [
        make_document(
            [
                Entity([[11, 17]], "entity", concepts=[Concept("c1", db_name="UMLS")]),
                Entity([[11, 17]], "entity", concepts=[Concept("c2", db_name="UMLS")]),
            ]
        )
    ]
    pred = [
        make_document(
            [
                Entity([[11, 17]], "entity", concepts=[Concept("c1", db_name="UMLS")]),
                Entity([[11, 17]], "entity", concepts=[Concept("c2", db_name="UMLS")]),
            ]
        )
    ]
    assert (evaluation.error_analysis(gt, pred).pred_index == 0).all()


def test_error_analysis_order():
    gt = [
        make_document(
            [
                Entity([[11, 17]], "entity", concepts=[Concept("c2", db_name="UMLS")]),
                Entity([[11, 17]], "entity", concepts=[Concept("c1", db_name="UMLS")]),
            ]
        )
    ]
    pred = [
        make_document(
            [
                Entity([[11, 17]], "entity", concepts=[Concept("c1", db_name="UMLS")]),
                Entity([[11, 17]], "entity", concepts=[Concept("c2", db_name="UMLS")]),
            ]
        )
    ]
    assert (evaluation.error_analysis(gt, pred).pred_index == 0).all()


def test_error_analysis_best_ranking():
    gt = [
        make_document(
            [
                Entity([[11, 17]], "entity", concepts=[Concept("c2", db_name="UMLS")]),
                Entity([[11, 17]], "entity", concepts=[Concept("c1", db_name="UMLS")]),
            ]
        )
    ]
    pred = [
        make_document(
            [
                Entity([[11, 17]], "entity", concepts=[Concept("c1", db_name="UMLS")]),
                Entity([[11, 17]], "entity", concepts=[Concept("c2", db_name="UMLS")]),
            ]
        )
    ]
    assert (evaluation.error_analysis(gt, pred).pred_index == 0).all()


def test_error_analysis_pred_index():
    gt = [
        make_document(
            [
                Entity([[11, 17]], "entity", concepts=[Concept("c1", db_name="UMLS")]),
            ]
        )
    ]
    pred = [
        make_document(
            [
                Entity(
                    [[11, 17]],
                    "entity",
                    concepts=[Concept("c1", db_name="UMLS", score=0.99), Concept("c2", db_name="UMLS", score=0.98)],
                ),
            ]
        )
    ]
    assert (evaluation.error_analysis(gt, pred).pred_index == 0).all()


def test_error_analysis_ner_boundaries():
    gt = [
        make_document(
            [
                Entity([[0, 5]], "entity", concepts=[]),
                Entity([[20, 30]], "entity", concepts=[]),
            ]
        )
    ]
    pred = [
        make_document(
            [
                Entity([[1, 4]], "entity", concepts=[]),
                Entity([[25, 35]], "entity", concepts=[]),
            ]
        )
    ]
    ea_df = evaluation.error_analysis(gt, pred)
    assert len(ea_df) == 2
    assert ea_df.ner_match_type.tolist() == ["be", "be"]


def test_all_tp():
    gt = [make_document([Entity([[0, 5]], "entity", concepts=[])])]
    pred = [make_document([Entity([[0, 5]], "entity", concepts=[])])]
    ea_df = evaluation.error_analysis(gt, pred)
    assert len(ea_df) == 1
    assert ea_df.ner_match_type.tolist() == ["tp"]


def test_all_fp():
    gt = [make_document([])]
    pred = [make_document([Entity([[0, 5]], "entity", concepts=[])])]
    ea_df = evaluation.error_analysis(gt, pred)
    assert len(ea_df) == 1
    assert ea_df.ner_match_type.tolist() == ["fp"]


def test_all_fn():
    gt = [make_document([Entity([[0, 5]], "entity", concepts=[])])]
    pred = [make_document([])]
    ea_df = evaluation.error_analysis(gt, pred)
    assert len(ea_df) == 1
    assert ea_df.ner_match_type.tolist() == ["fn"]


def test_mixed_errors():
    gt = [make_document([Entity([[0, 5]], "entity", concepts=[])])]
    pred = [make_document([Entity([[0, 4]], "entity", concepts=[]), Entity([[10, 15]], "entity", concepts=[])])]
    ea_df = evaluation.error_analysis(gt, pred)
    assert len(ea_df) == 2
    assert set(ea_df.ner_match_type.tolist()) == set(["be", "fp"])


def test_all_be():
    gt = [make_document([Entity([[0, 5]], "entity", concepts=[])])]
    pred = [make_document([Entity([[0, 4]], "entity", concepts=[])])]
    ea_df = evaluation.error_analysis(gt, pred)
    assert len(ea_df) == 1
    assert ea_df.ner_match_type.tolist() == ["be"]


def test_no_entities():
    gt = [make_document([])]
    pred = [make_document([])]
    ea_df = evaluation.error_analysis(gt, pred)
    assert len(ea_df) == 0


def test_multiple_same_errors():
    gt = [make_document([Entity([[0, 5]], "entity", concepts=[]), Entity([[6, 11]], "entity", concepts=[])])]
    pred = [make_document([Entity([[0, 4]], "entity", concepts=[]), Entity([[6, 10]], "entity", concepts=[])])]
    ea_df = evaluation.error_analysis(gt, pred)
    assert len(ea_df) == 2
    assert ea_df.ner_match_type.tolist() == ["be", "be"]


def test_multiple_tp():
    gt = [make_document([Entity([[0, 5]], "entity", concepts=[]), Entity([[6, 11]], "entity", concepts=[])])]
    pred = [make_document([Entity([[0, 5]], "entity", concepts=[]), Entity([[6, 11]], "entity", concepts=[])])]
    ea_df = evaluation.error_analysis(gt, pred)
    assert len(ea_df) == 2
    assert ea_df.ner_match_type.tolist() == ["tp", "tp"]


def test_multiple_types():
    gt = [make_document([Entity([[0, 5]], "entity", concepts=[]), Entity([[10, 15]], "entity", concepts=[])])]
    pred = [make_document([Entity([[0, 5]], "entity", concepts=[]), Entity([[15, 20]], "entity", concepts=[])])]
    ea_df = evaluation.error_analysis(gt, pred)
    assert ea_df.ner_match_type.tolist() == ["tp", "fn", "fp"]


def test_mixed_five_entities():
    gt = [
        make_document(
            [
                Entity([[0, 5]], "entity", concepts=[]),
                Entity([[10, 15]], "entity", concepts=[]),
                Entity([[20, 25]], "entity", concepts=[]),
                Entity([[30, 35]], "entity", concepts=[]),
                Entity([[40, 45]], "entity", concepts=[]),
            ]
        )
    ]
    pred = [
        make_document(
            [
                Entity([[0, 4]], "entity", concepts=[]),  # be
                Entity([[11, 15]], "entity", concepts=[]),  # be
                Entity([[20, 25]], "entity", concepts=[]),  # tp
                Entity([[36, 40]], "entity", concepts=[]),  # fp
                Entity([[45, 50]], "entity", concepts=[]),  # fp
            ]
        )
    ]
    ea_df = evaluation.error_analysis(gt, pred)
    assert len(ea_df) == 7
    assert ea_df.ner_match_type.tolist() == ["be", "be", "tp", "fn", "fn", "fp", "fp"]


def test_mixed_five_entities_le_lbe():
    gt = [
        make_document(
            [
                Entity([[0, 5]], "entity", concepts=[], entity_type="type1"),
                Entity([[10, 15]], "entity", concepts=[], entity_type="type2"),
                Entity([[20, 25]], "entity", concepts=[], entity_type="type2"),
                Entity([[30, 35]], "entity", concepts=[], entity_type="type1"),
                Entity([[40, 45]], "entity", concepts=[], entity_type="type1"),
            ]
        )
    ]
    pred = [
        make_document(
            [
                Entity([[0, 4]], "entity", concepts=[], entity_type="type1"),  # be
                Entity([[11, 15]], "entity", concepts=[], entity_type="type1"),  # be
                Entity([[20, 25]], "entity", concepts=[], entity_type="type1"),  # tp
                Entity([[36, 40]], "entity", concepts=[], entity_type="type2"),  # fp
                Entity([[45, 50]], "entity", concepts=[], entity_type="type1"),  # fp
            ]
        )
    ]
    ea_df = evaluation.error_analysis(gt, pred)
    assert len(ea_df) == 7
    assert ea_df.ner_match_type.tolist() == ["be", "lbe", "le", "fn", "fn", "fp", "fp"]


def test_mixed_five_entities_boundaries():
    gt = [
        make_document(
            [
                Entity([[0, 5]], "entity", concepts=[]),
                Entity([[6, 10]], "entity", concepts=[]),
                Entity([[11, 15]], "entity", concepts=[]),
                Entity([[30, 35]], "entity", concepts=[]),
                Entity([[40, 45]], "entity", concepts=[]),
            ]
        )
    ]
    pred = [
        make_document(
            [
                Entity([[0, 7]], "entity", concepts=[]),  # be
                Entity([[8, 13]], "entity", concepts=[]),  # be
                Entity([[36, 40]], "entity", concepts=[]),  # fp
                Entity([[45, 50]], "entity", concepts=[]),  # fp
            ]
        )
    ]
    ea_df = evaluation.error_analysis(gt, pred)
    assert len(ea_df) == 7
    assert ea_df.ner_match_type.tolist() == ["be", "be", "be", "fn", "fn", "fp", "fp"]


def test_mixed_five_entities_boundaries_best_match():
    gt = [
        make_document(
            [
                Entity([[0, 5]], "entity", concepts=[], entity_type="type2"),  # lbe
                Entity([[6, 10]], "entity", concepts=[], entity_type="type1"),
                Entity([[11, 15]], "entity", concepts=[], entity_type="type1"),
                Entity([[30, 35]], "entity", concepts=[], entity_type="type1"),
                Entity([[40, 45]], "entity", concepts=[], entity_type="type1"),
            ]
        )
    ]
    pred = [
        make_document(
            [
                Entity([[0, 7]], "entity", concepts=[], entity_type="type1"),  # be
                Entity([[8, 13]], "entity", concepts=[], entity_type="type1"),  # be
                Entity([[36, 40]], "entity", concepts=[], entity_type="type1"),  # fp
                Entity([[45, 50]], "entity", concepts=[], entity_type="type1"),  # fp
            ]
        )
    ]
    ea_df = evaluation.error_analysis(gt, pred)
    assert len(ea_df) == 8
    assert ea_df.ner_match_type.tolist() == ["lbe", "be", "be", "be", "fn", "fn", "fp", "fp"]


if __name__ == "__main__":
    test_error_analysis()
