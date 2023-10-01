from xmen import evaluation
from xmen.data import make_document, Entity, Concept


def test_empty():
    gt = []
    pred = []
    metrics = evaluation.evaluate(gt, pred, metrics="all")
    assert metrics["strict"]["recall"] == 0.0
    assert metrics["strict"]["recall"] == 0.0
    assert metrics["strict"]["fscore"] == 0.0


def test_equal():
    gt = [make_document([Entity([[11, 17]], "entity", concepts=[Concept("c1", db_name="UMLS")])])]
    pred = gt
    metrics = evaluation.evaluate(gt, pred, metrics="all")
    assert metrics["strict"]["precision"] == 1.0
    assert metrics["strict"]["recall"] == 1.0
    assert metrics["strict"]["fscore"] == 1.0
    assert metrics["strict"]["ptp"] == 1
    assert metrics["strict"]["fp"] == 0
    assert metrics["strict"]["rtp"] == 1
    assert metrics["strict"]["fn"] == 0

    assert metrics["partial"]["precision"] == 1.0
    assert metrics["partial"]["recall"] == 1.0
    assert metrics["partial"]["fscore"] == 1.0
    assert metrics["partial"]["ptp"] == 1
    assert metrics["partial"]["fp"] == 0
    assert metrics["partial"]["rtp"] == 1
    assert metrics["partial"]["fn"] == 0

    assert metrics["loose"]["precision"] == 1.0
    assert metrics["loose"]["recall"] == 1.0
    assert metrics["loose"]["fscore"] == 1.0
    assert metrics["loose"]["ptp"] == 1
    assert metrics["loose"]["fp"] == 0
    assert metrics["loose"]["rtp"] == 1
    assert metrics["loose"]["fn"] == 0


def test_false_positive():
    gt = []
    pred = [make_document([Entity([[11, 17]], "entity", concepts=[Concept("c1", db_name="UMLS")])])]

    metrics = evaluation.evaluate(gt, pred, metrics="all")
    assert metrics["strict"]["precision"] == 0.0
    assert metrics["strict"]["recall"] == 0.0
    assert metrics["strict"]["fscore"] == 0.0
    assert metrics["strict"]["ptp"] == 0
    assert metrics["strict"]["fp"] == 1
    assert metrics["strict"]["rtp"] == 0
    assert metrics["strict"]["fn"] == 0

    assert metrics["partial"]["precision"] == 0.0
    assert metrics["partial"]["recall"] == 0.0
    assert metrics["partial"]["fscore"] == 0.0
    assert metrics["partial"]["ptp"] == 0
    assert metrics["partial"]["fp"] == 1
    assert metrics["partial"]["rtp"] == 0
    assert metrics["partial"]["fn"] == 0

    assert metrics["loose"]["precision"] == 0.0
    assert metrics["loose"]["recall"] == 0.0
    assert metrics["loose"]["fscore"] == 0.0
    assert metrics["loose"]["ptp"] == 0
    assert metrics["loose"]["fp"] == 1
    assert metrics["loose"]["rtp"] == 0
    assert metrics["loose"]["fn"] == 0


def test_false_negative():
    gt = [make_document([Entity([[11, 17]], "entity", concepts=[Concept("c1", db_name="UMLS")])])]
    pred = []
    metrics = evaluation.evaluate(gt, pred, metrics="all")
    assert metrics["strict"]["precision"] == 0.0
    assert metrics["strict"]["recall"] == 0.0
    assert metrics["strict"]["fscore"] == 0.0
    assert metrics["strict"]["ptp"] == 0
    assert metrics["strict"]["fp"] == 0
    assert metrics["strict"]["rtp"] == 0
    assert metrics["strict"]["fn"] == 1

    assert metrics["partial"]["precision"] == 0.0
    assert metrics["partial"]["recall"] == 0.0
    assert metrics["partial"]["fscore"] == 0.0
    assert metrics["partial"]["ptp"] == 0
    assert metrics["partial"]["fp"] == 0
    assert metrics["partial"]["rtp"] == 0
    assert metrics["partial"]["fn"] == 1

    assert metrics["loose"]["precision"] == 0.0
    assert metrics["loose"]["recall"] == 0.0
    assert metrics["loose"]["fscore"] == 0.0
    assert metrics["loose"]["ptp"] == 0
    assert metrics["loose"]["fp"] == 0
    assert metrics["loose"]["rtp"] == 0
    assert metrics["loose"]["fn"] == 1


def test_false_partial_overlap():
    gt = [make_document([Entity([[11, 20]], "entity", concepts=[Concept("c1", db_name="UMLS")])])]
    pred = [make_document([Entity([[1, 20]], "entity", concepts=[Concept("c1", db_name="UMLS")])])]
    metrics = evaluation.evaluate(gt, pred, metrics="all")
    assert metrics["strict"]["precision"] == 0.0
    assert metrics["strict"]["recall"] == 0.0
    assert metrics["strict"]["fscore"] == 0.0
    assert metrics["strict"]["ptp"] == 0
    assert metrics["strict"]["fp"] == 1
    assert metrics["strict"]["rtp"] == 0
    assert metrics["strict"]["fn"] == 1

    assert metrics["partial"]["precision"] == 0.5
    assert metrics["partial"]["recall"] == 1.0
    assert metrics["partial"]["fscore"] == 2 / 3
    assert metrics["partial"]["ptp"] == 0.5
    assert metrics["partial"]["fp"] == 0.5
    assert metrics["partial"]["rtp"] == 1
    assert metrics["partial"]["fn"] == 0

    assert metrics["loose"]["precision"] == 1.0
    assert metrics["loose"]["recall"] == 1.0
    assert metrics["loose"]["fscore"] == 1.0
    assert metrics["loose"]["ptp"] == 1
    assert metrics["loose"]["fp"] == 0
    assert metrics["loose"]["rtp"] == 1
    assert metrics["loose"]["fn"] == 0


def test_multiple_candidates():
    gt = [make_document([Entity([[1, 20]], "entity", concepts=[Concept("c1", db_name="UMLS")])])]
    pred = [
        make_document(
            [
                Entity([[1, 20]], "entity", concepts=[Concept("c1", db_name="UMLS")]),
                Entity([[1, 20]], "entity", concepts=[Concept("c2", db_name="UMLS")]),
            ]
        )
    ]

    metrics = evaluation.evaluate(gt, pred, metrics="all", top_k_predictions=None)
    assert metrics["strict"]["precision"] == 0.5
    assert metrics["strict"]["recall"] == 1.0
    assert metrics["strict"]["fscore"] == 2 / 3
    assert metrics["strict"]["ptp"] == 1
    assert metrics["strict"]["fp"] == 1
    assert metrics["strict"]["rtp"] == 1
    assert metrics["strict"]["fn"] == 0

    assert metrics["partial"]["precision"] == 0.5
    assert metrics["partial"]["recall"] == 1.0
    assert metrics["partial"]["fscore"] == 2 / 3
    assert metrics["partial"]["ptp"] == 1
    assert metrics["partial"]["fp"] == 1
    assert metrics["partial"]["rtp"] == 1
    assert metrics["partial"]["fn"] == 0

    assert metrics["loose"]["precision"] == 0.5
    assert metrics["loose"]["recall"] == 1.0
    assert metrics["loose"]["fscore"] == 2 / 3
    assert metrics["loose"]["ptp"] == 1
    assert metrics["loose"]["fp"] == 1
    assert metrics["loose"]["rtp"] == 1
    assert metrics["loose"]["fn"] == 0


def test_multiple_candidates_predictions():
    gt = [make_document([Entity([[1, 20]], "entity", concepts=[Concept("c1", db_name="UMLS")])])]
    pred = [
        make_document(
            [
                Entity([[1, 20]], "entity", concepts=[Concept("c1", db_name="UMLS"), Concept("c2", db_name="UMLS")]),
            ]
        )
    ]
    metrics = evaluation.evaluate(gt, pred, metrics="all", top_k_predictions=None)
    assert metrics["strict"]["precision"] == 0.5
    assert metrics["strict"]["recall"] == 1
    assert metrics["strict"]["fscore"] == 2 / 3
    assert metrics["strict"]["ptp"] == 1
    assert metrics["strict"]["fp"] == 1
    assert metrics["strict"]["rtp"] == 1
    assert metrics["strict"]["fn"] == 0

    assert metrics["partial"]["precision"] == 0.5
    assert metrics["partial"]["recall"] == 1.0
    assert metrics["partial"]["fscore"] == 2 / 3
    assert metrics["partial"]["ptp"] == 1
    assert metrics["partial"]["fp"] == 1
    assert metrics["partial"]["rtp"] == 1
    assert metrics["partial"]["fn"] == 0

    assert metrics["loose"]["precision"] == 0.5
    assert metrics["loose"]["recall"] == 1.0
    assert metrics["loose"]["fscore"] == 2 / 3
    assert metrics["loose"]["ptp"] == 1
    assert metrics["loose"]["fp"] == 1
    assert metrics["loose"]["rtp"] == 1
    assert metrics["loose"]["fn"] == 0


def test_multiple_candidates_allow_multiple_gold_candidates():
    gt = [
        make_document(
            [Entity([[1, 20]], "entity", concepts=[Concept("c1", db_name="UMLS"), Concept("c2", db_name="UMLS")])]
        )
    ]
    pred = [make_document([Entity([[1, 20]], "entity", concepts=[Concept("c1", db_name="UMLS")])])]
    metrics = evaluation.evaluate(gt, pred, allow_multiple_gold_candidates=True, metrics="all")
    assert metrics["strict"]["precision"] == 1.0
    assert metrics["strict"]["recall"] == 1.0
    assert metrics["strict"]["fscore"] == 1.0
    assert metrics["strict"]["ptp"] == 1
    assert metrics["strict"]["fp"] == 0
    assert metrics["strict"]["rtp"] == 1
    assert metrics["strict"]["fn"] == 0

    assert metrics["partial"]["precision"] == 1.0
    assert metrics["partial"]["recall"] == 1.0
    assert metrics["partial"]["fscore"] == 1.0
    assert metrics["partial"]["ptp"] == 1
    assert metrics["partial"]["fp"] == 0
    assert metrics["partial"]["rtp"] == 1
    assert metrics["partial"]["fn"] == 0

    assert metrics["loose"]["precision"] == 1.0
    assert metrics["loose"]["recall"] == 1.0
    assert metrics["loose"]["fscore"] == 1.0
    assert metrics["loose"]["ptp"] == 1
    assert metrics["loose"]["fp"] == 0
    assert metrics["loose"]["rtp"] == 1
    assert metrics["loose"]["fn"] == 0


def test_multiple_candidates_allow_multiple_gold_candidates_order():
    gt = [
        make_document(
            [Entity([[1, 20]], "entity", concepts=[Concept("c2", db_name="UMLS"), Concept("c1", db_name="UMLS")])]
        )
    ]
    pred = [make_document([Entity([[1, 20]], "entity", concepts=[Concept("c1", db_name="UMLS")])])]
    metrics = evaluation.evaluate(gt, pred, allow_multiple_gold_candidates=True, metrics="all")
    assert metrics["strict"]["precision"] == 1.0
    assert metrics["strict"]["recall"] == 1.0
    assert metrics["strict"]["fscore"] == 1.0
    assert metrics["strict"]["ptp"] == 1
    assert metrics["strict"]["fp"] == 0
    assert metrics["strict"]["rtp"] == 1
    assert metrics["strict"]["fn"] == 0

    assert metrics["partial"]["precision"] == 1.0
    assert metrics["partial"]["recall"] == 1.0
    assert metrics["partial"]["fscore"] == 1.0
    assert metrics["partial"]["ptp"] == 1
    assert metrics["partial"]["fp"] == 0
    assert metrics["partial"]["rtp"] == 1
    assert metrics["partial"]["fn"] == 0

    assert metrics["loose"]["precision"] == 1.0
    assert metrics["loose"]["recall"] == 1.0
    assert metrics["loose"]["fscore"] == 1.0
    assert metrics["loose"]["ptp"] == 1
    assert metrics["loose"]["fp"] == 0
    assert metrics["loose"]["rtp"] == 1
    assert metrics["loose"]["fn"] == 0


def test_multiple_candidates_dont_allow_multiple_gold_candidates():
    gt = [
        make_document(
            [Entity([[1, 20]], "entity", concepts=[Concept("c1", db_name="UMLS"), Concept("c2", db_name="UMLS")])]
        )
    ]
    pred = [make_document([Entity([[1, 20]], "entity", concepts=[Concept("c1", db_name="UMLS")])])]
    metrics = evaluation.evaluate(gt, pred, allow_multiple_gold_candidates=False, metrics="all")
    assert metrics["strict"]["precision"] == 1.0
    assert metrics["strict"]["recall"] == 0.5
    assert metrics["strict"]["fscore"] == 2 / 3
    assert metrics["strict"]["ptp"] == 1
    assert metrics["strict"]["fp"] == 0
    assert metrics["strict"]["rtp"] == 1
    assert metrics["strict"]["fn"] == 1

    assert metrics["partial"]["precision"] == 1.0
    assert metrics["partial"]["recall"] == 0.5
    assert metrics["partial"]["fscore"] == 2 / 3
    assert metrics["partial"]["ptp"] == 1
    assert metrics["partial"]["fp"] == 0
    assert metrics["partial"]["rtp"] == 1
    assert metrics["partial"]["fn"] == 1

    assert metrics["loose"]["precision"] == 1.0
    assert metrics["loose"]["recall"] == 0.5
    assert metrics["loose"]["fscore"] == 2 / 3
    assert metrics["loose"]["ptp"] == 1
    assert metrics["loose"]["fp"] == 0
    assert metrics["loose"]["rtp"] == 1
    assert metrics["loose"]["fn"] == 1


def test_multiple_candidates_dont_allow_multiple_gold_candidates_order():
    gt = [
        make_document(
            [Entity([[1, 20]], "entity", concepts=[Concept("c2", db_name="UMLS"), Concept("c1", db_name="UMLS")])]
        )
    ]
    pred = [make_document([Entity([[1, 20]], "entity", concepts=[Concept("c1", db_name="UMLS")])])]
    metrics = evaluation.evaluate(gt, pred, allow_multiple_gold_candidates=False, metrics="all")
    assert metrics["strict"]["precision"] == 1.0
    assert metrics["strict"]["recall"] == 0.5
    assert metrics["strict"]["fscore"] == 2 / 3
    assert metrics["strict"]["ptp"] == 1
    assert metrics["strict"]["fp"] == 0
    assert metrics["strict"]["rtp"] == 1
    assert metrics["strict"]["fn"] == 1

    assert metrics["partial"]["precision"] == 1.0
    assert metrics["partial"]["recall"] == 0.5
    assert metrics["partial"]["fscore"] == 2 / 3
    assert metrics["partial"]["ptp"] == 1
    assert metrics["partial"]["fp"] == 0
    assert metrics["partial"]["rtp"] == 1
    assert metrics["partial"]["fn"] == 1

    assert metrics["loose"]["precision"] == 1.0
    assert metrics["loose"]["recall"] == 0.5
    assert metrics["loose"]["fscore"] == 2 / 3
    assert metrics["loose"]["ptp"] == 1
    assert metrics["loose"]["fp"] == 0
    assert metrics["loose"]["rtp"] == 1
    assert metrics["loose"]["fn"] == 1


def test_multiple_candidates_top_1():
    gt = [make_document([Entity([[1, 20]], "entity", concepts=[Concept("c1", db_name="UMLS")])])]
    pred = [
        make_document(
            [
                Entity(
                    [[1, 20]],
                    "entity",
                    concepts=[Concept("c1", 0.7, db_name="UMLS"), Concept("c2", 0.99, db_name="UMLS")],
                )
            ]
        )
    ]
    metrics = evaluation.evaluate(gt, pred, top_k_predictions=1, metrics="all")

    for m in ["loose", "partial", "strict"]:
        assert metrics["strict"]["precision"] == 0.0, m
        assert metrics["strict"]["recall"] == 0.0, m
        assert metrics["strict"]["fscore"] == 0, m
        assert metrics["strict"]["ptp"] == 0, m
        assert metrics["strict"]["fp"] == 1, m
        assert metrics["strict"]["rtp"] == 0, m
        assert metrics["strict"]["fn"] == 1, m


def test_multiple_candidates_top_2():
    gt = [make_document([Entity([[1, 20]], "entity", concepts=[Concept("c1", db_name="UMLS")])])]
    pred = [
        make_document(
            [
                Entity(
                    [[1, 20]],
                    "entity",
                    concepts=[Concept("c1", 0.7, db_name="UMLS"), Concept("c2", 0.99, db_name="UMLS")],
                )
            ]
        )
    ]
    metrics = evaluation.evaluate(gt, pred, top_k_predictions=2, metrics="all")

    for m in ["loose", "partial", "strict"]:
        assert metrics["strict"]["precision"] == 0.5, m
        assert metrics["strict"]["recall"] == 1.0, m
        assert metrics["strict"]["fscore"] == 2 / 3, m
        assert metrics["strict"]["ptp"] == 1, m
        assert metrics["strict"]["fp"] == 1, m
        assert metrics["strict"]["rtp"] == 1, m
        assert metrics["strict"]["fn"] == 0, m


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
    gt = [make_document([
        Entity([[0, 5]], "entity", concepts=[]),
        Entity([[10, 15]], "entity", concepts=[]),
        Entity([[20, 25]], "entity", concepts=[]),
        Entity([[30, 35]], "entity", concepts=[]),
        Entity([[40, 45]], "entity", concepts=[])
    ])]
    pred = [make_document([
        Entity([[0, 4]], "entity", concepts=[]),  # be
        Entity([[11, 15]], "entity", concepts=[]),  # be
        Entity([[20, 25]], "entity", concepts=[]),  # tp
        Entity([[36, 40]], "entity", concepts=[]),  # fp
        Entity([[45, 50]], "entity", concepts=[])  # fp
    ])]
    ea_df = evaluation.error_analysis(gt, pred)
    assert len(ea_df) == 7
    assert ea_df.ner_match_type.tolist() == ["be", "be", "tp", "fn", "fp", "fn", "fp"]

def test_mixed_five_entities_boundaries():
    gt = [make_document([
        Entity([[0, 5]], "entity", concepts=[]),
        Entity([[6, 10]], "entity", concepts=[]),
        Entity([[11, 15]], "entity", concepts=[]),
        Entity([[30, 35]], "entity", concepts=[]),
        Entity([[40, 45]], "entity", concepts=[])
    ])]
    pred = [make_document([
        Entity([[0, 7]], "entity", concepts=[]),  # be
        Entity([[8, 13]], "entity", concepts=[]),  # be
        Entity([[36, 40]], "entity", concepts=[]),  # fp
        Entity([[45, 50]], "entity", concepts=[])  # fp
    ])]
    ea_df = evaluation.error_analysis(gt, pred)
    assert len(ea_df) == 8
    assert ea_df.ner_match_type.tolist() == ["be", "be", "be", "be", "fn", "fp", "fn", "fp"]

if __name__ == "__main__":
    test_no_entities()
