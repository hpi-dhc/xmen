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

    metrics = evaluation.evaluate(gt, pred, metrics="all")
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
    metrics = evaluation.evaluate(gt, pred, metrics="all")
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
