from nemo_retriever.evidence.confidence import normalize_scores, corroboration, hit_confidence


def test_normalize_scores_minmax():
    assert normalize_scores([1.0, 3.0, 5.0]) == [0.0, 0.5, 1.0]


def test_normalize_scores_all_equal_is_one():
    assert normalize_scores([2.0, 2.0]) == [1.0, 1.0]


def test_normalize_scores_empty():
    assert normalize_scores([]) == []


def test_corroboration_fraction_of_other_hits_same_key():
    keys = ["docA:1", "docA:1", "docB:2"]
    # hit 0: of the 2 other hits, 1 shares key -> 0.5
    assert corroboration(0, keys) == 0.5
    # hit 2: of the 2 other hits, 0 share key -> 0.0
    assert corroboration(2, keys) == 0.0


def test_corroboration_single_hit_is_zero():
    assert corroboration(0, ["docA:1"]) == 0.0


def test_hit_confidence_weights_and_rounds():
    # 0.7*1.0 + 0.3*0.0 = 0.7
    assert hit_confidence(1.0, 0.0) == 0.7
    # 0.7*0.0 + 0.3*1.0 = 0.3
    assert hit_confidence(0.0, 1.0) == 0.3
