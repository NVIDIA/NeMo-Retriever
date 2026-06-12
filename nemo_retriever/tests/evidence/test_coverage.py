from nemo_retriever.evidence.coverage import coverage


def test_coverage_reports_missing_terms_as_thin_spots():
    texts = ["The premium desk fan costs 150 dollars."]
    cov = coverage(texts, "desk fan warranty period", best_norm_score=0.9)
    assert cov["thin_spots"] == ["period", "warranty"]
    assert cov["weak"] is False
    assert cov["n_hits"] == 1


def test_coverage_no_thin_spots_when_all_terms_present():
    texts = ["The desk fan costs money."]
    cov = coverage(texts, "desk fan costs", best_norm_score=0.5)
    assert cov["thin_spots"] == []


def test_coverage_weak_flag_when_best_score_low():
    cov = coverage(["irrelevant text"], "anything here", best_norm_score=0.1)
    assert cov["weak"] is True


def test_coverage_empty_hits_is_weak():
    cov = coverage([], "any query", best_norm_score=0.0)
    assert cov["weak"] is True
    assert cov["n_hits"] == 0
