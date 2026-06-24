from interpretation import interpret_profile


def test_norm_and_pathology_is_conflict():
    result = interpret_profile(
        {"NORM": 0.80, "MI": 0.71, "STTC": 0.10, "CD": 0.05, "HYP": 0.03},
        threshold=0.5,
    )
    assert result["code"] == "norm_pathology_conflict"
    assert result["level"] == "conflict"
    assert result["requires_physician_review"] is True


def test_multiple_pathologies_are_multilabel_not_conflict():
    result = interpret_profile(
        {"NORM": 0.02, "MI": 0.81, "STTC": 0.73, "CD": 0.10, "HYP": 0.08},
        threshold=0.5,
    )
    assert result["code"] == "multiple_pathology"
    assert result["pathology_classes"] == ["MI", "STTC"]


def test_norm_with_near_threshold_pathology_is_uncertain():
    result = interpret_profile(
        {"NORM": 0.91, "MI": 0.47, "STTC": 0.10, "CD": 0.05, "HYP": 0.03},
        threshold=0.5,
    )
    assert result["code"] == "norm_with_borderline_pathology"
    assert result["level"] == "uncertain"


def test_norm_only_does_not_claim_absence_of_disease():
    result = interpret_profile(
        {"NORM": 0.95, "MI": 0.08, "STTC": 0.09, "CD": 0.04, "HYP": 0.03},
        threshold=0.5,
    )
    assert result["code"] == "norm_only"
    assert "не исключает" in result["message"]


def test_no_class_is_indeterminate():
    result = interpret_profile(
        {"NORM": 0.30, "MI": 0.20, "STTC": 0.10, "CD": 0.15, "HYP": 0.12},
        threshold=0.5,
    )
    assert result["code"] == "no_class_detected"
    assert result["level"] == "uncertain"
