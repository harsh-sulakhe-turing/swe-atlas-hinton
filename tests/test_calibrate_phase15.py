from scripts.calibrate_phase15 import summarize_phase15

def _rec(results):
    return {"results": results}

def test_recall_counts_phase15_rejects_on_negatives():
    # one negative bundle where P01 rejected -> counts as recalled
    neg = [_rec([{"id": "P01", "severity": "reject", "passed": False, "needs_human": False}])]
    clean = []
    s = summarize_phase15(negatives=neg, cleans=clean)
    assert s["recall"] == 1.0

def test_reject_false_fires_counts_clean_rejects():
    clean = [_rec([{"id": "A04", "severity": "reject", "passed": False, "needs_human": False}])]
    s = summarize_phase15(negatives=[], cleans=clean)
    assert s["reject_false_fires"] == 1

def test_warn_and_disputed_are_not_false_fires():
    clean = [_rec([
        {"id": "A01", "severity": "warn", "passed": False, "needs_human": False},
        {"id": "P04", "severity": "reject", "passed": False, "needs_human": True}])]  # disputed -> human
    s = summarize_phase15(negatives=[], cleans=clean)
    assert s["reject_false_fires"] == 0

def test_per_check_recall_fires_separates_from_false_fires():
    # negative-side fire should appear only in recall_fires
    neg = [_rec([{"id": "P01", "severity": "reject", "passed": False, "needs_human": False}])]
    # clean-side fire should appear only in false_fires
    clean = [_rec([{"id": "A04", "severity": "reject", "passed": False, "needs_human": False}])]
    s = summarize_phase15(negatives=neg, cleans=clean)
    assert s["per_check_recall_fires"] == {"P01": 1}
    assert s["per_check_false_fires"] == {"A04": 1}

def test_per_check_maps_count_multiple_fires_per_check():
    # two negatives with same check -> count 2 in recall_fires
    neg = [
        _rec([{"id": "P02", "severity": "reject", "passed": False, "needs_human": False}]),
        _rec([{"id": "P02", "severity": "reject", "passed": False, "needs_human": False}])
    ]
    # two cleans with same check -> count 2 in false_fires
    clean = [
        _rec([{"id": "A05", "severity": "reject", "passed": False, "needs_human": False}]),
        _rec([{"id": "A05", "severity": "reject", "passed": False, "needs_human": False}])
    ]
    s = summarize_phase15(negatives=neg, cleans=clean)
    assert s["per_check_recall_fires"] == {"P02": 2}
    assert s["per_check_false_fires"] == {"A05": 2}
