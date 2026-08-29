from scripts.calibrate_phase15 import summarize_phase15

PHASE15_IDS = {"P01","P02","P03","P04","A01","A02","A03","A04","A05","A06","AL01","H01","Q13"}

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
