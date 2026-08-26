from autoqc.agent.engine import aggregate, adjudicate


def _f(cid, passed, ev="e"):
    return {"check_id": "Q07", "criterion_id": cid, "passed": passed, "evidence": [ev]}


def test_aggregate_unanimous_pass():
    sets = [[_f("a", True)], [_f("a", True)], [_f("a", True)]]
    agg = aggregate(sets, {"a"})
    assert agg["a"]["passed"] is True and agg["a"]["split"] is False


def test_aggregate_unanimous_fail():
    sets = [[_f("a", False)], [_f("a", False)], [_f("a", False)]]
    agg = aggregate(sets, {"a"})
    assert agg["a"]["passed"] is False and agg["a"]["split"] is False


def test_aggregate_split_on_disagreement():
    sets = [[_f("a", True)], [_f("a", False)], [_f("a", True)]]
    agg = aggregate(sets, {"a"})
    assert agg["a"]["passed"] is True   # 2/3 majority
    assert agg["a"]["split"] is True    # not unanimous


def test_aggregate_abstain_counts_as_split():
    sets = [[_f("a", True)], [], [_f("a", True)]]  # one pass abstained
    agg = aggregate(sets, {"a"})
    assert agg["a"]["split"] is True


def test_aggregate_no_votes_is_fail_and_split():
    agg = aggregate([[], [], []], {"a"})
    assert agg["a"]["passed"] is False and agg["a"]["split"] is True


def test_aggregate_collects_evidence():
    sets = [[_f("a", False, "ev1")], [_f("a", False, "ev2")]]
    agg = aggregate(sets, {"a"})
    assert "ev1" in agg["a"]["evidence"] and "ev2" in agg["a"]["evidence"]


def test_aggregate_dedupes_votes_per_pass():
    # one pass emits the same criterion 3x (all True); two passes abstain
    sets = [[_f("a", True), _f("a", True), _f("a", True)], [], []]
    agg = aggregate(sets, {"a"})
    assert agg["a"]["passed"] is True     # the one real vote is True
    assert agg["a"]["split"] is True      # but only 1 of 3 passes voted -> split, NOT false-unanimous


def test_adjudicate_clean_pass():
    agg = {"a": {"passed": True, "split": False, "evidence": []}}
    adj = adjudicate(agg, adversary_findings=[])
    assert adj["a"] == {"passed": True, "needs_human": False}


def test_adjudicate_split_needs_human():
    agg = {"a": {"passed": True, "split": True, "evidence": []}}
    adj = adjudicate(agg, [])
    assert adj["a"]["needs_human"] is True


def test_adjudicate_adversary_overturn_needs_human():
    agg = {"a": {"passed": False, "split": False, "evidence": []}}  # a reject
    adv = [{"check_id": "Q07", "criterion_id": "a", "passed": True, "evidence": ["actually fine"]}]
    adj = adjudicate(agg, adv)
    assert adj["a"]["passed"] is False and adj["a"]["needs_human"] is True  # disputed reject
