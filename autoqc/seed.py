from __future__ import annotations
import copy


def seed_bad_negative(items: list[dict]) -> tuple[list[dict], str | None]:
    """Inject a Q07 defect: rewrite the first negative criterion's title into a
    'Does not claim...' form. Returns (mutated_items, mutated_id). Does not mutate input."""
    mutated = copy.deepcopy(items)
    for it in mutated:
        if not isinstance(it, dict):
            continue
        ann = it.get("annotations")
        typ = ann.get("type", "") if isinstance(ann, dict) else ""
        if "negative" in str(typ):
            title = str(it.get("title", ""))
            if "Claims that" in title:
                it["title"] = title.replace("Claims that", "Does not claim that", 1)
            else:
                it["title"] = title + " (does not claim this)"
            return mutated, str(it.get("id")) if it.get("id") is not None else None
    return mutated, None


def seed_wildcard(items: list[dict]) -> tuple[list[dict], str | None]:
    """Inject a Q03 defect: append an open escape hatch to the first positive
    criterion's title. Returns (mutated_items, mutated_id). Does not mutate input."""
    mutated = copy.deepcopy(items)
    for it in mutated:
        if not isinstance(it, dict):
            continue
        ann = it.get("annotations")
        typ = ann.get("type", "") if isinstance(ann, dict) else ""
        if "positive" in str(typ):
            title = str(it.get("title", ""))
            it["title"] = title.rstrip(".") + ", or similar"
            return mutated, str(it.get("id")) if it.get("id") is not None else None
    return mutated, None
