"""Check evaluator labels and visual-ranking behavior."""

from pathlib import Path

import pytest

from scripts._evaluation import _score
from scripts.evaluate_rvl_cdip import _judgments


def test_rvl_pair_join_respects_judged_scope(tmp_path: Path) -> None:
    """Join path identities without treating unseen training pairs as negatives."""
    query = "imagesa/a/a/a/aaa.tif"
    positive = "imagesb/b/b/b/bbb.tif"
    negative = "imagesc/c/c/c/ccc.tif"
    documents = [{"id": "query", "rvlcdip_path": query, "has_train_duplicate": True}]
    pairs = [
        {"id": "query", "rvlcdip_path": query, "train_rvlcdip_path": positive, "duplicate": True},
        {"id": "query", "rvlcdip_path": query, "train_rvlcdip_path": negative, "duplicate": False},
    ]
    queries, positives, negatives, excluded = _judgments(documents, pairs, tmp_path)
    assert queries == [("query", tmp_path / query)]
    assert positives == {"query": {positive}}
    assert negatives == {"query": {negative}}
    assert excluded == 0
    with pytest.raises(ValueError, match="disagree"):
        _judgments([{**documents[0], "has_train_duplicate": False}], pairs, tmp_path)


def test_visual_score_respects_text_and_distinct_pages() -> None:
    """Require visual page evidence to respect text provenance and page identity."""
    left = {
        "raw": "a",
        "canonical": "a",
        "word": (),
        "char": (),
        "pages": ["ffffffffffffffff"],
        "page_count": 1,
        "native_text": True,
    }
    right = {**left, "raw": "b", "canonical": "b"}
    assert _score(left, right) == 0
    assert _score(left, {**right, "native_text": False}) == 1
    repeated = {**left, "pages": ["ffffffffffffffff"] * 2, "page_count": 2}
    mixed = {
        **right,
        "pages": ["ffffffffffffffff", "0000000000000000"],
        "page_count": 2,
        "native_text": False,
    }
    assert _score(repeated, mixed) == 0.5
