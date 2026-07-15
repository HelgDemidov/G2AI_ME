"""Тесты логики A/B-харнесса (hit@k) — без модели/сети, CI-safe."""
from __future__ import annotations

from ab_eval import CONTROL_QUERIES, hit_at_k


def test_hit_in_top1() -> None:
    assert hit_at_k(["about ACCOUNTABILITY here", "other"], ("account",), 1) is True


def test_case_insensitive() -> None:
    assert hit_at_k(["Human OVERSIGHT matters"], ("oversight",), 1) is True


def test_not_top1_but_topk() -> None:
    ranked = ["irrelevant chunk", "text about monitoring agents"]
    assert hit_at_k(ranked, ("monitor",), 1) is False
    assert hit_at_k(ranked, ("monitor",), 3) is True


def test_miss() -> None:
    assert hit_at_k(["a", "b", "c"], ("zzz",), 3) is False


def test_control_queries_wellformed() -> None:
    assert len(CONTROL_QUERIES) >= 3
    for cq in CONTROL_QUERIES:
        assert cq.query.strip()
        assert cq.expect
