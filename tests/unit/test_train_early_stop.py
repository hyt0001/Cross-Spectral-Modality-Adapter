"""train_csma 单项 AP 早停逻辑单元测试。"""

from __future__ import annotations

from src.train_csma import _update_class_ap_bests


def test_both_improve() -> None:
    """person 与 car 同时创新高。"""
    imp_p, imp_c, bp, bc = _update_class_ap_bests(
        {"ap_person": 0.7, "ap_car": 0.6},
        best_ap_person=0.5,
        best_ap_car=0.5,
    )
    assert imp_p and imp_c
    assert bp == 0.7
    assert bc == 0.6


def test_only_person_improves() -> None:
    """仅 person 创新高，仍应重置 patience。"""
    imp_p, imp_c, bp, bc = _update_class_ap_bests(
        {"ap_person": 0.72, "ap_car": 0.55},
        best_ap_person=0.70,
        best_ap_car=0.60,
    )
    assert imp_p and not imp_c
    assert bp == 0.72
    assert bc == 0.60


def test_neither_improves() -> None:
    """person/car 均未创新高。"""
    imp_p, imp_c, bp, bc = _update_class_ap_bests(
        {"ap_person": 0.68, "ap_car": 0.58},
        best_ap_person=0.70,
        best_ap_car=0.60,
    )
    assert not imp_p and not imp_c
    assert bp == 0.70
    assert bc == 0.60
