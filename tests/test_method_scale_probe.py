"""순위표 크기의 추정에서 기준을 밀었을 때의 탐침을 묶는다.

조용히 틀리는 자리는 나눗셈이다. 상한이 0 인 칸에서 비를 0 으로 적으면,
상한을 깬 칸이 표에서 가장 안전한 칸으로 들어간다. 그 자리를 두 경우로
가른다 — 이동도 0 이면 비가 없는 것이고, 이동이 0 이 아니면 위반이다.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

from method_scale_probe import (over_bound, same_segments,  # noqa: E402
                                shown)


def test_a_ratio_is_written_when_the_bound_is_positive():
    assert float(over_bound(2e-06, 8e-05, "x")) == pytest.approx(0.025)


def test_no_bound_and_no_movement_is_an_empty_cell_not_a_zero():
    # 0 으로 적으면 "상한의 0 퍼센트" 가 되는데, 그건 묻지 않은 비교의 답이다.
    assert over_bound(0.0, 0.0, "x") == ""


def test_no_bound_but_a_movement_stops_instead_of_reporting_zero():
    with pytest.raises(SystemExit):
        over_bound(1e-09, 0.0, "x")


def test_the_printed_value_keeps_the_board_decimals():
    assert shown(0.00149, 4) == "0.0015"
    assert shown(1.3149, 2) == "1.31"


def test_two_segment_lists_differ_when_a_requested_length_differs():
    a = [(0, 10, 100.0)]
    assert same_segments(a, [(0, 10, 100.0)])
    assert not same_segments(a, [(0, 10, 200.0)])
    assert not same_segments(a, [])
