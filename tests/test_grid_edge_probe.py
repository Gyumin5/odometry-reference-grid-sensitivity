"""격자 끝 탐침이 "사라진 칸" 을 실제로 가려내는지 본다.

이 탐침의 값어치는 한 가지다 — 부동소수 나눗셈이 정수 바로 아래로 떨어져
칸 하나를 통째로 잃는 자리를 찾는 것. 그래서 잃는 자리를 만들어 놓고 그걸
잡는지 보고, 안 잃는 자리에서는 조용한지도 같이 본다. 한쪽만 보면 "언제나
잃었다" 고 적는 탐침도 통과한다.
"""

import os
import sys
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

from grid_edge_probe import (exact_steps, float_steps,  # noqa: E402
                             rational_steps, row_for, ulp_to_integer,
                             window_end)
from protocol_sweep import RATIOS, coarse_grid  # noqa: E402


def test_the_float_count_can_be_one_too_many_but_never_too_few():
    """나눗셈이 정확히 반올림되므로 칸이 모자라지는 않는다.

    모자라려면 몫이 아래로 밀려야 하는데, 참 몫 바로 위의 정수는 그 자체로
    표현되는 수라 가장 가까운 값이 그 정수다. 반대쪽은 실제로 생긴다 —
    참 몫이 정수 바로 아래면 그 정수로 올라붙어 칸이 하나 더 선다. 끝이
    표본을 넘는 쪽이 이 방향이라 탐침이 보는 것도 이쪽이다.
    """
    step = 2.4169587419986867
    span = float(462 * step)
    assert float_steps(span, step) == 462
    assert exact_steps(span, step) == 461

    rng = np.random.default_rng(11)
    for _ in range(20000):
        s = float(rng.uniform(0.01, 3.0))
        n = int(rng.integers(10, 20000))
        a = float(n * s)
        assert float_steps(a, s) >= exact_steps(a, s)
        assert float_steps(a, s) <= exact_steps(a, s) + 1


def test_a_quotient_that_lands_on_an_integer_counts_the_same_either_way():
    """이진으로 떨어지는 자리는 두 셈이 같다."""
    assert float_steps(4.0, 0.25) == exact_steps(4.0, 0.25) == 16
    assert ulp_to_integer(4.0, 0.25) == 0.0


def test_the_exact_count_never_runs_past_the_samples():
    """정확한 바닥이어도 격자 끝이 마지막 표본을 넘지 않는다.

    넘으면 창이 외삽을 부르는 자리로 열린다. 바닥 연산이 지키는 것이
    그 성질이다.
    """
    rng = np.random.default_rng(7)
    for _ in range(200):
        span = float(rng.uniform(1.0, 1000.0))
        stepsize = float(rng.uniform(0.01, 3.0))
        n = exact_steps(span, stepsize)
        assert Fraction(n) * Fraction(stepsize) <= Fraction(span)
        assert Fraction(n + 1) * Fraction(stepsize) > Fraction(span)


def test_the_rational_step_count_uses_the_unrounded_step():
    """stepsize 를 유리수로 두면 그 반올림이 셈에서 빠진다."""
    dt, ratio = 0.1, 0.7
    span = 100.0
    assert rational_steps(span, dt, ratio) == int(
        Fraction(span) * Fraction(ratio) / Fraction(dt))
    assert rational_steps(span, dt, ratio) >= exact_steps(span, dt / ratio) - 1


def test_the_window_end_falls_back_to_the_grid_that_ends_first():
    """창 끝은 가장 먼저 끝나는 격자가 정한다 — 칸 수를 늘리면 안 줄어든다."""
    t = np.arange(400, dtype=float) * 0.1
    dt = 0.1
    base = [int(np.floor((t[-1] - t[0]) / (dt / r))) for r in RATIOS]
    wider = [n + 1 for n in base]
    assert window_end(t, dt, wider) >= window_end(t, dt, base)


def test_a_clean_sequence_reports_no_extra_step():
    """이진으로 떨어지는 0.25 초 격자에서는 칸이 더 서지 않는다."""
    t = np.arange(600, dtype=float) * 0.25
    rows = row_for("clean", t)
    assert len(rows) == len(RATIOS)
    assert all(r["extra_step"] == 0 for r in rows)
    assert all(r["window_frames"] == r["window_frames_exact"] for r in rows)


def test_a_grid_that_ends_past_the_samples_is_reported_as_such():
    """격자 끝이 마지막 표본을 넘으면 그 양이 양수로 적힌다.

    이 칸이 0 보다 크면 그 변형의 질의가 외삽이다 — 구면보간은 그 자리에서
    멈추고 최근접은 조용히 끝 표본을 쓴다. 둘 다 그 한 점만 다른 규약이
    된다.
    """
    t = np.arange(600, dtype=float) * 0.25
    rows = row_for("clean", t)
    assert all(float(r["grid_end_past_samples_s"]) <= 0.0 for r in rows)


def test_the_reported_grid_end_is_the_one_the_sweep_builds():
    """탐침이 적는 격자 끝이 실제 coarse_grid 의 끝과 같은 수다."""
    t = np.arange(500, dtype=float) * 0.1
    dt = float(np.median(np.diff(t)))
    rows = row_for("ten_hertz", t)
    for r, ratio in zip(rows, RATIOS):
        grid = coarse_grid(t, dt, ratio)
        assert len(grid) - 1 == r["n_steps_float"]
        assert float(r["grid_end_past_samples_s"]) <= 0.0
