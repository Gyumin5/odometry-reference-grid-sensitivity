"""격자를 절대 시각으로 만들 때 더해지는 반올림을 재는 탐침을 본다.

재는 양이 둘이라 섞이기 쉽다. 입력 시각 자체가 유닉스 기원 크기로 저장돼
있어 이미 놓인 눈금이 하나고, 격자를 그 기원 위에서 만들며 더해지는 밀림이
하나다. 앞쪽은 되돌릴 수 없고 뒤쪽은 원점을 빼면 없어진다. 탐침이 둘을 안
가르면 "원점을 빼도 소용없다" 와 "빼야 한다" 를 수로 구분할 수 없다.
"""

import os
import sys
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

from grid_origin_probe import (grid_drift, input_tick,  # noqa: E402
                               nearest, pick_flips, pooled_both,
                               pooled_by_rule, two_prod, two_sum)
from protocol_sweep import coarse_grid, window  # noqa: E402

EPOCH = 1.7e9


def test_the_two_error_terms_are_exact_not_estimates():
    """밀림을 유리수 기준과 맞댄다. 근사면 뒤의 비교가 전부 흔들린다."""
    a, b = 0.1 / 0.7, 31337.0
    p, e = two_prod(a, b)
    assert Fraction(p) + Fraction(e) == Fraction(a) * Fraction(b)
    s, e2 = two_sum(EPOCH, p)
    assert Fraction(s) + Fraction(e2) == Fraction(EPOCH) + Fraction(p)


def test_the_input_tick_follows_the_epoch_not_the_span():
    """눈금을 정하는 것은 경과 시간이 아니라 시각이 적힌 크기다.

    같은 1,000 초짜리 시퀀스라도 0 에서 시작하면 2e-16 초 눈금 위에 있고
    유닉스 기원에서 시작하면 2.4e-07 초 눈금 위에 있다.
    """
    span = np.arange(10000, dtype=float) * 0.1
    assert input_tick(span) < 1e-15
    assert 1e-7 < input_tick(EPOCH + span) < 1e-6


def test_moving_the_origin_shrinks_the_grid_drift_by_orders():
    """원점을 빼면 격자 밀림이 입력 눈금에서 경과 시간 눈금으로 내려간다."""
    t = EPOCH + np.arange(5000, dtype=float) * 0.1
    absolute = grid_drift(t, 0.1, 0.7)
    relative = grid_drift(t - t[0], 0.1, 0.7)
    assert absolute > 1e-8
    assert relative < absolute / 1e5


def test_a_binary_step_on_the_epoch_lattice_drifts_not_at_all():
    """0.25 초를 반으로 나눈 격자는 기원 위에서도 정확히 떨어진다.

    밀림을 "절대 시각이면 늘 생긴다" 로 적으면 이 자리를 틀리게 센다.
    """
    t = EPOCH + np.arange(4000, dtype=float) * 0.25
    assert grid_drift(t, 0.25, 0.5) == 0.0


def test_the_added_drift_stays_under_the_tick_the_input_already_sits_on():
    """격자 구성이 더하는 밀림이 입력이 이미 놓인 눈금보다 작다.

    이것이 두 손실의 크기 관계다. 더해지는 쪽이 이미 잃은 쪽보다 작으면,
    원점을 빼는 것이 입력의 정밀도를 되찾아 주지는 않는다.
    """
    t = EPOCH + np.arange(5000, dtype=float) * 0.1
    for r in (0.5, 0.7, 0.9):
        assert grid_drift(t, 0.1, r) <= input_tick(t)


def test_the_nearest_comparison_sits_on_the_inputs_own_lattice():
    """최근접 판정에 들어가는 두 간격의 차가 입력 눈금의 정수배다.

    질의도 격자점도 기원 크기의 배정밀도라 그 눈금 위에 있다. 그래서 그
    차는 0 이거나 적어도 한 눈금이고, 원점을 빼서 생기는 어긋남은 한 눈금을
    못 넘는다 — 동률이 아닌 자리는 뒤집힐 수 없다는 뜻이다.
    """
    t = EPOCH + np.arange(5000, dtype=float) * 0.1
    tick = input_tick(t)
    lo, hi = window(t, 0.1)
    tw = t[(t >= lo) & (t <= hi)]
    tc = coarse_grid(t, 0.1, 0.7)
    j = np.clip(np.searchsorted(tc, tw), 1, len(tc) - 1)
    diff = (tw - tc[j - 1]) - (tc[j] - tw)
    off = np.abs(diff / tick - np.round(diff / tick))
    assert off.max() < 1e-6
    nonzero = np.abs(diff)[diff != 0.0]
    assert nonzero.min() >= 2.0 * grid_drift(t, 0.1, 0.7)


def test_an_epoch_sequence_picks_the_same_samples_either_way():
    # 위 두 검사가 말하는 것을 실제 고른 색인으로 확인한다.
    t = EPOCH + np.arange(5000, dtype=float) * 0.1
    assert pick_flips(t, 0.1, 0.7) == (0, 4998)


def test_a_grid_shifted_inside_the_tick_does_flip_the_pick():
    """밀림이 동률을 깨면 색인이 실제로 바뀐다. 못 일어나는 일이 아니다.

    실제 릴리스에서 안 밟히는 것과 원리상 못 밟는 것은 다른 말이라,
    밟히는 자리를 하나 만들어 둔다. 여기서는 격자를 한 눈금의 반만큼 밀어
    한가운데 질의의 동률을 깬다.
    """
    src = np.arange(0.0, 100.0, 1.0)
    tq = np.arange(0.5, 99.0, 1.0)          # 정확히 두 표본 한가운데
    assert np.array_equal(nearest(src, tq), np.searchsorted(src, tq) - 1)
    assert np.array_equal(nearest(src - 1e-9, tq),
                          np.searchsorted(src, tq))


def test_too_short_a_sequence_is_counted_as_nothing_rather_than_guessed():
    assert pick_flips(np.arange(5, dtype=float), 1.0, 0.5) == (0, 0)


def arc(n=1200, dt=0.1, speed=10.0, radius=200.0, epoch=EPOCH):
    """반지름 200 m 호를 10 m/s 로 도는 합성 궤적. 유닉스 기원 시각이다."""
    t = epoch + np.arange(n, dtype=float) * dt
    ang = np.arange(n, dtype=float) * dt * speed / radius
    pos = np.stack([radius * np.sin(ang), radius * (1.0 - np.cos(ang)),
                    np.zeros(n)], axis=1)
    rot = np.zeros((n, 3, 3))
    rot[:, 0, 0] = rot[:, 1, 1] = np.cos(ang)
    rot[:, 0, 1] = -np.sin(ang)
    rot[:, 1, 0] = np.sin(ang)
    rot[:, 2, 2] = 1.0
    return t, rot, pos


def test_the_control_arm_is_not_the_same_run_measured_twice():
    """대조 경로가 실제로 다른 산술을 돈다. 이게 깨지면 차이 0 은 무의미하다.

    measure 가 안에서 원점을 빼므로, 입력에서 미리 빼 두는 방식으로는 두 판이
    같은 것을 재고 차이가 늘 0 으로 나온다. 그 0 은 "원점이 수를 안 가른다" 가
    아니라 "묻지 않았다" 다. 그래서 차이가 나야 하는 합성 입력을 하나 두고,
    거기서 차이가 검출되는 것을 못박는다.
    """
    a, b = pooled_both([arc()])
    assert a and b
    differ = [k for k in a if a[k] != b.get(k)]
    assert len(differ) > 10
    assert "rate_0.7_nearest" in differ


def test_a_trajectory_too_short_for_the_segment_rule_is_measured_by_the_other():
    """100~800 m 가 안 서는 궤적을 빈 칸으로 두면 "원점이 수를 안 가른다" 로
    읽힌다. 실제로는 그 규칙으로 잴 것이 없었다는 뜻이라, 그 릴리스가 실제로
    평가받는 1 m 규칙으로 다시 재서 수를 채운다.
    """
    short = arc(n=300, speed=0.3, radius=5.0)   # 9 m — 100 m 구간이 안 선다
    rule, a, b = pooled_by_rule([short])
    assert rule == "rpe1m"
    assert a and b
    assert pooled_both([short])[0] == {}
    rule, a, _ = pooled_by_rule([arc()])
    assert rule == "seg"
    assert a


def test_a_sequence_that_starts_at_zero_gives_the_two_arms_nothing_to_differ():
    """기원이 0 이면 뺄 것이 없어 두 판이 같아야 한다. 대조의 반대쪽이다."""
    t, rot, pos = arc(epoch=0.0)
    a, b = pooled_both([(t, rot, pos)])
    assert a and b
    assert all(a[k] == b[k] for k in a)
