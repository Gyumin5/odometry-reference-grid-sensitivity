"""추정기 주기 비율을 훑는 측정이 무엇을 보는지 고정한다.

주장은 둘이다. 곡선이 단조가 아니라 약분되는 비율에서 파인다는 것, 그리고 그
파임의 깊이가 공개 격자가 얼마나 고른지를 읽는다는 것. 둘 다 깨지면 이 측정은
0.7 이라는 상수를 다른 상수들로 바꿔 적은 것에 지나지 않는다.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

from interp_rate_sweep import MAX_SAMPLES, sweep_one  # noqa: E402
from resample_sensitivity import trajectory  # noqa: E402

FS = 50.0
N = 1200
RATIOS = np.round(np.arange(0.25, 2.5001, 0.025), 4)


def traj(kind, fs=FS, n=N, seed=7):
    return trajectory(n, fs, kind, np.random.default_rng(seed))


def test_a_motionless_reference_cannot_move_at_any_ratio():
    d, fs = sweep_one(*traj("still"), RATIOS)
    assert abs(fs - FS) < 1e-6
    assert len(d) == len(RATIOS)
    assert max(d.values()) < 1e-9


def test_commensurate_ratios_fall_into_a_dip():
    # 1/r 이 정수면 질의가 공개 시각에 그대로 얹혀 가장 가까운 표본이 정확한
    # 값이다. 고른 격자에서 그 자리가 0 이 아니면 측정 자체가 틀린 것이다.
    d, _ = sweep_one(*traj("broadband"), RATIOS)
    for r in (0.25, 0.5, 1.0):
        assert d[r] < 1e-9, "r=%s 에서 안 파인다: %g" % (r, d[r])
    rest = [v for r, v in d.items() if r not in (0.25, 0.5, 1.0)]
    assert min(rest) > 1e-6


def test_the_curve_is_not_monotone_in_the_ratio():
    # 단조라면 "추정기가 느릴수록 나쁘다" 한 줄로 끝나고 비율을 훑을 이유가 없다.
    d, _ = sweep_one(*traj("broadband"), RATIOS)
    v = [d[r] for r in RATIOS]
    ups = sum(1 for a, b in zip(v, v[1:]) if b > a)
    downs = sum(1 for a, b in zip(v, v[1:]) if b < a)
    assert ups > 3 and downs > 3


def test_grid_jitter_fills_the_dip_in_and_that_is_the_point():
    # 공개 시각이 흔들리면 약분이 되는 비율에서도 질의가 표본에 정확히 안 얹힌다.
    # 그 파임의 깊이가 격자가 얼마나 고른지를 읽는 값이라, 흔들린 격자에서는
    # 0 이 아니어야 한다 — 다만 여전히 약분 안 되는 자리보다는 낮다.
    t, rot = traj("broadband")
    rng = np.random.default_rng(3)
    jit = np.sort(t + rng.normal(0.0, 0.2 / FS, size=len(t)))
    d, _ = sweep_one(jit, rot, RATIOS)
    assert d[1.0] > 1e-7
    rest = sorted(v for r, v in d.items() if r not in (0.25, 0.5, 1.0))
    assert d[1.0] < rest[len(rest) // 2]


def test_a_denser_publication_moves_less_at_the_same_ratio():
    slow = sweep_one(*traj("broadband", fs=10.0), RATIOS)[0]
    fast = sweep_one(*traj("broadband", fs=200.0), RATIOS)[0]
    for r in (0.7, 1.3, 1.7):
        assert fast[r] < slow[r], r


def test_faster_motion_moves_more_at_the_same_ratio():
    calm = sweep_one(*traj("smooth"), RATIOS)[0]
    quick = sweep_one(*traj("fast"), RATIOS)[0]
    for r in (0.7, 1.3, 1.7):
        assert quick[r] > calm[r], r


def test_long_files_are_truncated_deterministically():
    # 자르는 길이를 바꾸면 값이 바뀌므로, 자른다는 사실이 검사에 박혀 있어야 한다.
    t, rot = traj("broadband", n=MAX_SAMPLES + 5000, fs=200.0)
    whole = sweep_one(t, rot, [0.7])[0]
    cut = sweep_one(t[:MAX_SAMPLES], rot[:MAX_SAMPLES], [0.7])[0]
    assert abs(whole[0.7] - cut[0.7]) < 1e-12


def test_too_short_a_file_is_refused_rather_than_guessed():
    d, fs = sweep_one(*traj("broadband", n=32), RATIOS)
    assert d is None and np.isnan(fs)


def test_ratios_that_give_too_few_pairs_are_left_out_not_guessed():
    # 아주 느린 추정기는 질의가 몇 개 안 나와 평균이 못 선다. 그 비율은 빠지고
    # 값이 지어지지 않아야 한다.
    t, rot = traj("broadband", n=200)
    d, _ = sweep_one(t, rot, [0.02, 0.7])
    assert 0.02 not in d
    assert 0.7 in d
