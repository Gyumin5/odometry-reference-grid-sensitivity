"""시각 어긋남 추정기와 그 구간 폭이 믿을 만한지 본다.

각속도 계열 하나를 만들고 그것을 적분해 자세를 만든 뒤, 각속도 쪽 시각만 알려진
양만큼 밀어 둔다. 추정기는 그 양을 되찾아야 하고, 이동 구간 부트스트랩이 내는
95% 구간은 그 참값을 품어야 한다.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from reference_resolution_audit import ypr_to_rot  # noqa: E402
from reference_timing_audit import (  # noqa: E402
    best_tau,
    bootstrap_tau,
    rate_from_attitude,
)


def make_pair(fs, duration, delay, seed, sigma=0.0):
    """자세 계열과, 그 자세와 같은 운동을 d 초 이르게 찍은 각속도 계열."""
    n = int(fs * duration)
    t = np.arange(n) / fs
    rng = np.random.default_rng(seed)
    w = np.stack([0.05 * np.sin(2 * math.pi * 0.31 * t),
                  0.08 * np.sin(2 * math.pi * 0.17 * t + 0.4),
                  0.40 * np.sin(2 * math.pi * 0.23 * t + 1.1)], axis=1)
    w = w + rng.normal(0.0, sigma, size=w.shape)
    # 자세 증분이 구간 가운데 각속도와 맞도록 사다리꼴로 적분한다. 앞값만 쓰면
    # 반 표본(여기서는 10 ms)만큼 치우친 답이 나온다.
    step = np.zeros_like(w)
    step[1:] = 0.5 * (w[1:] + w[:-1]) / fs
    ang = np.cumsum(step, axis=0)
    rot = ypr_to_rot(ang[:, 2], ang[:, 1], ang[:, 0])
    return t, rot, t - delay, w


def test_recovers_an_injected_delay():
    fs, delay = 50.0, 0.030
    t, rot, tg, w = make_pair(fs, 300.0, delay, seed=5)
    tc, w_ref = rate_from_attitude(t, rot)
    m = (tc > tc[0] + 0.3) & (tc < tc[-1] - 0.3)
    tau, _, _ = best_tau(tc[m], w_ref[m], tg, w, -0.2, 0.2, 0.002)
    # tau 는 자세 쪽 시각에 더해 짝이 되는 각속도 표본을 찾는 양이다. 각속도가
    # d 만큼 이르게 찍혀 있으면 tau = -d 로 나온다.
    assert abs(tau + delay) < 0.003, tau


def test_bootstrap_interval_covers_the_truth_and_is_not_inflated():
    """구간을 합집합 마스크로 모으면 회차마다 표본이 줄어 폭이 부풀어 오른다.
    되돌려 놓고 뽑아 원래 길이만큼 이어 붙이면 그 부풀음이 없어진다."""
    fs, delay = 50.0, 0.030
    t, rot, tg, w = make_pair(fs, 300.0, delay, seed=9, sigma=0.004)
    tc, w_ref = rate_from_attitude(t, rot)
    m = (tc > tc[0] + 0.3) & (tc < tc[-1] - 0.3)
    tc, w_ref = tc[m], w_ref[m]

    boot = bootstrap_tau(tc, w_ref, tg, w, -0.2, 0.2, 0.002,
                         block_s=10.0, draws=60, seed=1)
    assert boot is not None and len(boot) == 60
    lo, hi = np.percentile(boot, [2.5, 97.5])
    assert lo <= -delay <= hi, (lo, -delay, hi)
    # 격자가 2 ms 다. 참값을 품으면서도 그 몇 배 안에 들어와야 쓸모가 있다.
    assert (hi - lo) < 0.020, hi - lo


def test_bootstrap_refuses_a_series_shorter_than_three_blocks():
    fs = 50.0
    t, rot, tg, w = make_pair(fs, 20.0, 0.0, seed=3)
    tc, w_ref = rate_from_attitude(t, rot)
    assert bootstrap_tau(tc, w_ref, tg, w, -0.2, 0.2, 0.002,
                         block_s=10.0, draws=10, seed=1) is None
