"""한 파일 안의 자세 열과 각속도 열을 맞대는 측정이 맞는 답을 내는지 본다.

자세를 적분해 만들고, 각속도 열은 그 자세의 각속도를 정해진 회전만큼 돌리고 정해진
시간만큼 민 사본으로 만든다. 측정이 그 시간 밀림을 되찾고 잔차를 0 근처로 내려야 한다.
반대로 아무 상관 없는 열을 주면 잔차가 신호만큼 남아야 한다 — 두 방향을 다 본다.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from rate_channel_consistency import compare  # noqa: E402
from reference_resolution_audit import ypr_to_rot  # noqa: E402


class Args:
    tau_max = 0.1
    tau_step = 0.002


def make_pose(fs=50.0, duration=200.0, seed=3):
    n = int(fs * duration)
    t = np.arange(n) / fs
    rng = np.random.default_rng(seed)
    w = np.zeros((n, 3))
    w[:, 2] = 0.3 * np.sin(2 * math.pi * 0.15 * t) + 0.1 * rng.normal(size=n)
    w[:, 1] = 0.05 * np.sin(2 * math.pi * 0.4 * t)
    w[:, 0] = 0.03 * np.cos(2 * math.pi * 0.25 * t)
    # 사다리꼴로 적분한다. 그래야 표본 사이의 자세 증분이 그 구간 가운데의 각속도와
    # 같아지고, 측정기가 증분을 가운데 시각에 놓는 것과 규약이 맞는다.
    step = np.zeros_like(w)
    step[1:] = 0.5 * (w[1:] + w[:-1]) / fs
    ang = np.cumsum(step, axis=0)
    return t, ypr_to_rot(ang[:, 2], ang[:, 1], ang[:, 0]), w


def test_recovers_injected_delay_between_two_columns():
    fs, delay = 50.0, 0.030
    t, rot, w = make_pose(fs)
    # 각속도 열은 자세보다 delay 만큼 앞서 실린 사본이다. 축도 한 번 돌려 둔다.
    a = math.radians(20.0)
    c = np.array([[math.cos(a), -math.sin(a), 0.0],
                  [math.sin(a), math.cos(a), 0.0],
                  [0.0, 0.0, 1.0]])
    t_rate = t - delay
    w_rate = w @ c.T
    r = compare(t, rot, t_rate, w_rate, Args())
    assert r is not None
    # tau 는 자세 쪽 시각에 더해 짝이 되는 각속도 표본을 찾는 양이다. 각속도 열을
    # delay 만큼 이르게 찍어 뒀으니 tau = -delay 가 나와야 한다.
    assert abs(r["tau_ms"] + delay * 1000) < 6.0, r["tau_ms"]
    assert float(r["residual_frac"]) < 0.05, r["residual_frac"]
    assert abs(float(r["corr_z"])) > 0.99, r["corr_z"]


def test_unrelated_column_leaves_the_whole_signal():
    t, rot, _ = make_pose()
    rng = np.random.default_rng(99)
    w_rate = rng.normal(0.0, 0.3, size=(len(t), 3))
    r = compare(t, rot, t, w_rate, Args())
    assert r is not None
    assert float(r["residual_frac"]) > 0.9, r["residual_frac"]


def test_flat_axis_reports_no_correlation_instead_of_nan():
    """평면 자세는 x·y 각속도가 정확히 0 이라 상관을 낼 수 없다. 빈칸으로 둔다."""
    fs = 50.0
    n = int(fs * 200)
    t = np.arange(n) / fs
    w = np.zeros((n, 3))
    w[:, 2] = 0.3 * np.sin(2 * math.pi * 0.15 * t)
    step = np.zeros_like(w)
    step[1:] = 0.5 * (w[1:] + w[:-1]) / fs
    ang = np.cumsum(step, axis=0)
    rot = ypr_to_rot(ang[:, 2], ang[:, 1], ang[:, 0])
    r = compare(t, rot, t, w, Args())
    assert r is not None
    assert r["corr_x"] == "" and r["corr_y"] == ""
    assert abs(float(r["corr_z"])) > 0.99
