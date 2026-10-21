"""답을 아는 궤적에 측정 경로를 걸어 보는 쪽을 묶는다.

조용히 틀리는 자리는 답이 0 인 칸이다. 재는 경로를 통째로 꺼도 0 이 나오므로,
0 짜리 칸만으로는 그 경로가 도는지를 못 가른다. 그래서 답이 0 이 아닌 칸이
반드시 있어야 하고, 그 칸은 경로를 끄면 깨져야 한다.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

from kitti_segment_D import common_window, measure, segments  # noqa: E402
from known_answer_probe import (RATE_HZ, SPINUP_ACCEL_RAD_PER_S2,  # noqa: E402
                                SPINUP_LENGTHS, SPINUP_STEP, spinup,
                                spinup_reference)


def spinup_segments():
    t, rot, pos = spinup()
    keep = common_window(t)
    return t, rot, pos, segments(pos[:keep], SPINUP_LENGTHS, SPINUP_STEP)


def test_the_two_protocols_have_different_non_zero_answers():
    # 둘이 같은 수면 한쪽 경로가 다른 쪽 자리에 들어가 있어도 통과한다.
    _, _, _, idx = spinup_segments()
    want = spinup_reference(idx)
    dt = 1.0 / RATE_HZ
    assert want["phase_half"] > 0.0 and want["decim2"] > 0.0
    assert want["phase_half"] / want["decim2"] > 2.0
    # 구간이 세 프레임을 건너므로 위상 쪽이 솎기 쪽의 세 배다.
    assert abs(want["phase_half"] / want["decim2"] - 3.0) < 1e-9
    assert abs(want["decim2"] - 0.5 * SPINUP_ACCEL_RAD_PER_S2 * dt * dt
               / SPINUP_LENGTHS[0]) < 1e-18


def test_every_segment_crosses_an_odd_number_of_frames():
    # 짝수로 건너면 반 주기 솎기에서 두 끝이 같은 양만큼 밀려 차가 0 이 된다.
    # 그러면 그 칸이 다시 "경로를 꺼도 통과하는" 칸으로 돌아간다.
    _, _, _, idx = spinup_segments()
    assert idx
    assert all((j - i) % 2 == 1 for i, j, _ in idx)


def test_the_measured_path_lands_on_the_closed_form():
    t, rot, pos, idx = spinup_segments()
    got, _ = measure(t, rot, pos, lengths=SPINUP_LENGTHS, step=SPINUP_STEP)
    want = spinup_reference(idx)
    for name in ("phase_half", "decim2"):
        assert abs(got[name] - want[name]) / want[name] < 1e-4


def test_a_resampling_path_that_does_nothing_fails_the_non_zero_cells():
    # 재표집을 끈 꼴을 흉내 낸다 — 돌려주는 것이 원본 그대로면 이동량이 0 이다.
    # 답이 0 인 칸은 그래도 통과하고, 이 두 칸만 깨진다.
    _, _, _, idx = spinup_segments()
    want = spinup_reference(idx)
    for name in ("phase_half", "decim2"):
        assert abs(0.0 - want[name]) / want[name] > 0.99


def test_the_trajectory_turns_with_a_constant_angular_acceleration():
    # 닫힌 꼴의 전제다. 각이 2 차가 아니면 위 세 검사가 전부 다른 것을 잰다.
    t, rot, _ = spinup()
    k = len(t) // 2
    ang = np.arccos(np.clip(
        (np.trace(rot[0].T @ rot[k]) - 1.0) / 2.0, -1.0, 1.0))
    want = 0.5 * SPINUP_ACCEL_RAD_PER_S2 * t[k] ** 2
    assert abs(ang - want % (2.0 * np.pi)) < 1e-9 or \
        abs(ang - (2.0 * np.pi - want % (2.0 * np.pi))) < 1e-9
