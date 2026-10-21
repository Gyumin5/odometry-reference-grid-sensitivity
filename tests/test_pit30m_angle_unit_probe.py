"""각도 단위 탐침이 단위를 실제로 가리는지 본다.

이 탐침이 하는 말은 하나다 — 저장된 각도를 도로 읽어야 궤적과 맞는다.
그 말이 값어치를 가지려면 판정이 양쪽으로 다 움직여야 한다. 도로 적힌 자료에
"도" 라고 답하는 것만으로는 부족하고, 라디안으로 적힌 자료에는 "라디안" 이라고
답해야 한다. 한쪽으로만 답하는 판정식은 자료를 안 보고도 맞는다.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

from pit30m_angle_unit_probe import (DEGREE, RADIAN, forward_axis_error,  # noqa: E402
                                     rot_from_fields, row_for, verdict)


def rot_z(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0.0],
                     [np.sin(a), np.cos(a), 0.0],
                     [0.0, 0.0, 1.0]])


def drive(n=400, fs=10.0, speed=9.0, turn=0.35, climb=0.02):
    """도 단위로 적힌 합성 주행. 돌면서 완만하게 오르내린다."""
    t = np.arange(n, dtype=float) / fs
    yaw = turn * np.sin(0.05 * t)
    x = np.concatenate([[0.0], np.cumsum(np.cos(yaw[:-1]) * speed / fs)])
    y = np.concatenate([[0.0], np.cumsum(np.sin(yaw[:-1]) * speed / fs)])
    z = climb * np.sin(0.03 * t) * 100.0
    pos = np.stack([x, y, z], axis=1)
    dz = np.gradient(z, t)
    pitch = -np.arctan2(dz, speed)
    roll = np.zeros_like(t)
    ypr_rad = np.stack([yaw, pitch, roll], axis=1)
    return t, pos, ypr_rad


def test_a_trajectory_written_in_degrees_is_read_as_degrees():
    """도로 적힌 자료에 도라고 답한다 — 벌어진 각이 1도 아래다."""
    t, pos, ypr_rad = drive()
    row = row_for("deg", t, np.degrees(ypr_rad), pos)
    assert row["verdict"] == "degree"
    assert float(row["median_deg_as_degree"]) < 1.0
    assert float(row["median_deg_as_radian"]) > 20.0


def test_a_trajectory_written_in_radians_is_read_as_radians():
    """같은 궤적을 라디안으로 적으면 반대로 답한다.

    이 검사가 없으면 판정식이 "언제나 도" 여도 통과한다.
    """
    t, pos, ypr_rad = drive()
    row = row_for("rad", t, ypr_rad, pos)
    assert row["verdict"] == "radian"
    assert float(row["median_deg_as_radian"]) < 1.0
    assert float(row["median_deg_as_degree"]) > 10.0


def test_a_pose_that_does_not_face_the_road_is_not_decided():
    """어느 읽기로도 안 맞으면 둘 중 하나를 고르지 않는다."""
    assert verdict(41.0, 62.0) == "undecided"
    assert verdict(float("nan"), float("nan")) == "undecided"


def test_the_measured_angle_is_the_one_between_two_directions():
    """앞축이 길을 그대로 가리키면 0, 직각으로 틀면 90도."""
    t = np.arange(5, dtype=float)
    pos = np.stack([t * 10.0, np.zeros(5), np.zeros(5)], axis=1)
    same = np.repeat(np.eye(3)[None], 5, axis=0)
    assert np.allclose(forward_axis_error(t, same, pos), 0.0)
    side = np.repeat(rot_z(90.0)[None], 5, axis=0)
    assert np.allclose(np.degrees(forward_axis_error(t, side, pos)), 90.0)


def test_standing_still_is_not_counted_as_a_direction():
    """멈춰 있는 구간은 이동 방향이 없어서 한 걸음도 안 센다.

    세면 가만히 선 차의 바퀴 밑 잡음이 방향 행세를 한다.
    """
    t = np.arange(50, dtype=float) / 10.0
    pos = np.zeros((50, 3))
    pos[:, 0] = np.linspace(0.0, 0.5, 50)        # 0.1 m/s
    rot = np.repeat(np.eye(3)[None], 50, axis=0)
    assert len(forward_axis_error(t, rot, pos)) == 0


def test_an_undecided_cell_says_which_kind_of_empty_it_is():
    """선 기록과 한 점짜리 기록을 같은 칸으로 묶지 않는다.

    실제 자료에 둘 다 있는 게 아니라 한 점짜리만 둘 있다 — 그래서 묶어두면
    "가만히 선 기록" 이라고 잘못 읽힌다.
    """
    t = np.arange(20, dtype=float) / 10.0
    still = row_for("still", t, np.zeros((20, 3)), np.zeros((20, 3)))
    assert still["n_steps"] == 0
    assert still["verdict"] == "undecided"
    assert still["median_deg_as_degree"] == "nan"
    assert still["empty_reason"] == "임계보다 빠른 걸음이 없다"

    one = row_for("one", t[:1], np.zeros((1, 3)), np.zeros((1, 3)))
    assert one["empty_reason"] == "자세가 한 점뿐이라 이동 방향이 없다"

    t_d, pos_d, ypr_rad = drive()
    assert row_for("deg", t_d, np.degrees(ypr_rad), pos_d)["empty_reason"] == ""


def test_the_two_readings_differ_only_by_the_factor_between_them():
    """두 읽기가 같은 칸을 환산만 달리 쓴다 — 한쪽을 미리 곱해두면 같아진다."""
    ypr = np.array([[30.0, 2.0, 1.0], [-120.0, -3.0, 0.5]])
    a = rot_from_fields(ypr, DEGREE)
    b = rot_from_fields(ypr * DEGREE, RADIAN)
    assert np.allclose(a, b)
