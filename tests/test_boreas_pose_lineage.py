"""센서별 자세가 한 해에서 나왔는지 가르는 측정을 묶는다.

약한 고리 둘이다. 하나는 회전 규약 — 잘못 잡으면 같은 해인데도 잔차가 도 단위로
나와서 "따로 만든 해" 라는 반대 결론이 나온다. 둘은 판정의 비어 있음 — 어떤 입력을
넣어도 "한 해에서 나왔다" 가 나오면 그 측정은 아무것도 안 가른다. 그래서 심은 답을
되찾는 시험과 일부러 다른 해를 넣는 시험을 같이 둔다.
"""

import os
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

from boreas_pose_lineage import (  # noqa: E402
    ROOT, interpolation_yardstick, lineage, load_poses, ypr_to_rot,
)


def devkit_rot(y, p, r):
    """pyboreas utils.py 의 세 기본 행렬을 그대로 적은 것."""
    roll = np.array([[1, 0, 0],
                     [0, np.cos(r), np.sin(r)],
                     [0, -np.sin(r), np.cos(r)]])
    pitch = np.array([[np.cos(p), 0, -np.sin(p)],
                      [0, 1, 0],
                      [np.sin(p), 0, np.cos(p)]])
    yaw = np.array([[np.cos(y), np.sin(y), 0],
                    [-np.sin(y), np.cos(y), 0],
                    [0, 0, 1]])
    return roll @ pitch @ yaw


def smooth_solution(n=4000, rate=200.0, seed=5):
    """천천히 도는 조밀한 해. 200 Hz 에서 보간이 잘 듣는 매끄러움이다."""
    t = np.arange(n) / rate
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    ang = 0.3 * np.sin(2 * np.pi * 0.11 * t) + 0.05 * t
    rot = Rotation.from_rotvec(np.outer(ang, axis))
    pos = np.stack([10.0 * t, 2.0 * np.sin(0.3 * t), 0.1 * t], axis=1)
    return t, rot, pos


def test_the_rotation_matches_the_devkit_composition():
    y, p, r = 0.7, -0.2, 0.15
    got = ypr_to_rot(np.array([y]), np.array([p]), np.array([r]))[0]
    assert np.max(np.abs(got - devkit_rot(y, p, r))) < 1e-14


def test_the_rotation_is_a_proper_rotation():
    rng = np.random.default_rng(1)
    a = rng.uniform(-np.pi, np.pi, size=(20, 3))
    m = ypr_to_rot(a[:, 0], a[:, 1], a[:, 2])
    assert np.max(np.abs(np.einsum("nab,nac->nbc", m, m)
                         - np.eye(3))) < 1e-13
    assert np.allclose(np.linalg.det(m), 1.0)


def test_zero_angles_give_the_identity():
    z = np.zeros(1)
    assert np.max(np.abs(ypr_to_rot(z, z, z)[0] - np.eye(3))) < 1e-15


def test_the_columns_are_read_in_the_published_order(tmp_path):
    # 열을 하나만 밀려 읽어도 프로그램은 돌아가고 수만 틀린다.
    p = tmp_path / "x.csv"
    p.write_text("GPSTime,easting,northing,altitude,vel_east,vel_north,"
                 "vel_up,roll,pitch,heading,angvel_z,angvel_y,angvel_x\n"
                 "1606417097.5,11,22,33,0,0,0,0.1,0.2,0.3,0,0,0\n")
    t, rot, pos = load_poses(str(p))
    assert t[0] == pytest.approx(1606417097.5)
    assert list(pos[0]) == [11.0, 22.0, 33.0]
    assert np.max(np.abs(rot.as_matrix()[0] - devkit_rot(0.3, 0.2, 0.1))) < 1e-14


def test_microsecond_stamps_become_seconds(tmp_path):
    # 센서 파일은 마이크로초, 조밀한 해는 초다. 안 맞추면 겹치는 구간이 0 이 된다.
    p = tmp_path / "x.csv"
    p.write_text("GPSTime,easting,northing,altitude,vel_east,vel_north,"
                 "vel_up,roll,pitch,heading\n"
                 "1606417097502930,0,0,0,0,0,0,0,0,0\n"
                 "1606417097602930,0,0,0,0,0,0,0,0,0\n")
    t, _, _ = load_poses(str(p))
    assert t[0] == pytest.approx(1606417097.50293)
    assert t[1] - t[0] == pytest.approx(0.1, abs=1e-6)


def test_a_sensor_file_cut_from_the_same_solution_comes_back_as_such():
    t, rot, pos = smooth_solution()
    mark_med, _ = interpolation_yardstick(t, rot)

    fixed = Rotation.from_rotvec([0.0, 0.0, np.radians(42.6)])
    lever = np.array([0.1, -0.05, 1.2])
    ts = t[5:-5:20] + 0.0013          # 조밀한 격자에 안 걸치는 시각
    at = Rotation.from_rotvec(
        np.stack([np.interp(ts, t, rot.as_rotvec()[:, k])
                  for k in range(3)], axis=1))
    ps = np.stack([np.interp(ts, t, pos[:, k]) for k in range(3)], axis=1) \
        + np.einsum("nab,b->na", at.as_matrix(), lever)

    got = lineage(t, rot, pos, ts, at * fixed, ps)
    assert got["offset_deg"] == pytest.approx(42.6, abs=1e-3)
    assert got["residual_median_deg"] < mark_med
    assert got["lever_norm_m"] == pytest.approx(np.linalg.norm(lever),
                                                abs=1e-6)
    assert got["lever_spread_m"] < 1e-6


def test_a_sensor_file_from_another_solution_does_not():
    # 비어 있지 않은지 보는 자리다. 자세에 느린 흔들림을 얹으면 고정 회전
    # 하나로는 못 흡수하고 잔차가 눈금 위로 올라와야 한다.
    t, rot, pos = smooth_solution()
    mark_med, _ = interpolation_yardstick(t, rot)
    ts = t[5:-5:20] + 0.0013
    at = Rotation.from_rotvec(
        np.stack([np.interp(ts, t, rot.as_rotvec()[:, k])
                  for k in range(3)], axis=1))
    wobble = Rotation.from_rotvec(
        np.outer(np.radians(0.02) * np.sin(2 * np.pi * 0.7 * ts),
                 [0.0, 1.0, 0.0]))
    ps = np.stack([np.interp(ts, t, pos[:, k]) for k in range(3)], axis=1)

    got = lineage(t, rot, pos, ts, at * wobble, ps)
    assert got["residual_median_deg"] > 10 * mark_med


def test_the_yardstick_grows_when_the_solution_turns_faster():
    t, rot, _ = smooth_solution()
    fast = Rotation.from_rotvec(4.0 * rot.as_rotvec())
    assert interpolation_yardstick(t, fast)[0] > \
        interpolation_yardstick(t, rot)[0]


def test_the_released_boreas_files_come_from_one_solution():
    seq = "boreas-2020-11-26-13-58"
    d = os.path.join(ROOT, seq, "applanix")
    if not os.path.exists(os.path.join(d, "lidar_poses.csv")):
        pytest.skip("보레아스 자료가 이 기계에 없다")
    t_d, rot_d, pos_d = load_poses(os.path.join(d, "gps_post_process.csv"))
    mark_med, _ = interpolation_yardstick(t_d, rot_d)
    for s in ("lidar", "radar", "camera"):
        t_s, rot_s, pos_s = load_poses(os.path.join(d, "%s_poses.csv" % s))
        got = lineage(t_d, rot_d, pos_d, t_s, rot_s, pos_s)
        assert got["residual_median_deg"] < mark_med
        assert got["lever_spread_m"] < 1e-4
