"""공식 이진파일에 넘기는 입력이 우리가 잰 것과 같은 궤적인지 본다.

이 대조의 약한 고리는 비교식이 아니라 입력이다. 자세 파일을 잘못 쓰면 공식
프로그램은 멀쩡히 다른 궤적을 평가하고 그럴듯한 수를 찍는다. 그래서 쓰는 쪽과
읽는 쪽을 왕복시켜 묶고, 공식 소스를 자르는 자리도 바이트로 묶는다.
"""

import os
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

from kitti_official_binary_check import (  # noqa: E402
    drift_poses, drift_reference, run_official, run_official_raw,
    single_precision_angle, single_precision_mean, truncate_before_main,
    write_poses,
)
from kitti_segment_D import load_poses  # noqa: E402


def rotations_of_fixed_angle(theta, n, seed):
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=(n, 3))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    return Rotation.from_rotvec(axis * theta).as_matrix()


def test_the_cut_keeps_everything_before_main_byte_for_byte(tmp_path):
    src = tmp_path / "evaluate_odometry.cpp"
    head = b"#include <stdio.h>\n\nfloat lengths[] = {100,200};\n\n"
    tail = b"int32_t main (int32_t argc,char *argv[]) {\n  return 0;\n}\n"
    src.write_bytes(head + tail)
    dst = tmp_path / "funcs.cpp"
    cut = truncate_before_main(str(src), str(dst))
    assert cut == 4
    assert dst.read_bytes() == head


def test_the_cut_refuses_a_file_without_a_main(tmp_path):
    # 원본 구조가 바뀌면 조용히 빈 파일을 빌드하지 말고 멈춰야 한다.
    src = tmp_path / "x.cpp"
    src.write_bytes(b"void f() {}\n")
    with pytest.raises(SystemExit):
        truncate_before_main(str(src), str(tmp_path / "out.cpp"))


def test_what_we_hand_the_official_program_reads_back_as_the_same_trajectory(
        tmp_path):
    rng = np.random.default_rng(7)
    n = 40
    rot = Rotation.from_rotvec(rng.normal(0.0, 0.5, size=(n, 3))).as_matrix()
    pos = np.cumsum(rng.normal(0.0, 1.0, size=(n, 3)), axis=0)
    p = tmp_path / "poses" / "00.txt"
    p.parent.mkdir()
    write_poses(str(p), rot, pos)
    _, rot2, pos2 = load_poses(str(p))
    # 깎지 않고 적는다. 깎으면 공식 프로그램이 읽는 수가 우리가 잰 수와
    # 다르고, 두 출력의 남은 차이에 그 몫이 섞여 공식 산술의 몫과 안 갈린다.
    assert np.array_equal(rot2, rot)
    assert np.array_equal(pos2, pos)


def test_the_rows_are_in_the_official_three_by_four_order(tmp_path):
    # 공식 loadPoses 는 행 우선 3x4 로 읽는다. 열을 섞으면 회전과 위치가
    # 뒤바뀌는데도 프로그램은 돌아간다.
    rot = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
    pos = np.array([[10.0, 11.0, 12.0]])
    p = tmp_path / "one.txt"
    write_poses(str(p), rot, pos)
    v = [float(x) for x in p.read_text().split()]
    assert v == [1.0, 2.0, 3.0, 10.0, 4.0, 5.0, 6.0, 11.0,
                 7.0, 8.0, 9.0, 12.0]


def wandering_ride(n=600, seed=3):
    """1 미터/프레임으로 가면서 천천히 도는 궤적."""
    rng = np.random.default_rng(seed)
    yaw = np.cumsum(rng.normal(0.0, 0.01, size=n))
    rot = Rotation.from_euler("z", yaw).as_matrix()
    step = np.stack([np.cos(yaw), np.sin(yaw), np.zeros(n)], axis=1)
    return rot, np.cumsum(step, axis=0)


def test_the_drift_turns_each_frame_by_its_own_index_times_eps():
    # 흔들림의 크기를 우리가 안다는 것이 탐침의 전제다. k 번째 프레임이
    # 몸체축으로 정확히 k*eps 만큼 더 돌아가 있어야 한다.
    rot, _ = wandering_ride(n=50)
    eps = 1e-3
    axis = (0.0, 1.0, 0.0)
    drifted = drift_poses(rot, eps, axis=axis)
    for k in (0, 1, 17, 49):
        extra = np.asarray(rot[k]).T @ drifted[k]
        ang = Rotation.from_matrix(extra).as_rotvec()
        assert np.allclose(ang, np.asarray(axis) * (k * eps), atol=1e-12)


def test_no_drift_means_no_reference_error():
    # 정확히 0 은 아니다. 각이 0 이면 대각합이 3 에 붙는데 acos 의 기울기가
    # 거기서 발산해서 배정밀도 반올림이 1e-11 라디안쯤으로 번진다. 재려는
    # 크기(1e-4 이상)보다 일곱 자리 아래라 바닥으로 묶는다.
    rot, pos = wandering_ride()
    assert drift_reference(rot, pos, 0.0) < 1e-9


def test_the_reference_grows_with_the_drift_we_put_in():
    # 기준값을 닫힌 꼴로 예측하지 않고 재는 이유가 여기 있다 — 예측은
    # 축이 다 같을 때만 맞는다. 대신 작은 eps 에서는 선형이어야 한다.
    rot, pos = wandering_ride()
    one = drift_reference(rot, pos, 1e-6)
    two = drift_reference(rot, pos, 2e-6)
    assert one > 0.0
    assert abs(two / one - 2.0) < 0.01


def test_single_precision_loses_a_small_angle_outright():
    # 공식 구현의 바닥이 왜 생기는지 그대로 보여주는 자리다. 1e-05 라디안이면
    # 대각합이 단정밀도에서 정확히 3 이 되어 acos 가 0 을 돌려준다.
    err = rotations_of_fixed_angle(1e-5, 8, seed=1)
    assert np.max(single_precision_angle(err)) == 0.0


def test_single_precision_keeps_a_large_angle():
    theta = 0.5
    err = rotations_of_fixed_angle(theta, 8, seed=2)
    got = single_precision_angle(err)
    assert np.max(np.abs(got.astype(np.float64) - theta)) < 1e-6


def test_the_two_accumulators_are_read_from_the_same_run(tmp_path):
    # 누산형만 바꾼 차를 재려면 두 값이 같은 구간 오차에서 나와야 한다.
    # 구동부가 네 수를 한 줄에 찍고, 읽는 쪽이 그 줄을 다 읽는지 묶는다.
    fake = tmp_path / "official_mean"
    fake.write_text("#!/bin/sh\necho '14562 0.00123456789 0.00456789 "
                    "0.00123456 0.0045679'\n")
    fake.chmod(0o755)
    got = run_official_raw(str(fake), "gt.txt", "est.txt")
    assert got[0] == 14562
    assert got[1] == 0.00123456789 and got[3] == 0.00123456
    assert got[2] == 0.00456789 and got[4] == 0.0045679
    # 옛 호출부는 세 칸만 받는다. 다섯 칸을 돌려주면 그 자리가 깨진다.
    assert run_official(str(fake), "gt.txt", "est.txt") == got[:3]


def test_the_single_precision_mean_is_the_mean_of_angle_over_length():
    ang = np.asarray([0.3, 0.7, 1.1, 0.5], dtype=np.float32)
    lengths = [100.0, 200.0, 400.0, 800.0]
    want = float(np.mean(np.asarray(ang, dtype=np.float64)
                         / np.asarray(lengths)))
    assert abs(single_precision_mean(ang, lengths) - want) < 1e-9
