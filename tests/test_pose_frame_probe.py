"""자세 규약 탐침이 실제로 가르는지, 못 가릴 때 그렇다고 말하는지 본다.

이 탐침이 잡는 결함은 조용하다. 자세를 통째로 전치해서 읽으면 상대 회전이
켤레로 옮겨갈 뿐이라 회전 수치가 한 비트도 안 바뀌고, 구간 상대 병진 한 줄만
틀린다. 그래서 탐침이 고장나도 다른 어느 검사도 안 운다 — 여기서 직접 묶는다.

묶는 것 셋이다. 바르게 읽은 궤적을 바르다고 하는가, 전치해서 넣으면 전치라고
하는가, 가를 수 없는 움직임(거의 안 도는 주행)에 함부로 판정을 내리지 않는가.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

from pose_frame_probe import (RATIO, TIGHT_DEG, TURN_DEG,  # noqa: E402
                              direction_spread, fast_mask, turn_coverage)


def drive(n=600, turn=np.pi, forward=(1.0, 0.0, 0.0), step=2.0):
    """앞으로만 가면서 방위가 turn 만큼 도는 주행. (회전, 위치) 를 준다.

    회전은 본체를 세계로 보내는 쪽이다. 전진축을 인자로 받는 이유는 릴리스마다
    앞이 x 이기도 하고 z 이기도 해서다(키티는 카메라 좌표라 z 가 앞이다) —
    다만 여기서 도는 축은 z 라, 그 축과 나란한 전진축은 전치해도 방향이 안
    바뀌므로 애초에 가릴 수 없는 자리다.

    속도를 일부러 흔든다. 탐침이 빠른 쪽 일부만 쓰는데 걸음이 똑같으면 그
    "빠른 쪽" 이 서지 않는다.
    """
    yaw = np.linspace(0.0, turn, n)
    c, s = np.cos(yaw), np.sin(yaw)
    z, o = np.zeros(n), np.ones(n)
    rot = np.stack([c, -s, z, s, c, z, z, z, o], -1).reshape(-1, 3, 3)
    # 여러 주기로 흔든다. 한 주기만 흔들면 빠른 쪽 표본이 한 봉우리에 몰려서
    # 주행이 다 돌았는데도 쓴 표본은 안 돈 상태가 된다.
    speed = step * (1.0 + 0.3 * np.sin(np.linspace(0.0, 70.0, n)))
    pos = np.cumsum(speed[:, None]
                    * np.einsum("nij,j->ni", rot, np.asarray(forward)), 0)
    return rot, pos


def verdict(rot, pos, lag=1):
    """probe() 가 쓰는 판정을 같은 순서로 재현한다."""
    got = direction_spread(rot, pos, lag)
    flipped = direction_spread(np.transpose(rot, (0, 2, 1)), pos, lag)
    m, _ = fast_mask(pos, lag)
    assert got is not None and flipped is not None and m is not None
    if turn_coverage(rot[:-lag][m]) < TURN_DEG:
        return "indecisive"
    if got < TIGHT_DEG and flipped > RATIO * got:
        return "as_read"
    if flipped < TIGHT_DEG and got > RATIO * flipped:
        return "transposed"
    return "indecisive"


def test_a_correctly_read_drive_is_called_as_read():
    for forward in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.6, 0.8, 0.0)):
        rot, pos = drive(forward=forward)
        assert verdict(rot, pos) == "as_read"


def test_a_transposed_drive_is_caught():
    rot, pos = drive()
    assert verdict(np.transpose(rot, (0, 2, 1)), pos) == "transposed"


def test_a_constant_mounting_rotation_does_not_change_the_verdict():
    # 장치가 돌아 달려 있어도 방향은 다른 상수에 머물 뿐이다. 상수는 우리가
    # 재는 양에 안 섞인다 — 같은 회전을 양쪽에 곱하면 차의 크기가 안 변한다.
    rot, pos = drive()
    mount = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert verdict(rot @ mount, pos) == "as_read"


def test_a_drive_that_barely_turns_is_left_undecided():
    # 안 도는 주행은 전치해도 방향이 안 벌어진다. 그 자리에서 "맞다" 고 하면
    # 검사가 아니라 통과 도장이 된다.
    rot, pos = drive(turn=np.radians(5.0))
    assert verdict(rot, pos) == "indecisive"
    assert verdict(np.transpose(rot, (0, 2, 1)), pos) == "indecisive"


def test_the_turn_is_measured_on_the_samples_that_were_used():
    # 전체로는 많이 돌았는데 빠른 쪽 표본만 한 직선에 몰린 궤적. 쓴 표본으로
    # 재야 미정이 되고, 전체로 재면 두 읽기가 똑같이 좁은데도 판정이 나온다.
    straight_rot, straight_pos = drive(n=400, turn=np.radians(2.0), step=5.0)
    turn_rot, turn_pos = drive(n=400, turn=np.pi, step=0.2)
    rot = np.concatenate([straight_rot, turn_rot])
    pos = np.concatenate([straight_pos, turn_pos + straight_pos[-1]])
    used, _ = fast_mask(pos, 1)
    assert turn_coverage(rot[:-1][used]) < TURN_DEG <= turn_coverage(rot)
    assert verdict(rot, pos) == "indecisive"
