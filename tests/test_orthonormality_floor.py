"""직교 이탈이 점수에 남기는 바닥을 재는 쪽을 묶는다.

조용히 틀리는 자리는 둘이다. 이탈폭을 0 으로 잘못 재면 "문제 없음" 으로
보이고, 바닥을 구간 수로 안 가중하면 짧은 시퀀스가 긴 시퀀스를 눌러 버린다.
"""

import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

from kitti_segment_D import measure, proper, segments  # noqa: E402
from orthonormality_floor import deviation, floor_of  # noqa: E402

LEN1 = ((1.0,), 1)


def track(n=600, dt=0.1, v=10.0):
    t = np.arange(n) * dt
    yaw = 0.4 * np.sin(0.7 * t)
    rot = Rotation.from_euler("z", yaw).as_matrix()
    dpos = np.stack([v * np.cos(yaw), v * np.sin(yaw), np.zeros(n)],
                    axis=1) * dt
    pos = np.cumsum(dpos, axis=0) - dpos[0]
    return t, rot, pos


def test_a_real_rotation_has_no_deviation_and_no_floor():
    t, rot, pos = track()
    assert deviation(rot) < 1e-14
    d, n = floor_of(t, rot, pos, *LEN1)
    assert n > 0
    # 0 이 아니라 1e-8 쪽이다. 공식 구현이 쓰는 역코사인은 항등 근처에서
    # 배정밀도 끝자리를 제곱근으로 키우므로 정확한 회전끼리도 그만큼 남는다.
    # 실제로 잰 값보다 다섯 자릿수 아래라 판정을 안 바꾼다.
    assert d < 1e-7


def test_truncating_the_published_digits_shows_up_as_a_deviation():
    # 공개 파일이 소수 자리를 자르는 것을 흉내 낸다.
    t, rot, pos = track()
    cut = np.round(rot, 7)
    assert deviation(cut) > 1e-9
    d, _ = floor_of(t, cut, pos, *LEN1)
    assert d > 0


def test_the_floor_grows_faster_than_the_deviation_itself():
    # 측지각이 항등 근처에서 제곱근으로 키운다 — 이탈폭 e 에 대해 바닥이
    # e 가 아니라 대략 sqrt(e) 쪽으로 간다. 선형으로 보면 과소평가한다.
    t, rot, pos = track()
    coarse, fine = np.round(rot, 5), np.round(rot, 7)
    dc, _ = floor_of(t, coarse, pos, *LEN1)
    df, _ = floor_of(t, fine, pos, *LEN1)
    assert dc > df
    assert dc / df < (deviation(coarse) / deviation(fine)) / 2.0


def test_the_floor_is_counted_on_the_same_segments_as_what_it_is_compared_to():
    # 본문은 바닥을 같은 릴리스·같은 규칙의 상한 옆에 놓고 "상한이 바닥의 몇
    # 배" 로 읽는다. 두 수가 다른 구간 집합에서 나오면 그 비가 직교 이탈이
    # 아니라 창 차이를 섞어 재게 된다. 공개 시각처럼 기원이 큰 쪽에서 특히
    # 갈린다 — 배정밀도 칸이 2.4e-07 초라 창 경계가 프레임 단위로 밀린다.
    t, rot, pos = track()
    epoch = t + 1.7e9
    _, n_measure = measure(epoch, rot, pos, lengths=(1.0,), step=1)
    _, n_floor = floor_of(epoch, rot, pos, *LEN1)
    assert n_floor == n_measure
    # 창을 안 자르던 때의 수는 이것보다 크다 — 그 차이가 섞이던 몫이다.
    assert len(segments(np.asarray(pos, dtype=float), (1.0,), 1)) > n_floor


def test_pressing_the_matrix_down_leaves_a_real_rotation():
    _, rot, _ = track()
    cut = np.round(rot, 5)
    assert deviation(proper(cut)) < 1e-14
