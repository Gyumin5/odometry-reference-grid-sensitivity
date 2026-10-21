"""거리 공리를 세는 쪽을 묶는다.

조용히 틀리는 자리는 0 이다. 자기거리가 0 으로 보이는 데에는 두 길이 있고 —
정말 0 인 것과 arccos 인자가 1 을 넘어 잘린 것 — 둘을 안 가르면 뒤쪽이 공리를
지킨 것으로 세어진다. 삼각부등식도 같다. 무작위 삼조는 각이 큰 자리를 주로
뽑는데, 역코사인이 못 견디는 곳은 항등 근처다.
"""

import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

from metric_axioms_probe import (clipped_self,  # noqa: E402
                                 near_identity_triangle, self_distance)


def slowly_turning(n=400, rate=0.001, seed=5):
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    return Rotation.from_rotvec(
        np.outer(np.arange(n) * rate, axis)).as_matrix()


def test_a_swollen_matrix_is_counted_as_clipped_not_as_a_zero():
    # 행렬을 아주 조금 부풀리면 tr(A^T A) 가 3 을 넘어 arccos 인자가 1 을
    # 넘는다. clip 이 그걸 1 로 눌러 자기거리가 0 으로 나오는데, 그건 공리를
    # 지킨 게 아니라 가린 것이다.
    rot = slowly_turning(n=5) * (1.0 + 1e-9)
    assert np.max(self_distance(rot)) == 0.0
    n, over = clipped_self(rot)
    assert n == 5
    assert over > 0.0


def test_a_shrunken_matrix_is_not_counted_as_clipped():
    # 반대쪽으로 벗어나면 인자가 1 보다 작아져 자기거리가 0 이 아니게 나온다.
    # 그 자리는 이미 보이므로 잘린 칸으로 세면 안 된다.
    rot = slowly_turning(n=5) * (1.0 - 1e-9)
    assert np.max(self_distance(rot)) > 0.0
    assert clipped_self(rot)[0] == 0


def test_a_real_rotation_clips_only_by_the_last_bit():
    # 정확한 회전도 끝자리 때문에 인자가 1 을 아주 조금 넘는다. 그래서 세는
    # 수만으로는 못 가르고 넘은 폭을 같이 적는다 — 폭이 끝자리 크기면 식의
    # 반올림이고, 그보다 크면 행렬이 회전이 아니라는 뜻이다.
    rot = slowly_turning(n=5)
    assert clipped_self(rot)[1] < 1e-14
    assert clipped_self(rot * (1.0 + 1e-9))[1] > 1e-10


def test_the_near_identity_triples_are_consecutive_and_counted_whole():
    rot = slowly_turning(n=10)
    got = near_identity_triangle(rot)
    assert got.size == 8
    # 같은 축으로 같은 양씩 도는 궤적이면 세 각이 정확히 더해져서 여유가 0 이다.
    assert np.max(np.abs(got)) < 1e-9


def test_the_near_identity_test_sees_what_random_triples_miss():
    # 이웃끼리는 각이 1e-3 라디안쯤이라 역코사인 끝자리가 제곱근으로 커진다.
    # 멀리 떨어진 삼조는 각이 커서 그 자리를 안 밟는다.
    rot = slowly_turning(n=300)
    near = near_identity_triangle(rot)
    far = (np.arccos(np.clip((np.trace(rot[0].T @ rot[299]) - 1) / 2, -1, 1))
           - np.arccos(np.clip((np.trace(rot[0].T @ rot[150]) - 1) / 2, -1, 1))
           - np.arccos(np.clip((np.trace(rot[150].T @ rot[299]) - 1) / 2,
                               -1, 1)))
    assert abs(near).max() >= abs(far) or abs(far) < 1e-12
