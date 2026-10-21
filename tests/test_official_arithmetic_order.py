"""공식 구간 고르기를 옮길 때는 산술 순서까지 옮겨야 한다.

evaluate_odometry.cpp 의 Matrix 는 원소가 double 이고(matrix.h 46줄
typedef double FLOAT), trajectoryDistances 는 그 double 좌표를 뺀 뒤
결과를 float 변수에 담는다. 그래서 자르는 자리가 좌표가 아니라 차다.
좌표를 먼저 자르고 빼면 마지막 자리가 달라지고, 누적 거리가 갈리고,
lastFrameFromSegmentLength 가 고르는 끝 프레임이 달라진다 — 구간 수는
그대로인 채 구간이 달라지므로 개수만 세면 안 보인다.

문턱도 같다. 원본은 dist 와 len 이 둘 다 float 이라 그 합이 float 인데,
길이를 파이썬 수로 두면 덧셈이 배정밀도로 올라가서 경계에 걸린 프레임의
판정이 뒤집힌다.

두 순서가 실제로 다른 답을 내는 좌표를 만들어 고정한다. 다르지 않다면
이 검사는 아무것도 지키지 못하므로, 먼저 두 순서가 갈린다는 것부터 센다.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "scripts"))

from kitti_official_crosscheck import (  # noqa: E402
    official_distances, official_last_frame,
)


def _coords_where_the_two_orders_differ():
    """차를 자르는 것과 좌표를 자르는 것이 갈리는 x 좌표 쌍을 찾는다."""
    rng = np.random.default_rng(20261021)
    for _ in range(100000):
        a = float(rng.uniform(100.0, 1000.0))
        b = a + float(rng.uniform(1e-3, 2.0))
        diff_then_cast = np.float32(b - a)
        cast_then_diff = np.float32(b) - np.float32(a)
        if diff_then_cast != cast_then_diff:
            return a, b, diff_then_cast, cast_then_diff
    raise AssertionError("두 순서가 갈리는 좌표를 못 찾았다")


def test_the_two_rounding_orders_really_disagree():
    a, b, diff_then_cast, cast_then_diff = _coords_where_the_two_orders_differ()
    assert diff_then_cast != cast_then_diff
    assert abs(a) > 0 and b > a


def test_the_port_rounds_the_difference_not_the_coordinates():
    a, b, diff_then_cast, _ = _coords_where_the_two_orders_differ()
    pos = np.array([[a, 0.0, 0.0], [b, 0.0, 0.0]], dtype=np.float64)
    dist = official_distances(pos, "float32")
    assert dist[1] == np.float32(diff_then_cast)


def test_the_threshold_adds_in_the_accumulation_precision():
    # 누적 거리가 단정밀도면 문턱 합도 단정밀도여야 한다. 배정밀도로 더하면
    # 같은 값이 "넘지 않았다" 로 읽혀 끝 프레임이 한 칸 뒤로 밀린다.
    step = np.float32(1.0)
    dist = [np.float32(0.0)]
    for i in range(1, 300):
        dist.append(np.float32(dist[i - 1] + step))
    length = 100.0
    got = official_last_frame(dist, 0, length)
    assert isinstance(got, int) and got > 0
    assert dist[got] > np.float32(dist[0] + np.float32(length))
    assert dist[got - 1] <= np.float32(dist[0] + np.float32(length))


def test_a_differing_list_is_caught_even_when_the_counts_match():
    # 개수만 세면 못 잡는 경우를 고정한다. 길이가 같고 내용이 다른 두 목록.
    ours = [(0, 10, 100.0), (10, 21, 100.0)]
    theirs = [(0, 10, 100.0), (10, 22, 100.0)]
    assert len(ours) == len(theirs)
    differing = [(a, b) for a, b in zip(ours, theirs) if a != b]
    assert len(differing) == 1
