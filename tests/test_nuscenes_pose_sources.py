"""누씬즈 자세 두 벌을 맞대는 탐침을 묶는다.

묶는 것 셋이다. 기가 단위 표를 통째로 안 올리고 한 원소씩 흘려 읽는가,
같은 시각끼리 짝지을 때 허용 폭 밖의 짝을 버리는가, 두 벌이 어긋나 있을 때
그 어긋남이 수로 나오는가.
"""

import json
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

from nuscenes_pose_sources import (MATCH_S, best_shift, gap,  # noqa: E402
                                   rate, stream)


def write_pretty(path, rows):
    """누씬즈 본 묶음과 같은 꼴로 찍는다 — 원소마다 줄을 나눈 배열."""
    with open(path, "w") as f:
        f.write("[\n")
        for k, r in enumerate(rows):
            f.write(json.dumps(r, indent=0))
            f.write(",\n" if k + 1 < len(rows) else "\n")
        f.write("]")


def test_the_table_is_read_one_element_at_a_time(tmp_path):
    rows = [{"token": "a", "timestamp": 1, "translation": [1.0, 2.0, 0.0]},
            {"token": "b", "timestamp": 2, "translation": [3.0, 4.0, 0.0]}]
    p = str(tmp_path / "ego_pose.json")
    write_pretty(p, rows)
    got = list(stream(p))
    assert got == rows


def traj(t, x, deg=None):
    t = np.asarray(t, float)
    pos = np.stack([np.asarray(x, float), np.zeros(len(t)),
                    np.zeros(len(t))], 1)
    ang = np.zeros(len(t)) if deg is None else np.asarray(deg, float)
    rot = Rotation.from_euler("z", ang, degrees=True).as_matrix()
    return t, rot, pos


def test_a_pair_outside_the_window_is_not_counted():
    """허용 폭 밖의 짝은 걸음 자체가 차이로 섞여서 버린다."""
    a = traj([0.0, 1.0], [0.0, 10.0])
    far = traj([0.0 + 2 * MATCH_S, 1.0 + 2 * MATCH_S], [0.0, 10.0])
    n, _ = gap(a, far)
    assert n == 0
    near = traj([0.0 + MATCH_S / 2, 1.0 + MATCH_S / 2], [0.0, 10.0])
    n, _ = gap(a, near)
    assert n == 2


def test_the_two_series_are_compared_in_position_and_in_heading():
    a = traj([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], deg=[0.0, 0.0, 0.0])
    b = traj([0.0, 1.0, 2.0], [0.0, 1.5, 2.5], deg=[0.0, 2.0, 4.0])
    n, g = gap(a, b)
    assert n == 3
    # 위치는 0, 0.5, 0.5 라 중앙값 0.5, 방향은 0, 2, 4 도라 중앙값 2 도다.
    assert abs(g[0] - 0.5) < 1e-9
    assert abs(g[2] - 2.0) < 1e-6


def test_an_identical_pair_shows_no_gap():
    a = traj([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
    n, g = gap(a, a)
    assert n == 3 and max(g) < 1e-9


def test_a_pure_clock_offset_is_found_and_closed():
    """같은 길을 다른 시계로 적은 두 벌은 밀면 차이가 사라진다."""
    t = np.arange(0, 4, 0.02)
    a = traj(t, t * 5.0)
    b = traj(t + 0.05, t * 5.0)
    s, dm, _ = best_shift(a, b)
    # 짝짓기 허용 폭이 표본 간격보다 넓어서 이웃한 밀기 몇 개가 다 0 이
    # 된다 — 찾은 자리가 참값에서 한 간격 안이면 맞은 것으로 본다.
    assert abs(s - 0.05) <= 0.0201
    assert dm < 1e-9
    # 밀기 전에는 5 미터 앞뒤로 벌어져 있다.
    assert gap(a, b)[1][0] > 0.2


def test_a_sideways_offset_does_not_close_with_a_shift():
    """시계가 아니라 풀이가 다르면 어디로 밀어도 차이가 남는다."""
    t = np.arange(0, 4, 0.02)
    a = traj(t, t * 5.0)
    tb, rb, pb = traj(t, t * 5.0)
    pb[:, 1] += 0.3
    s, dm, _ = best_shift(a, (tb, rb, pb))
    assert abs(dm - 0.3) < 1e-6
    assert abs(s) <= 0.2


def test_the_rate_is_read_from_the_median_interval():
    assert abs(rate(np.arange(10) * 0.05) - 20.0) < 1e-9
    assert np.isnan(rate(np.asarray([1.0])))
