"""키티를 공개 시각으로 읽을 때와 공식 0.1 초 격자로 읽을 때를 가르는 탐침을 묶는다.

이 릴리스는 자세 파일에 시각 칸이 없고 시퀀스마다 times.txt 를 따로 싣는데,
그 간격이 0.1 초가 아니다. 공식 평가 프로그램은 그 파일을 안 읽고 0.1 초를
꽂아 쓰므로 읽는 쪽에 축이 둘 열려 있다.

묶는 것 셋이다. 시각 파일이 있으면 그걸 쓰는가, 없으면 격자로 가는가,
속도 흔들림이 자세와 안 맞물린 축에서 실제로 커지는가.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

from kitti_segment_D import load_poses  # noqa: E402
from kitti_time_axis import NOMINAL_DT, speed_jitter  # noqa: E402


def write_sequence(root, seq, times):
    """poses/<seq>.txt 와 (주면) sequences/<seq>/times.txt 를 만든다."""
    os.makedirs(os.path.join(root, "poses"), exist_ok=True)
    n = len(times) if times is not None else 5
    rows = []
    for k in range(n):
        m = np.eye(4)[:3]
        m[0, 3] = 2.0 * k
        rows.append(" ".join("%.9f" % v for v in m.reshape(-1)))
    p = os.path.join(root, "poses", "%s.txt" % seq)
    with open(p, "w") as f:
        f.write("\n".join(rows) + "\n")
    if times is not None:
        d = os.path.join(root, "sequences", seq)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "times.txt"), "w") as f:
            f.write("\n".join("%.6e" % t for t in times) + "\n")
    return p


def test_the_published_times_are_used_when_the_release_ships_them(tmp_path):
    times = [0.0, 0.1036, 0.2073, 0.3109, 0.4146]
    p = write_sequence(str(tmp_path), "00", times)
    t, rot, pos = load_poses(p)
    assert np.allclose(t, times)
    assert len(rot) == 5 and len(pos) == 5
    # 공식 프로그램이 꽂아 쓰는 간격과 다르다는 것이 이 탐침의 전제다.
    assert abs(float(np.median(np.diff(t))) - NOMINAL_DT) > 1e-4


def test_a_sequence_without_times_falls_back_to_the_nominal_grid(tmp_path):
    p = write_sequence(str(tmp_path), "11", None)
    t, _, _ = load_poses(p)
    assert np.allclose(t, np.arange(5) * NOMINAL_DT)


def test_the_jitter_grows_on_an_axis_that_does_not_match_the_motion():
    """고르게 걸은 궤적을 들쭉날쭉한 축에 얹으면 겉보기 속도가 튄다."""
    pos = np.stack([np.arange(400) * 1.0, np.zeros(400), np.zeros(400)], 1)
    even = np.arange(400) * 0.1
    uneven = np.cumsum(np.r_[0.0, np.tile([0.09, 0.11], 199), 0.09])
    assert speed_jitter(even, pos) < 1e-9
    # 0.09 와 0.11 을 오가면 이웃 속도가 한 번에 20 퍼센트쯤 갈린다.
    assert speed_jitter(uneven, pos) > 0.15


def test_the_jitter_is_not_reported_when_the_platform_barely_moves():
    """움직인 자리가 스무 곳도 안 되면 수를 내지 않는다."""
    pos = np.zeros((400, 3))
    assert np.isnan(speed_jitter(np.arange(400) * 0.1, pos))
