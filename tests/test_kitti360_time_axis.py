"""키티-360 의 시간축이 프레임 격자인지, 시각 파일은 골라야만 쓰이는지 본다.

이 릴리스만 시각을 안 싣고 프레임 번호를 싣는다. 축이 둘 서는데 자세는 그중
프레임 격자 위에 있다 — oxts 시각으로 읽으면 1 m 규칙의 병진 상한이 두 배에서
열 배로 뛴다(results/next/kitti360_time_axis.csv). 그래서 기본값이 조용히
시각 파일 쪽으로 넘어가면 표의 수가 통째로 바뀌고, 바뀐 사실은 아무 데도
안 남는다.

묶는 것 넷이다. 나노초가 부동소수로 옮겨 가며 뭉개지지 않는가, 기본 축이
프레임 번호를 공개 주기로 나눈 것인가, 골랐을 때만 시각 파일을 쓰는가,
번호가 그 파일 밖으로 나가면 격자로 돌아오는가.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

from kitti_segment_D import (kitti360_timestamps,  # noqa: E402
                             load_poses_kitti360)

STAMPS = ["2013-05-28 08:46:11.906941184",
          "2013-05-28 08:46:12.006953472",
          "2013-05-28 08:46:12.106965760",
          "2013-05-28 08:46:12.207978048"]


def write_drive(root, frames, stamps=STAMPS):
    """poses.txt 하나와 (주면) oxts/timestamps.txt 하나를 만든다."""
    d = os.path.join(root, "2013_05_28_drive_0000_sync")
    os.makedirs(d, exist_ok=True)
    rows = []
    for k, n in enumerate(frames):
        m = np.eye(4)[:3]
        m[0, 3] = 2.0 * k
        rows.append("%d " % n + " ".join("%.9f" % v for v in m.reshape(-1)))
    p = os.path.join(d, "poses.txt")
    with open(p, "w") as f:
        f.write("\n".join(rows) + "\n")
    if stamps is not None:
        os.makedirs(os.path.join(d, "oxts"), exist_ok=True)
        with open(os.path.join(d, "oxts", "timestamps.txt"), "w") as f:
            f.write("\n".join(stamps) + "\n")
    return p


def test_the_nanosecond_field_survives_the_conversion(tmp_path):
    p = write_drive(str(tmp_path), [0, 1, 2, 3])
    t = kitti360_timestamps(p)
    # 간격은 나노초 그대로여야 한다. 1.37e+18 나노초를 그대로 배정밀도에
    # 담으면 여기서 수십 나노초가 날아간다.
    assert np.allclose(np.diff(t), [0.100012288, 0.100012288, 0.101012288],
                       atol=1e-12)
    assert abs(t[0] - 1369730771.906941184) < 1e-6


def test_the_default_axis_is_the_frame_grid(tmp_path):
    # 시각 파일이 있어도 기본값은 번호를 공개 주기로 나눈 격자다. 번호가
    # 비어도 그 자리를 메우지 않는다 — 2 번이 빠졌으면 1 번과 3 번 사이가
    # 두 칸이다.
    p = write_drive(str(tmp_path), [0, 1, 3])
    t, rot, pos = load_poses_kitti360(p)
    assert np.allclose(t, [0.0, 0.1, 0.3])
    assert len(rot) == 3 and len(pos) == 3


def test_the_stamp_file_is_used_only_when_asked(tmp_path):
    p = write_drive(str(tmp_path), [0, 1, 3])
    full = kitti360_timestamps(p)
    t, _, _ = load_poses_kitti360(p, stamped=True)
    assert np.allclose(t, full[[0, 1, 3]])


def test_a_frame_number_past_the_stamp_file_falls_back(tmp_path):
    # 번호가 시각 파일보다 멀리 가면 짝이 안 맞는 것이다. 그 자리에서
    # 색인 오류를 내거나 엉뚱한 줄을 집지 않고 통째로 격자로 간다.
    p = write_drive(str(tmp_path), [0, 1, 99])
    t, _, _ = load_poses_kitti360(p, stamped=True)
    assert np.allclose(t, [0.0, 0.1, 9.9])


def test_a_drive_without_stamps_stays_on_the_grid(tmp_path):
    p = write_drive(str(tmp_path), [0, 1, 3], stamps=None)
    assert kitti360_timestamps(p) is None
    t, _, _ = load_poses_kitti360(p, stamped=True)
    assert np.allclose(t, [0.0, 0.1, 0.3])
