"""옥스퍼드 로봇카 RTK 적재기를 묶는다.

묶는 것 넷이다. 머리글만 있는 파일에서 빈 궤적을 돌려주는가, UTM 구역이
바뀌는 자리에서 끊고 긴 토막만 읽는가, 위치와 자세를 적힌 칸에서 제대로
가져오는가, 시각이 끊긴 걸음을 그 주행 자신의 간격으로 세는가.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

from kitti_segment_D import load_poses_oxford, longest_run  # noqa: E402
from oxford_rtk_quality import gap_count, is_gap  # noqa: E402

HEAD = ("timestamp,latitude,longitude,altitude,northing,easting,down,"
        "utm_zone,velocity_north,velocity_east,velocity_down,roll,pitch,yaw")


def write(path, rows):
    """(시각us, 북, 동, 아래, 구역, 방위) 를 공개 파일 꼴로 적는다."""
    with open(path, "w") as f:
        f.write(HEAD + "\n")
        for t, n, e, d, z, yaw in rows:
            f.write("%d,51.0,-1.0,100.0,%.6f,%.6f,%.6f,%s,1.0,0.0,0.0,"
                    "0.0,0.0,%.6f\n" % (t, n, e, d, z, yaw))


def test_a_file_with_only_a_header_reads_as_empty(tmp_path):
    p = str(tmp_path / "rtk.csv")
    with open(p, "w") as f:
        f.write(HEAD + "\n")
    t, rot, pos = load_poses_oxford(p)
    assert len(t) == 0 and len(rot) == 0 and len(pos) == 0


def test_the_longer_side_of_a_zone_change_is_the_one_read(tmp_path):
    """구역이 바뀌면 북·동의 원점이 달라서 이어 붙이면 안 된다."""
    rows = [(1000000 + 100000 * k, 10.0 * k, 0.0, 0.0, "30T", 0.0)
            for k in range(3)]
    rows += [(1000000 + 100000 * (3 + k), 5000.0 + 10.0 * k, 0.0, 0.0,
              "30U", 0.0) for k in range(6)]
    p = str(tmp_path / "rtk.csv")
    write(p, rows)
    t, rot, pos = load_poses_oxford(p)
    assert len(t) == 6
    # 긴 쪽(30U)만 남았으므로 5,000 m 짜리 뜀이 궤적에 안 들어온다.
    assert pos[0, 0] == 5000.0
    assert np.linalg.norm(np.diff(pos, axis=0), axis=1).max() < 11.0


def test_the_columns_land_where_the_header_says(tmp_path):
    rows = [(2000000, 7.0, 8.0, -9.0, "30U", 0.0),
            (2100000, 7.5, 8.0, -9.0, "30U", np.pi / 2)]
    p = str(tmp_path / "rtk.csv")
    write(p, rows)
    t, rot, pos = load_poses_oxford(p)
    assert np.allclose(t, [2.0, 2.1])
    assert np.allclose(pos[0], [7.0, 8.0, -9.0])
    # 방위 90 도면 본체의 앞축이 동쪽을 본다.
    # 파일에 소수 여섯 자리로 적히므로 그 자리만큼 느슨하게 본다.
    assert np.allclose(rot[1] @ np.array([1.0, 0.0, 0.0]),
                       [0.0, 1.0, 0.0], atol=1e-6)


def test_a_time_gap_is_counted_against_the_traversals_own_interval():
    """문턱을 밖에서 들고 오지 않는다 — 그 주행 간격의 두 배가 기준이다."""
    dt = np.r_[np.full(20, 0.1), [0.3], np.full(20, 0.1)]
    assert gap_count(dt) == 1
    # 느리게 적힌 주행에서 같은 0.3 초는 틈이 아니다.
    assert gap_count(np.r_[np.full(20, 0.5), [0.3], np.full(20, 0.5)]) == 0


def test_the_gap_mask_is_what_sets_the_no_gap_worst_case_aside():
    """끊긴 자리를 뺀 최악값이 본문에 들어가므로 마스크가 자리를 맞춰야 한다."""
    dt = np.r_[np.full(20, 0.1), [0.3], np.full(20, 0.1)]
    resid = np.zeros(41)
    resid[20] = 900.0   # 끊긴 걸음의 잔차는 빠진다.
    resid[3] = 7.0
    assert is_gap(dt).sum() == 1
    assert resid[~is_gap(dt)].max() == 7.0


def test_the_longest_run_is_the_one_returned():
    a = np.arange(10).reshape(10, 1)
    broken = np.zeros(9, dtype=bool)
    broken[1] = broken[5] = True
    got = longest_run(a, broken)
    assert got.ravel().tolist() == [2, 3, 4, 5]
