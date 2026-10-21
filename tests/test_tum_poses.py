"""튬 꼴 기준 궤적 읽기를 묶는다.

조용히 틀리는 자리는 둘이다. 동작포착 구멍을 안 끊으면 없는 자세를 지어내
상한이 작게 나오고, 쿼터니언 칸 차례를 잘못 잡으면 궤적이 통째로 돌아간 채
그럴듯한 수가 나온다.
"""

import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

from tum_poses import pieces, read_tum  # noqa: E402

HEAD = "# ground truth trajectory\n# timestamp tx ty tz qx qy qz qw\n"


def line(t, p, q):
    return "%.4f %.4f %.4f %.4f %.6f %.6f %.6f %.6f\n" % (
        (t,) + tuple(p) + tuple(q))


def write(path, t, p, q):
    with open(path, "w") as f:
        f.write(HEAD)
        for i in range(len(t)):
            f.write(line(t[i], p[i], q[i]))


def sample(n=200, dt=0.01):
    t = np.arange(n) * dt
    yaw = 0.5 * np.sin(2.0 * t)
    q = Rotation.from_euler("z", yaw).as_quat()
    p = np.stack([t, np.sin(t), np.zeros(n)], axis=1)
    return t, p, q


def test_the_last_four_columns_are_read_as_x_y_z_w(tmp_path):
    # 튬 파일은 qx qy qz qw 차례다. w 를 앞으로 읽으면 자세가 통째로 돌아간다.
    f = tmp_path / "a-groundtruth.txt"
    t, p, q = sample()
    write(str(f), t, p, q)
    _, qq, _ = read_tum(str(f))
    assert np.allclose(qq, q, atol=1e-6)
    (_, rot, _), = list(pieces(t, qq, p))
    assert np.allclose(rot[0], np.eye(3), atol=1e-5)


def test_a_motion_capture_dropout_splits_the_track(tmp_path):
    f = tmp_path / "b-groundtruth.txt"
    t, p, q = sample()
    t = t.copy()
    t[100:] += 2.0           # 2 초짜리 구멍
    write(str(f), t, p, q)
    out = list(pieces(*read_tum(str(f))))
    assert len(out) == 2
    assert len(out[0][0]) == 100 and len(out[1][0]) == 100
    for tt, _, _ in out:
        assert np.max(np.diff(tt)) < 0.05


def test_a_scrap_too_short_to_hold_a_segment_is_dropped(tmp_path):
    f = tmp_path / "c-groundtruth.txt"
    t, p, q = sample()
    t = t.copy()
    t[190:] += 2.0           # 열 프레임짜리 토막이 남는다
    write(str(f), t, p, q)
    out = list(pieces(*read_tum(str(f))))
    assert len(out) == 1
    assert len(out[0][0]) == 190


def test_a_file_with_only_a_header_yields_nothing(tmp_path):
    f = tmp_path / "d-groundtruth.txt"
    with open(f, "w") as fh:
        fh.write(HEAD)
    assert list(pieces(*read_tum(str(f)))) == []


def test_a_repeated_timestamp_is_dropped_so_resampling_accepts_the_track(
        tmp_path):
    # 같은 시각이 두 번 적힌 줄이 실제 파일에 있다. 남겨 두면 재표집이
    # "시각이 엄격히 늘어야 한다" 로 거절해서 그 시퀀스가 통째로 빠진다.
    f = tmp_path / "f-groundtruth.txt"
    t, p, q = sample()
    t = t.copy()
    t[50] = t[49]
    write(str(f), t, p, q)
    (tt, _, _), = list(pieces(*read_tum(str(f))))
    assert np.all(np.diff(tt) > 0)
    assert len(tt) == len(t) - 1


def test_rows_out_of_order_are_sorted_before_the_gaps_are_judged(tmp_path):
    f = tmp_path / "e-groundtruth.txt"
    t, p, q = sample()
    idx = np.r_[np.arange(100, 200), np.arange(100)]
    write(str(f), t[idx], p[idx], q[idx])
    out = list(pieces(*read_tum(str(f))))
    assert len(out) == 1
    assert np.all(np.diff(out[0][0]) > 0)
