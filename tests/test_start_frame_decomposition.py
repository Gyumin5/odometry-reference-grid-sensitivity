"""솎기가 구간 시작 자세를 건드리는 몫과 끝점만 건드리는 몫을 가르는 쪽.

이 분해는 두 위상이 같은 구간 집합을 쓸 때만 뜻이 있고, 그 집합이 공식
집합의 부분집합일 때만 공식 평가에 대해 말하는 것이 된다. 창 안의 자세를
다시 훑어 구간을 고르면 시작점 격자가 창의 첫 프레임에서 다시 세어져서,
크기만 비슷하고 원소는 하나도 안 겹치는 집합이 나온다. 그렇게 되면 어느
위상이 시작점을 보존하는지가 뒤집히는데, 표는 멀쩡해 보인다.
"""

import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

from kitti_segment_D import LENGTHS, STEP, segments  # noqa: E402
from protocol_sweep import window, window_segments  # noqa: E402
from start_frame_decomposition import decompose  # noqa: E402


def wiggly(n=1400, dt=0.1, v=10.0):
    """각속도와 속도가 함께 변하는 궤적. 어떤 복원도 정확하지 않다."""
    t = np.arange(n) * dt
    yaw = 0.6 * np.sin(0.8 * t) + 0.2 * t
    speed = v * (1.0 + 0.4 * np.sin(1.3 * t))
    rot = Rotation.from_euler("z", yaw).as_matrix()
    dpos = np.stack([speed * np.cos(yaw), speed * np.sin(yaw),
                     np.zeros(n)], axis=1) * dt
    pos = np.cumsum(dpos, axis=0) - dpos[0]
    return t, rot, pos


def test_the_two_phases_share_one_segment_set():
    # 집합이 다르면 두 줄의 차이가 위상 차이가 아니라 집합 차이가 된다.
    rows = decompose(*wiggly())
    assert len(rows) == 2
    assert rows[0]["n_segments"] == rows[1]["n_segments"]
    assert rows[0]["n_segments"] > 0


def test_the_segments_are_the_official_ones_that_fall_inside_the_window():
    t, _, pos = wiggly()
    dt = float(np.median(np.diff(t)))
    lo, hi = window(t, dt)
    m = (t >= lo) & (t <= hi)
    off = int(np.argmax(m))
    got = window_segments(pos, m, LENGTHS, STEP)
    assert got
    # 창 기준 색인을 원본으로 되돌리면 공식 집합의 원소여야 한다.
    official = set(segments(pos, lengths=LENGTHS, step=STEP))
    assert {(i + off, j + off, L) for (i, j, L) in got} <= official


def test_exactly_one_phase_keeps_the_official_start_frames():
    # 공식 시작점은 보폭의 배수라 전부 짝수 색인이다. 짝수 솎기는 그 표본을
    # 그대로 들고 있고 홀수 솎기는 하나도 안 들고 있다 — 둘 다 보존하거나
    # 둘 다 잃는 결과가 나오면 색인 공간이 어긋난 것이다.
    rows = {r["phase"]: r for r in decompose(*wiggly())}
    assert rows["even"]["start_parity_retained"] == 1
    assert rows["odd"]["start_parity_retained"] == 0


def test_the_phase_that_keeps_the_start_frames_moves_them_not_at_all():
    rows = {r["phase"]: r for r in decompose(*wiggly())}
    kept, lost = rows["even"], rows["odd"]
    assert kept["start_pos_err_m"] == 0.0
    # 회전은 0 이 아니라 역코사인 바닥에 앉는다(5.47e-05 도/100 m).
    assert kept["start_rot_err_deg"] < 5.47e-05
    assert lost["start_pos_err_m"] > 0.0
    assert lost["start_rot_err_deg"] > kept["start_rot_err_deg"]


def test_a_segment_whose_two_ends_are_both_retained_moves_by_nothing():
    """시작도 끝도 보존하는 구간에서는 남는 몫이 없어야 한다.

    이 칸이 0 이 아니면 그 위상이 실제로는 그 표본을 안 들고 있는 것이고,
    0 이 엉뚱한 칸(끝 홀수)에 앉아 있으면 끝점 홀짝을 원본이 아니라 창
    기준으로 세고 있는 것이다.
    """
    rows = {r["phase"]: r for r in decompose(*wiggly())}
    kept = rows["even"]
    assert kept["D_trans_pct_end_even"] < 1e-10
    assert kept["D_trans_pct_end_odd"] > 1e-6
    # 끝점만 잃는 몫은 시작 자세까지 잃는 몫보다 작다. 그 비가 분해의 값이다.
    assert kept["D_trans_pct"] < rows["odd"]["D_trans_pct"]
