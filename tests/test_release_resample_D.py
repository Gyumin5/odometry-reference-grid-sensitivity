"""릴리스별 D 측정기가 무엇을 재고 무엇을 안 재는지 고정한다.

상한 자체는 test_resample_sensitivity.py 가 본다. 여기서 보는 것은 측정 규약이다 —
움직이지 않는 궤적에서 0 이 나오는가, 표집이 촘촘해지면 줄어드는가, 표본을 그대로
집어오는 규약이 왜 뺄 수밖에 없었는가.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

from release_resample_D import measure, pairs_of, protocol_pair  # noqa: E402
from resample_sensitivity import (  # noqa: E402
    displacement, geodesic, relative, resample, trajectory, weights,
)

FS = 50.0
N = 600


def traj(kind, fs=FS, n=N, seed=11):
    return trajectory(n, fs, kind, np.random.default_rng(seed))


def test_a_motionless_reference_cannot_move_under_resampling():
    t, rot = traj("still")
    d, npairs, fs = measure(t, rot)
    assert npairs > 0
    assert abs(fs - FS) < 1e-6
    for k, v in d.items():
        assert v < 1e-9, "%s 가 0 이 아니다: %g" % (k, v)


def test_every_protocol_compares_the_same_number_of_pairs():
    t, rot = traj("broadband")
    dt = float(np.median(np.diff(t)))
    for name in ("phase_half", "decim2", "interp"):
        q, qp = protocol_pair(t, rot, dt, name)
        assert len(q) == len(qp) > 0
        assert q.shape[1:] == (3, 3)


def test_picking_whole_samples_does_not_move_the_increments():
    # 뺀 규약의 근거. 민 격자에서 '가장 가까운 표본' 을 집으면 집어오는 값이
    # 공개 표본 그 자체라 상대회전이 글자 그대로 같다. 이게 성립하지 않으면
    # 그 규약을 뺀 이유가 사라진다.
    t, rot = traj("broadband")
    dt = float(np.median(np.diff(t)))
    pr = pairs_of(len(t) - 1)
    q = relative(rot, pr)
    qp = relative(resample(t, rot, t + dt / 3.0, "nearest"), pr)
    assert displacement(q, qp, weights(len(q))) < 1e-9


def test_a_denser_publication_moves_less():
    slow = measure(*traj("broadband", fs=10.0))[0]
    fast = measure(*traj("broadband", fs=200.0))[0]
    for k in ("phase_half", "decim2", "interp"):
        assert fast[k] < slow[k], k


def test_faster_motion_moves_more_at_the_same_rate():
    calm = measure(*traj("smooth"))[0]
    quick = measure(*traj("fast"))[0]
    for k in ("phase_half", "decim2", "interp"):
        assert quick[k] > calm[k], k


def test_the_increment_column_is_the_real_rotation_between_neighbours():
    # D 를 읽으려면 몸체가 이웃 표본 사이에 실제로 얼마나 도는지를 알아야 한다.
    # 일정한 각속도를 넣으면 그 값은 omega/fs 여야 한다.
    from scipy.spatial.transform import Rotation

    fs, n, omega = 50.0, 500, 0.4
    t = np.arange(n) / fs
    rot = Rotation.from_rotvec(np.outer(t, [0, 0, omega])).as_matrix()
    d = measure(t, rot)[0]
    assert abs(d["increment"] - omega / fs) < 1e-9
    # 고르게 도는 궤적은 보간이 정확하므로 규약이 가르는 몫이 거의 없다.
    assert d["phase_half"] < 1e-6 * d["increment"]


def test_too_short_a_file_is_refused_rather_than_guessed():
    t, rot = traj("broadband", n=32)
    d, npairs, fs = measure(t, rot)
    assert d is None and npairs == 0 and np.isnan(fs)


def test_the_half_sample_shift_is_the_worst_phase_within_one_interval():
    # 반 표본이 최악이라는 전제를 고정한다. 같은 보간으로 1/8, 1/4, 1/2 를
    # 재서 반 표본이 가장 큰 이동을 주지 않으면 규약 선택의 근거가 없다.
    t, rot = traj("broadband")
    dt = float(np.median(np.diff(t)))
    pr = pairs_of(len(t) - 1)
    q = relative(rot, pr)
    w = weights(len(q))
    ds = [displacement(q, relative(resample(t, rot, t + f * dt, "slerp"), pr), w)
          for f in (0.125, 0.25, 0.5)]
    assert ds[0] < ds[1] < ds[2]


def test_the_measured_displacement_is_a_real_rotation_angle():
    # 보고하는 값이 각도인지. 알려진 각만큼 통째로 돌려 놓으면 D 가 그 각이어야
    # 한다. 아니면 밀리도 환산이 다른 양을 각도처럼 적고 있는 것이다.
    from scipy.spatial.transform import Rotation

    t, rot = traj("smooth")
    pr = pairs_of(len(t) - 1)
    q = relative(rot, pr)
    ang = 0.017
    turned = np.einsum("nab,bc->nac", q,
                       Rotation.from_rotvec([0, 0, ang]).as_matrix())
    assert abs(displacement(q, turned, weights(len(q))) - ang) < 1e-9
    assert np.max(geodesic(q, q)) < 1e-12
