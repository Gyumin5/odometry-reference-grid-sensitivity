"""쌍별 상쇄를 재는 모형이 자기가 재겠다는 양을 실제로 재는지 본다.

여기서 조용히 틀리는 자리는 둘이다. 하나는 상쇄를 못 재는 것 — 두 방법의
오차장을 똑같이 주면 차이의 변화는 정확히 0 이어야 하는데, 섭동을 두 번
따로 뽑으면 0 이 안 나오고 그 잔차가 "상쇄되지 않는 몫" 으로 보고된다.
다른 하나는 상한을 넘는 것 — 차이의 변화는 어떤 조건에서도 2D 를 못 넘는다.
넘으면 상한 쪽 증명이나 이 모형 중 하나가 틀린 것이다.
"""

import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

import kitti_segment_D as K  # noqa: E402
from pair_cancellation import _perturb, _unit, shifts, sweep  # noqa: E402


def _case(rng, n=40):
    """구간 n 개짜리 가짜 기준 하나와 그것을 흔든 기준 하나."""
    q = Rotation.from_rotvec(rng.normal(size=(n, 3)) * 0.3).as_matrix()
    qp = _perturb(q, rng.normal(size=(n, 3)) * 0.01)
    w = rng.random(n)
    return q, qp, w / w.sum()


def test_a_perturbation_keeps_the_rotations_rotations():
    rng = np.random.default_rng(1)
    q = Rotation.from_rotvec(rng.normal(size=(12, 3))).as_matrix()
    got = _perturb(q, rng.normal(size=(12, 3)) * 0.2)
    eye = np.einsum("nab,ncb->nac", got, got)
    assert np.allclose(eye, np.eye(3), atol=1e-12)
    assert np.allclose(np.linalg.det(got), 1.0, atol=1e-12)


def test_the_error_directions_are_unit_vectors():
    v = _unit(np.random.default_rng(2), 500)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-12)


def test_two_identical_estimates_cancel_exactly():
    # 닮음도 1, 오차 크기도 같으면 두 추정은 같은 추정이다. 점수가 각자
    # 아무리 밀려도 차이는 한 비트도 안 움직여야 한다.
    rng = np.random.default_rng(3)
    q, qp, w = _case(rng)
    amp = np.full(len(w), 0.004)
    s = shifts(q, qp, w, amp, amp, 1.0, rng, trials=20)
    assert np.max(np.abs(s)) < 1e-12


def test_the_shift_of_the_difference_never_exceeds_twice_D():
    # 상한의 쓰임새 자체. 넘는 조건이 하나라도 나오면 2D 판정이 무너진다.
    rng = np.random.default_rng(4)
    for _ in range(30):
        q, qp, w = _case(rng)
        d = float(np.dot(w, K.geodesic(q, qp)))
        amp_a = rng.random(len(w)) * 0.05
        amp_b = rng.random(len(w)) * 0.05
        rho = float(rng.random())
        s = shifts(q, qp, w, amp_a, amp_b, rho, rng, trials=8)
        assert np.max(np.abs(s)) <= 2.0 * d + 1e-12


def test_the_realized_shift_stays_well_under_the_worst_case():
    # 이 실험이 말하려는 것. 최악의 경우를 가정한 2D 와 실제로 움직인 양은
    # 자릿수가 다르다. 여기서 1 에 가까워지면 상쇄 주장을 접어야 한다.
    rng = np.random.default_rng(5)
    q, qp, w = _case(rng, n=200)
    d = float(np.dot(w, K.geodesic(q, qp)))
    amp = np.full(len(w), 0.01)
    s = np.abs(shifts(q, qp, w, amp, amp * 1.1, 0.0, rng, trials=40))
    assert np.median(s) < 0.5 * (2.0 * d)


def test_independent_errors_shift_more_than_similar_ones():
    # 닮은 오차일수록 공통으로 밀리는 몫이 커져 차이에서 더 많이 지워진다.
    # 방향이 반대로 나오면 닮음도를 거꾸로 쓴 것이다.
    rng = np.random.default_rng(6)
    q, qp, w = _case(rng, n=200)
    amp = np.full(len(w), 0.02)
    lo = np.abs(shifts(q, qp, w, amp, amp, 0.0, rng, trials=40))
    hi = np.abs(shifts(q, qp, w, amp, amp, 0.99, rng, trials=40))
    assert np.median(hi) < np.median(lo)


def test_the_sweep_reports_a_flip_rate_for_every_condition():
    rng = np.random.default_rng(7)
    n = 400
    t = np.arange(n) * 0.1
    rot = Rotation.from_rotvec(
        np.cumsum(rng.normal(size=(n, 3)) * 0.002, axis=0)).as_matrix()
    pos = np.cumsum(np.tile([2.0, 0.0, 0.0], (n, 1)), axis=0)
    got = sweep(t, rot, pos, rng, trials=4, rhos=(0.0, 0.9),
                scores=(0.1, 0.3), gaps=(0.01, 0.05))
    assert got is not None
    recs, two_d, nseg = got
    assert nseg > 0 and two_d > 0.0
    assert len(recs) == 8
    for r in recs:
        assert 0.0 <= r["flip_rate"] <= 1.0
        assert r["shift_median"] <= r["shift_p95"] <= r["shift_max"]
        assert r["frac_of_twoD_median"] >= 0.0


def test_a_flip_needs_the_signed_shift_not_its_size():
    # 차이가 더 벌어진 쪽(부호가 반대인 s)을 뒤집힘으로 세면 확률이 두 배로
    # 부풀고, 그 수가 "2D 는 몇 배 보수적인가" 의 근거가 된다.
    rng = np.random.default_rng(11)
    n = 400
    t = np.arange(n) * 0.1
    rot = Rotation.from_rotvec(
        np.cumsum(rng.normal(size=(n, 3)) * 0.002, axis=0)).as_matrix()
    pos = np.cumsum(np.tile([2.0, 0.0, 0.0], (n, 1)), axis=0)
    recs, _, _ = sweep(t, rot, pos, rng, trials=200, rhos=(0.0,),
                       scores=(0.15,), gaps=(0.01,))
    r = recs[0]
    # 부호 있는 판정은 대칭 분포에서 |s| 판정의 절반쯤이 된다. 같은 수가
    # 나오면 절댓값을 그대로 쓰고 있다는 뜻이다.
    assert r["flip_rate"] < 0.5


def test_a_trajectory_too_short_for_a_segment_is_skipped():
    # 공식 구간은 100 m 부터다. 그보다 짧은 기록에서 0 으로 나누지 않는다.
    rng = np.random.default_rng(8)
    t = np.arange(20) * 0.1
    rot = np.tile(np.eye(3), (20, 1, 1))
    pos = np.cumsum(np.tile([0.5, 0.0, 0.0], (20, 1)), axis=0)
    assert sweep(t, rot, pos, rng, trials=2) is None
