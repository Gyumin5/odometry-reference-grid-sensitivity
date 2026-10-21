"""창이 구간 집합을 바꿀 때 점수가 어디로 가는지 재는 쪽을 묶는다.

이 측정의 값은 하나다 — 공식 프로그램이 있는 릴리스는 한 줄뿐인데, 나머지
여섯에서도 같은 양을 같은 꼴로 내야 "제외율만 적고 영향은 안 쟀다" 를 닫을 수
있다. 그러려면 두 가지가 지켜져야 한다. 공식 집계 규칙대로 구간을 한 자루에
담아 평균할 것, 그리고 창 집합이 전체 집합의 부분집합일 것.
"""

import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

from kitti_segment_D import (LENGTHS, STEP, geodesic,  # noqa: E402
                             relative, segments)
from window_exclusion import (per_segment, pooled_from,  # noqa: E402
                              sets_for)

LEN1 = ((1.0,), 1)


def wiggly(n=400, dt=0.1, v=10.0):
    """각속도와 속도가 함께 변하는 궤적. 어떤 복원도 정확하지 않다."""
    t = np.arange(n) * dt
    yaw = 0.6 * np.sin(0.8 * t) + 0.2 * t
    speed = v * (1.0 + 0.4 * np.sin(1.3 * t))
    rot = Rotation.from_euler("z", yaw).as_matrix()
    dpos = np.stack([speed * np.cos(yaw), speed * np.sin(yaw),
                     np.zeros(n)], axis=1) * dt
    pos = np.cumsum(dpos, axis=0) - dpos[0]
    return t, rot, pos


def test_an_exact_estimate_scores_at_the_arccos_floor_not_above_it():
    # 답을 아는 입력. 병진은 정확히 0 이어야 하고, 회전은 0 이 아니라 바닥에
    # 앉는다 — 측지각이 항등 근처에서 제곱근으로 키우기 때문이다. 그 바닥은
    # 본문이 따로 재서 적은 값(5.47e-05 도/100 m = 5.47e-07 도/m)이고, 여기서
    # 보는 것은 잔차가 그 아래에 있다는 것뿐이다.
    _, rot, pos = wiggly()
    idx = segments(pos, lengths=LENGTHS, step=STEP)
    r, p = per_segment(rot, pos, rot, pos, idx)
    assert len(r) == len(idx) and len(p) == len(idx)
    assert float(np.max(np.abs(r))) < 5.47e-07
    assert float(np.max(np.abs(p))) < 1e-12


def test_each_segment_error_is_already_divided_by_its_length():
    # 공식 지표는 구간 길이로 나눈 값을 한 자루에 담는다. 길이로 안 나누면
    # 긴 구간이 자루를 끌고 가고, 묶음 값이 집합 크기에 따라 휜다.
    _, rot, pos = wiggly()
    idx = segments(pos, lengths=LENGTHS, step=STEP)
    extra = Rotation.from_rotvec(
        np.outer(np.arange(len(rot)) * 2e-4, [0.0, 1.0, 0.0])).as_matrix()
    est_rot = np.einsum("nab,nbc->nac", rot, extra)
    r, _ = per_segment(rot, pos, est_rot, pos, idx)
    L = np.asarray([q[2] for q in idx], dtype=float)
    raw = np.degrees(geodesic(relative(rot, idx), relative(est_rot, idx)))
    assert np.allclose(r * L, raw, rtol=0, atol=1e-15)
    assert float(np.min(L)) < float(np.max(L))   # 길이가 실제로 여럿이다


def test_the_window_sets_are_subsets_of_the_full_set():
    t, _, pos = wiggly(n=600)
    got = sets_for(t, pos, LENGTHS, STEP)
    full = set(got["full"])
    assert full
    for key in ("main", "sweep"):
        assert got[key], key
        assert set(got[key]) <= full, key
    # 훑기 창은 주표 창보다 더 덜어낸다 — 앞뒤를 다 자르기 때문이다.
    assert len(got["sweep"]) <= len(got["main"]) <= len(got["full"])


def test_sequences_go_into_one_bag_not_a_mean_of_means():
    # 길이가 다른 두 시퀀스를 넣는다. 한 자루 평균과 평균의 평균이 다른
    # 상황을 일부러 만들고, 코드가 한 자루 쪽을 내는지 본다.
    short = wiggly(n=200)
    long = wiggly(n=600)
    eps = 2e-4
    got = pooled_from([short, long], LENGTHS, STEP, eps)
    each = [pooled_from([s], LENGTHS, STEP, eps) for s in (short, long)]

    n = [e["full"][2] for e in each]
    assert n[0] != n[1]
    assert got["full"][2] == n[0] + n[1]

    bag = got["full"][0]
    mean_of_means = float(np.mean([e["full"][0] for e in each]))
    weighted = float(np.dot([e["full"][0] for e in each], n) / sum(n))
    assert abs(bag - weighted) < 1e-12
    assert abs(bag - mean_of_means) > 1e-12


def test_a_release_without_segments_under_a_rule_reports_nothing():
    # 100~800 m 규칙이 안 서는 짧은 궤적. 빈 칸을 0 점수로 적으면 그 릴리스가
    # 창의 영향을 안 받는 것처럼 보인다.
    t, rot, pos = wiggly(n=40, v=0.2)
    got = pooled_from([(t, rot, pos)], LENGTHS, STEP, 2e-4)
    assert got["full"] is None
    assert got["main"] is None and got["sweep"] is None


def test_the_window_moves_the_score_by_a_finite_measurable_amount():
    # 집합을 바꾸면 점수가 움직인다. 0 이 아니고 발산하지도 않아야 한다 —
    # 0 이면 측정이 집합을 안 바꾼 것이고, 발산하면 자루가 섞인 것이다.
    seqs = [wiggly(n=600), wiggly(n=500, v=8.0)]
    got = pooled_from(seqs, LENGTHS, STEP, 2e-4)
    full_r, full_t, _ = got["full"]
    for key in ("main", "sweep"):
        r, p, _ = got[key]
        assert np.isfinite(r) and np.isfinite(p)
        assert abs(r - full_r) < abs(full_r)
        assert abs(p - full_t) < abs(full_t)
    assert got["sweep"][2] < got["full"][2]
