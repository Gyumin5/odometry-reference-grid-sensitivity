"""재표집 이동량 D 가 평균 각거리 점수 변화의 상한인지 본다.

주장은 |E' - E| <= D 다. 이게 깨지면 제4갈래는 그 자리에서 죽는다 — 그래서
맞는 조건을 확인하는 것보다 깨질 만한 조건을 모아 때리는 쪽에 무게를 둔다.
같이 확인하는 것 둘: 상한이 실제로 닿는가(닿지 않으면 느슨해서 쓸모가 없다),
적용 범위 밖이라고 적어 둔 점수들에서는 정말로 안 서는가.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

from resample_sensitivity import (  # noqa: E402
    check_once,
    displacement,
    falsify,
    geodesic,
    relative,
    resample,
    score,
    trajectory,
    weights,
)

N = 200
FS = 50.0


def test_the_distance_is_a_metric_on_rotations():
    rng = np.random.default_rng(1)
    _, a = trajectory(60, FS, "broadband", rng)
    _, b = trajectory(60, FS, "fast", rng)
    _, c = trajectory(60, FS, "smooth", rng)
    ab, ba = geodesic(a, b), geodesic(b, a)
    assert np.allclose(ab, ba)
    assert np.allclose(geodesic(a, a), 0.0, atol=1e-12)
    assert np.all(ab >= -1e-12)
    # 삼각부등식이 상한의 유일한 근거다.
    assert np.all(ab <= geodesic(a, c) + geodesic(c, b) + 1e-9)


def test_resampling_onto_the_original_times_changes_nothing():
    rng = np.random.default_rng(2)
    t, rot = trajectory(N, FS, "broadband", rng)
    again = resample(t, rot, t, "slerp")
    assert np.max(geodesic(rot, again)) < 1e-9


def test_the_bound_holds_on_every_adversarial_condition():
    bad = []
    seed = 0
    for traj in ("broadband", "fast", "smooth", "still"):
        for est in ("exact", "bias", "noisy", "random", "antipodal"):
            for grid in ("phase", "decimate", "irregular", "gap"):
                for method in ("slerp", "nearest", "zoh"):
                    seed += 1
                    d, diff, slack = check_once(
                        traj, est, grid, method, N, FS, seed, seed % 2 == 0)
                    if slack < -1e-9:
                        bad.append((traj, est, grid, method, d, diff))
    assert not bad, "상한 위반 %d 건: %s" % (len(bad), bad[:3])


def test_the_bound_is_attained_so_it_is_not_merely_loose():
    # 추정값이 원래 기준과 같으면 E = 0 이고 E' 는 정의상 D 다. 상한이 닿는
    # 조건이 실재한다는 뜻이라 "맞지만 쓸모없는 상한" 이 아니다.
    rng = np.random.default_rng(3)
    t, rot = trajectory(N, FS, "broadband", rng)
    rot_q = resample(t, rot, t + 0.5 / FS, "slerp")
    pairs = np.stack([np.arange(N - 1), np.arange(1, N)], axis=1)
    q, qp = relative(rot, pairs), relative(rot_q, pairs)
    w = weights(N - 1)
    d = displacement(q, qp, w)
    e, ep = score(q, q, w), score(q, qp, w)
    assert d > 1e-6
    assert abs(abs(ep - e) - d) < 1e-9


def test_a_squared_error_score_breaks_the_bound():
    d, diff, slack = falsify("squared")
    assert diff > d
    assert slack < 0


def test_a_position_score_breaks_the_bound():
    d, diff, slack = falsify("position")
    assert diff > d
    assert slack < 0


def test_realigning_each_pair_breaks_the_bound():
    d, diff, slack = falsify("realigned")
    assert diff > d
    assert slack < 0


def test_a_longer_lever_arm_breaks_the_position_bound_further():
    # 위치오차가 왜 범위 밖인지의 이유를 수치로 묶어 둔다 — 같은 회전 이동이
    # 지렛대 길이에 비례해 커지므로 회전 거리 하나로는 덮을 수 없다.
    near = falsify("position", arm_m=1.0)[1]
    far = falsify("position", arm_m=20.0)[1]
    assert far > 10 * near


def test_nearest_does_not_build_a_distance_matrix():
    # 긴 파일에서 터지던 자리. 6만 표본이면 거리 행렬이 28 기가라 못 만든다 —
    # 이 검사가 통과한다는 것 자체가 이분 탐색으로 돌고 있다는 뜻이다.
    n = 60000
    rng = np.random.default_rng(9)
    t, rot = trajectory(n, 200.0, "smooth", rng)
    got = resample(t, rot, t + 0.3 / 200.0, "nearest")
    assert got.shape == (n, 3, 3)
    # 0.3 표본만 밀면 가장 가까운 것은 언제나 원래 표본이다.
    assert np.max(geodesic(rot, got)) < 1e-12


def test_weights_are_non_negative_and_sum_to_one():
    rng = np.random.default_rng(4)
    for w in (weights(37), weights(37, rng)):
        assert len(w) == 37
        assert np.all(w >= 0)
        assert abs(w.sum() - 1.0) < 1e-12


def test_a_denser_grid_moves_the_reference_less():
    # 같은 궤적을 반 표본만큼 밀면, 표집이 촘촘할수록 이동량이 줄어야 한다.
    # 줄지 않으면 D 는 재표집이 아니라 다른 것을 재고 있는 것이다.
    coarse = check_once("broadband", "exact", "phase", "slerp", N, 10.0, 5, False)[0]
    fine = check_once("broadband", "exact", "phase", "slerp", N, 200.0, 5, False)[0]
    assert fine < coarse
