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


def test_a_position_score_breaks_a_bound_read_off_another_distance():
    """지렛대 위치는 형태를 깨는 게 아니라 거리함수가 다른 사례다.

    심사 25 회차가 갈랐다. 점수는 미터로 재는데 D 는 라디안으로 재서 둘을
    맞대니 넘는 것이다. 같은 거리로 양쪽을 재면 삼각부등식이라 안 깨진다 —
    아래 test_the_position_score_obeys_its_own_distance 가 그걸 고정한다.
    """
    d, diff, slack = falsify("position")
    assert diff > d
    assert slack < 0


def test_the_position_score_obeys_its_own_distance():
    # 고정점 l 에 대한 d_l(A,Q) = ||Al - Ql|| 로 양쪽을 통일하면 선다.
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(7)
    worst = -9.0
    for _ in range(200):
        m = int(rng.integers(1, 6))
        q = Rotation.random(m, random_state=int(rng.integers(1000000))).as_matrix()
        qp = np.einsum("nab,nbc->nac", Rotation.from_rotvec(
            rng.normal(scale=0.3, size=(m, 3))).as_matrix(), q)
        a = Rotation.random(m, random_state=int(rng.integers(1000000))).as_matrix()
        arm = rng.normal(scale=5.0, size=3)
        w = rng.random(m)
        w /= w.sum()

        def dl(x, y):
            return np.linalg.norm(np.einsum("nab,b->na", x, arm)
                                  - np.einsum("nab,b->na", y, arm), axis=1)

        d = float(np.dot(w, dl(q, qp)))
        e, ep = float(np.dot(w, dl(a, q))), float(np.dot(w, dl(a, qp)))
        worst = max(worst, abs(ep - e) - d)
    assert worst <= 1e-9, worst


def test_a_whole_trajectory_rigid_alignment_leaves_the_score_alone():
    """궤적 전체의 강체 정렬은 상대자세에서 통째로 소거된다 — 원고의 주장.

    세계좌표계 자세를 H 로 돌리면 상대자세는 (H P_i)^T (H P_j) = P_i^T P_j 라
    한 톨도 안 움직인다. 그래서 기준마다 정렬을 다시 맞춰도 점수가 아예 안
    바뀐다 — "최소값이라 D 를 못 넘는다" 보다 강한 말이고, 심사 25 회차가
    그 최소값 논증을 쓰지 말라고 한 자리다. 이 원고가 최소화하는 목적함수가
    아니기 때문이다.
    """
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(5)
    for _ in range(50):
        n = int(rng.integers(3, 12))
        poses = Rotation.random(n, random_state=int(rng.integers(1000000))).as_matrix()
        pairs = np.stack([np.arange(n - 1), np.arange(1, n)], axis=1)
        h = Rotation.random(1, random_state=int(rng.integers(1000000))).as_matrix()[0]
        turned = np.einsum("ab,nbc->nac", h, poses)
        assert np.max(np.abs(relative(poses, pairs) - relative(turned, pairs))) < 1e-12


def test_minimising_over_a_common_rotation_is_stable():
    """상대자세에 공통 회전을 걸고 최소를 취하는 점수도 D 를 못 넘는다.

    위 검사와 다른 명제다 — 이건 궤적 정렬이 아니라 상대회전을 직접 최소화하는
    점수이고, 1 차 립시츠라 두 최소의 차가 피최소량의 최대 변화를 못 넘는다.
    원고는 이 논증을 정렬에 갖다 쓰지 않는다(심사 25 회차). 옛 "realigned"
    가지가 이 자리에서 변경 전후를 다른 잣대로 재고 있었다(24 회차).
    """
    from scipy.spatial.transform import Rotation

    H = Rotation.random(4000, random_state=11).as_matrix()

    def refit(a, q, w):
        v = np.zeros(len(H))
        for k in range(len(q)):
            tgt = np.repeat(q[k][None], len(H), axis=0)
            v += w[k] * geodesic(np.einsum("nab,bc->nac", H, a[k]), tgt)
        return float(v.min())

    rng = np.random.default_rng(3)
    for _ in range(8):
        m = int(rng.integers(2, 5))
        q = Rotation.random(m, random_state=int(rng.integers(1000000))).as_matrix()
        pert = Rotation.from_rotvec(
            rng.normal(scale=0.15, size=(m, 3))).as_matrix()
        qp = np.einsum("nab,nbc->nac", pert, q)
        a = Rotation.random(m, random_state=int(rng.integers(1000000))).as_matrix()
        w = rng.random(m)
        w /= w.sum()
        d = float(np.dot(w, geodesic(q, qp)))
        # 훑기로 잡은 최소라 참값보다 조금 크다. 여유를 둔다.
        assert abs(refit(a, qp, w) - refit(a, q, w)) <= d + 1e-9


def test_exactly_two_falsify_branches_exist():
    """falsify 의 갈래는 둘이다 — 제곱과 지렛대 위치.

    형태를 깨는 반례는 제곱 하나뿐이다. 지렛대는 거리함수를 다르게 쓴
    사례라 본문이 반례로 부르지 않는다(심사 25 회차).

    본문이 이 둘을 이름으로 부르는지는 tests/test_paper_matches_code.py 가
    따로 본다. 원고는 공개 나무에 안 담기므로 그 대조를 여기 두면 이 파일이
    통째로 공개 나무에서 빠진다 — 상한의 적대 검사까지 같이 빠진다.
    """
    import inspect
    import re

    src = inspect.getsource(falsify)
    kinds = set(re.findall(r'score_kind == "(\w+)"', src))
    assert kinds == {"squared", "position"}, kinds


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
