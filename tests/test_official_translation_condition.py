"""공식 병진 점수에 상한을 옮기려면 추정 회전이 직교여야 한다.

제VI-A절은 두 문장을 나눠 적는다. 추상적인 정리는 거리함수가 추정값을 담는
집합 위의 것이기만 하면 되고, 병진은 유클리드 거리라 그 조건이 따로 없다.
공식 프로그램의 병진 점수는 그와 별개다 — evaluate_odometry.cpp 가
pose_error = inv(delta_result) * delta_gt 를 만들어 그 이동 성분의 길이를
쓰므로, 점수는 R̂^-1 (t_g - t_r) 의 길이다. 그 길이가 ||t_g - t_r|| 과 같아야
D 의 증명이 병진 칸으로 옮겨가고, 같아지는 것은 R̂ 이 직교일 때뿐이다.

본문이 "가정" 이라고 적은 것이 실제로 가정인지를 두 쪽에서 묶는다. 직교인
추정에서는 길이가 보존되는지, 직교를 벗어난 추정에서는 무조건적인 상한이
실제로 깨지는지다. 둘째가 없으면 그 조건이 형식적인 단서인지 점수를 바꾸는
조건인지 글만 보고는 못 가른다.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def pose(rot, trans):
    """회전과 이동으로 4x4 자세."""
    m = np.eye(4)
    m[:3, :3] = rot
    m[:3, 3] = trans
    return m


def official_translation_error(delta_result, delta_gt):
    """공식 t_err — inv(추정 상대자세) * 기준 상대자세 의 이동 성분 길이."""
    err = np.linalg.inv(delta_result) @ delta_gt
    return float(np.linalg.norm(err[:3, 3]))


def test_an_orthogonal_estimate_leaves_the_official_score_euclidean():
    rng = np.random.default_rng(20261021)
    for _ in range(50):
        r_hat = Rotation.from_rotvec(rng.normal(size=3)).as_matrix()
        t_r = rng.normal(size=3)
        t_g = rng.normal(size=3)
        gt = pose(Rotation.from_rotvec(rng.normal(size=3)).as_matrix(), t_g)
        got = official_translation_error(pose(r_hat, t_r), gt)
        assert np.isclose(got, np.linalg.norm(t_g - t_r), atol=1e-12)


def test_a_non_orthogonal_estimate_breaks_the_unconditional_bound():
    # 기준 병진이 eps 만큼 움직이면 D 의 그 구간 몫도 eps 다. 추정 회전이
    # 0.5 I 면 역행렬이 2 I 라 점수는 2 eps 움직인다 — 상한이 깨진다.
    # 기준 변화는 두 병진의 차와 같은 방향으로 준다. 비스듬히 주면 길이가
    # 그 성분만큼만 늘어 배수가 2 보다 작게 나오고, 그건 상한이 서는지와
    # 상관없는 기하다 — 가장 나쁜 쪽에서 재야 깨지는지가 보인다.
    eps = 1e-3
    t_r = np.array([1.0, 2.0, 3.0])
    t_g = t_r + np.array([0.5, 0.0, 0.0])
    t_g_moved = t_g + np.array([eps, 0.0, 0.0])

    scaled = pose(0.5 * np.eye(3), t_r)
    before = official_translation_error(scaled, pose(np.eye(3), t_g))
    after = official_translation_error(scaled, pose(np.eye(3), t_g_moved))

    moved = abs(after - before)
    assert np.isclose(moved, 2.0 * eps, rtol=1e-9)
    assert moved > eps

    # 같은 기준 변화를 직교 추정에 주면 상한 안에 들어온다.
    proper = pose(Rotation.from_euler("z", 0.3).as_matrix(), t_r)
    kept = abs(official_translation_error(proper, pose(np.eye(3), t_g_moved))
               - official_translation_error(proper, pose(np.eye(3), t_g)))
    assert kept <= eps + 1e-12


def test_the_scaling_factor_is_the_operator_norm_of_the_inverse():
    # 본문이 "직교를 벗어나면 구간별 역행렬의 연산자 노름을 실은 상한이
    # 필요하다" 고 적은 자리. 늘어나는 배수가 실제로 그 노름인지 본다.
    rng = np.random.default_rng(7)
    eps = 1e-4
    for _ in range(20):
        a = Rotation.from_rotvec(rng.normal(size=3)).as_matrix()
        scale = np.diag(rng.uniform(0.4, 2.5, size=3))
        r_hat = a @ scale
        t_r = rng.normal(size=3)
        t_g = rng.normal(size=3)
        step = rng.normal(size=3)
        step = eps * step / np.linalg.norm(step)

        before = official_translation_error(pose(r_hat, t_r),
                                            pose(np.eye(3), t_g))
        after = official_translation_error(pose(r_hat, t_r),
                                           pose(np.eye(3), t_g + step))
        norm = np.linalg.norm(np.linalg.inv(r_hat), 2)
        assert abs(after - before) <= norm * eps + 1e-12
