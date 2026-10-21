#!/usr/bin/env python3
"""재표집이 상대회전을 얼마나 움직이는지 재고, 그 값이 평가점수 변화의 상한인지 본다.

재는 것은 하나다. 공개된 기준 궤적을 미리 정해 둔 규약으로 다시 표집하면 평가에
쓰이는 상대회전 Q_ij 가 Q'_ij 로 바뀐다. 그 이동량의 가중합

    D = sum_ij w_ij * d(Q_ij, Q'_ij),   d 는 SO(3) 측지거리, w >= 0, sum w = 1

은 그 릴리스 파일 하나만 읽어서 나온다. 추정기도 평가 대상도 필요 없다.

쓰임새는 상한이다. 고정된 추정값 A_ij 에 대해 평균 각거리 점수

    E  = sum_ij w_ij * d(A_ij, Q_ij)
    E' = sum_ij w_ij * d(A_ij, Q'_ij)

를 두면 |E' - E| <= D 다. 측지거리는 양불변 거리라 삼각부등식이 성립하고,
각 항에서 |d(A,Q') - d(A,Q)| <= d(Q,Q') 이기 때문이다. 그러므로 두 방법의
점수 차가 2D 보다 크면 어느 규약으로 재표집하든 순위가 안 뒤집힌다.

적용 범위는 좁다. 여기서 상한이 서는 것은 평균 각거리 점수뿐이다.
  - 제곱오차 점수에는 안 선다. 제곱은 거리의 함수라 삼각부등식이 그대로 안 넘어간다.
  - 위치오차에는 안 선다. 회전 이동량이 위치를 얼마나 미는지는 지렛대 길이에 달렸다.
아래 falsify() 가 이 둘을 반례로 직접 확인한다 — 안 서는 것을 안 선다고 보이기
위해서다.

정렬을 다시 맞추는 점수는 이 목록에 없다. 기준마다 다시 맞춰도 궤적 전체에 거는
강체 정렬은 상대자세에서 상쇄되고, 정렬을 훑어 최소를 취하는 점수도 최소 연산이
1 차 립시츠라 D 를 못 넘는다. 가르는 기준은 "재적합이라 부르는가" 가 아니라
"추정의 상대자세가 실제로 바뀌는가" 다. 척도를 품은 정렬과 구간 재선택이
그쪽이다.

python3 scripts/resample_sensitivity.py 로 돌리면 적대적 조건을 훑어
results/next/resample_bound_check.csv 에 위반 여부를 적는다.
"""

import argparse
import csv
import os

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


# ------------------------------------------------------------------ 거리와 점수

def geodesic(a, b):
    """회전 묶음 두 개 사이의 측지거리(라디안). (N,3,3) 둘 -> (N,)."""
    ra = Rotation.from_matrix(np.asarray(a, dtype=float))
    rb = Rotation.from_matrix(np.asarray(b, dtype=float))
    return (ra.inv() * rb).magnitude()


def relative(rot, pairs):
    """Q_ij = R_i^T R_j. pairs 는 (M,2) 정수 배열."""
    rot = np.asarray(rot, dtype=float)
    i, j = np.asarray(pairs)[:, 0], np.asarray(pairs)[:, 1]
    return np.einsum("nba,nbc->nac", rot[i], rot[j])


def weights(m, rng=None):
    """균등 가중. rng 를 주면 음이 아닌 난수 가중을 합 1 로 정규화한다."""
    if rng is None:
        return np.full(m, 1.0 / m)
    w = rng.random(m)
    return w / w.sum()


def displacement(q, qp, w):
    """D — 재표집 전후 상대회전의 가중 평균 이동량."""
    return float(np.dot(w, geodesic(q, qp)))


def score(a, q, w):
    """E — 고정 추정값에 대한 가중 평균 각거리."""
    return float(np.dot(w, geodesic(a, q)))


# ------------------------------------------------------------------ 재표집

def resample(t, rot, tq, method="slerp"):
    """시각 tq 에서의 회전. 구간 밖은 끝값으로 자른다."""
    t = np.asarray(t, dtype=float)
    tq = np.clip(np.asarray(tq, dtype=float), t[0], t[-1])
    r = Rotation.from_matrix(np.asarray(rot, dtype=float))
    if method == "slerp":
        return Slerp(t, r)(tq).as_matrix()
    if method == "nearest":
        # 거리 행렬을 만들면 긴 파일에서 터진다 — 200 Hz 보레아스 한 건이
        # 209 기가를 요구했다. 정렬된 시각이라 이분 탐색이면 충분하다.
        # 정확히 가운데인 질의는 어느 쪽이든 같은 거리라 반올림이 가른다. 그런
        # 격자에서는 이 규약 자체가 정해진 답을 안 주므로 쓰지 않는다.
        j = np.clip(np.searchsorted(t, tq), 1, len(t) - 1)
        pick = np.where(tq - t[j - 1] <= t[j] - tq, j - 1, j)
        return r[pick].as_matrix()
    if method == "zoh":
        return r[np.clip(np.searchsorted(t, tq, side="right") - 1,
                         0, len(t) - 1)].as_matrix()
    raise ValueError("모르는 재표집 방식: %s" % method)


# ------------------------------------------------------------------ 합성 궤적

def trajectory(n, fs, kind, rng):
    """회전 궤적 하나. (t, rot) 반환."""
    t = np.arange(n) / fs
    if kind == "broadband":
        w = rng.normal(0, 1.5, (n, 3))
    elif kind == "smooth":
        f = np.array([0.3, 0.7, 1.1])
        w = 0.8 * np.sin(2 * np.pi * np.outer(t, f))
    elif kind == "fast":
        w = 6.0 * np.sin(2 * np.pi * np.outer(t, [3.0, 5.0, 7.0]))
    elif kind == "still":
        w = np.zeros((n, 3))
    else:
        raise ValueError("모르는 궤적: %s" % kind)
    inc = Rotation.from_rotvec(w / fs)
    rot = np.empty((n, 3, 3))
    cur = Rotation.identity()
    for k in range(n):
        rot[k] = cur.as_matrix()
        cur = cur * inc[k]
    return t, rot


def estimates(q, kind, rng):
    """고정된 추정값 A_ij. 평가 대상 궤적이 아니라 '어떤 고정값' 이면 된다."""
    m = len(q)
    if kind == "exact":
        return q.copy()
    if kind == "bias":
        b = Rotation.from_rotvec([0.05, -0.03, 0.08]).as_matrix()
        return np.einsum("ab,nbc->nac", b, q)
    if kind == "noisy":
        e = Rotation.from_rotvec(rng.normal(0, 0.02, (m, 3))).as_matrix()
        return np.einsum("nab,nbc->nac", q, e)
    if kind == "random":
        return Rotation.random(m, random_state=rng.integers(1 << 31)).as_matrix()
    if kind == "antipodal":
        # 측지거리가 pi 근처라 arccos 가 가장 둔한 자리.
        e = Rotation.from_rotvec(
            (np.pi - 1e-4) * np.eye(3)[rng.integers(0, 3, m)]).as_matrix()
        return np.einsum("nab,nbc->nac", q, e)
    raise ValueError("모르는 추정값: %s" % kind)


def grid(t, kind, rng, fs):
    """재표집할 시각들."""
    if kind == "phase":
        return t + 0.5 / fs
    if kind == "decimate":
        return t[::3]
    if kind == "irregular":
        return np.sort(rng.uniform(t[0], t[-1], len(t)))
    if kind == "gap":
        keep = np.ones(len(t), bool)
        keep[len(t) // 3:len(t) // 3 + len(t) // 10] = False
        return t[keep]
    if kind == "same":
        return t.copy()
    raise ValueError("모르는 시각 격자: %s" % kind)


# ------------------------------------------------------------------ 반증 시험

def check_once(traj, est, gridkind, method, n, fs, seed, use_random_weights):
    """조건 하나에서 상한이 서는지 본다. (D, |E'-E|, 여유) 반환."""
    rng = np.random.default_rng(seed)
    t, rot = trajectory(n, fs, traj, rng)
    tq = grid(t, gridkind, rng, fs)
    rot_q = resample(t, rot, tq, method)

    # 비교할 쌍은 둘 다에서 같은 표본 번호로 고른다 — 이 상한은 '같은 쌍 목록'
    # 위에서만 성립한다. 쌍 목록 자체가 바뀌면 D 가 설명하는 양이 아니다.
    m = min(len(t), len(tq)) - 1
    pairs = np.stack([np.arange(m), np.arange(1, m + 1)], axis=1)
    q = relative(rot[:m + 1], pairs)
    qp = relative(rot_q[:m + 1], pairs)

    w = weights(m, rng if use_random_weights else None)
    a = estimates(q, est, rng)

    d = displacement(q, qp, w)
    e, ep = score(a, q, w), score(a, qp, w)
    return d, abs(ep - e), d - abs(ep - e)


def falsify(score_kind, delta=0.01, arm_m=10.0):
    """상한이 안 서는 쪽 둘. 여기서는 위반이 나오는 게 정상이다.

    난수 궤적으로는 이걸 못 보인다 — 쌍이 수백 개면 부호가 엇갈려 상쇄되고,
    안 서는 점수도 우연히 서 버린다. 그래서 쌍 하나로 직접 만든다.
    기준 Q = I, 재표집된 기준 Q' = z 축 delta 회전이라 D = delta 다.

    셋째 갈래였던 "정렬 재적합" 은 뺐다. 심사 24 회차가 잡았는데, 그 가지는
    변경 전은 정렬 안 한 점수로 재고 변경 후는 0 으로 박아 서로 다른 잣대를
    비교하고 있었다 — delta=0 이라 재표집이 아예 없어도 "위반" 2 가 나왔다.
    제대로 양쪽에 같은 정렬을 걸면 상한은 안 깨진다. 최소를 취하는 연산이
    1 차 립시츠라 두 최소의 차가 피최소량의 최대 변화, 곧 D 를 못 넘는다.
    test_a_common_refit_cannot_break_the_bound 가 그걸 수치로 고정한다.
    """
    q = np.eye(3)[None]
    qp = Rotation.from_rotvec([0, 0, delta]).as_matrix()[None]
    w = np.array([1.0])
    d = displacement(q, qp, w)

    if score_kind == "squared":
        # 추정값을 같은 축으로 2 라디안 틀어 둔다. 제곱 점수의 차는
        # (d' + d)(d' - d) 라 거리가 1 라디안을 넘으면 그 앞의 인수가 1 을
        # 넘어 d(Q,Q') 를 넘긴다.
        a = Rotation.from_rotvec([0, 0, 2.0]).as_matrix()[None]
        e = float(np.dot(w, geodesic(a, q) ** 2))
        ep = float(np.dot(w, geodesic(a, qp) ** 2))
    elif score_kind == "position":
        # 회전 이동량이 위치를 얼마나 미는지는 지렛대 길이에 비례한다.
        a, arm = q.copy(), np.array([arm_m, 0.0, 0.0])
        e = float(np.dot(w, np.linalg.norm(
            np.einsum("nab,b->na", a, arm) - np.einsum("nab,b->na", q, arm), axis=1)))
        ep = float(np.dot(w, np.linalg.norm(
            np.einsum("nab,b->na", a, arm) - np.einsum("nab,b->na", qp, arm), axis=1)))
    else:
        raise ValueError("모르는 점수: %s" % score_kind)
    return d, abs(ep - e), d - abs(ep - e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/next/resample_bound_check.csv")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--fs", type=float, default=50.0)
    args = ap.parse_args()

    rows = []
    seed = 0
    for traj in ("broadband", "smooth", "fast", "still"):
        for est in ("exact", "bias", "noisy", "random", "antipodal"):
            for gk in ("phase", "decimate", "irregular", "gap", "same"):
                for method in ("slerp", "nearest", "zoh"):
                    for rw in (False, True):
                        seed += 1
                        d, diff, slack = check_once(
                            traj, est, gk, method, args.n, args.fs, seed, rw)
                        rows.append({
                            "trajectory": traj, "estimate": est, "grid": gk,
                            "method": method,
                            "weights": "random" if rw else "uniform",
                            "D": "%.6g" % d, "abs_dE": "%.6g" % diff,
                            "slack": "%.3g" % slack,
                            # 상한이 서는 것과 쓸모 있는 것은 다르다. 이 비가
                            # 작으면 D 는 맞지만 느슨해서 순위 보존을 못 말한다.
                            # 생성기가 이 열의 최댓값을 "최대 …" 로 내므로 잰
                            # 값을 그대로 적는다 — 네 자리로 적으면 그 깎기가
                            # 뒷단의 바깥 깎기보다 먼저 와서 보장을 지운다.
                            "tightness": repr(float(diff / d)) if d > 1e-12 else "",
                            "violated": int(slack < -1e-12),
                        })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    bad = [r for r in rows if r["violated"]]
    worst = min(rows, key=lambda r: float(r["slack"]))
    tight = [(float(r["tightness"]), r) for r in rows if r["tightness"]]
    print("조건 %d 건, 상한 위반 %d 건" % (len(rows), len(bad)))
    if tight:
        tight.sort(key=lambda p: p[0])
        hi, r = tight[-1]
        print("가장 꽉 찬 비 |E'-E|/D = %.4f  (%s/%s/%s/%s), 중앙 %.4f"
              % (hi, r["trajectory"], r["estimate"], r["grid"], r["method"],
                 tight[len(tight) // 2][0]))
    print("가장 빠듯한 조건: %s/%s/%s/%s/%s  D=%s  |E'-E|=%s  여유=%s"
          % (worst["trajectory"], worst["estimate"], worst["grid"],
             worst["method"], worst["weights"], worst["D"], worst["abs_dE"],
             worst["slack"]))
    for r in bad[:10]:
        print("  위반 %s/%s/%s/%s" % (r["trajectory"], r["estimate"],
                                      r["grid"], r["method"]))
    print()
    for kind in ("squared", "position"):
        d, diff, slack = falsify(kind)
        print("범위 밖 %-10s D=%.4g  |E'-E|=%.4g  %s"
              % (kind, d, diff, "상한 안 섬" if slack < 0 else "이 조건에선 우연히 섬"))
    print("\n%d 행 -> %s" % (len(rows), args.out))


if __name__ == "__main__":
    main()
