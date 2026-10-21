#!/usr/bin/env python3
"""두 점수가 같이 밀려 상쇄되는 몫을 재서 2D 가 얼마나 보수적인지 본다.

2D 는 최악의 경우다. 등수는 점수 하나가 아니라 두 점수의 차이로 정해지는데,
기준을 바꾸면 둘 다 움직이고 같은 방향으로 움직인 만큼은 차이에서 지워진다.
2D 는 둘이 정확히 반대로 밀린다고 가정한 값이라 실제보다 보수적이다.

제대로 재려면 추정기 둘을 돌려 각자의 오차 방향을 봐야 하는데, 추정기를 안
돌리는 것이 이 설계의 전제다. 대신 오차를 모형으로 두고 닮음도를 훑는다.

    A 의 구간 오차  e_k  (등방 무작위, 크기는 목표 점수에 맞춘다)
    B 의 구간 오차  f_k = rho*e_k + sqrt(1-rho^2)*g_k   (g 는 독립)

재는 것은 차이가 실제로 움직인 양이다.

    s = (E'_A - E_A) - (E'_B - E_B)

B 가 A 보다 g 만큼 나쁘다고 두면 기준을 바꾼 뒤의 차이는 g - s 다. 등수가
뒤집히는 것은 그 값이 음수일 때, 곧 부호 있는 s 가 g 를 넘을 때다. |s| > g
가 아니다 — 그렇게 세면 차이가 더 벌어진 쪽(s < -g)까지 뒤집힘으로 센다.
크기는 |s| 로 적고 뒤집힘은 부호 있는 s 로 센다.

이건 모형이지 실측이 아니다. 실제 추정기의 오차가 어떤 rho 를 갖는지는 모르고,
등방 무작위라는 가정도 실제 정합 오차의 모양이 아니다. 얻는 것은 "최악의 경우
하나" 대신 "조건에 따른 범위" 다.
"""

import argparse
import csv
import glob
import os

import numpy as np
from scipy.spatial.transform import Rotation

import kitti_segment_D as K

RHOS = (0.0, 0.5, 0.9, 0.99)
# 점수 수준은 공식 순위표에서 가져온다. 키티 주행거리계 공식 순위표 상위 30 의
# 회전오차가 0.09~0.22 도/100 m 이고 중앙이 0.15 다. 다른 논문의 결과표는 쓰지
# 않는다 — 어느 방법을 싣느냐로 수준이 달라진다.
SCORES_DEG_PER_100M = (0.09, 0.15, 0.22)
# 뒤집힘을 읽을 간격 셋. 0.01 은 그 표의 표시 자릿수에서 이웃이 갈리는 최소
# 간격이고(실제로는 같은 값으로 묶인 쌍이 더 많다), 0.06 은 1 등과 중앙의
# 차이, 0.13 은 상위 30 전체가 퍼진 폭이다.
GAPS = (0.01, 0.06, 0.13)
TRIALS = 200


def _perturb(q, rotvec):
    return np.einsum("nab,nbc->nac", q, Rotation.from_rotvec(rotvec).as_matrix())


def _unit(rng, n):
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def shifts(q, qp, w, amp_a, amp_b, rho, rng, trials):
    """시행마다 s = (E'_A - E_A) - (E'_B - E_B) 를 라디안 가중평균 단위로."""
    n = len(w)
    out = np.empty(trials)
    root = np.sqrt(max(0.0, 1.0 - rho * rho))
    for i in range(trials):
        e = _unit(rng, n)
        f = rho * e + root * _unit(rng, n)
        f /= np.linalg.norm(f, axis=1, keepdims=True)
        a = _perturb(q, e * amp_a[:, None])
        b = _perturb(q, f * amp_b[:, None])
        da = np.dot(w, K.geodesic(a, qp)) - np.dot(w, K.geodesic(a, q))
        db = np.dot(w, K.geodesic(b, qp)) - np.dot(w, K.geodesic(b, q))
        out[i] = da - db
    return out


def sweep(t, rot, pos, rng, trials=TRIALS, rhos=RHOS,
          scores=SCORES_DEG_PER_100M, gaps=GAPS):
    """한 시퀀스에서 (점수 수준 × rho) 마다 s 분포를 낸다. 단위는 도/100 m."""
    dt = float(np.median(np.diff(t)))
    # 주표와 같은 공통 창·같은 구간 생성 절차를 쓴다. 보조 모형이 끝을 잘라
    # 붙이면 2D 가 주표의 2D 와 다른 수가 되고, 그 몫을 "2D 대비" 로 적는
    # 순간 두 수의 분모가 어긋난다.
    keep = K.common_window(t, dt)
    t_src, rot_src, pos_src = t, rot, pos
    t, rot, pos = t[:keep], rot[:keep], pos[:keep]
    idx = K.segments(pos)
    if not idx:
        return None
    L = np.asarray([p[2] for p in idx], dtype=float)
    w = K.segment_weights(idx)
    q = K.relative(rot, idx)
    hr = K.resample(t_src, rot_src, pos_src, t + 0.5 * dt, "slerp")[0]
    qp = K.relative(hr, idx)
    two_d = 2.0 * float(np.dot(w, K.geodesic(q, qp))) * np.degrees(1.0) * 100.0
    if two_d <= 0.0:
        return None

    recs = []
    for score in scores:
        for gap in gaps:
            # 점수는 각/길이의 가중 평균이므로 구간마다 (점수 × 길이) 의 오차를
            # 주면 그 수준이 나온다. B 는 A 보다 그 간격만큼 나쁘게 둔다.
            amp_a = np.radians(score / 100.0) * L
            amp_b = np.radians((score + gap) / 100.0) * L
            for rho in rhos:
                raw = shifts(q, qp, w, amp_a, amp_b, rho, rng, trials)
                signed = np.degrees(raw) * 100.0
                mag = np.abs(signed)
                recs.append({
                    "score_deg_per_100m": score,
                    "gap_deg_per_100m": gap,
                    "rho": rho,
                    "shift_median": float(np.median(mag)),
                    "shift_p95": float(np.quantile(mag, 0.95)),
                    "shift_max": float(np.max(mag)),
                    "frac_of_twoD_median": float(np.median(mag)) / two_d,
                    # 부호 있는 s 가 간격을 넘을 때만 등수가 뒤집힌다.
                    "flip_rate": float(np.mean(signed > gap)),
                })
    return recs, two_d, len(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti-root", default="/mnt/Data/velref/kitti/dataset")
    ap.add_argument("--out", default="results/next/pair_cancellation.csv")
    ap.add_argument("--trials", type=int, default=TRIALS)
    ap.add_argument("--seed", type=int, default=20261019)
    args = ap.parse_args()

    rows = []
    for p in sorted(glob.glob(os.path.join(args.kitti_root, "poses",
                                           "*.txt"))):
        seq = os.path.basename(p)[:-4]
        # 주표와 같은 읽기 경로여야 한다. 공개 행렬을 날로 쓰면 직교 이탈이
        # 남긴 바닥이 이 모형의 이동량에 섞여서, 같은 자료를 두 방식으로 읽은
        # 수가 한 논문에 같이 실린다.
        t, rot, pos = K.load_poses(p)
        rot = K.proper(rot)
        got = sweep(t, rot, pos, np.random.default_rng(args.seed),
                    trials=args.trials)
        if got is None:
            continue
        recs, two_d, nseg = got
        for r in recs:
            rows.append({
                "sequence": seq, "n_segments": nseg,
                # 잰 값을 그대로 적는다. 결과표는 측정이 낸 수를 담는
                # 자리이고, 깎는 것은 본문에 실을 때 방향을 정해 한 번만
                # 한다 — 저장 단계에서 먼저 깎으면 그 방향이 사라진다.
                "twoD_deg_per_100m": repr(float(two_d)),
                "score_deg_per_100m": repr(float(r["score_deg_per_100m"])),
                "gap_deg_per_100m": repr(float(r["gap_deg_per_100m"])),
                "rho": repr(float(r["rho"])),
                "shift_median": repr(float(r["shift_median"])),
                "shift_p95": repr(float(r["shift_p95"])),
                "shift_max": repr(float(r["shift_max"])),
                "frac_of_twoD_median": repr(float(r["frac_of_twoD_median"])),
                "flip_rate": repr(float(r["flip_rate"])),
            })
        print("[%s] 구간 %5d  2D=%.4g" % (seq, nseg, two_d))

    if not rows:
        print("잴 것이 없다")
        return
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n점수 수준 × 간격 × rho 별 요약 (시퀀스 가로질러 중앙)")
    print("%-8s %-8s %-6s %10s %10s %9s %9s"
          % ("점수", "간격", "rho", "|s| 중앙", "|s| 95분위", "2D 대비",
             "뒤집힘"))
    for score in SCORES_DEG_PER_100M:
        for gap in GAPS:
            for rho in RHOS:
                sub = [r for r in rows
                       if abs(float(r["score_deg_per_100m"]) - score) < 1e-9
                       and abs(float(r["gap_deg_per_100m"]) - gap) < 1e-9
                       and abs(float(r["rho"]) - rho) < 1e-9]
                print("%-8.2f %-8.2f %-6.2f %10.4f %10.4f %9.3f %9.3f"
                      % (score, gap, rho,
                         np.median([float(r["shift_median"]) for r in sub]),
                         np.median([float(r["shift_p95"]) for r in sub]),
                         np.median([float(r["frac_of_twoD_median"])
                                    for r in sub]),
                         np.median([float(r["flip_rate"]) for r in sub])))
    print("\n뒤집힘은 B 가 A 보다 그 간격만큼 나쁠 때 기준을 바꿔 순서가"
          " 바뀔 확률이다. 모형 값이고 실제 방법의 뒤집힘률이 아니다.")
    print("%d 행 -> %s" % (len(rows), args.out))


if __name__ == "__main__":
    main()
