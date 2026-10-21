#!/usr/bin/env python3
"""공개 자세 행렬 위에서 측지각 식이 거리의 공리를 지키는지 직접 센다.

상한의 증명은 d 가 거리라는 데에만 기댄다. 그 전제가 공개 자료에서 실제로
성립하는지는 식을 믿을 일이 아니라 잴 일이다. 식 자체는

    d(A,B) = arccos((tr(A^T B) - 1) / 2)

인데, 이게 거리인 것은 두 인자가 모두 회전일 때다. 공개 행렬은 직교에서
조금 벗어나 있으므로 여기서 둘을 센다.

  자기거리  d(A,A) 가 0 인가. 회전이면 tr(A^T A) = 3 이라 정확히 0 이다.
            직교에서 벗어나면 tr(A^T A) 가 3 에서 밀리고, 3 보다 크면
            arccos 의 인자가 1 을 넘어 잘려서 0 으로 보이고, 작으면 0 이
            아닌 값이 나온다. 잘려서 0 으로 보이는 쪽도 거리를 지킨 게
            아니라 clip 이 가린 것이다.

  삼각부등식  d(A,C) <= d(A,B) + d(B,C) 인가. 세 자세를 뽑아 센다.

가장 가까운 회전으로 한 번 누른 뒤에 같은 것을 다시 센다. 누른 쪽에서
위반이 사라지면, 본문이 적은 "그 값은 회전 변위가 아니라 SO(3) 밖으로
나간 입력에 SO(3) 식을 적용한 수치 잔여" 가 측정으로 뒷받침된다.
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kitti_segment_D import geodesic, load_poses, proper  # noqa: E402


def self_distance(rot):
    """d(A,A) 들. 회전이면 전부 0 이다."""
    return geodesic(rot, rot)


def clipped_self(rot):
    """자기거리에서 arccos 인자가 1 을 넘은 수와 넘은 폭.

    d(A,A) 가 0 으로 보이는 데에는 두 길이 있다. 행렬이 회전이라 정말 0 인
    것과, 직교에서 벗어나 인자가 1 을 넘었는데 clip 이 1 로 눌러 0 처럼
    보이는 것이다. 세어 두지 않으면 뒤쪽이 "공리를 지켰다" 로 세어진다 —
    자른 자리는 식이 거리라는 증거가 아니라 그 반대의 증거다.
    """
    gram = np.einsum("nba,nbc->nac", rot, rot)
    arg = (np.einsum("nii->n", gram) - 1.0) / 2.0
    over = arg - 1.0
    return int(np.count_nonzero(over > 0.0)), float(max(0.0, np.max(over)))


def near_identity_triangle(rot, gap=1):
    """이웃한 세 자세의 d(A,C) - d(A,B) - d(B,C).

    무작위 삼조는 각이 큰 자리를 주로 뽑는다. 역코사인이 못 견디는 곳은 그
    반대쪽 — 항등 근처라 대각합이 3 에 붙는 자리다. 거기서 끝자리가 제곱근
    으로 커지므로, 공리가 깨진다면 먼저 깨지는 데가 거기다.
    """
    a, b, c = rot[:-2 * gap], rot[gap:-gap], rot[2 * gap:]
    return geodesic(a, c) - geodesic(a, b) - geodesic(b, c)


def triangle_violations(rot, rng, trials):
    """무작위 삼조 (A,B,C) 에서 d(A,C) - d(A,B) - d(B,C) 의 양수 쪽."""
    n = len(rot)
    i = rng.integers(0, n, size=trials)
    j = rng.integers(0, n, size=trials)
    k = rng.integers(0, n, size=trials)
    ac = geodesic(rot[i], rot[k])
    ab = geodesic(rot[i], rot[j])
    bc = geodesic(rot[j], rot[k])
    return ac - ab - bc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti-root", default="/mnt/Data/velref/kitti/dataset")
    ap.add_argument("--out", default="results/next/metric_axioms_probe.csv")
    ap.add_argument("--trials", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=20261020)
    args = ap.parse_args()

    rows = []
    for p in sorted(glob.glob(os.path.join(args.kitti_root, "poses",
                                           "*.txt"))):
        seq = os.path.basename(p)[:-4]
        _, raw, _ = load_poses(p)
        for label, rot in (("released", raw), ("projected", proper(raw))):
            rng = np.random.default_rng(args.seed)
            d0 = self_distance(rot)
            slack = triangle_violations(rot, rng, args.trials)
            bad = slack > 0.0
            # 직교 이탈 자체도 같이 적는다. 위반 크기가 그 이탈을 따라가는지
            # 보려면 두 수가 한 줄에 있어야 한다.
            gram = np.einsum("nba,nbc->nac", rot, rot)
            off = float(np.max(np.abs(gram - np.eye(3))))
            n_clip, clip_over = clipped_self(rot)
            near = near_identity_triangle(rot)
            rows.append({
                "sequence": seq, "matrices": label,
                "n_poses": len(rot),
                # 잰 값을 그대로 적는다. 본문이 이 열들에서 "최대 이만큼"
                # 을 뽑으므로 저장 단계에서 깎으면 그 상한이 안쪽이 된다.
                "orthonormality_deviation": repr(float(off)),
                "self_distance_max_deg": repr(float(np.degrees(np.max(d0)))),
                "self_distance_nonzero": int(np.count_nonzero(d0)),
                # 잘려서 0 으로 보이는 칸. 위 두 수와 다른 물음이다.
                "self_distance_clipped": n_clip,
                "self_distance_clip_excess": repr(float(clip_over)),
                "triangle_trials": args.trials,
                "triangle_violations": int(np.count_nonzero(bad)),
                "triangle_worst_deg":
                    repr(float(np.degrees(max(0.0, float(np.max(slack)))))),
                "near_triangle_trials": int(near.size),
                "near_triangle_violations": int(np.count_nonzero(near > 0.0)),
                "near_triangle_worst_deg":
                    repr(float(np.degrees(max(0.0, float(np.max(near)))))),
            })
            print("[%s %-9s] 직교이탈 %.3g  자기거리 최대 %.3g 도 (0 아닌 것"
                  " %d, 잘린 것 %d)  삼각부등식 위반 %d / %d  항등 근처 위반"
                  " %d / %d"
                  % (seq, label, off, np.degrees(np.max(d0)),
                     np.count_nonzero(d0), n_clip, np.count_nonzero(bad),
                     args.trials, np.count_nonzero(near > 0.0), near.size))

    if not rows:
        print("잴 것이 없다")
        return
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    for label in ("released", "projected"):
        sub = [r for r in rows if r["matrices"] == label]
        print("%-9s: 자기거리 최대 %.3g 도 (잘린 칸 %d), 삼각부등식 위반"
              " %d / %d, 항등 근처 %d / %d"
              % (label,
                 max(float(r["self_distance_max_deg"]) for r in sub),
                 sum(r["self_distance_clipped"] for r in sub),
                 sum(r["triangle_violations"] for r in sub),
                 sum(r["triangle_trials"] for r in sub),
                 sum(r["near_triangle_violations"] for r in sub),
                 sum(r["near_triangle_trials"] for r in sub)))
    print("%d 행 -> %s" % (len(rows), args.out))


if __name__ == "__main__":
    main()
