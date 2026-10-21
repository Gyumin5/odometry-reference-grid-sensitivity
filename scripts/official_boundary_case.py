#!/usr/bin/env python3
"""공식 평가기의 단정밀도 산술이 상한을 얼마나 넘길 수 있는지 공식 코드로 잰다.

상한은 이상적인 산술 위에서 선다. 공식 프로그램은 요약을 단정밀도로
누산하므로, 기준을 아주 조금 옮겼을 때 찍히는 수의 변화가 D 를 넘을 수
있다. 그 틈이 실제로 얼마인지를 추정하지 않고 공식 코드로 잰다.

구간 수가 적은 짧은 궤적을 쓴다. 구간이 많으면 반올림이 서로 지워져서 격자
효과가 평균에 묻힌다. 기본값인 201 프레임 곧은 궤적에서는 구간이 열 선다 —
공식 평가기가 100 미터부터 재기 때문이다. 실제 구간 수는 결과 표의
segments 칸에 적힌다.

빌드와 실행은 kitti_official_binary_check.py 의 것을 그대로 쓴다.
"""

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kitti_official_binary_check import DRIFT_EPS, build  # noqa: E402
from kitti_segment_D import (  # noqa: E402
    geodesic, relative, segment_weights, segments,
)

STEP_M = 1.0


def straight(n):
    """x 축으로 1 미터씩 가는 곧은 궤적."""
    rot = np.tile(np.eye(3), (n, 1, 1))
    pos = np.zeros((n, 3))
    pos[:, 0] = np.arange(n) * STEP_M
    return rot, pos


def write_poses(path, rot, pos):
    with open(path, "w") as f:
        for r, p in zip(rot, pos):
            m = np.hstack([r, p.reshape(3, 1)]).reshape(-1)
            f.write(" ".join("%.12e" % v for v in m) + "\n")


def yaw(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devkit",
                    default="state/devkit-20261020/extracted/devkit/cpp")
    ap.add_argument("--frames", type=int, default=201)
    # 11 시퀀스 드리프트 탐침과 같은 값을 쓴다. 숫자를 여기 또 적으면 한쪽만
    # 바뀌어도 본문이 둘 중 어느 것을 가리키는지 알 수 없게 된다.
    ap.add_argument("--eps", type=float, default=DRIFT_EPS,
                    help="추정이 가진 프레임당 회전 흔들림(라디안)")
    # 찍히는 값이 격자 위에 있어서, 격자 한 칸을 건너뛰는 자리가 가장 불리하다.
    # 그 자리를 못 밟는 성긴 그물로는 "넘지 않는다" 가 나온다 — 격자가 0 으로
    # 떨어지는 구간을 0.1e-4 간격으로 훑는다.
    ap.add_argument("--shift", type=float, nargs="+",
                    default=[1e-7, 1e-6, 1e-5, 1e-4, 2e-4, 3e-4, 4e-4,
                             5.0e-4, 5.2e-4, 5.4e-4, 5.6e-4, 5.8e-4,
                             5.9e-4, 6.0e-4, 7e-4, 8e-4, 9e-4,
                             1e-3, 2e-3, 5e-3],
                    help="기준을 옮기는 양(라디안). D 가 이 크기로 나온다")
    ap.add_argument("--out",
                    default="results/next/official_boundary_case.csv")
    args = ap.parse_args()

    work = tempfile.mkdtemp(prefix="boundary-")
    binary, _ = build(args.devkit, work)

    n = args.frames
    rot, pos = straight(n)

    # 추정: 프레임마다 쌓이는 흔들림.
    est = np.einsum("nab,nbc->nac", rot,
                    np.stack([yaw(k * args.eps) for k in range(n)]))
    gt_a = os.path.join(work, "gt_a.txt")
    gt_b = os.path.join(work, "gt_b.txt")
    res = os.path.join(work, "res.txt")
    write_poses(gt_a, rot, pos)
    write_poses(res, est, pos)

    idx = segments(pos)
    if not len(idx):
        raise SystemExit("구간이 안 선다 — 궤적을 더 길게 잡아라")

    def official(gt):
        r = subprocess.run([binary, gt, res], capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit("공식 이진파일이 멈췄다: %s" % r.stderr.strip())
        f = r.stdout.split()
        return float(f[1]), float(f[3])   # 배정밀도 누산, 단정밀도 누산

    base_d, base_f = official(gt_a)
    rows = []
    for shift in args.shift:
        # 뒤쪽 절반만 돌린다. 구간의 한쪽 끝만 움직여야 상대자세가 바뀐다.
        moved = rot.copy()
        moved[n // 2:] = np.einsum("nab,bc->nac", moved[n // 2:], yaw(shift))
        write_poses(gt_b, moved, pos)
        got_d, got_f = official(gt_b)
        d = float(np.dot(segment_weights(idx),
                         geodesic(relative(rot, idx), relative(moved, idx))))
        ratio = abs(got_d - base_d) / d if d > 0.0 else float("nan")
        rows.append({"shift_rad": repr(shift), "segments": len(idx),
                     "official_rot": repr(got_d),
                     "official_rot_single": repr(got_f),
                     "D_rot_per_m": repr(d), "change_over_D": repr(ratio),
                     "change_minus_D": repr(abs(got_d - base_d) - d)})
        print("옮김 %7.1e  공식 %-16.9g  D %9.3e  변화/D %6.3f"
              % (shift, got_d, d, ratio))

    # 찍히는 값들이 올라서는 격자의 한 칸. 0 바로 위 칸이 가장 거친 칸이고,
    # D 를 넘는 초과분이 그 한 칸 안에 드는지가 고쳐진 보장식의 판정이다.
    levels = sorted({float(r["official_rot"]) for r in rows} | {base_d})
    step = min(b - a for a, b in zip(levels, levels[1:])) if len(levels) > 1 \
        else float("nan")
    coarse = min(v for v in levels if v > 0.0)
    for r in rows:
        r["grid_step"] = repr(step)
        r["excess_over_coarsest"] = repr(float(r["change_minus_D"]) / coarse)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("바탕 %.9g (단정밀도 %.9g)" % (base_d, base_f))
    print("격자 칸 %.6g, 0 바로 위 칸 %.6g, D 초과분 최대 %.6g (칸의 %.3f)"
          % (step, coarse, max(float(r["change_minus_D"]) for r in rows),
             max(float(r["excess_over_coarsest"]) for r in rows)))
    print("%d 행 -> %s" % (len(rows), args.out))


if __name__ == "__main__":
    main()
