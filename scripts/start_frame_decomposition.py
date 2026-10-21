#!/usr/bin/env python3
"""구간 병진 이동량을 시작 자세 몫과 끝점 위치 몫으로 가른다.

규약 훑기에서 솎기 두 위상이 병진만 쉰일곱 배 갈렸다. 회전은 두 배 남짓이고
1 m 규칙에서는 둘이 같다. 그 비대칭이 주기 때문이 아니라는 것을 보이려면
두 위상이 각각 어느 프레임을 정확히 되살리는지, 그리고 그 프레임이 시작이냐
끝이냐에 따라 이동량이 어떻게 달라지는지를 같은 자료에서 재야 한다.

재는 것 넷.
  구간 시작에서의 위치 오차와 자세 오차
  구간 끝에서의 위치 오차
  끝 프레임 위상으로 가른 두 무리의 병진 이동량
그 둘이 같으면 끝점은 기여가 없다는 뜻이고, 그러면 남는 것은 시작뿐이다.
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kitti_segment_D import (LENGTHS, STEP, euclidean,  # noqa: E402
                             gather, geodesic, relative,
                             relative_translation)
from protocol_sweep import resample, window, window_segments  # noqa: E402

PHASES = (("even", slice(None, None, 2)), ("odd", slice(1, None, 2)))


def decompose(t, rot, pos, lengths=LENGTHS, step=STEP):
    """위상마다 한 묶음. 구간 집합은 두 위상이 함께 쓴다.

    구간은 창 안의 자세를 다시 훑어 고르지 않는다. 그렇게 하면 시작점 격자가
    창의 첫 프레임에서 다시 세어지는데 이 창은 원본 색인 1 에서 시작하므로,
    공식 집합과 크기만 비슷하고 원소는 하나도 안 겹치는 집합이 나온다. 원본
    격자에 고정한 뒤 창 안에 온전히 드는 것만 남겨 공식 집합의 부분집합으로
    만들고, 색인만 창 기준으로 옮겨 아래 배열에 바로 쓴다. 끝점 홀짝은 그
    색인에 창의 시작 위치를 더해 원본 기준으로 되돌려 센다.
    """
    dt = float(np.median(np.diff(t)))
    lo, hi = window(t, dt)
    m = (t >= lo) & (t <= hi)
    tw, rw, pw = t[m], rot[m], pos[m]
    idx = window_segments(pos, m, lengths, step)
    if not idx:
        return []
    offset = int(np.argmax(m))
    starts = np.asarray([a for a, _, _ in idx])
    ends = np.asarray([b for _, b, _ in idx])
    L = np.asarray([p[2] for p in idx], dtype=float)
    end_parity = (ends + offset) % 2

    out = []
    for name, sl in PHASES:
        ar, ap = resample(t[sl], rot[sl], pos[sl], tw, "slerp")
        moved = euclidean(relative_translation(rw, pw, idx),
                          relative_translation(ar, ap, idx)) / L
        row = {
            "phase": name,
            "n_segments": len(idx),
            "start_parity_retained":
                int(np.all((starts + offset) % 2 == (1 if name == "odd"
                                                     else 0))),
            "start_pos_err_m": float(np.mean(np.linalg.norm(
                ap[starts] - pw[starts], axis=1))),
            "start_rot_err_deg": float(np.degrees(np.mean(geodesic(
                rw[starts], ar[starts])))),
            "end_pos_err_m": float(np.mean(np.linalg.norm(
                ap[ends] - pw[ends], axis=1))),
            "D_trans_pct": float(100.0 * np.mean(moved)),
            "D_trans_pct_end_even": float(
                100.0 * np.mean(moved[end_parity == 0]))
            if np.any(end_parity == 0) else float("nan"),
            "D_trans_pct_end_odd": float(
                100.0 * np.mean(moved[end_parity == 1]))
            if np.any(end_parity == 1) else float("nan"),
            "D_rot_deg_per_100m": float(
                100.0 * np.degrees(np.mean(geodesic(
                    relative(rw, idx), relative(ar, idx)) / L))),
        }
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti-root", default="/mnt/Data/velref/kitti/dataset")
    ap.add_argument("--sequence", default="00")
    ap.add_argument("--out",
                    default="results/next/start_frame_decomposition.csv")
    args = ap.parse_args()

    rows = []
    for name, (t, rot, pos) in gather("kitti", args.kitti_root):
        if name != args.sequence:
            continue
        for row in decompose(t, rot, pos):
            row["sequence"] = name
            rows.append(row)
    if not rows:
        raise SystemExit("그 시퀀스를 못 찾았다")

    keys = ["sequence"] + [k for k in rows[0] if k != "sequence"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("%.6g" % r[k] if isinstance(r[k], float)
                            else r[k]) for k in keys})
    for r in rows:
        print("[%s] 시작 위치오차 %.4g m, 시작 자세오차 %.4g 도,"
              " 끝 위치오차 %.4g m" % (r["phase"], r["start_pos_err_m"],
                                       r["start_rot_err_deg"],
                                       r["end_pos_err_m"]))
        print("      병진 %.5g %%  (끝 짝수 %.5g, 끝 홀수 %.5g)"
              % (r["D_trans_pct"], r["D_trans_pct_end_even"],
                 r["D_trans_pct_end_odd"]))
    print("\n%d 줄 -> %s" % (len(rows), args.out))


if __name__ == "__main__":
    main()
