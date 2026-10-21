#!/usr/bin/env python3
"""옥스퍼드 로봇카 RTK 자세에서 읽히지 않는 부분이 주행마다 얼마인지 적는다.

이 묶음은 위치(북·동·아래)와 속도(북·동·아래)를 같은 줄에 같이 싣는다. 둘은
같은 해를 두 가지로 적은 것이라 서로 맞아야 하는데, 어떤 주행은 위치가 한
걸음에 천 미터씩 뛰는 동안 속도 칸은 걷는 속도 그대로다. UTM 구역을 넘는
주행도 하나 있어서 그 자리의 북·동 값은 원점이 다르다.

적재기(kitti_segment_D.load_poses_oxford)는 그 두 자리에서 끊고 가장 긴
토막만 읽는다. 이 대본은 그래서 주행마다 무엇이 얼마나 빠졌는지를 남긴다 —
빠진 양을 안 적으면 표의 한 줄이 어느 구간에서 나온 수인지 알 수 없다.

실행:
    PYTHONPATH=src python3 scripts/oxford_rtk_quality.py
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kitti_segment_D import (OXFORD_ROOT, exact,  # noqa: E402
                             load_poses_oxford)

OUT = "results/next/oxford_rtk_quality.csv"
FIELDS = ("traversal", "rows_published", "rows_read", "kept_frac",
          "zones", "inconsistent_steps", "inconsistent_frac",
          "gap_steps", "resid_p50_m", "resid_p99_m", "resid_max_m",
          "resid_max_nogap_m",
          "duration_s", "median_hz", "distance_m", "max_speed_mps")


def is_gap(dt):
    """시각이 끊긴 걸음 — 그 주행 자신의 간격 중앙값의 두 배를 넘는 자리.

    문턱을 밖에서 들고 오지 않으려고 주행마다 자기 간격으로 잰다. 이 자리를
    따로 표시하는 이유는 큰 잔차가 녹화가 끊긴 데서 오는지 보려는 것이다.
    끊긴 데서 온다면 거기서 끊는 규칙은 임의가 아니다.
    """
    return dt > 2.0 * np.median(dt)


def gap_count(dt):
    return int(is_gap(dt).sum())


def raw(path):
    """파일에 적힌 그대로 — 구역 이름과 시각·위치·속도."""
    zone = np.atleast_1d(np.genfromtxt(path, delimiter=",", skip_header=1,
                                       usecols=(7,), dtype=str))
    a = np.atleast_2d(np.genfromtxt(path, delimiter=",", skip_header=1,
                                    usecols=(0, 4, 5, 6, 8, 9, 10)))
    return zone, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=OXFORD_ROOT)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    rows = []
    for p in sorted(glob.glob(os.path.join(args.root, "*", "rtk.csv"))):
        name = os.path.basename(os.path.dirname(p))
        zone, a = raw(p)
        if a.size == 0 or len(a) < 2:
            print("%s 줄이 없다" % name)
            continue
        t, pos, vel = a[:, 0] * 1e-6, a[:, 1:4], a[:, 4:7]
        dt = np.diff(t)
        step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        allow = 0.5 * (np.linalg.norm(vel[:-1], axis=1)
                       + np.linalg.norm(vel[1:], axis=1)) * dt
        resid = np.abs(step - allow)
        off = int((resid > allow).sum())

        rt, _, rpos = load_poses_oxford(p)
        if len(rt) < 2:
            print("%s 읽을 토막이 없다" % name)
            continue
        d = np.diff(rt)
        rstep = np.linalg.norm(np.diff(rpos, axis=0), axis=1)
        rows.append({
            "traversal": name,
            "rows_published": len(a), "rows_read": len(rt),
            "kept_frac": exact(len(rt) / len(a)),
            "zones": len(set(zone.tolist())),
            "inconsistent_steps": off,
            "inconsistent_frac": exact(off / len(step)),
            "gap_steps": gap_count(dt),
            "resid_p50_m": exact(float(np.percentile(resid, 50))),
            "resid_p99_m": exact(float(np.percentile(resid, 99))),
            "resid_max_m": exact(float(resid.max())),
            "resid_max_nogap_m": exact(float(resid[~is_gap(dt)].max())),
            "duration_s": exact(float(rt[-1] - rt[0])),
            "median_hz": exact(1.0 / float(np.median(d))),
            "distance_m": exact(float(rstep.sum())),
            "max_speed_mps": exact(float((rstep / np.maximum(d, 1e-12)).max())),
        })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(FIELDS))
        w.writeheader()
        w.writerows(rows)

    off = [float(r["inconsistent_frac"]) for r in rows]
    dirty = [r for r in rows if float(r["inconsistent_frac"]) > 0]
    print("주행 %d, 속도 칸과 어긋나는 걸음이 있는 주행 %d"
          % (len(rows), len(dirty)))
    print("어긋나는 몫 중앙 %.4f 최대 %.3f"
          % (float(np.median(off)), max(off)))
    print("가장 빠른 걸음 %.1f m/s" % max(float(r["max_speed_mps"])
                                          for r in rows))
    print("잔차 중앙의 중앙 %.4f m, 99 분위의 최대 %.1f m, 최대 %.1f m"
          % (float(np.median([float(r["resid_p50_m"]) for r in rows])),
             max(float(r["resid_p99_m"]) for r in rows),
             max(float(r["resid_max_m"]) for r in rows)))
    print("시각이 끊긴 걸음 합 %d, 안 끊긴 걸음의 잔차 최대 %.1f m"
          % (sum(int(r["gap_steps"]) for r in rows),
             max(float(r["resid_max_nogap_m"]) for r in rows)))
    print("%d 행 -> %s" % (len(rows), args.out))


if __name__ == "__main__":
    main()
