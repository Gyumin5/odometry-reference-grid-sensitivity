#!/usr/bin/env python3
"""키티를 공개된 시각으로 읽을 때와 공식 프로그램이 쓰는 0.1 초 격자로 읽을 때
상한이 얼마나 갈리는지 잰다.

이 릴리스는 자세 파일에 시각 칸이 없고 시퀀스마다 times.txt 를 따로 싣는다.
그 시각의 간격은 0.1 초가 아니다. 공식 평가 프로그램은 그 파일을 안 읽고
프레임마다 0.1 초를 꽂아 쓰므로, 읽는 쪽에 축이 둘 열려 있다. 어느 쪽이
틀렸다고 가릴 자료는 없다 — 둘 다 그 릴리스가 공개한 것이다. 그래서 가르지
않고 둘에서 각각 재서 수가 얼마나 갈리는지만 적는다.

키티-360 쪽 시험(이동량비)은 여기 안 쓴다. 그 시험은 간격이 두 값을 번갈아
낼 때만 서는데, 이 릴리스의 간격은 한 값 둘레에 퍼져 있다.

실행:
    PYTHONPATH=src python3 scripts/kitti_time_axis.py
"""

import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kitti_segment_D import exact, load_poses, measure, proper  # noqa: E402

ROOT = "/mnt/Data/velref/kitti/dataset/poses"
OUT = "results/next/kitti_time_axis.csv"
# 공식 평가 프로그램이 프레임마다 꽂아 쓰는 간격.
NOMINAL_DT = 0.1
RULES = (("seg", (100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0), 10),
         ("rpe1m", (1.0,), 1))
KEYS = ("phase_half", "t_phase_half")
FIELDS = (("sequence", "n_frames", "stamp_median_dt_s", "stamp_min_dt_s",
           "stamp_max_dt_s", "speed_jitter_stamp", "speed_jitter_nominal")
          + tuple("%s_%s_%s" % (rule, k, axis)
                  for rule, _, _ in RULES for k in KEYS
                  for axis in ("stamp", "nominal", "ratio")))
MOVING_M = 0.3


def speed_jitter(t, pos):
    """이웃한 겉보기 속도의 상대 변화 중앙값. 축이 자세와 맞물리면 작다."""
    d = np.diff(t)
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    v = step / np.maximum(d, 1e-12)
    moving = step[:-1] > MOVING_M
    if moving.sum() < 20:
        return float("nan")
    rel = np.abs(np.diff(v))[moving] / np.maximum(v[:-1][moving], 1e-9)
    return float(np.median(rel))


def main():
    rows = []
    for p in sorted(glob.glob(os.path.join(ROOT, "*.txt"))):
        seq = os.path.basename(p)[:-4]
        stamp, rot, pos = load_poses(p)
        if len(stamp) < 2:
            continue
        rot = proper(rot)
        nominal = np.arange(len(stamp)) * NOMINAL_DT
        d = np.diff(stamp)
        row = {
            "sequence": seq,
            "n_frames": len(stamp),
            "stamp_median_dt_s": exact(float(np.median(d))),
            "stamp_min_dt_s": exact(float(d.min())),
            "stamp_max_dt_s": exact(float(d.max())),
            "speed_jitter_stamp": exact(speed_jitter(stamp, pos)),
            "speed_jitter_nominal": exact(speed_jitter(nominal, pos)),
        }
        for rule, lengths, step in RULES:
            a, _ = measure(stamp, rot, pos, lengths=lengths, step=step)
            b, _ = measure(nominal, rot, pos, lengths=lengths, step=step)
            for k in KEYS:
                av = a[k] if a and k in a else float("nan")
                bv = b[k] if b and k in b else float("nan")
                row["%s_%s_stamp" % (rule, k)] = exact(av)
                row["%s_%s_nominal" % (rule, k)] = exact(bv)
                row["%s_%s_ratio" % (rule, k)] = exact(bv / av if av
                                                       else float("nan"))
        rows.append(row)
        print("%-3s 프레임 %5d  간격 중앙값 %.5f (%.5f~%.5f)  속도흔들림 "
              "시각 %.4f 격자 %.4f  1 m 병진비 %6.3f"
              % (seq, len(stamp), float(row["stamp_median_dt_s"]),
                 float(row["stamp_min_dt_s"]), float(row["stamp_max_dt_s"]),
                 float(row["speed_jitter_stamp"]),
                 float(row["speed_jitter_nominal"]),
                 float(row["rpe1m_t_phase_half_ratio"])))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(FIELDS))
        w.writeheader()
        w.writerows(rows)
    print("\n%d 행 -> %s" % (len(rows), OUT))


if __name__ == "__main__":
    main()
