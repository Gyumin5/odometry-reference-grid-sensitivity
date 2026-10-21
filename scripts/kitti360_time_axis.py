#!/usr/bin/env python3
"""키티-360 자세가 어느 시간축 위에 있는지 가리고, 틀린 축을 고르면 얼마나
옮겨가는지 잰다.

이 릴리스의 poses.txt 는 시각이 아니라 프레임 번호를 싣는다. 번호가 드문드문
비므로 번호를 공개 주기 10 Hz 로 나눈 축이 하나 서고, 따로 받는
data_poses_oxts 의 oxts/timestamps.txt 가 또 하나 선다. 둘 중 어느 쪽이 이
자세의 축인지는 적혀 있지 않아서 재야 안다.

가르는 시험은 하나다. 공식 시각의 간격은 0.100 초와 0.110 초를 오간다. 달리는
중이면 0.110 초짜리 구간의 이동량이 이웃한 0.100 초짜리보다 10 퍼센트 커야
한다. 안 커지면 자세는 그 시각이 아니라 고른 프레임 격자 위에 있는 것이다.

가린 뒤에도 틀린 축의 값을 같이 적는다. 그 축을 고르는 것이 읽는 쪽에
열려 있는 선택이고, 선택이 수를 얼마나 옮기는지가 이 논문이 재는 양이다.

실행:
    PYTHONPATH=src python3 scripts/kitti360_time_axis.py
"""

import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kitti_segment_D import (exact, kitti360_timestamps,  # noqa: E402
                             load_poses_kitti360, measure, proper)

ROOT = "/mnt/Data/velref/kitti360"
OUT = "results/next/kitti360_time_axis.csv"
RULES = (("seg", (100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0), 10),
         ("rpe1m", (1.0,), 1))
KEYS = ("phase_half", "t_phase_half")
FIELDS = (("sequence", "has_stamps", "n_frames", "n_stamps",
           "grid_median_dt_s", "stamp_median_dt_s", "stamp_max_gap_s",
           "stamp_short_dt_s", "stamp_long_dt_s",
           "step_ratio_median", "step_ratio_pairs",
           "speed_jitter_grid", "speed_jitter_stamp")
          + tuple("%s_%s_%s" % (rule, k, axis)
                  for rule, _, _ in RULES for k in KEYS
                  for axis in ("grid", "stamp", "ratio")))
# 0.100 초와 0.110 초를 가르는 창. 공식 간격이 이 둘에 거의 전부 몰린다.
SHORT = (0.0995, 0.1005)
LONG = (0.1095, 0.1105)
MOVING_M = 0.3


def mode_dt(t, window):
    """그 창 안에 든 간격들의 중앙값. 두 자리 값을 본문이 글자로 안 들게."""
    d = np.diff(t)
    sel = d[(d > window[0]) & (d < window[1])]
    return float(np.median(sel)) if len(sel) else float("nan")


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


def step_ratio(t, pos):
    """간격이 0.100 에서 0.110 으로 바뀌는 자리의 이동량비 중앙값과 쌍 수.

    같은 속도로 달리는 중이면 1.10 이 나와야 한다. 1.00 이 나오면 자세가 그
    시각을 안 따른다. 두 방향을 같이 세어 한쪽으로 치우치지 않게 한다.
    """
    d = np.diff(t)
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    short = (d > SHORT[0]) & (d < SHORT[1])
    longi = (d > LONG[0]) & (d < LONG[1])
    moving = step[:-1] > MOVING_M
    up = short[:-1] & longi[1:] & moving
    down = longi[:-1] & short[1:] & moving
    ratios = np.concatenate([step[1:][up] / step[:-1][up],
                             step[:-1][down] / step[1:][down]])
    if len(ratios) < 40:
        return float("nan"), len(ratios)
    return float(np.median(ratios)), len(ratios)


def main():
    rows = []
    for p in sorted(glob.glob(os.path.join(ROOT, "*_sync", "poses.txt"))):
        seq = os.path.basename(os.path.dirname(p)).split("_drive_")[1][:4]
        grid, rot, pos = load_poses_kitti360(p)
        stamp, _, _ = load_poses_kitti360(p, stamped=True)
        rot = proper(rot)
        stamps = kitti360_timestamps(p)
        has = stamps is not None and not np.array_equal(grid, stamp)
        ratio, pairs = step_ratio(stamp, pos) if has else (float("nan"), 0)
        row = {
            "sequence": seq,
            "has_stamps": "yes" if has else "no",
            "n_frames": len(grid),
            "n_stamps": len(stamps) if stamps is not None else 0,
            "grid_median_dt_s": exact(float(np.median(np.diff(grid)))),
            "stamp_median_dt_s": exact(float(np.median(np.diff(stamp)))),
            "stamp_max_gap_s": exact(float(np.max(np.diff(stamp)))),
            "stamp_short_dt_s": exact(mode_dt(stamp, SHORT)),
            "stamp_long_dt_s": exact(mode_dt(stamp, LONG)),
            "step_ratio_median": exact(ratio),
            "step_ratio_pairs": pairs,
            "speed_jitter_grid": exact(speed_jitter(grid, pos)),
            "speed_jitter_stamp": exact(speed_jitter(stamp, pos)),
        }
        for rule, lengths, step in RULES:
            a, _ = measure(grid, rot, pos, lengths=lengths, step=step)
            b, _ = measure(stamp, rot, pos, lengths=lengths, step=step)
            for k in KEYS:
                av = a[k] if a and k in a else float("nan")
                bv = b[k] if b and k in b else float("nan")
                row["%s_%s_grid" % (rule, k)] = exact(av)
                row["%s_%s_stamp" % (rule, k)] = exact(bv)
                row["%s_%s_ratio" % (rule, k)] = exact(bv / av if av
                                                       else float("nan"))
        rows.append(row)
        print("%-6s 시각파일 %-3s 프레임 %6d  이동량비 %6.4f (%5d 쌍)  "
              "속도흔들림 격자 %.4f 시각 %.4f  1 m 병진비 %6.3f"
              % (seq, row["has_stamps"], len(grid), ratio, pairs,
                 float(row["speed_jitter_grid"]),
                 float(row["speed_jitter_stamp"]),
                 float(row["rpe1m_t_phase_half_ratio"])))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(FIELDS))
        w.writeheader()
        w.writerows(rows)
    print("\n%d 행 -> %s" % (len(rows), OUT))


if __name__ == "__main__":
    main()
