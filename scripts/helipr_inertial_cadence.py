#!/usr/bin/env python3
"""헬리퍼 관성 파일의 주기와 열 구성을 확인한다.

목록에 미확인으로 올라 있던 항목이다. 겸해서 그 릴리스가 같은 항법 해를 두 주기로
싣고 있는지 본다 — INSPVA 는 자세를 직접 싣고 LiDAR_GT 는 같은 해를 라이다 시각으로
뽑은 10 Hz 궤적이다. 같은 해가 두 주기로 실려 있으면 보레아스에서 보려다 못 본
'느리게 실으면 조용해 보인다' 를 같은 릴리스 안에서 다시 잴 수 있다.

INSPVA 열: 시각(ns), 위도, 경도, 고도, 북/동/상 속도, 롤, 피치, 방위(도), 상태.
"""

import argparse
import csv
import glob
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reference_resolution_audit import audit, ypr_to_rot  # noqa: E402


def load_inspva(path):
    """시각(s)과 자세 회전행렬. 방위는 북기준 시계방향이라 요로 쓸 때 부호를 뒤집는다."""
    t, ypr = [], []
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) < 11:
                continue
            try:
                t.append(float(row[0]) * 1e-9)
                ypr.append([-math.radians(float(row[9])),
                            math.radians(float(row[8])),
                            math.radians(float(row[7]))])
            except ValueError:
                continue
    a = np.array(ypr)
    return np.array(t), ypr_to_rot(a[:, 0], a[:, 1], a[:, 2]), a.tolist()


def cadence(t):
    dt = np.diff(t)
    med = float(np.median(dt))
    inlier = dt[dt <= 3 * med]
    return {
        "n": len(t),
        "fs_hz": round(1.0 / med, 3),
        "dt_med_ms": round(med * 1000, 4),
        "jitter_rms_ms": round(float(np.std(inlier) * 1000), 4),
        "gap_max_s": round(float(dt.max()), 4),
        "duration_s": round(float(t[-1] - t[0]), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/Data/velref/helipr")
    ap.add_argument("--band", default="2,4")
    ap.add_argument("--win-s", type=float, default=16.0)
    ap.add_argument("--out", default="results/next/helipr_inertial_cadence.csv")
    args = ap.parse_args()
    band_lo, band_hi = (float(v) for v in args.band.split(","))

    rows = []
    for p in sorted(glob.glob(os.path.join(args.root, "*", "Inertial_data", "inspva.csv"))):
        seq = os.path.basename(os.path.dirname(os.path.dirname(p)))
        try:
            t, rot, _ = load_inspva(p)
        except Exception as e:
            print("[%s] 읽기 실패: %s" % (seq, e))
            continue
        if len(t) < 64:
            print("[%s] 표본 %d 개라 건너뛴다" % (seq, len(t)))
            continue
        rec = {"sequence": seq}
        rec.update(cadence(t))
        r = audit(t, rot, seq, 1.0, 10.0, band_lo, band_hi, args.win_s, "x") or {}
        rec["model"] = r.get("model", "")
        rec["band_slope"] = r.get("band_slope", "")
        rec["S0_total"] = r.get("S0_total_rad2ps2perHz", "")
        rec["ARW_deg_per_sqrt_hr"] = r.get("ARW_deg_per_sqrt_hr", "")
        rows.append(rec)
        print("%-14s n=%-7d %7.2f Hz  흔들림 %6.3f ms  기울기 %7s  판정 %s"
              % (seq, rec["n"], rec["fs_hz"], rec["jitter_rms_ms"],
                 rec["band_slope"], rec["model"] or "-"))

    if not rows:
        print("읽을 것이 없다")
        return
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("%d 건 -> %s" % (len(rows), args.out))

    fs = sorted(r["fs_hz"] for r in rows)
    jt = sorted(r["jitter_rms_ms"] for r in rows)
    sl = sorted(float(r["band_slope"]) for r in rows if r["band_slope"] != "")
    print("주기 중앙 %.2f Hz (%.2f~%.2f), 시각 흔들림 중앙 %.3f ms"
          % (fs[len(fs) // 2], fs[0], fs[-1], jt[len(jt) // 2]))
    if sl:
        print("기울기 중앙 %.2f (%.2f~%.2f), 백색 판정 %d/%d"
              % (sl[len(sl) // 2], sl[0], sl[-1],
                 sum(r["model"] == "백색" for r in rows), len(rows)))


if __name__ == "__main__":
    main()
