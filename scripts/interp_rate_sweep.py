#!/usr/bin/env python3
"""추정기 주기가 기준 주기의 몇 배냐에 따라 재표집 이동량이 어떻게 변하는지 쓴다.

release_resample_D.py 의 interp 규약은 비율 하나(0.7)를 골라 썼다. 고른 상수라
그 자리만 보면 "왜 하필 0.7 이냐" 를 못 막는다. 여기서는 비율을 훑어 그 열을
곡선으로 바꾼다.

재는 것은 같은 격자 위의 두 보간 방식 차이다.

    tq = t0, t0 + dt/r, t0 + 2*dt/r, ...        r = 기준주기 / 추정기주기
    D(r) = sum w * d( slerp(tq), nearest(tq) )

질의 시각은 dt/r 간격이라, 1/r 이 정수면(0.25·0.5·1.0) 모든 질의가 공개 시각에
그대로 얹힌다. 그 자리에서는 가장 가까운 표본이 정확한 값이라 D 가 0 으로 파인다.
r 이 정수면 그중 1/r 만 얹히므로 부분적으로만 파인다. 곡선이 단조가 아니라는 뜻이고,
평가 코드가 고르는 주기가 공개 주기와 약분되느냐가 점수를 가른다. 고른 상수가 아니라
측정된 성질이다.

시퀀스는 릴리스마다 이름 순 앞쪽 몇 개만 쓴다. 비율 수만큼 되풀이하는 계산이라
전수로 돌릴 값이 아니다 — 모양을 보는 측정이다.
"""

import argparse
import csv
import os

import numpy as np

from reference_resolution_audit import gather
from release_resample_D import pairs_of
from resample_sensitivity import displacement, relative, resample, weights

MAX_SAMPLES = 20000
MIN_PAIRS = 32


def sweep_one(t, rot, ratios):
    """한 파일에서 비율마다 D(r). (dict, fs) 반환, 못 재면 (None, nan)."""
    t = np.asarray(t, dtype=float)
    rot = np.asarray(rot, dtype=float)
    if len(t) > MAX_SAMPLES:
        t, rot = t[:MAX_SAMPLES], rot[:MAX_SAMPLES]
    if len(t) < 64:
        return None, float("nan")
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        return None, float("nan")

    out = {}
    for r in ratios:
        tq = np.arange(t[0], t[-1], dt / r)
        if len(tq) < MIN_PAIRS:
            continue
        pr = pairs_of(len(tq))
        q = relative(resample(t, rot, tq, "slerp"), pr)
        qp = relative(resample(t, rot, tq, "nearest"), pr)
        out[float(r)] = displacement(q, qp, weights(len(q)))
    if not out:
        return None, float("nan")
    return out, 1.0 / dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--helipr-root", default="/mnt/Data/velref/helipr")
    ap.add_argument("--boreas-root", default="data/boreas")
    ap.add_argument("--kitti-root", default="/mnt/Data/velref/kitti/dataset")
    ap.add_argument("--kitti360-root", default="/mnt/Data/velref/kitti360")
    ap.add_argument("--pit30m-root", default="/mnt/Data/velref/pit30m")
    ap.add_argument("--nuscenes-can", default="/mnt/Data/velref/nuscenes/can_bus")
    ap.add_argument("--nuscenes-limit", type=int, default=0)
    ap.add_argument("--per-release", type=int, default=4,
                    help="릴리스마다 쓸 시퀀스 수(이름 순 앞쪽)")
    ap.add_argument("--out", default="results/next/interp_rate_sweep.csv")
    args = ap.parse_args()

    ratios = np.round(np.arange(0.25, 2.5001, 0.025), 4)

    todo = {}
    for release, name, path, loader in gather(args):
        todo.setdefault(release, []).append((name, path, loader))

    rows = []
    for release in sorted(todo):
        for name, path, loader in sorted(todo[release])[:args.per_release]:
            try:
                t, rot, _ = loader(path)
            except Exception as e:
                print("[%s] %s 읽기 실패: %s" % (release, name, e))
                continue
            d, fs = sweep_one(t, rot, ratios)
            if d is None:
                continue
            for r, v in d.items():
                rows.append({
                    "release": release, "sequence": name,
                    "fs_hz": round(fs, 3), "rate_ratio": r,
                    "D_rad": "%.6g" % v,
                    "D_mdeg": "%.4g" % (np.degrees(v) * 1000.0),
                })
            at = {r: d[r] for r in (0.5, 1.0, 2.0) if r in d}
            print("[%s] %-24s fs=%7.2f  %s" % (
                release, name, fs,
                "  ".join("D(%.1f)=%.3g mdeg" % (r, np.degrees(v) * 1000.0)
                          for r, v in sorted(at.items()))))

    if not rows:
        print("잴 것이 없다")
        return
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 약분되는 자리(1/r 이 정수)에서 실제로 파이는지, 그 사이가 얼마나 높은지.
    print()
    head = "%-15s %5s %-20s %-20s %-20s %s" % (
        "릴리스", "n", "약분자리(mdeg)", "그 사이 중앙(mdeg)", "최대(mdeg)", "사이/약분")
    print(head)
    print("-" * len(head))
    by = {}
    for r in rows:
        by.setdefault(r["release"], []).append(r)
    for release, rs in sorted(by.items()):
        exact, rest = [], []
        for r in rs:
            inv = 1.0 / r["rate_ratio"]
            (exact if abs(inv - round(inv)) < 1e-6 else rest).append(
                float(r["D_mdeg"]))
        if not exact or not rest:
            continue
        exact.sort()
        rest.sort()
        me, mr = exact[len(exact) // 2], rest[len(rest) // 2]
        print("%-15s %5d %-20s %-20s %-20s %s" % (
            release, len(rs),
            "%.3g (%.3g~%.3g)" % (me, exact[0], exact[-1]),
            "%.3g (%.3g~%.3g)" % (mr, rest[0], rest[-1]),
            "%.3g" % rest[-1],
            "%.3g" % (mr / me) if me else "무한"))
    print("\n%d 행 -> %s" % (len(rows), args.out))


if __name__ == "__main__":
    main()
