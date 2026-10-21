#!/usr/bin/env python3
"""릴리스 하나의 스펙트럼을 전 시퀀스에 걸쳐 묶어 대역 모양을 판정한다.

시퀀스 한 건으로 백색인지 아닌지를 가르는 것은 길이가 짧으면 성립하지 않는다.
누씬즈 한 장면은 20 초라 창 하나가 겨우 들어가고, 그러면 대역의 각 칸이 자유도
둘짜리라 기울기가 씨앗마다 -2 에서 +2 까지 흩어진다. 실제로 통제 실험에서 참으로
백색인 조건이 그 길이에서 200 번 중 101 번만 백색으로 판정됐고, 두 표본 이동평균을
먹인 조건도 98 번 백색으로 판정됐다 — 둘이 구별이 안 된다.

그래서 판정을 시퀀스마다 하지 않고 릴리스마다 한 번 한다. 시퀀스마다 제 스펙트럼을
구한 뒤 평균하고, 그 평균 위에서 기울기와 굽힘을 잰다. 불확실성은 시퀀스를 복원
추출해 다시 평균하는 방식으로 잡는다 — 자유도를 늘려 준 것이 시퀀스 수이므로
흔들림도 시퀀스 단위로 재야 한다.

출력은 릴리스마다 한 줄이다. 기울기와 굽힘의 신뢰구간이 0 을 품는지가 판정이고,
"백색이다" 라고 말할 수 있는 것은 기울기 구간이 0 을 품고 굽힘 구간도 0 을 품을
때뿐이다.
"""

import argparse
import csv
import os

import numpy as np

from reference_resolution_audit import gather, rate_segments, welch_psd_segments

MAX_SEQ_SAMPLES = 200000
BOOT = 400


def sequence_spectrum(t, rot, nseg):
    """한 시퀀스의 축 합산 스펙트럼. (freqs, s_tot) 또는 (None, None)."""
    segs, _, fs, _, _ = rate_segments(t, rot)
    if not segs:
        return None, None
    total, freqs = None, None
    for k in range(3):
        if all(np.allclose(s[:, k], 0.0) for s in segs):
            continue  # 고정된 축. 잡음이 아니라 내용이 없다.
        f, s = welch_psd_segments(segs, k, fs, nseg)
        if len(f) == 0:
            continue
        freqs = f
        total = s if total is None else total + s
    return freqs, total


def fit(freqs, s, band):
    lf, ls = np.log(freqs[band]), np.log(s[band])
    slope = float(np.polyfit(lf, ls, 1)[0])
    curv = float(np.polyfit(lf, ls, 2)[0])
    return slope, curv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--helipr-root", default="/mnt/Data/velref/helipr")
    ap.add_argument("--boreas-root", default="data/boreas")
    ap.add_argument("--kitti-root", default="/mnt/Data/velref/kitti/dataset")
    ap.add_argument("--kitti360-root", default="/mnt/Data/velref/kitti360")
    ap.add_argument("--pit30m-root", default="/mnt/Data/velref/pit30m")
    ap.add_argument("--nuscenes-can", default="/mnt/Data/velref/nuscenes/can_bus")
    ap.add_argument("--nuscenes-limit", type=int, default=0)
    ap.add_argument("--band", default="2,4")
    ap.add_argument("--win-s", type=float, default=16.0)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--out", default="results/next/pooled_band_fit.csv")
    args = ap.parse_args()
    band_lo, band_hi = (float(v) for v in args.band.split(","))

    jobs = {}
    for release, name, path, loader in gather(args):
        jobs.setdefault(release, []).append((name, path, loader))

    rng = np.random.default_rng(args.seed)
    rows = []
    for release, items in sorted(jobs.items()):
        spectra, freqs, rates, nseg = [], None, [], 0
        for name, path, loader in items:
            try:
                t, rot, _ = loader(path)
            except Exception as e:
                print("[%s] %s 읽기 실패: %s" % (release, name, e))
                continue
            t, rot = np.asarray(t, float), np.asarray(rot, float)
            if len(t) > MAX_SEQ_SAMPLES:
                t, rot = t[:MAX_SEQ_SAMPLES], rot[:MAX_SEQ_SAMPLES]
            if len(t) < 64:
                continue
            fs = 1.0 / float(np.median(np.diff(t)))
            if not nseg:
                # 창 길이는 릴리스 안에서 하나로 고정한다. 시퀀스마다 다시 정하면
                # 주파수 격자가 달라져 평균 자체가 뜻을 잃는다.
                nseg = max(16, int(round(args.win_s * fs)))
            rates.append(fs)
            f, s = sequence_spectrum(t, rot, nseg)
            if f is None or s is None:
                continue
            if freqs is None:
                freqs = f
            if len(f) != len(freqs):
                continue
            spectra.append(s)
        if not spectra or freqs is None:
            continue

        sp = np.asarray(spectra)
        band = (freqs >= band_lo) & (freqs <= min(band_hi, np.median(rates) / 2))
        if band.sum() < 5:
            print("[%s] 대역 칸이 모자라다" % release)
            continue
        slope, curv = fit(freqs, sp.mean(axis=0), band)

        bs, bc = [], []
        for _ in range(BOOT):
            pick = rng.integers(0, len(sp), len(sp))
            a, b = fit(freqs, sp[pick].mean(axis=0), band)
            bs.append(a)
            bc.append(b)
        bs, bc = np.sort(bs), np.sort(bc)
        lo, hi = int(0.025 * BOOT), int(0.975 * BOOT)
        white = bs[lo] <= 0 <= bs[hi] and bc[lo] <= 0 <= bc[hi]
        rows.append({
            "release": release, "n_sequences": len(sp),
            "fs_hz": round(float(np.median(rates)), 3),
            "band_bins": int(band.sum()),
            "slope": round(slope, 3),
            "slope_lo": round(float(bs[lo]), 3), "slope_hi": round(float(bs[hi]), 3),
            "curv": round(curv, 3),
            "curv_lo": round(float(bc[lo]), 3), "curv_hi": round(float(bc[hi]), 3),
            "white": "백색" if white else "아님",
        })
        print("%-15s n=%4d  기울기 %7.3f [%7.3f,%7.3f]  굽힘 %7.3f [%7.3f,%7.3f]  %s"
              % (release, len(sp), slope, bs[lo], bs[hi], curv, bc[lo], bc[hi],
                 "백색" if white else "아님"))

    if not rows:
        print("잴 것이 없다")
        return
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\n%d 릴리스 -> %s" % (len(rows), args.out))


if __name__ == "__main__":
    main()
