#!/usr/bin/env python3
"""공개 기준 자세와 따로 실린 자이로 사이의 시각 어긋남을 잰다.

2단계 측정이다. 독립 관성 장치를 같이 공개한 릴리스에서만 쓴다.

기준 자세에서 얻은 각속도 w_ref 와 자이로 각속도 w_gyro 는 같은 운동을 보지만
좌표축이 다르고(C), 치우침이 있고(b), 시각이 어긋나 있다(tau). 모형은

    w_ref(t) = C [ w_gyro(t + tau) - b ]

이고, tau 격자마다 C 와 b 를 최소제곱으로 풀어 잔차를 재고 가장 작은 곳을 고른다.
tau 는 기준 쪽 시각에 더해 짝이 되는 자이로 표본을 찾는 양이다 — 자이로가 d 만큼
이르게 찍혀 있으면 tau = -d 로 나온다.
구간 부트스트랩으로 tau 의 95% 구간을 얻고, 그 폭(ms)을 종속변수로 쓴다.

C 를 미리 주지 않는 이유는 외부 보정값이 바로 그 공개본이 주장하는 값이고,
여기서 재려는 것은 공개본 자신의 정합성이기 때문이다.
"""

import argparse
import csv
import glob
import json
import math
import os

import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reference_resolution_audit import (  # noqa: E402
    load_boreas_lidar_poses,
    load_nuscenes_pose,
    load_tum,
    so3_log,
)


# ---------------------------------------------------------------- 자이로 읽기

def gyro_helipr_xsens(path):
    """ts_ns, q(4), euler(3), gyro(3), accel(3), mag(3)"""
    t, g = [], []
    with open(path) as f:
        for line in f:
            c = line.strip().split(",")
            if len(c) < 14:
                continue
            t.append(float(c[0]) * 1e-9)
            g.append([float(c[8]), float(c[9]), float(c[10])])
    return np.array(t), np.array(g)


def gyro_boreas_raw(path):
    """GPSTime(s), angvel_z, angvel_y, angvel_x, accel..."""
    t, g = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            t.append(float(row["GPSTime"]))
            g.append([float(row["angvel_x"]), float(row["angvel_y"]), float(row["angvel_z"])])
    return np.array(t), np.array(g)


def gyro_nuscenes(path):
    with open(path) as f:
        rows = json.load(f)
    t = np.array([r["utime"] * 1e-6 for r in rows])
    g = np.array([r["rotation_rate"] for r in rows])
    return t, g


# ---------------------------------------------------------------- 맞추기

def rate_from_attitude(t, rot):
    dt = np.diff(t)
    inc = np.einsum("nji,njk->nik", rot[:-1], rot[1:])
    return t[:-1] + dt / 2, so3_log(inc) / dt[:, None]


def fit_at_tau(tc, w_ref, tg, w_gyro, tau):
    """시각을 tau 만큼 밀어 맞춘 뒤 C 와 b 를 풀고 잔차 벡터를 돌려준다."""
    shifted = np.stack([np.interp(tc + tau, tg, w_gyro[:, k]) for k in range(3)], axis=1)
    b = shifted.mean(axis=0)
    a = shifted - b
    y = w_ref - w_ref.mean(axis=0)
    # 최적 회전 C: Procrustes. y ~ C a.
    u, _, vt = np.linalg.svd(y.T @ a)
    d = np.sign(np.linalg.det(u @ vt))
    c = u @ np.diag([1.0, 1.0, d]) @ vt
    res = y - a @ c.T
    return res, c


def best_tau(tc, w_ref, tg, w_gyro, lo, hi, step):
    taus = np.arange(lo, hi + step / 2, step)
    cost = np.empty(len(taus))
    for i, tau in enumerate(taus):
        res, _ = fit_at_tau(tc, w_ref, tg, w_gyro, tau)
        cost[i] = float((res ** 2).mean())
    i = int(np.argmin(cost))
    # 최소 주변 세 점의 포물선 꼭짓점으로 격자보다 가늘게 잡는다.
    if 0 < i < len(taus) - 1:
        y0, y1, y2 = cost[i - 1], cost[i], cost[i + 1]
        den = y0 - 2 * y1 + y2
        off = 0.5 * (y0 - y2) / den if den != 0 else 0.0
        return float(taus[i] + off * step), cost, taus
    return float(taus[i]), cost, taus


def bootstrap_tau(tc, w_ref, tg, w_gyro, lo, hi, step, block_s, draws, seed):
    """이동 구간 부트스트랩. 시계열이 상관돼 있으므로 표본 하나씩 뽑지 않는다.

    구간은 되돌려 놓고 뽑아(중복 허용) 원래 길이만큼 이어 붙인다. 뽑은 구간을
    합집합 마스크로 모으면 겹친 구간이 한 번만 남아서 회차마다 표본 수가 들쭉날쭉
    줄어든다 — 그러면 퍼짐이 원래 길이가 아니라 더 짧은 계열의 퍼짐이 되어
    구간 폭이 부풀어 오른다.
    """
    rng = np.random.default_rng(seed)
    n = len(tc)
    span = tc[-1] - tc[0]
    if span < 3 * block_s or n < 64:
        return None
    blk = max(8, int(round(block_s * (n - 1) / span)))
    if blk >= n:
        return None
    nblk = int(np.ceil(n / blk))
    out = []
    for _ in range(draws):
        starts = rng.integers(0, n - blk + 1, size=nblk)
        idx = np.concatenate([np.arange(s, s + blk) for s in starts])[:n]
        tau, _, _ = best_tau(tc[idx], w_ref[idx], tg, w_gyro, lo, hi, step)
        out.append(tau)
    return np.array(out) if out else None


# ---------------------------------------------------------------- 묶기

def run_pair(name, t_ref, rot_ref, tg, w_gyro, args):
    tc, w_ref = rate_from_attitude(t_ref, rot_ref)
    lo_t = max(tc[0], tg[0] + abs(args.tau_max))
    hi_t = min(tc[-1], tg[-1] - abs(args.tau_max))
    m = (tc >= lo_t) & (tc <= hi_t)
    if m.sum() < 128:
        return None
    tc, w_ref = tc[m], w_ref[m]

    tau, cost, taus = best_tau(tc, w_ref, tg, w_gyro, -args.tau_max, args.tau_max, args.tau_step)
    res, c = fit_at_tau(tc, w_ref, tg, w_gyro, tau)
    rms = float(np.sqrt((res ** 2).mean()))
    amp = float(np.sqrt((w_ref ** 2).mean()))

    row = {
        "pair": name,
        "n": int(m.sum()),
        "overlap_s": round(float(tc[-1] - tc[0]), 2),
        "tau_ms": round(tau * 1000, 3),
        "residual_rms_radps": "%.4g" % rms,
        "ref_rms_radps": "%.4g" % amp,
        "residual_frac": "%.4g" % (rms / amp if amp else float("nan")),
        "cost_min": "%.4g" % float(cost.min()),
        "cost_at_zero": "%.4g" % float(cost[int(np.argmin(np.abs(taus)))]),
        "axis_angle_deg": round(math.degrees(float(np.arccos(
            np.clip((np.trace(c) - 1) / 2, -1.0, 1.0)))), 3),
    }

    boot = bootstrap_tau(tc, w_ref, tg, w_gyro, -args.tau_max, args.tau_max,
                         args.tau_step, args.block_s, args.draws, args.seed)
    if boot is not None and len(boot) >= 20:
        lo, hi = np.percentile(boot, [2.5, 97.5])
        row["tau_lo_ms"] = round(float(lo) * 1000, 3)
        row["tau_hi_ms"] = round(float(hi) * 1000, 3)
        row["tau_ci95_width_ms"] = round(float(hi - lo) * 1000, 3)
        row["boot_draws"] = int(len(boot))
    return row


def gather(args):
    jobs = []
    if os.path.isdir(args.helipr_root):
        for seq in sorted(os.listdir(args.helipr_root)):
            imu = os.path.join(args.helipr_root, seq, "Inertial_data", "xsens_imu.csv")
            if not os.path.exists(imu):
                continue
            for gt in sorted(glob.glob(os.path.join(args.helipr_root, seq,
                                                    "LiDAR_GT", "*_gt.txt"))):
                sensor = os.path.basename(gt).replace("_gt.txt", "")
                if sensor.startswith("global_"):
                    # 전역 좌표판은 같은 궤적을 상수 회전만큼 돌린 것이라 회전 증분이
                    # 같다. 두 번 세지 않는다 — 두 판이 같은 값을 주는 것 자체가
                    # 이 측정이 전역 자세에 안 묶인다는 확인이다.
                    continue
                jobs.append(("%s/%s vs xsens" % (seq, sensor), gt, load_tum,
                             imu, gyro_helipr_xsens))
    if os.path.isdir(args.boreas_root):
        for seq in sorted(os.listdir(args.boreas_root)):
            ref = os.path.join(args.boreas_root, seq, "applanix", "lidar_poses.csv")
            imu = os.path.join(args.boreas_root, seq, "applanix", "imu_raw.csv")
            if os.path.exists(ref) and os.path.exists(imu):
                jobs.append(("%s/lidar_poses vs imu_raw" % seq, ref,
                             load_boreas_lidar_poses, imu, gyro_boreas_raw))
    if os.path.isdir(args.nuscenes_can):
        paths = sorted(glob.glob(os.path.join(args.nuscenes_can, "scene-*_pose.json")))
        if args.nuscenes_limit:
            paths = paths[:args.nuscenes_limit]
        for p in paths:
            imu = p.replace("_pose.json", "_ms_imu.json")
            if os.path.exists(imu):
                jobs.append((os.path.basename(p).split("_")[0] + " vs ms_imu", p,
                             load_nuscenes_pose, imu, gyro_nuscenes))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--helipr-root", default="/mnt/Data/velref/helipr")
    ap.add_argument("--boreas-root", default="data/boreas")
    ap.add_argument("--nuscenes-can", default="/mnt/Data/velref/nuscenes/can_bus")
    ap.add_argument("--nuscenes-limit", type=int, default=20)
    ap.add_argument("--tau-max", type=float, default=0.2, help="훑는 시각 어긋남 범위(s)")
    ap.add_argument("--tau-step", type=float, default=0.002)
    ap.add_argument("--block-s", type=float, default=10.0)
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--out", default="results/next/reference_timing.csv")
    args = ap.parse_args()

    rows = []
    for name, ref_path, ref_loader, imu_path, imu_loader in gather(args):
        try:
            t_ref, rot_ref, _ = ref_loader(ref_path)
            tg, wg = imu_loader(imu_path)
        except Exception as e:
            print("[%s] 읽기 실패: %s" % (name, e))
            continue
        row = run_pair(name, t_ref, rot_ref, tg, wg, args)
        if row:
            rows.append(row)
            print("%-44s tau %8.2f ms  95%% 폭 %8s ms  잔차 %7s (%s 배)"
                  % (name, row["tau_ms"], row.get("tau_ci95_width_ms", "-"),
                     row["residual_rms_radps"], row["residual_frac"]))

    if not rows:
        print("맞댈 쌍이 없다")
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("%d 쌍 -> %s" % (len(rows), args.out))


if __name__ == "__main__":
    main()
