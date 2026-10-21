#!/usr/bin/env python3
"""한 공개 파일 안에서 자세와 각속도 열이 서로 맞는지 본다.

여러 릴리스가 같은 파일에 자세와 각속도를 같이 싣는다. 둘은 같은 해의 두 표현이므로
자세를 미분한 값과 실린 각속도가 축 규약을 맞춘 뒤 일치해야 한다. 여기서 재는 것은
그 일치도 하나다 — 바깥 자료도 다른 센서도 안 쓴다. 모형은 2단계와 같다.

    w_pose(t) = C [ w_rate(t + tau) - b ]

tau 는 자세 쪽 시각에 더해 짝이 되는 각속도 표본을 찾는 양이다 — 각속도 열이 d 만큼
이르게 찍혀 있으면 tau = -d 로 나온다. tau 를 같이 훑는 이유는 두 열이 같은 파일 안에서 시각이 밀려 실리는 경우가 있기
때문이다. 같은 파일 안의 두 열 사이에 0 이 아닌 tau 가 나오면 그 자체가 결함이다.

종속변수 둘. 잔차 대 신호 비(맞춘 뒤 남는 양)와 tau(ms).
"""

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reference_resolution_audit import (  # noqa: E402
    load_boreas_gps_post,
    load_boreas_lidar_poses,
    load_nuscenes_pose,
    load_pit30m_poses,
)
from reference_timing_audit import (  # noqa: E402
    best_tau,
    fit_at_tau,
    rate_from_attitude,
)


# ------------------------------------------------- 실린 각속도 열 읽기

def rate_pit30m(path):
    """continuous 의 roll_rate/pitch_rate/yaw_rate. 자세와 같은 행에 실린다."""
    d = np.load(path, allow_pickle=True)["data"]
    d = d[d["continuous"]["valid"] & d["poses_and_differentials_valid"]]
    c = d["continuous"]
    return (d["capture_time"].astype(float),
            np.stack([c["roll_rate"].astype(float),
                      c["pitch_rate"].astype(float),
                      c["yaw_rate"].astype(float)], axis=1))


def rate_nuscenes(path):
    rows = json.load(open(path))
    return (np.array([r["utime"] * 1e-6 for r in rows]),
            np.array([r["rotation_rate"] for r in rows]))


def rate_boreas(path):
    t, g = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            t.append(float(row["GPSTime"]))
            g.append([float(row["angvel_x"]), float(row["angvel_y"]),
                      float(row["angvel_z"])])
    t = np.array(t)
    if t[0] > 1e12:          # lidar_poses 는 마이크로초로 실린다
        t = t * 1e-6
    return t, np.array(g)


def gather(args):
    jobs = []
    if os.path.isdir(args.pit30m_root):
        for log in sorted(os.listdir(args.pit30m_root)):
            p = os.path.join(args.pit30m_root, log, "all_poses.npz")
            if os.path.exists(p) and os.path.getsize(p) > 0:
                jobs.append(("pit30m", log[:8], p, load_pit30m_poses, rate_pit30m))
    if os.path.isdir(args.boreas_root):
        for seq in sorted(os.listdir(args.boreas_root)):
            for fn, ld in (("gps_post_process.csv", load_boreas_gps_post),
                           ("lidar_poses.csv", load_boreas_lidar_poses)):
                p = os.path.join(args.boreas_root, seq, "applanix", fn)
                if os.path.exists(p):
                    jobs.append(("boreas/" + fn[:-4], seq, p, ld, rate_boreas))
    if os.path.isdir(args.nuscenes_can):
        paths = sorted(glob.glob(os.path.join(args.nuscenes_can, "scene-*_pose.json")))
        if args.nuscenes_limit:
            paths = paths[:args.nuscenes_limit]
        for p in paths:
            jobs.append(("nuscenes", os.path.basename(p).split("_")[0], p,
                         load_nuscenes_pose, rate_nuscenes))
    return jobs


def compare(t_pose, rot, t_rate, w_rate, args):
    if len(t_pose) < 258 or len(t_rate) < 2:
        return None
    tc, w_pose = rate_from_attitude(t_pose, rot)
    lo = max(tc[0], t_rate[0] + abs(args.tau_max))
    hi = min(tc[-1], t_rate[-1] - abs(args.tau_max))
    m = (tc >= lo) & (tc <= hi)
    if m.sum() < 256:
        return None
    tc, w_pose = tc[m], w_pose[m]
    tau, _, _ = best_tau(tc, w_pose, t_rate, w_rate, -args.tau_max,
                         args.tau_max, args.tau_step)
    res, _ = fit_at_tau(tc, w_pose, t_rate, w_rate, tau)
    rms = float(np.sqrt((res ** 2).mean()))
    amp = float(np.sqrt((w_pose ** 2).mean()))
    ref = np.stack([np.interp(tc + tau, t_rate, w_rate[:, k]) for k in range(3)], axis=1)
    out = {
        "n": int(m.sum()),
        "span_s": round(float(tc[-1] - tc[0]), 1),
        "tau_ms": round(tau * 1000, 3),
        "residual_frac": "%.4g" % (rms / amp if amp else float("nan")),
        "pose_rms_radps": "%.4g" % amp,
        "rate_rms_radps": "%.4g" % float(np.sqrt((ref ** 2).mean())),
    }
    # 축마다 따로 본다. 지상 차량은 요 축만 신호를 싣고 롤·피치 축의 자세 미분은
    # 대개 1단계에서 잰 잡음 바닥이라, 그 축의 상관이 낮은 것은 결함이 아니라
    # 바닥이 맞다는 확인이다. 분산이 0 인 축(평면 자세)은 상관을 못 낸다.
    for k, ax in enumerate("xyz"):
        a, b = w_pose[:, k], ref[:, k]
        out["pose_rms_" + ax] = "%.4g" % float(np.sqrt((a ** 2).mean()))
        ok = a.std() > 0 and b.std() > 0
        out["corr_" + ax] = "%.4f" % float(np.corrcoef(a, b)[0, 1]) if ok else ""
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pit30m-root", default="/mnt/Data/velref/pit30m")
    ap.add_argument("--boreas-root", default="data/boreas")
    ap.add_argument("--nuscenes-can", default="/mnt/Data/velref/nuscenes/can_bus")
    ap.add_argument("--nuscenes-limit", type=int, default=30)
    ap.add_argument("--tau-max", type=float, default=0.2)
    ap.add_argument("--tau-step", type=float, default=0.002)
    ap.add_argument("--out", default="results/next/rate_channel_consistency.csv")
    args = ap.parse_args()

    rows = []
    for release, name, path, pose_loader, rate_loader in gather(args):
        try:
            t_pose, rot, _ = pose_loader(path)
            t_rate, w_rate = rate_loader(path)
        except Exception as e:
            print("[%s] %s 읽기 실패: %s" % (release, name, e))
            continue
        r = compare(t_pose, rot, t_rate, w_rate, args)
        if not r:
            continue
        r["release"], r["sequence"] = release, name
        rows.append(r)
        print("%-22s %-12s tau %7.1f ms  잔차/신호 %7s  상관 %s %s %s"
              % (release, name, r["tau_ms"], r["residual_frac"],
                 r["corr_x"], r["corr_y"], r["corr_z"]))

    if not rows:
        print("맞댈 것이 없다")
        return
    keys = ["release", "sequence"] + [k for k in rows[0] if k not in ("release", "sequence")]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("%d 건 -> %s" % (len(rows), args.out))

    by = {}
    for r in rows:
        by.setdefault(r["release"], []).append(float(r["residual_frac"]))
    for rel, v in sorted(by.items()):
        v = sorted(v)
        print("%-22s n=%-4d 잔차/신호 중앙 %.3f  (%.3f~%.3f)"
              % (rel, len(v), v[len(v) // 2], v[0], v[-1]))


if __name__ == "__main__":
    main()
