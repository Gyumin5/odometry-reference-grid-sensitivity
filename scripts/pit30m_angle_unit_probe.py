#!/usr/bin/env python3
"""핏서티엠 자세의 각도 단위를 그 릴리스의 다른 칸으로 가린다.

공개 묶음 all_poses.npz 의 dtype 에는 단위 표기가 없고, 공식 적재기
(pit30m/data/log_reader.py)도 yaw·pitch·roll 을 변환 없이 그대로 넘긴다 —
두 곳 어디에도 도인지 라디안인지가 안 적혀 있다. 그래서 값의 크기로 넘겨짚는
대신 같은 레코드의 독립된 칸인 위치(x, y, z)로 가린다.

자세의 앞축과 실제 이동 방향은 달리는 동안 거의 같은 쪽을 가리킨다. 각도를
도로 읽으면 그 둘이 몇 분의 일 도 안에서 맞고, 라디안으로 읽으면 수십 도로
벌어진다. 이 판정은 단위가 없는 양(방향 사이의 각)만 쓰므로 위치 칸의 단위나
좌표판 원점에 기대지 않는다.

가려지는 것은 앞축을 세우는 yaw 와 pitch 다. roll 은 궤적만으로는 관측되지
않아 같은 레코드에 나란히 실린 값이라는 이유로 같은 단위로 읽는다.
"""

import argparse
import csv
import glob
import os

import numpy as np
from scipy.spatial.transform import Rotation

FAST = 3.0
DEGREE = np.pi / 180.0
RADIAN = 1.0


def rot_from_fields(ypr, scale):
    """저장된 (yaw, pitch, roll) 묶음을 주어진 환산으로 읽어 회전으로."""
    return Rotation.from_euler("ZYX", np.asarray(ypr, dtype=float) * scale
                               ).as_matrix()


def forward_axis_error(t, rot, pos, fast=FAST):
    """달리는 구간에서 자세 앞축과 이동 방향이 벌어진 각(라디안)."""
    t = np.asarray(t, dtype=float)
    pos = np.asarray(pos, dtype=float)
    rot = np.asarray(rot, dtype=float)
    if len(t) < 2:
        return np.empty(0)
    dt = np.diff(t)
    dp = np.diff(pos, axis=0)
    n = np.linalg.norm(dp, axis=1)
    ok = (dt > 0) & (n > 0)
    speed = np.zeros_like(n)
    speed[ok] = n[ok] / dt[ok]
    take = ok & (speed > fast)
    if not take.any():
        return np.empty(0)
    d = dp[take] / n[take][:, None]
    f = rot[:-1][take][:, :, 0]
    return np.arccos(np.clip((d * f).sum(axis=1), -1.0, 1.0))


def read_fields(path):
    """valid 가 선 구간의 (시각, (yaw, pitch, roll), 위치)."""
    d = np.load(path, allow_pickle=True)["data"]
    d = d[d["continuous"]["valid"] & d["poses_and_differentials_valid"]]
    c = d["continuous"]
    ypr = np.stack([c["yaw"].astype(float), c["pitch"].astype(float),
                    c["roll"].astype(float)], axis=1)
    pos = np.stack([c["x"].astype(float), c["y"].astype(float),
                    c["z"].astype(float)], axis=1)
    return d["capture_time"].astype(float), ypr, pos


def verdict(deg_median, rad_median):
    """어느 읽기가 궤적과 맞는지. 둘 다 멀면 판정하지 않는다."""
    if deg_median < 5.0 and deg_median < rad_median:
        return "degree"
    if rad_median < 5.0 and rad_median < deg_median:
        return "radian"
    return "undecided"


def empty_reason(n_samples, n_steps):
    """못 가린 칸이 어느 쪽으로 빈 것인지. 둘을 한 칸으로 묶지 않는다."""
    if n_steps:
        return ""
    if n_samples < 2:
        return "자세가 한 점뿐이라 이동 방향이 없다"
    return "임계보다 빠른 걸음이 없다"


def row_for(name, t, ypr, pos, fast=FAST):
    out = {"log": name, "n_samples": len(t)}
    med = {}
    for key, scale in (("degree", DEGREE), ("radian", RADIAN)):
        e = np.degrees(forward_axis_error(t, rot_from_fields(ypr, scale), pos,
                                          fast))
        out["n_steps"] = len(e)
        med[key] = float(np.median(e)) if len(e) else float("nan")
        out["median_deg_as_%s" % key] = "%.6f" % med[key]
        out["p90_deg_as_%s" % key] = ("%.6f" % np.percentile(e, 90)
                                      if len(e) else "nan")
    out["verdict"] = (verdict(med["degree"], med["radian"]) if out["n_steps"]
                      else "undecided")
    out["empty_reason"] = empty_reason(out["n_samples"], out["n_steps"])
    return out


def rows_for(root, fast=FAST):
    rows = []
    for p in sorted(glob.glob(os.path.join(root, "*", "all_poses.npz"))):
        if os.path.getsize(p) == 0:
            continue
        name = os.path.basename(os.path.dirname(p))[:8]
        t, ypr, pos = read_fields(p)
        rows.append(row_for(name, t, ypr, pos, fast))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pit30m-root", default="/mnt/Data/velref/pit30m")
    ap.add_argument("--out", default="results/next/pit30m_angle_unit.csv")
    args = ap.parse_args()

    rows = rows_for(args.pit30m_root)
    if not rows:
        raise SystemExit("읽은 기록이 없다: %s" % args.pit30m_root)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    for r in rows:
        print("[%s] %7d 걸음  도 %.4f°  라디안 %.4f°  판정 %s %s"
              % (r["log"], r["n_steps"], float(r["median_deg_as_degree"]),
                 float(r["median_deg_as_radian"]), r["verdict"],
                 r["empty_reason"]))
    judged = [r for r in rows if r["n_steps"]]
    votes = {r["verdict"] for r in judged}
    worst = max(float(r["median_deg_as_degree"]) for r in judged)
    print("기록 %d 가운데 가린 것 %d, 판정 %s, 도로 읽었을 때 가장 큰 중앙값 %.4f°"
          % (len(rows), len(judged), "/".join(sorted(votes)), worst))


if __name__ == "__main__":
    main()
