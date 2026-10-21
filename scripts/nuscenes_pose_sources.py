#!/usr/bin/env python3
"""누씬즈가 같은 주행에 자세를 두 벌 싣는다 — 둘이 얼마나 다른지 잰다.

읽는 쪽에 선택이 둘이다. 차량통신망 확장팩의 scene-*_pose.json 은 50 Hz 로
위치와 자세를 싣고, 본 묶음의 ego_pose.json 은 각 센서 표본 시각마다 한 줄씩
싣는다(라이다 쪽만 추리면 20 Hz). 둘 다 그 릴리스가 공개한 자차 자세이고
어느 쪽이 기준이라고 적혀 있지 않다.

재는 것 둘이다. 하나는 같은 시각에서 두 벌이 서로 얼마나 벌어지나 — 이건
자세 두 벌만 있으면 바로 읽힌다. 다른 하나는 어느 쪽을 읽느냐에 따라 상한이
얼마나 갈리나 — 상한은 자세 한 벌에서 나오므로 두 벌이면 두 수가 나온다.

라이다 자세만 쓰는 이유는 ego_pose 가 센서마다 다른 시각에 찍혀서, 전부
합치면 한 플랫폼의 궤적이 아니라 여섯 궤적을 겹친 것이 되기 때문이다.

실행:
    PYTHONPATH=src python3 scripts/nuscenes_pose_sources.py
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kitti_segment_D import (exact, load_poses_nuscenes, measure,  # noqa: E402
                             proper)

META = "/mnt/Data/velref/nuscenes/meta_x/v1.0-trainval"
CAN = "/mnt/Data/velref/nuscenes/can_bus"
OUT = "results/next/nuscenes_pose_sources.csv"
RULES = (("seg", (100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0), 10),
         ("rpe1m", (1.0,), 1))
KEYS = ("phase_half", "t_phase_half")
# 두 벌을 같은 시각에서 맞댈 때 허용하는 어긋남. 50 Hz 와 20 Hz 라 둘 중
# 성긴 쪽의 반 주기(25 ms)를 넘겨 짝지으면 걸음 자체가 차이로 섞인다.
MATCH_S = 0.025
# 두 벌이 같은 길을 다른 시계로 적은 것인지 가르려고 한쪽을 밀어 본다.
# 두 벌의 성긴 주기(50 ms)의 네 배까지 5 ms 씩 민다.
SHIFTS = np.arange(-0.2, 0.2001, 0.005)
FIELDS = (("scene", "n_meta", "n_can", "hz_meta", "hz_can", "matched",
           "gap_median_m", "gap_p95_m", "gap_median_deg", "gap_p95_deg",
           "best_shift_s", "gap_median_m_shifted", "gap_median_deg_shifted")
          + tuple("%s_%s_%s" % (rule, k, src)
                  for rule, _, _ in RULES for k in KEYS
                  for src in ("meta", "can", "ratio")))


def stream(path):
    """줄바꿈으로 예쁘게 찍힌 JSON 배열을 한 원소씩 흘려 읽는다.

    누씬즈 본 묶음의 표는 한 파일이 1 기가가 넘어서 통째로 올리면 자리를
    몇 기가 먹는다. 원소 경계가 들여쓰기 없는 `}` 줄이라 그 줄에서 끊는다.

    바깥 괄호는 원소를 모으고 있지 않을 때만 버린다. 안쪽에도 `]` 로만 된
    줄이 있어서(rotation·translation 이 줄을 나눠 찍힌다) 무조건 버리면
    원소가 닫히지 않은 채로 넘어간다.
    """
    buf = []
    with open(path) as f:
        for line in f:
            s = line.rstrip("\n")
            if not buf and s in ("[", "]", ""):
                continue
            buf.append(s)
            if s in ("},", "}"):
                yield json.loads("".join(buf).rstrip(","))
                buf = []


def lidar_rows(meta):
    """장면 이름 -> 라이다 표본의 (시각, ego_pose 토큰) 목록."""
    scene_name = {s["token"]: s["name"] for s in stream(
        os.path.join(meta, "scene.json"))}
    sample_scene = {s["token"]: s["scene_token"] for s in stream(
        os.path.join(meta, "sample.json"))}
    want = {}
    for r in stream(os.path.join(meta, "sample_data.json")):
        if "/LIDAR_TOP/" not in r["filename"]:
            continue
        name = scene_name.get(sample_scene.get(r["sample_token"]))
        if name is None:
            continue
        want.setdefault(name, []).append((r["timestamp"] * 1e-6,
                                          r["ego_pose_token"]))
    return want


def meta_poses(meta, want):
    """ego_pose.json 에서 필요한 토큰만 골라 장면마다 궤적으로 세운다."""
    need = {tok for rows in want.values() for _, tok in rows}
    got = {}
    for r in stream(os.path.join(meta, "ego_pose.json")):
        if r["token"] in need:
            got[r["token"]] = (r["rotation"], r["translation"])
    out = {}
    for name, rows in want.items():
        rows = sorted(r for r in rows if r[1] in got)
        if len(rows) < 2:
            continue
        t = np.asarray([a for a, _ in rows])
        q = np.asarray([got[tok][0] for _, tok in rows])
        pos = np.asarray([got[tok][1] for _, tok in rows])
        rot = Rotation.from_quat(np.stack([q[:, 1], q[:, 2], q[:, 3],
                                           q[:, 0]], axis=1)).as_matrix()
        out[name] = (t, rot, pos)
    return out


def gap(a, b):
    """두 궤적을 가까운 시각끼리 짝지어 위치·자세 차이를 낸다."""
    ta, ra, pa = a
    tb, rb, pb = b
    j = np.clip(np.searchsorted(tb, ta), 1, len(tb) - 1)
    j = np.where(np.abs(tb[j - 1] - ta) <= np.abs(tb[j] - ta), j - 1, j)
    ok = np.abs(tb[j] - ta) <= MATCH_S
    if ok.sum() < 2:
        return 0, [float("nan")] * 4
    i, j = np.flatnonzero(ok), j[ok]
    d = np.linalg.norm(pa[i] - pb[j], axis=1)
    ang = Rotation.from_matrix(
        np.matmul(np.transpose(ra[i], (0, 2, 1)), rb[j])).magnitude()
    return int(ok.sum()), [float(np.median(d)), float(np.percentile(d, 95)),
                           float(np.degrees(np.median(ang))),
                           float(np.degrees(np.percentile(ang, 95)))]


def best_shift(a, b):
    """한쪽을 밀어 가며 위치 차이가 가장 작아지는 자리를 찾는다.

    두 벌이 같은 길을 다른 시계로 적은 것이면 어떤 밀기에서 차이가 뚝
    떨어진다. 밀어도 안 줄면 시계가 아니라 풀이가 다른 것이다.
    """
    ta, ra, pa = a
    out = (float("nan"), float("nan"), float("nan"))
    best = float("inf")
    for s in SHIFTS:
        n, g = gap((ta + s, ra, pa), b)
        if n >= 2 and g[0] < best:
            best, out = g[0], (float(s), g[0], g[2])
    return out


def rate(t):
    return 1.0 / float(np.median(np.diff(t))) if len(t) > 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default=META)
    ap.add_argument("--can-bus", default=CAN)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    want = lidar_rows(args.meta)
    print("라이다 표본이 있는 장면 %d" % len(want))
    metas = meta_poses(args.meta, want)
    print("자세가 선 장면 %d" % len(metas))

    rows = []
    for name in sorted(metas):
        p = os.path.join(args.can_bus, "%s_pose.json" % name)
        if not os.path.exists(p):
            continue
        tm, rm, pm = metas[name]
        rm = proper(rm)
        tc, rc, pc = load_poses_nuscenes(p)
        rc = proper(rc)
        n, g = gap((tm, rm, pm), (tc, rc, pc))
        row = {"scene": name, "n_meta": len(tm), "n_can": len(tc),
               "hz_meta": exact(rate(tm)), "hz_can": exact(rate(tc)),
               "matched": n}
        for key, v in zip(("gap_median_m", "gap_p95_m", "gap_median_deg",
                           "gap_p95_deg"), g):
            row[key] = exact(v)
        for key, v in zip(("best_shift_s", "gap_median_m_shifted",
                           "gap_median_deg_shifted"),
                          best_shift((tm, rm, pm), (tc, rc, pc))):
            row[key] = exact(v)
        for rule, lengths, step in RULES:
            a, _ = measure(tm, rm, pm, lengths=lengths, step=step)
            b, _ = measure(tc, rc, pc, lengths=lengths, step=step)
            for k in KEYS:
                av = a[k] if a and k in a else float("nan")
                bv = b[k] if b and k in b else float("nan")
                row["%s_%s_meta" % (rule, k)] = exact(av)
                row["%s_%s_can" % (rule, k)] = exact(bv)
                row["%s_%s_ratio" % (rule, k)] = exact(av / bv if bv
                                                       else float("nan"))
        rows.append(row)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(FIELDS))
        w.writeheader()
        w.writerows(rows)
    print("%d 장면 -> %s" % (len(rows), args.out))


if __name__ == "__main__":
    main()
