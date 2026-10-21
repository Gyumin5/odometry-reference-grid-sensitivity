#!/usr/bin/env python3
"""규약 훑기 열두 장에서 묶음 줄만 뽑아 한 장으로 세운다.

scripts/protocol_sweep.py 가 릴리스·규칙마다 한 장씩 남기는데, 거기서 쓸 것은
시퀀스를 다 합친 ALL 줄 하나다. 그 줄을 릴리스 여섯 × 규칙 둘 × 변형 스물넷으로
길게 편다 — 변형 이름을 가로로 두면 칸이 마흔여덟이라 사람이 못 읽고, 변형이
하나 늘 때마다 표 머리가 바뀐다.

변형 이름은 네 갈래로 쪼개 적는다(axis, param). 위상(shift)·솎기(decim)·
주기(rate)·절대 어긋남(tau) 이고, 주기만 보간 방식(method)이 따로 붙는다.
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kitti_segment_D import RELEASES  # noqa: E402

RULES = (("seg", "segment"), ("rpe1m", "rpe1m"))
POOLED = "ALL"
SUFFIXES = (("_deg_per_100m", "rot"), ("_pct", "trans"))


def split_variant(name):
    """D_shift_0.5_pct -> ('shift', '0.5'). 축과 값을 갈라 둔다."""
    body = name[len("D_"):]
    for suffix, _ in SUFFIXES:
        if body.endswith(suffix):
            body = body[: -len(suffix)]
            break
    axis, _, param = body.partition("_")
    return axis, param


def pooled_row(path):
    """그 장의 ALL 줄. 장이 없거나 ALL 이 없으면 None."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get("sequence") == POOLED:
                return row
    return None


def flatten(release, rule, row):
    """묶음 줄 하나를 변형당 한 줄로 편다."""
    pairs = {}
    for key, value in row.items():
        if not key.startswith("D_"):
            continue
        for suffix, unit in SUFFIXES:
            if key.endswith(suffix):
                axis, param = split_variant(key)
                pairs.setdefault((axis, param), {})[unit] = value
                break
    out = []
    for (axis, param), units in sorted(pairs.items()):
        out.append({
            "release": release, "rule": rule, "axis": axis, "param": param,
            "n_segments": row["n_segments"],
            "D_rot_deg_per_100m": units.get("rot", ""),
            "D_trans_pct": units.get("trans", ""),
        })
    return out


def collect(root, releases=RELEASES, rules=RULES):
    out = []
    for release in releases:
        for tag, stem in rules:
            row = pooled_row(os.path.join(
                root, "%s_%s_sweep.csv" % (release, stem)))
            if row is None:
                continue
            out += flatten(release, tag, row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/next")
    ap.add_argument("--out", default="results/next/protocol_sweep_table.csv")
    args = ap.parse_args()

    rows = collect(args.root)
    if not rows:
        raise SystemExit("잴 것이 없다 — 훑기 장을 먼저 만들어라")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("%d 줄 -> %s" % (len(rows), args.out))


if __name__ == "__main__":
    main()
