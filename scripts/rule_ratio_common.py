#!/usr/bin/env python3
"""두 구간 규칙 사이의 배율을 같은 시퀀스 위에서 다시 잰다.

릴리스 상한 표의 묶음 행은 규칙마다 자기가 설 수 있는 시퀀스를 전부 쓴다.
그래서 두 규칙의 묶음 값을 바로 나누면 구간 길이의 효과에 모집단 변화가
섞인다 — 100~800 m 가 안 서는 짧은 주행이 1 m 쪽에만 들어오기 때문이다
(누씬즈 527 대 910, 유로크 2 대 12, 튬 브이아이 5 대 7).

여기서는 릴리스마다 두 규칙에 다 있는 시퀀스만 남겨 각 규칙을 그 시퀀스
위에서 다시 묶고(구간 수 가중, 공식 점수와 같은 꼴) 배율을 낸다. 두 배율을
나란히 적어 두면 섞인 몫이 얼마인지가 표 안에서 읽힌다.
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kitti_segment_D import RELEASES, exact  # noqa: E402

POOLED = "ALL"
KEY = "twoD_t_phase_half_pct"


def read(path):
    """(묶음 값, {시퀀스: (값, 구간 수)}). 파일이 없으면 (None, {})."""
    if not os.path.exists(path):
        return None, {}
    pooled, per = None, {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r[KEY] in ("", "nan"):
                continue
            if r["sequence"] == POOLED:
                pooled = float(r[KEY])
            else:
                per[r["sequence"]] = (float(r[KEY]), int(r["n_segments"]))
    return pooled, per


def pool(per, names):
    """구간 수로 가중한 평균. 묶음 행이 쓰는 꼴 그대로다."""
    num = sum(per[n][0] * per[n][1] for n in names)
    den = sum(per[n][1] for n in names)
    return num / den if den else None


def collect(root, releases=RELEASES):
    out = []
    for rel in releases:
        seg_all, seg = read(os.path.join(root, "%s_segment_D.csv" % rel))
        rpe_all, rpe = read(os.path.join(root, "%s_rpe1m_D.csv" % rel))
        if seg_all is None or rpe_all is None or not (seg and rpe):
            continue
        both = sorted(set(seg) & set(rpe))
        if not both:
            continue
        a, b = pool(seg, both), pool(rpe, both)
        out.append({
            "release": rel,
            "n_seg_rule": str(len(seg)),
            "n_rpe_rule": str(len(rpe)),
            "n_common": str(len(both)),
            "ratio_all": exact(rpe_all / seg_all),
            "ratio_common": exact(b / a),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/next")
    ap.add_argument("--out", default="results/next/rule_ratio_common.csv")
    args = ap.parse_args()

    rows = collect(args.root)
    if not rows:
        print("잴 것이 없다 — kitti_segment_D.py 를 먼저 돌려라")
        return

    print("%-10s %8s %8s %8s %10s %10s"
          % ("릴리스", "100-800m", "1m", "공통", "배율(전체)", "배율(공통)"))
    for r in rows:
        print("%-10s %8s %8s %8s %10.4g %10.4g"
              % (r["release"], r["n_seg_rule"], r["n_rpe_rule"],
                 r["n_common"], float(r["ratio_all"]),
                 float(r["ratio_common"])))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\n%d 행 -> %s" % (len(rows), args.out))


if __name__ == "__main__":
    main()
