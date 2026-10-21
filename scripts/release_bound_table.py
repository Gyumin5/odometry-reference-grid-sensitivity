#!/usr/bin/env python3
"""릴리스마다 잰 상한을 사람들이 표에 적는 단위로 한 장에 모은다.

kitti_segment_D.py 는 릴리스 하나와 구간 규칙 하나마다 파일을 쓴다. 논문
표는 그것들을 가로질러 한 장이어야 한다 — 어느 릴리스를 쓰든 자기 표에서
두 칸을 빼 여기 적힌 2D 와 견주면 되게.

구간 규칙 둘을 나란히 둔다.
  seg   100~800 m 구간, 시작 프레임 열 칸마다. 키티 주행거리계 공식 규칙.
  rpe1m 1 m 간격, 모든 프레임이 시작점. 키티-360 공식 규칙이고, 상대오차를
        짧은 간격에서 재는 평가 코드들이 쓰는 꼴이다.

값은 묶음 행(ALL)에서 가져온다. 시퀀스별 값을 다시 평균하면 구간 수가 적은
시퀀스가 과하게 세어져 공식 점수의 꼴과 달라진다.

묶음 값 옆에 시퀀스별 범위를 같이 적는다. 한 수만 적으면 그 수가 어느 폭을
가운데로 접은 것인지 읽는 쪽이 알 길이 없다.
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kitti_segment_D import RELEASES, exact  # noqa: E402

RULES = (("seg", "segment", "100~800 m"), ("rpe1m", "rpe1m", "1 m"))
POOLED = "ALL"
PROTOCOL = "phase_half"


def read_rows(path):
    """(묶음 행, 시퀀스별 행들). 파일이 없으면 (None, [])."""
    if not os.path.exists(path):
        return None, []
    pooled, per_seq = None, []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["sequence"] == POOLED:
                pooled = r
            else:
                per_seq.append(r)
    return pooled, per_seq


def spread(rows, key):
    """그 열의 시퀀스별 최소·최대. 줄 수가 하나면 둘이 같다.

    깎지 않고 적는다 — 이 표는 되읽히는 중간물이고, 표시 반올림은 본문
    매크로를 만들 때 한 번만 한다.
    """
    vals = [float(r[key]) for r in rows if r[key] not in ("", "nan")]
    if not vals:
        return "", ""
    return exact(min(vals)), exact(max(vals))


def collect(root, releases=RELEASES, protocol=PROTOCOL):
    """(릴리스, 규칙) 마다 한 줄. 못 찾은 조합은 빼고 센다."""
    out = []
    for rel in releases:
        for tag, suffix, label in RULES:
            row, per_seq = read_rows(os.path.join(
                root, "%s_%s_D.csv" % (rel, suffix)))
            if row is None:
                continue
            # 훑기는 밀린 변형이 자료 밖으로 안 나가게 양 끝을 잘라 구간을
            # 짓는다. 그러고 나면 구간이 하나도 안 남는 시퀀스가 있어서 두
            # 표의 모집단이 갈린다 — 그 차이를 여기서 같이 적어 둔다.
            _, sweep_seq = read_rows(os.path.join(
                root, "%s_%s_sweep.csv" % (rel, suffix)))
            rot_lo, rot_hi = spread(
                per_seq, "twoD_%s_deg_per_100m" % protocol)
            trans_lo, trans_hi = spread(per_seq, "twoD_t_%s_pct" % protocol)
            out.append({
                "release": rel,
                "rule": tag,
                "segment_length": label,
                "n_sequences": str(len(per_seq)),
                "n_sequences_sweep":
                    str(len({r["sequence"] for r in sweep_seq})),
                "n_frames": row["n_frames"],
                "n_segments": row["n_segments"],
                "twoD_rot_deg_per_100m":
                    row["twoD_%s_deg_per_100m" % protocol],
                "twoD_trans_pct": row["twoD_t_%s_pct" % protocol],
                "twoD_rot_rms_deg_per_100m":
                    row["twoD_%s_rms_deg_per_100m" % protocol],
                "twoD_trans_rms_pct": row["twoD_t_%s_rms_pct" % protocol],
                "twoD_rot_seq_min_deg_per_100m": rot_lo,
                "twoD_rot_seq_max_deg_per_100m": rot_hi,
                "twoD_trans_seq_min_pct": trans_lo,
                "twoD_trans_seq_max_pct": trans_hi,
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/next")
    ap.add_argument("--out", default="results/next/release_bound_table.csv")
    args = ap.parse_args()

    rows = collect(args.root)
    if not rows:
        print("잴 것이 없다 — kitti_segment_D.py 를 먼저 돌려라")
        return

    print("%-10s %-6s %-10s %9s %12s %10s"
          % ("릴리스", "규칙", "구간", "구간 수", "2D 회전", "2D 병진"))
    for r in rows:
        print("%-10s %-6s %-10s %9s %9s 도/100m %7s %%"
              % (r["release"], r["rule"], r["segment_length"],
                 r["n_segments"], r["twoD_rot_deg_per_100m"],
                 r["twoD_trans_pct"]))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\n%d 행 -> %s" % (len(rows), args.out))


if __name__ == "__main__":
    main()
