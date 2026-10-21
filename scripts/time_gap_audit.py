#!/usr/bin/env python3
"""공개 기준 궤적의 내부 시간 공백이 채점 구간에 들어가는지 잰다.

대본마다 공백을 다르게 다룬다 — 튬 계열을 읽는 쪽은 중앙 간격의 다섯 배를
넘는 자리에서 궤적을 끊고, 구간 점수를 내는 쪽은 끊지 않고 사이를 메운다.
그 차이가 릴리스·표집률·복원 방식의 효과에 섞이면 안 되므로, 모든
릴리스에 같은 판정 규칙을 씌워 공백이 실제 채점 구간에 들어가는지를 센다.

규칙 하나다. 이웃한 두 표본의 간격이 그 시퀀스 중앙 간격의 배수 문턱을
넘으면 공백으로 본다. 공백이 하나라도 걸치는 구간은 그 공백이 없었다면
안 나왔을 값을 들고 있는 구간이다.
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np

from kitti_segment_D import RELEASES, gather


def median_dt(t):
    """이웃 표본 간격의 중앙값. 축이 프레임 번호면 그대로 1 이 나온다.

    표본이 셋보다 적으면 간격이 안 선다. 그 줄은 0 으로 적는데, 0 초
    간격이라는 뜻이 아니라 잴 것이 없다는 뜻이다 — 같은 줄의 samples 칸이
    왜인지를 말한다. 핏서티엠 두 주행이 표본 한 개짜리라 여기 든다.
    """
    if len(t) < 3:
        return 0.0
    return float(np.median(np.diff(t)))


def gaps(t, factor):
    """공백 자리의 색인과 그 길이. 중앙 간격의 factor 배를 넘는 것만."""
    if len(t) < 3:
        return []
    d = np.diff(t)
    med = float(np.median(d))
    if med <= 0.0:
        return []
    idx = np.nonzero(d > factor * med)[0]
    return [(int(i), float(d[i]), med) for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor", type=float, default=5.0,
                    help="중앙 간격의 몇 배를 넘으면 공백으로 보나")
    ap.add_argument("--out", default="results/next/time_gap_audit.csv")
    args = ap.parse_args()

    rows = []
    for release in RELEASES:
        n_seq = n_with = 0
        worst = 0.0
        total = 0
        for name, (t, _rot, _pos) in gather(release):
            n_seq += 1
            g = gaps(np.asarray(t, dtype=float), args.factor)
            if g:
                n_with += 1
                total += len(g)
                worst = max(worst, max(x[1] for x in g))
            rows.append({"release": release, "sequence": name,
                         "samples": len(t), "gaps": len(g),
                         "max_gap_s": repr(max((x[1] for x in g), default=0.0)),
                         "median_dt_s": repr(median_dt(
                             np.asarray(t, dtype=float)))})
        print("%-9s 시퀀스 %3d  공백 있는 것 %3d  공백 %4d  최대 %.3f 초"
              % (release, n_seq, n_with, total, worst))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["release", "sequence", "samples",
                                          "gaps", "max_gap_s",
                                          "median_dt_s"])
        w.writeheader()
        w.writerows(rows)
    print("%d 행 -> %s" % (len(rows), args.out))


if __name__ == "__main__":
    main()
