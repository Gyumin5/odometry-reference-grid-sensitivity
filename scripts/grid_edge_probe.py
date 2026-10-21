#!/usr/bin/env python3
"""성긴 격자의 마지막 칸이 공개 표본 끝을 넘어서는지 본다.

protocol_sweep.coarse_grid 은 칸 수를 int(np.floor(span / stepsize)) 로 정한다.
나눗셈이 정확히 반올림되므로 이 바닥은 참 몫의 바닥보다 작아지지 않는다 —
대신 한 칸 커질 수 있다. 참 몫이 정수 바로 아래일 때 나눗셈이 그 정수로
올라붙기 때문이고, 그러면 마지막 격자점이 마지막 공개 표본과 같거나 그 뒤에
놓인다. 뒤에 놓이면 그 점을 묻는 재표집이 외삽이 된다 — 구면보간은 그 자리에서
멈추고, 최근접은 조용히 끝 표본으로 눌러 붙인다. 그래서 짐작하지 말고 센다.

칸 수는 셋으로 센다. 부동소수 바닥(지금 쓰는 것), 같은 stepsize 로 구한 참
몫의 바닥(나눗셈과 바닥 연산만 걷어낸 것), stepsize 를 dt/ratio 의 정확한
유리수로 둔 바닥. 가운데를 따로 두는 까닭은 stepsize 의 반올림이 격자를
만드는 쪽과 세는 쪽에 똑같이 들어가서, 그것까지 넣으면 한 칸의 차이가 아니라
다른 격자를 세게 되기 때문이다.
"""

import argparse
import csv
import math
import os
import sys
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kitti_segment_D import gather, relative_time  # noqa: E402
from protocol_sweep import (RATIOS, SHIFTS, TAUS_MS,  # noqa: E402
                            coarse_grid, window)
from tie_rule_probe import RELEASES  # noqa: E402


def float_steps(span, stepsize):
    """coarse_grid 이 쓰는 칸 수. 부동소수 나눗셈 뒤 바닥."""
    return int(np.floor(span / stepsize))


def exact_steps(span, stepsize):
    """같은 두 부동소수의 참 몫의 바닥. 나눗셈 반올림만 걷어낸다."""
    return math.floor(Fraction(float(span)) / Fraction(float(stepsize)))


def rational_steps(span, dt, ratio):
    """stepsize 를 dt/ratio 의 정확한 유리수로 둔 칸 수."""
    return math.floor(Fraction(float(span)) * Fraction(float(ratio))
                      / Fraction(float(dt)))


def ulp_to_integer(span, stepsize):
    """부동소수 몫이 가장 가까운 정수에서 몇 눈금 떨어져 있나."""
    q = float(span) / float(stepsize)
    near = round(q)
    step = float(np.spacing(abs(q))) if q else float(np.spacing(1.0))
    return abs(q - near) / step


def window_end(t, dt, steps, ratios=RATIOS):
    """window() 의 끝을 주어진 칸 수로 다시 낸다. 나머지 항은 그대로다."""
    hi = float(t[-1]) - max(max(SHIFTS) * dt, max(TAUS_MS) * 1e-3)
    hi = min(hi, float(t[::2][-1]), float(t[1::2][-1]))
    for r, n in zip(ratios, steps):
        hi = min(hi, float(t[0] + (dt / r) * n))
    return hi


def row_for(name, t, ratios=RATIOS):
    # 시각 표현을 재는 쪽에 맞춘다(protocol_sweep.measure 와 같은 relative_time).
    # 안 맞추면 창과 격자가 다른 눈금 위에서 서서, 여기서 센 격자 끝이 실제
    # 측정이 쓴 격자 끝이 아니다.
    t = relative_time(t)
    dt = float(np.median(np.diff(t)))
    span = float(t[-1] - t[0])
    lo, hi = window(t, dt)
    n_float = [float_steps(span, dt / r) for r in ratios]
    n_exact = [exact_steps(span, dt / r) for r in ratios]
    hi_exact = window_end(t, dt, n_exact, ratios)
    frames = int(((t >= lo) & (t <= hi)).sum())
    frames_exact = int(((t >= lo) & (t <= hi_exact)).sum())
    out = []
    for r, nf, ne in zip(ratios, n_float, n_exact):
        stepsize = dt / r
        grid = coarse_grid(t, dt, r)
        out.append({
            "sequence": name,
            "ratio": "%g" % r,
            "n_frames": len(t),
            "quotient_ulp_from_integer": "%.3g" % ulp_to_integer(span,
                                                                 stepsize),
            "n_steps_float": nf,
            "n_steps_exact": ne,
            "n_steps_rational": rational_steps(span, dt, r),
            "extra_step": int(nf > ne),
            "grid_end_long_by_s": "%.6g" % max(0.0, (nf - ne) * stepsize),
            "grid_end_past_samples_s": "%.6g" % (float(grid[-1])
                                                 - float(t[-1])),
            "window_end_set_by_this_grid": int(float(grid[-1]) == hi),
            "window_frames": frames,
            "window_frames_exact": frames_exact,
        })
    return out


def summarise(tag, rows):
    extra = [r for r in rows if r["extra_step"]]
    past = [r for r in rows if float(r["grid_end_past_samples_s"]) > 0.0]
    tight = min(float(r["quotient_ulp_from_integer"]) for r in rows)
    moved = [r for r in rows
             if r["window_frames"] != r["window_frames_exact"]]
    print("[%s] 줄 %d, 칸이 하나 더 선 자리 %d, 격자 끝이 표본을 넘은 자리 %d, "
          "창 프레임이 달라진 줄 %d, 정수에 가장 가까운 몫 %.3g 눈금"
          % (tag, len(rows), len(extra), len(past), len(moved), tight))
    for r in (past or extra)[:5]:
        print("  %s ratio %s: 부동소수 %d 칸, 정확 %d 칸, 표본 끝 대비 %s 초"
              % (r["sequence"], r["ratio"], r["n_steps_float"],
                 r["n_steps_exact"], r["grid_end_past_samples_s"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--releases", default=",".join(RELEASES))
    ap.add_argument("--out", default="results/next/grid_edge_probe.csv")
    args = ap.parse_args()

    rows = []
    for release in args.releases.split(","):
        got = []
        for name, (t, _rot, _pos) in gather(release):
            if len(t) < 3:
                continue
            got.extend(dict(r, release=release) for r in row_for(name, t))
        if not got:
            print("[%s] 읽은 시퀀스가 없다" % release)
            continue
        summarise(release, got)
        rows.extend(got)
    if not rows:
        raise SystemExit("읽은 시퀀스가 없다")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["release"]
                           + [k for k in rows[0] if k != "release"])
        w.writeheader()
        w.writerows(rows)
    summarise("전체", rows)


if __name__ == "__main__":
    main()
