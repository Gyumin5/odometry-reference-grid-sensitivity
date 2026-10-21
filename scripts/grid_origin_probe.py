#!/usr/bin/env python3
"""성긴 격자를 절대 시각으로 만드는 것이 수를 가르나.

coarse_grid 는 격자점을 t[0] + stepsize*k 로 만든다. 시각이 유닉스 기원이면
t[0] 이 1.7e9 대라 그 덧셈의 눈금이 2.4e-07 초다. 격자점은 그 눈금 위로
눌린다. 원점을 먼저 빼고 만들면 눌리는 눈금이 경과 시간 크기(1e3 대)의
눈금, 곧 1e-13 초가 된다.

이 둘을 가르는 것이 이 탐침이다. 섞이기 쉬운 두 가지를 떼어 놓는다.

  ⑴ 입력이 이미 잃은 정밀도. 공개 시각 t 자체가 1.7e9 대로 저장돼 있어
     2.4e-07 눈금 위에 있다. 원점을 빼도 안 돌아온다 — 뺄셈은 정확하지만
     없던 자릿수가 생기지는 않는다.
  ⑵ 격자를 만들면서 잃는 정밀도. 이쪽은 우리가 만든 수라 되돌릴 수 있다.
     t[0] 을 빼고 만들면 격자점이 제자리에 선다.

⑵가 실제로 D 를 바꾸는지 묻는다. 바꾸지 않으면 절대 시각으로 두어도 되고,
바꾸면 상대 시각으로 옮겨야 한다. 판정은 같은 시퀀스를 두 번 재서 변형마다
묶음 D 를 맞대는 것이다 — 원점을 빼는 것은 정확한 연산이라 두 판이 같은
질문을 재고 있다.
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import protocol_sweep as P  # noqa: E402
from kitti_segment_D import RELEASES, gather  # noqa: E402
from protocol_sweep import RATIOS, coarse_grid, window  # noqa: E402

def input_tick(t):
    """입력 시각이 놓인 눈금(초). t[0] 크기에서의 배정밀도 한 칸이다.

    이것은 ⑴ 이다. 격자를 어떻게 만들든 안 줄어든다.
    """
    t0 = float(t[0])
    return float(np.spacing(abs(t0))) if t0 else float(np.spacing(1.0))


def two_prod(a, b):
    """곱의 반올림 오차까지 낸다. (떨어진 값, 그 값이 놓친 양)."""
    p = a * b
    split = 134217729.0                 # 2^27 + 1
    ah = a * split
    ah = ah - (ah - a)
    al = a - ah
    bh = b * split
    bh = bh - (bh - b)
    bl = b - bh
    return p, ((ah * bh - p) + ah * bl + al * bh) + al * bl


def two_sum(a, b):
    """합의 반올림 오차까지 낸다. (떨어진 값, 그 값이 놓친 양)."""
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def grid_drift(t, dt, ratio):
    """격자점이 제 위치에서 밀린 최대 초. 이것이 ⑵ 다.

    제 위치는 t[0] + stepsize*k 를 정확히 계산한 값이다. 그 값은 배정밀도에
    안 들어가므로 빼서 재지 않고, 두 번의 반올림을 각각 정확히 꺼내 더한다 —
    곱 stepsize*k 가 놓친 양과 그 뒤 덧셈이 놓친 양이다. 둘의 합이 곧 밀림이고
    격자점 전부에서 한꺼번에 난다.

    stepsize 자신이 dt/ratio 를 반올림한 값인 것은 여기 안 센다. 그 반올림은
    원점을 빼도 그대로라 두 구성에 똑같이 들어가고, 여기서 가르려는 것은
    원점이 만드는 차이뿐이다.
    """
    tc = coarse_grid(t, dt, ratio)
    if len(tc) < 2:
        return 0.0
    stepsize = float(dt) / float(ratio)
    k = np.arange(len(tc), dtype=float)
    p, e_mul = two_prod(stepsize, k)
    _, e_add = two_sum(float(t[0]), p)
    return float(np.abs(e_mul + e_add).max())


def pick_flips(t, dt, ratio):
    """두 산술에서 최근접 색인이 갈리는 질의 수와 전체 질의 수.

    질의는 창 안의 원본 프레임이고 원본은 성긴 격자다 — protocol_sweep 의
    rate_*_nearest 가 묻는 그 자리다.
    """
    t = np.asarray(t, dtype=float)
    if len(t) < 8:
        return 0, 0
    lo, hi = window(t, dt)
    if not (hi > lo):
        return 0, 0
    tw = t[(t >= lo) & (t <= hi)]
    tc = coarse_grid(t, dt, ratio)
    if tw.size == 0 or len(tc) < 2:
        return 0, 0
    rel = coarse_grid(t - t[0], dt, ratio)
    return int(np.count_nonzero(nearest(tc, tw) != nearest(rel, tw - t[0]))), \
        int(tw.size)


def nearest(src, tq):
    """동률이면 앞 표본을 고르는 최근접 색인. resample 의 그 규칙이다."""
    j = np.clip(np.searchsorted(src, tq), 1, len(src) - 1)
    return np.where(tq - src[j - 1] <= src[j] - tq, j - 1, j)


def pooled_both(seqs, lengths=P.LENGTHS, step=P.STEP):
    """같은 시퀀스를 절대 시각과 상대 시각으로 각각 재서 (절대, 상대).

    절대 판은 measure 의 대조 경로(subtract_origin=False)로 돈다. 입력에서
    미리 빼 두는 것으로는 이 질문을 못 묻는다 — measure 가 안에서 다시 빼므로
    두 판이 같은 것을 재고 차이가 0 으로 나온다. 그 0 은 원점이 수를 안 가른다는
    뜻이 아니라 묻지 않았다는 뜻이다.
    """
    out = []
    for relative in (False, True):
        raws = []
        for t, rot, pos in seqs:
            raw = {}
            d, _ = P.measure(np.asarray(t, dtype=float), rot, pos, raw=raw,
                             lengths=lengths, step=step,
                             subtract_origin=relative)
            if d is not None:
                raws.append(raw)
        out.append(P.pool(raws) if raws else {})
    return out[0], out[1]


def pooled_by_rule(seqs):
    """(규칙 이름, 절대, 상대). 공식 구간 규칙이 한 구간도 안 세우면 1 m 로 간다.

    실내 궤적은 100~800 m 구간이 안 서서 그 규칙으로는 두 판이 다 비고, 표에
    빈 칸이 남는다. 그 빈 칸은 "원점이 수를 안 가른다" 로 읽히는데 실제로는
    "그 규칙으로는 잴 것이 없었다" 다. 그 릴리스가 실제로 평가받는 규칙으로
    다시 재서 수를 채운다.
    """
    abs_d, rel_d = pooled_both(seqs)
    if abs_d:
        return "seg", abs_d, rel_d
    abs_d, rel_d = pooled_both(seqs, lengths=(1.0,), step=1)
    return "rpe1m", abs_d, rel_d


def worst_gap(abs_d, rel_d):
    """두 산술의 D 가 가장 크게 갈리는 변형. (이름, 절대, 상대, 상대 퍼센트)."""
    best = None
    for k in sorted(abs_d):
        a, b = abs_d[k], rel_d.get(k, abs_d[k])
        if not a:
            continue
        pct = abs(b - a) / abs(a) * 100.0
        if best is None or pct > best[3]:
            best = (k, a, b, pct)
    return best


def rows_for():
    rows = []
    for release in RELEASES:
        seqs, flips, queries, drift, tick = [], 0, 0, 0.0, None
        drift_rel = 0.0
        for _, (t, rot, pos) in gather(release):
            seqs.append((t, rot, pos))
            tt = np.asarray(t, dtype=float)
            if len(tt) < 8:
                continue
            dt = float(np.median(np.diff(tt)))
            tick = input_tick(tt) if tick is None else max(tick,
                                                           input_tick(tt))
            for r in RATIOS:
                drift = max(drift, grid_drift(tt, dt, r))
                drift_rel = max(drift_rel, grid_drift(tt - tt[0], dt, r))
                f, q = pick_flips(tt, dt, r)
                flips += f
                queries += q
        rule, abs_d, rel_d = pooled_by_rule(seqs)
        w = worst_gap(abs_d, rel_d)
        rows.append({
            "release": release,
            "rule": rule,
            "epoch_s": "%.6g" % float(seqs[0][0][0]) if seqs else "",
            "input_tick_s": "" if tick is None else "%.3g" % tick,
            "grid_drift_s": "%.3g" % drift,
            "grid_drift_relative_s": "%.3g" % drift_rel,
            "n_queries": queries,
            "n_pick_flips": flips,
            "worst_variant": w[0] if w else "",
            "D_absolute": "%.6g" % w[1] if w else "",
            "D_relative": "%.6g" % w[2] if w else "",
            "D_rel_diff_pct": "%.3g" % w[3] if w else "",
        })
        print("[%s/%s] 기원 %s  입력 눈금 %s  격자 밀림 %s (원점 빼면 %s)  "
              "질의 %d  바뀐 색인 %d  가장 크게 갈린 변형 %s %s %%"
              % (release, rule, rows[-1]["epoch_s"], rows[-1]["input_tick_s"],
                 rows[-1]["grid_drift_s"], rows[-1]["grid_drift_relative_s"],
                 queries, flips, rows[-1]["worst_variant"],
                 rows[-1]["D_rel_diff_pct"]))
        sys.stdout.flush()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/next/grid_origin_probe.csv")
    args = ap.parse_args()

    rows = rows_for()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\n%d 행 -> %s" % (len(rows), args.out))
    print("색인이 갈리는 릴리스 %d / %d."
          % (len([r for r in rows if r["n_pick_flips"]]), len(rows)))


if __name__ == "__main__":
    main()
