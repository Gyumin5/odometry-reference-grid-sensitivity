#!/usr/bin/env python3
"""최근접 재표집의 동률 규칙이 실제 자료에서 수를 가르나.

최근접은 질의 시각에 가장 가까운 표본을 고른다. 두 이웃이 똑같이 떨어져
있으면 어느 쪽을 고를지는 부등호 하나가 정한다. 그 자리가 밟히지 않는다면
규칙은 적을 필요도 없는 세부지만, 밟힌다면 복원에 규약 하나가 더 필요해진다.
둘 중 어느 쪽인지 잰다.

두 가지를 묻는다.

  - 동률 질의가 몇 개인가. 규칙을 뒤집어(앞 표본 대신 뒤 표본) 고른 색인이
    달라지는 질의를 센다.
  - 그래서 보고하는 D 가 얼마나 달라지나. 같은 뒤집기로 묶음 D 를 다시 내
    견준다. 색인이 달라져도 두 표본의 자세가 비슷하면 D 는 거의 안 움직인다.

동률이 우연이 아닌 이유는 성긴 격자에 있다. 비율 0.5 는 표본을 하나 걸러
고르므로, 남은 질의가 정확히 두 표본 한가운데에 놓인다.
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import protocol_sweep as P  # noqa: E402
from kitti_segment_D import (LENGTHS, RELEASES, STEP,  # noqa: E402
                             gather, relative_time)
from protocol_sweep import RATIOS, coarse_grid, window  # noqa: E402

# 상한 표가 쓰는 두 구간 규칙 그대로다. 한쪽만 재면 그 규칙에서 구간이 안
# 서는 릴리스(튬 알지비디의 100~800 m)가 통째로 빈 줄이 되고, 동률이 거기서
# 수를 가르는지 아닌지를 모른 채로 남는다.
RULES = (("seg", LENGTHS, STEP), ("rpe1m", (1.0,), 1))
ORIG = P.resample


def resample_tie_high(t, rot, pos, tq, method):
    """동률에서 뒤 표본을 고르는 판. 최근접 가지만 부등호가 다르다."""
    if method != "nearest":
        return ORIG(t, rot, pos, tq, method)
    t = np.asarray(t, dtype=float)
    tq = np.asarray(tq, dtype=float)
    j = np.clip(np.searchsorted(t, tq), 1, len(t) - 1)
    pick = np.where(tq - t[j - 1] < t[j] - tq, j - 1, j)
    return np.asarray(rot)[pick], np.asarray(pos)[pick]


def flips(t):
    """그 시퀀스에서 (질의 수, 규칙을 뒤집어 바뀐 색인 수, 가장 작은 간격차).

    최근접이 실제로 쓰이는 자리만 센다 — 성긴 격자를 원본으로 삼고 창 안
    프레임을 질의로 삼는 protocol_sweep 의 rate_*_nearest 다. 시각 표현도
    그쪽을 따른다(relative_time) — 안 맞추면 창이 한두 프레임 다르게 잡혀
    세는 질의가 재는 질의와 달라진다.
    """
    t = relative_time(t)
    if len(t) < 8:
        return 0, 0, None
    dt = float(np.median(np.diff(t)))
    lo, hi = window(t, dt)
    if not (hi > lo):
        return 0, 0, None
    tw = t[(t >= lo) & (t <= hi)]
    if tw.size == 0:
        return 0, 0, None
    n_q = n_diff = 0
    closest = None
    for r in RATIOS:
        tc = coarse_grid(t, dt, r)
        if len(tc) < 2:
            continue
        j = np.clip(np.searchsorted(tc, tw), 1, len(tc) - 1)
        a, b = tw - tc[j - 1], tc[j] - tw
        n_q += tw.size
        n_diff += int(np.count_nonzero((a <= b) != (a < b)))
        c = float(np.abs(a - b).min())
        closest = c if closest is None else min(closest, c)
    return n_q, n_diff, closest


def pooled_both(seqs, lengths=None, step=None):
    """같은 시퀀스를 두 동률 규칙으로 각각 재서 (앞 표본 D, 뒤 표본 D, 구간 수).

    시퀀스를 한 번만 읽고 두 번 재도록 자세를 들고 있는다. 핏서티엠은
    시퀀스 하나가 이십만 프레임이라 두 번 읽으면 그만큼 더 든다.

    구간 수를 같이 내는 것은 빈 칸의 까닭을 적기 위해서다. D 가 비는 데는
    이유가 둘이고 — 그 구간 규칙이 그 궤적에서 한 구간도 안 서는 것과 재다
    실패하는 것 — 빈 칸만 보면 둘을 못 가른다.
    """
    lengths = LENGTHS if lengths is None else lengths
    step = STEP if step is None else step
    out, nseg = {}, 0
    for tie_high in (False, True):
        P.resample = resample_tie_high if tie_high else ORIG
        raws, total = [], 0
        for t, rot, pos in seqs:
            raw = {}
            d, n = P.measure(t, rot, pos, raw=raw, lengths=lengths, step=step)
            if d is not None:
                raws.append(raw)
                total += n
        out[tie_high] = P.pool(raws) if raws else {}
        nseg = max(nseg, total)
    P.resample = ORIG
    return out[False], out[True], nseg


def worst(low, high):
    """두 규칙의 D 가 가장 크게 갈리는 변형. (이름, 앞, 뒤, 상대 퍼센트).

    최근접이 들어간 변형만 본다 — 나머지는 규칙과 무관해서 정확히 같다.
    """
    best = None
    for k in sorted(low):
        if "nearest" not in k and not k.startswith(("method_", "t_method_")):
            continue
        a, b = low[k], high.get(k, low[k])
        if not a:
            continue
        pct = abs(b - a) / abs(a) * 100.0
        if best is None or pct > best[3]:
            best = (k, a, b, pct)
    return best


def method_gap(low, high):
    """복원 방식의 몫이 동률 규칙을 뒤집어도 남나.

    논문이 방식의 몫으로 드는 것은 같은 주기에서 구면보간과 최근접을 맞댄
    차다. 동률 규칙은 그 최근접 쪽만 건드리므로, 규칙을 뒤집었을 때 둘의
    비가 그대로면 방식의 몫은 규칙이 만든 것이 아니다. 비가 1 쪽으로 무너지면
    그 몫의 일부가 규칙이다.

    (주기, 규칙 앞의 비, 규칙 뒤의 비, 순서가 뒤집힌 변형 쌍 수) 를 낸다.
    비는 최근접 D 를 구면보간 D 로 나눈 값이고, 순서 뒤집힘은 두 규칙에서
    변형들을 D 로 줄 세웠을 때 자리가 바뀐 쌍의 수다.
    """
    best = None
    for r in RATIOS:
        s, n = low.get("rate_%g_slerp" % r), low.get("rate_%g_nearest" % r)
        s2, n2 = high.get("rate_%g_slerp" % r), high.get("rate_%g_nearest" % r)
        if not s or not n or not s2 or not n2:
            continue
        a, b = n / s, n2 / s2
        if best is None or abs(b - a) > abs(best[2] - best[1]):
            best = (r, a, b)
    shared = [k for k in low if k in high]
    order_low = sorted(shared, key=lambda k: low[k])
    rank_high = {k: i for i, k in enumerate(sorted(shared,
                                                   key=lambda k: high[k]))}
    swaps = sum(1 for i in range(len(order_low))
                for j in range(i + 1, len(order_low))
                if rank_high[order_low[i]] > rank_high[order_low[j]])
    return best, swaps


def reason(nseg, best):
    """빈 칸의 까닭. 채워진 칸이면 빈 글자다.

    비는 칸을 그냥 두면 읽는 쪽이 "안 쟀다" 와 "잴 것이 없었다" 를 못
    가른다. 둘은 다른 말이다 — 뒤쪽은 그 규칙이 그 궤적에서 한 구간도 안
    세운다는 측정 결과다.
    """
    if best:
        return ""
    if nseg == 0:
        return "그 구간 규칙이 이 릴리스에서 한 구간도 안 세운다"
    return "구간은 섰는데 최근접이 들어간 변형이 없다"


def rows_for():
    rows = []
    for release in RELEASES:
        seqs, n_q, n_diff, closest = [], 0, 0, None
        for _, (t, rot, pos) in gather(release):
            seqs.append((t, rot, pos))
            q, d, c = flips(t)
            n_q += q
            n_diff += d
            if c is not None:
                closest = c if closest is None else min(closest, c)
        for rule, lengths, step in RULES:
            low, high, nseg = pooled_both(seqs, lengths, step)
            w = worst(low, high)
            g, swaps = method_gap(low, high)
            rows.append({
                "release": release,
                "rule": rule,
                "n_segments": nseg,
                "n_queries": n_q,
                "n_index_flips": n_diff,
                "closest_tie_gap_s": ("" if closest is None
                                      else "%.3g" % closest),
                "worst_variant": w[0] if w else "",
                "D_tie_low": "%.6g" % w[1] if w else "",
                "D_tie_high": "%.6g" % w[2] if w else "",
                "D_rel_diff_pct": "%.3g" % w[3] if w else "",
                "method_ratio_rate": "%g" % g[0] if g else "",
                "method_ratio_tie_low": "%.4g" % g[1] if g else "",
                "method_ratio_tie_high": "%.4g" % g[2] if g else "",
                "n_order_swaps": swaps,
                "empty_reason": reason(nseg, w),
            })
            print("[%s %s] 구간 %d  질의 %d  바뀐 색인 %d  가장 작은 간격차 "
                  "%s  가장 크게 갈린 변형 %s %s %%  방식 비 %s -> %s  "
                  "순서 뒤집힘 %d %s"
                  % (release, rule, nseg, n_q, n_diff,
                     rows[-1]["closest_tie_gap_s"],
                     rows[-1]["worst_variant"], rows[-1]["D_rel_diff_pct"],
                     rows[-1]["method_ratio_tie_low"],
                     rows[-1]["method_ratio_tie_high"], swaps,
                     rows[-1]["empty_reason"]))
            sys.stdout.flush()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/next/tie_rule_probe.csv")
    args = ap.parse_args()

    rows = rows_for()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\n%d 행 -> %s" % (len(rows), args.out))
    seen = {r["release"] for r in rows if r["n_index_flips"]}
    print("동률이 밟히는 릴리스 %d / %d." % (len(seen), len(RELEASES)))
    for r in rows:
        if r["empty_reason"]:
            print("빈 칸: %s %s — %s"
                  % (r["release"], r["rule"], r["empty_reason"]))


if __name__ == "__main__":
    main()
