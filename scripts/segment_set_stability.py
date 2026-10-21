#!/usr/bin/env python3
"""공식 구간 선택이 기준 궤적을 바꿔도 같은 자리를 고르는지 센다.

상한 D = sum w d(Q,Q') 는 네 가지가 고정됐을 때 선다 — 구간 집합, 대응,
가중치, 추정 상대자세다. 우리 측정은 구간을 공개 자세에서 한 번 고르고
규약을 바꾼 쪽에 그대로 다시 쓰므로 그 조건을 지킨다. 공식 평가 프로그램은
안 그렇다. evaluate_odometry.cpp 의 lastFrameFromSegmentLength 는 누적
이동거리를 그때그때 기준 궤적에서 다시 재고, 그 거리가 요구 길이를 넘는 첫
프레임을 끝으로 잡는다. 기준 궤적이 달랐다면 그 "첫 프레임" 이 옮겨갈 수
있고, 옮겨가면 가중치가 고정이라는 전제가 깨진다.

그래서 전제를 글로 적어 두는 대신 센다. 공개 자세에서 고른 시작점마다 끝점을
두 번 구한다 — 공개 자세에서 한 번, 규약을 바꾼 자세에서 한 번. 둘이 다른
줄이 몇인지, 가중치로 얼마인지, 최대 몇 프레임 옮겨가는지가 이 표다.

옮겨간 줄이 0 이면 그 릴리스·그 규약에서는 공식 선택이 우리 집합과 같다 —
상한의 전제가 실제 자료 위에서 지켜진다는 뜻이다. 0 이 아니면 그만큼이
상한 밖이고 그 수를 본문에 적는다.

실행:
    PYTHONPATH=src python3 scripts/segment_set_stability.py [--release kitti]
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kitti_segment_D import (LENGTHS, RELEASES, STEP,  # noqa: E402
                             exact, gather, relative_time)
from protocol_sweep import variants, window, window_segments  # noqa: E402

OUT = "results/next/segment_set_stability.csv"
# 키티-360 공식 순위표가 쓰는 1 m 규칙도 같이 본다. 간격이 좁을수록 후보
# 프레임이 촘촘해서 끝점이 옮겨갈 자리가 많다 — 어느 쪽이 더 흔들리는지는
# 재야 안다.
RULES = (("100-800m", LENGTHS, STEP), ("1m", (1.0,), 1))
FIELDS = ("release", "rule", "convention", "sequences", "pairs", "moved",
          "moved_frac", "weight_moved_frac", "max_shift_frames",
          "dropped", "max_len_change_rel")


def arc(pos):
    return np.concatenate([[0.0],
                           np.cumsum(np.linalg.norm(np.diff(pos, axis=0),
                                                    axis=1))])


def endpoints(pos, pairs):
    """공식 규칙 그대로 (시작, 요구길이) 마다 끝 프레임. 없으면 -1.

    `dist[i] > dist[first] + len` 이라 딱 닿는 자리는 안 받는다(side="right").
    """
    d = arc(pos)
    out = []
    for i, L in pairs:
        j = int(np.searchsorted(d, d[i] + L, side="right"))
        out.append(j if j < len(d) else -1)
    return np.asarray(out), d


def tally(release, lengths, step, source=None):
    """규약마다 (옮겨간 줄, 전체 줄, 가중치 몫, 최대 이동, 버려진 줄, 길이변화).

    source 를 주면 그 자세들을 읽는다. 검사에서만 쓴다 — 시퀀스를 가로질러
    가중치를 더하는 자리가 틀리기 쉬운데 실제 릴리스로는 그 자리를 못 짚는다.
    """
    acc = {}
    nseq = 0
    for _name, (t, rot, pos) in (gather(release) if source is None
                                 else source):
        t = relative_time(np.asarray(t, dtype=float))
        if len(t) < 8:
            continue
        dt = float(np.median(np.diff(t)))
        lo, hi = window(t, dt)
        if not (hi > lo):
            continue
        m = (t >= lo) & (t <= hi)
        pw = np.asarray(pos)[m]
        idx = window_segments(pos, m, lengths, step)
        if not idx:
            continue
        nseq += 1
        pairs = [(i, L) for (i, _j, L) in idx]
        # 가중치는 1/L_k 다. 공식 꼴은 1/(N L_k) 인데 N 은 릴리스를 한 자루에
        # 담았을 때의 전체 구간 수라 몫에서 약분된다 — 시퀀스마다 그 시퀀스의
        # 구간 수로 나누면 상한 표와 다른 것을 재게 된다(상한 표는 모든
        # 시퀀스의 구간을 한 자루에 담아 센다). 옮겨간 줄이 어느 쪽 길이에
        # 몰리는지가 보이게 수가 아니라 가중치로도 센다.
        w = np.asarray([1.0 / L for _i, L in pairs])
        for v in variants(t, rot, pos, dt, lo, hi):
            if len(v) == 3:
                name, _ar, ap = v
                base = pw
            else:
                name, _ar, ap, _br, bp = v
                base = bp
            j0, d0 = endpoints(base, pairs)
            j1, d1 = endpoints(ap, pairs)
            live = (j0 >= 0) & (j1 >= 0)
            moved = live & (j0 != j1)
            a = acc.setdefault(name, [0, 0, 0.0, 0, 0, 0.0, 0.0])
            a[0] += int(moved.sum())
            a[1] += int(live.sum())
            # 분모는 정규화돼 있지 않으므로 같이 쌓아 두고 마지막에 나눈다.
            a[2] += float(w[moved].sum())
            a[6] += float(w.sum())
            if moved.any():
                a[3] = max(a[3], int(np.abs(j1 - j0)[moved].max()))
            a[4] += int((~live).sum())
            if live.any():
                i0 = np.asarray([i for i, _L in pairs])[live]
                L0 = d0[j0[live]] - d0[i0]
                L1 = d1[j1[live]] - d1[i0]
                a[5] = max(a[5], float(np.max(np.abs(L1 - L0)
                                              / np.maximum(L0, 1e-12))))
    return nseq, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", action="append")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    rows = []
    for rel in (args.release or list(RELEASES)):
        for rule, lengths, step in RULES:
            nseq, acc = tally(rel, lengths, step)
            if not acc:
                print("%-9s %-8s 구간 없음" % (rel, rule))
                continue
            worst = 0
            for name in sorted(acc):
                mv, tot, wm, sh, dr, lc, wtot = acc[name]
                worst = max(worst, mv)
                rows.append({
                    "release": rel, "rule": rule, "convention": name,
                    "sequences": nseq, "pairs": tot, "moved": mv,
                    "moved_frac": exact(mv / tot if tot else float("nan")),
                    "weight_moved_frac": exact(wm / wtot if wtot
                                               else float("nan")),
                    "max_shift_frames": sh, "dropped": dr,
                    "max_len_change_rel": exact(lc),
                })
            print("%-9s %-8s 시퀀스 %3d  규약 %2d  옮겨간 줄 최대 %d"
                  % (rel, rule, nseq, len(acc), worst))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(FIELDS))
        w.writeheader()
        w.writerows(rows)
    print("\n%d 행 -> %s" % (len(rows), args.out))


if __name__ == "__main__":
    main()
