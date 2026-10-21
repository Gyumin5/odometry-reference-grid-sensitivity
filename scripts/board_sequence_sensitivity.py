#!/usr/bin/env python3
"""순위표 환산이 공개 시퀀스 하나에 얼마나 묶여 있는지 잰다.

본문은 순위표 칸의 폭을 어긋남 밀리초로 바꿔 적는다. 그 환산은 공개
시퀀스를 묶은 값 하나에서 나오는데, 묶음이 시퀀스 하나에 끌려가면 그
수는 공개 자료의 성질이 아니라 그 시퀀스의 성질이다. 그래서 둘을 낸다 —
시퀀스마다의 기울기, 그리고 시퀀스를 하나씩 뺀 묶음으로 다시 환산한 값의
범위. 범위가 좁으면 환산이 한 시퀀스에 안 묶여 있다는 뜻이고, 넓으면
본문이 그 폭을 같이 적어야 한다.

환산 자체는 offset_tolerance.py 가 정본이고, 본문이 적는 밀리초는 잰 점들
사이를 되짚어 얻는다. 그래서 두 축으로 낸다. 기울기는 시퀀스마다의 민감도를
한 수로 보여 주는 값이고, 되짚어 얻는 span_ms 는 본문이 적는 바로 그 수다.
기울기 범위만 내면 본문 수치의 안정성과는 다른 통계량을 보이게 된다.
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from offset_tolerance import (  # noqa: E402
    LEADERBOARDS, pairs, tolerance_ms,
)

TAUS_MS = (1.0, 2.0, 5.0, 10.0, 25.0, 50.0)
POOLED = "ALL"

# 순위표 칸마다 (훑기 열 꼬리, 단위 배율, offset_tolerance 의 칸 이름,
# 그 칸이 찍는 자리수). 배율은 훑기 표의 D 를 순위표 단위의 2D 로 옮긴다 —
# offset_tolerance.UNITS 가 쓰는 값과 같다. 기울기 쪽 수치는 비라서 배율이
# 지워지고, 되짚기 쪽은 순위표 차이와 직접 견주므로 배율이 있어야 한다.
COLUMNS = {
    "trans": ("pct", 2.0, "translation", 2),
    "rot": ("deg_per_100m", 0.02, "rotation", 4),
}


def read_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def slope(points):
    """원점을 지나는 직선의 기울기. 점이 없으면 None."""
    den = sum(x * x for x, _ in points)
    if den <= 0.0:
        return None
    return sum(x * y for x, y in points) / den


def sequence_points(row, tail, scale):
    out = []
    for ms in TAUS_MS:
        key = "D_tau_%gms_%s" % (ms, tail)
        if key not in row or row[key] in ("", None):
            continue
        out.append((ms, scale * float(row[key])))
    return sorted(out)


def pooled_points(rows, tail, scale, skip=None):
    """구간 수로 무게를 준 묶음의 (밀리초, 2D) 점들. skip 은 빼는 시퀀스."""
    acc, tot = [], 0.0
    for r in rows:
        if r["sequence"] == POOLED or r["sequence"] == skip:
            continue
        n = float(r.get("n_segments") or 0.0)
        if n <= 0.0:
            continue
        acc.append((n, sequence_points(r, tail, scale)))
        tot += n
    if tot <= 0.0:
        return []
    merged = {}
    for n, pts in acc:
        for ms, v in pts:
            merged[ms] = merged.get(ms, 0.0) + n * v / tot
    return sorted(merged.items())


def span_ms(points, board_key, metric, decimals):
    """표 전체 폭을 지우는 어긋남. offset_tolerance 와 같은 되짚기다."""
    if not points:
        return None, None
    board = LEADERBOARDS[board_key]
    tol = [tolerance_ms(g, points)
           for _, _, _, _, g in pairs(board, metric, decimals)]
    if not tol:
        return None, None
    worst = max(tol, key=lambda t: (t[1] == "above", t[0]))
    return worst[0], worst[1]


def leave_one_out(rows, tail, scale):
    """시퀀스를 하나씩 뺀 묶음의 기울기들. 묶음은 구간 수로 무게를 준다."""
    named = [r for r in rows if r["sequence"] != POOLED]
    out = []
    for drop in named:
        acc, tot = [], 0.0
        for r in named:
            if r["sequence"] == drop["sequence"]:
                continue
            n = float(r.get("n_segments") or 0.0)
            if n <= 0.0:
                continue
            acc.append((n, sequence_points(r, tail, scale)))
            tot += n
        if tot <= 0.0:
            continue
        merged = {}
        for n, pts in acc:
            for ms, v in pts:
                merged[ms] = merged.get(ms, 0.0) + n * v / tot
        k = slope(sorted(merged.items()))
        if k is not None:
            out.append((drop["sequence"], k))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="results/next/kitti_segment_sweep.csv")
    ap.add_argument("--out",
                    default="results/next/board_sequence_sensitivity.csv")
    args = ap.parse_args()

    rows = read_rows(args.sweep)
    named = [r for r in rows if r["sequence"] != POOLED]
    out = []
    for metric, (tail, scale, column, decimals) in sorted(COLUMNS.items()):
        for r in named:
            k = slope(sequence_points(r, tail, scale))
            if k is None:
                continue
            out.append({"metric": metric, "kind": "sequence",
                        "label": r["sequence"], "slope_per_ms": repr(k),
                        "span_ms": "", "span_where": ""})
        for name, k in leave_one_out(rows, tail, scale):
            ms, where = span_ms(pooled_points(rows, tail, scale, skip=name),
                                "kitti", column, decimals)
            out.append({"metric": metric, "kind": "drop",
                        "label": name, "slope_per_ms": repr(k),
                        "span_ms": "" if ms is None else repr(ms),
                        "span_where": where or ""})
        ms, where = span_ms(pooled_points(rows, tail, scale),
                            "kitti", column, decimals)
        out.append({"metric": metric, "kind": "pooled", "label": POOLED,
                    "slope_per_ms": repr(slope(pooled_points(
                        rows, tail, scale)) or float("nan")),
                    "span_ms": "" if ms is None else repr(ms),
                    "span_where": where or ""})

    if not out:
        raise SystemExit("기울기가 하나도 안 섰다 — 훑기 파일을 확인해라")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "kind", "label",
                                          "slope_per_ms", "span_ms",
                                          "span_where"])
        w.writeheader()
        w.writerows(out)

    for metric in sorted(COLUMNS):
        seq = [float(r["slope_per_ms"]) for r in out
               if r["metric"] == metric and r["kind"] == "sequence"]
        drop = [float(r["slope_per_ms"]) for r in out
                if r["metric"] == metric and r["kind"] == "drop"]
        if not seq:
            continue
        print("%-6s 시퀀스 %2d  기울기 %.6g~%.6g  하나씩 빼면 %.6g~%.6g"
              % (metric, len(seq), min(seq), max(seq), min(drop), max(drop)))
        span = [float(r["span_ms"]) for r in out
                if r["metric"] == metric and r["kind"] == "drop"
                and r["span_ms"]]
        full = [r for r in out
                if r["metric"] == metric and r["kind"] == "pooled"]
        if span and full and full[0]["span_ms"]:
            print("       본문 환산(되짚기) %.4g ms, 하나씩 빼면 %.4g~%.4g ms"
                  % (float(full[0]["span_ms"]), min(span), max(span)))
    print("%d 행 -> %s" % (len(out), args.out))


if __name__ == "__main__":
    main()
