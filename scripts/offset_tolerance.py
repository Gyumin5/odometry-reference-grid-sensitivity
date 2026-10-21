#!/usr/bin/env python3
"""공식 순위표의 칸 차이를 시간 어긋남의 밀리초로 환산한다.

kitti_rank_check.py 는 규약 하나(반 표본 위상)에서 "이 차이가 상한보다 큰가"
를 묻는다. 그 물음에는 규약이 하나 박혀 있어서, 다른 어긋남에서 어떻게 되는지
가 안 보인다. 여기서는 물음을 뒤집는다 — 이 차이를 지우려면 기준 시각이
얼마나 어긋나야 하나.

환산은 잰 점들 위에서 한다. 절대 어긋남 1·2·5·10·25·50 밀리초에서 잰 2D 를
놓고, 차이 g 와 같아지는 어긋남을 그 점들 사이에서 되짚는다. 잰 범위 밖으로는
안 늘린다 — 가장 작은 어긋남에서 이미 상한이 더 크면 "1 밀리초 미만", 가장 큰
어긋남에서도 상한이 더 작으면 "50 밀리초 초과" 로 적는다.

되짚는 대상은 적힌 차이 그대로가 아니라 표시값 반올림 폭을 뺀 차이다. 순위표는
자리수를 잘라 찍으므로 두 항목이 반대쪽으로 잘렸으면 실제 간격이 적힌 간격보다
폭 하나만큼 좁다. 그래서 2D 가 (차이 − h) 와 같아지는 어긋남을 찾는다.
판정 쪽 문턱 2D + (h_A+h_B)/2 를 어긋남 축으로 옮긴 것과 같은 식이다.

기울기 하나로 나누지 않는 까닭은 되짚을 점들이 꼭 직선 위에 있다는 보장이
없기 때문이다. 실제로는 병진도 회전도 어긋남에 그대로 비례한다 — 직선에서
벗어난 폭이 1e-3 아래다. 그래도 점들 사이에서 되짚는 쪽을 쓴다. 휜 폭은 따로
적어 그 가정이 깨지면 보이게 한다.

인증이 아니다. 순위는 비공개 시험 구간에서 매기고 상한은 공개 구간에서 쟀다.
여기서 나오는 밀리초는 그 순위표의 간격이 어느 규모의 시간 어긋남에 해당하는
가를 읽는 값이다.
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kitti_rank_check import (LEADERBOARDS, display_term,  # noqa: E402
                              exceeds, rounds_to_display, scores)

TAUS_MS = (1.0, 2.0, 5.0, 10.0, 25.0, 50.0)

# 훑기 표가 적는 단위를 순위표의 단위로 옮긴다. 둘 다 2D 로 적으므로 ×2.
UNITS = {
    "translation": {"column": "D_trans_pct", "scale": 2.0, "unit": "%"},
    "rotation": {"column": "D_rot_deg_per_100m", "scale": 0.02,
                 "unit": "도/미터"},
}

RULE = {"kitti": "seg", "kitti360": "rpe1m"}


def sweep_taus(path, release, rule, metric, taus_ms=TAUS_MS):
    """그 릴리스·규칙의 (어긋남 밀리초, 2D) 들. 순위표 단위로 옮겨 낸다."""
    spec = UNITS[metric]
    want = {"%gms" % ms: ms for ms in taus_ms}
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if (r["release"], r["rule"], r["axis"]) != (release, rule, "tau"):
                continue
            if r["param"] in want:
                out.append((want[r["param"]],
                            spec["scale"] * float(r[spec["column"]])))
    return sorted(out)


def slope(points):
    """원점을 지나는 직선의 기울기와, 그 직선에서 가장 많이 벗어난 비율.

    환산에는 안 쓴다. 그 칸이 비례하는지 아닌지를 적어 두려고 잰다.
    """
    num = sum(x * y for x, y in points)
    den = sum(x * x for x, _ in points)
    if den <= 0.0:
        raise ValueError("어긋남 점이 없다")
    k = num / den
    dev = max(abs(y - k * x) / y for x, y in points if y > 0.0)
    return k, dev


def tolerance_ms(gap, points):
    """그 차이와 상한이 같아지는 어긋남. 잰 범위 밖은 경계로 적는다.

    돌려주는 것은 (값, 꼬리표) 다. 꼬리표는 "잰 범위 안"·"아래로 벗어남"·
    "위로 벗어남" 셋이고, 벗어난 쪽은 값 대신 그 경계를 들고 있다.
    """
    pts = sorted(points)
    if not pts:
        raise ValueError("어긋남 점이 없다")
    if gap <= 0.0:
        # 어긋남이 0 이어도 안 갈리는 쌍이다. 찍는 자리수만으로 겹치므로
        # 시간과 아무 상관이 없다 — 시간 때문에 덮인 쌍과 같은 칸에 세면
        # 어긋남 민감도가 그만큼 부풀어 보인다.
        return 0.0, "dead"
    # 세 비교가 같은 규칙을 쓴다. 아래 끝만 exceeds 로 재고 위 끝과 사이를
    # 날 부등호로 재면, 잰 점과 사실상 같은 차이가 한쪽에서는 경계로 다른
    # 쪽에서는 내부로 갈린다 — 그 두 길이 내놓는 꼬리표가 다르다.
    if not exceeds(gap, pts[0][1]):
        return pts[0][0], "below"
    if not exceeds(pts[-1][1], gap):
        return pts[-1][0], "above"
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if not exceeds(y0, gap) and not exceeds(gap, y1):
            if y1 == y0:
                return x0, "inside"
            # 느슨해진 만큼 바깥으로 나갈 수 있으니 구간 안으로 눌러 둔다.
            f = min(1.0, max(0.0, (gap - y0) / (y1 - y0)))
            return x0 + f * (x1 - x0), "inside"
    raise ValueError("되짚을 자리가 없다 — 상한이 어긋남에 대해 증가하지 않는다")


def pairs(board, metric, decimals):
    """(등수a, 등수b, 이름a, 이름b, 쓸 수 있는 차이) 들. 전수를 본다.

    차이에서 표시값 반올림 폭을 뺀다. 적힌 두 수가 반대쪽으로 잘렸을 때까지
    버티는 값이라야 충분조건이 된다. 뺀 뒤 0 이하면 0 으로 둔다 — 음수 차이는
    뜻이 없고, 어느 어긋남에서도 안 버틴다는 뜻은 그대로 남는다.

    그 뺄셈은 표시 격자의 정수 칸으로 한다. 적힌 두 수를 이진 부동소수로
    빼면 정확히 한 칸 떨어진 쌍에서 0 대신 1e-18 쯤이 남고, 그 부호는 두
    값이 무엇이냐에 따라 양수도 음수도 된다. 양수로 남은 쌍은 어긋남이 0
    인데도 엄격한 부등호를 통과한다 — 두 수가 서로 반대쪽으로 잘렸으면
    겹치는 쌍이라 통과하면 안 되는 자리다. 두 칸의 자리수가 같으므로
    (h_A+h_B)/2 는 정확히 한 칸이고, 쓸 수 있는 차이는 "떨어진 칸 수 빼기
    하나" 다.
    """
    s = scores(board, metric)
    grid, unit = 10 ** decimals, 10.0 ** (-decimals)
    out = []
    for ia in range(len(s)):
        for ib in range(ia + 1, len(s)):
            steps = abs(round(s[ia][1] * grid) - round(s[ib][1] * grid)) - 1
            out.append((ia + 1, ib + 1, s[ia][0], s[ib][0],
                        max(steps, 0) * unit))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep",
                    default="results/next/protocol_sweep_table.csv")
    ap.add_argument("--out", default="results/next/offset_tolerance.csv")
    ap.add_argument("--summary",
                    default="results/next/offset_tolerance_summary.csv")
    args = ap.parse_args()

    rows = []
    summary = []
    for key in sorted(LEADERBOARDS):
        board = LEADERBOARDS[key]
        print("\n출처: %s" % board["source"])
        for metric, label, decimals in board["columns"]:
            pts = sweep_taus(args.sweep, key, RULE[key], metric)
            if not pts:
                print("  %s: 훑기 값이 없다 — 건너뛴다" % label)
                continue
            if not rounds_to_display(board, metric, decimals):
                raise SystemExit("%s %s: 적어 둔 점수가 소수 %d 자리로 안 찍힌다"
                                 % (key, metric, decimals))
            h = display_term(decimals)
            # 그 칸이 실제로 찍고 있는 값의 폭. 합성 추정이 순위표 칸을
            # 덮었다고 적으려면 덮을 대상의 양 끝이 어디인지가 자료에 있어야
            # 한다.
            vals = [e[1][metric] for e in board["entries"] if metric in e[1]]
            k, dev = slope(pts)
            by_ms = dict(pts)
            # 어긋남 0 은 훑기에 없다 — 어긋남을 안 주면 재표집이 같은 자세를
            # 돌려줘서 D 가 정의상 0 이고, 잴 것이 없다. 그래도 한 줄로 둔다.
            # 그 줄에서 이미 탈락하는 쌍은 찍는 자릿수만으로 안 갈리는 쌍이라,
            # 어긋남 때문에 못 가른 쌍과 섞으면 안 된다. 그 0 이 잰 값으로
            # 읽히지 않게 줄마다 출처를 같이 적는다.
            by_ms[0.0] = 0.0
            source_of = dict.fromkeys(by_ms, "swept")
            source_of[0.0] = "exact"
            ps = pairs(board, metric, decimals)
            tol = [(a, b, na, nb, g) + tolerance_ms(g, pts)
                   for a, b, na, nb, g in ps]
            near = [t for t in tol if t[1] - t[0] == 1]
            inside = [t for t in tol if t[6] == "inside"]
            print("\n칸: %s (%s), 표시값 항 %.4g" % (label,
                                                 UNITS[metric]["unit"], h))
            print("  2D: %s"
                  % ", ".join("%g ms %.4g" % (ms, by_ms[ms])
                              for ms in sorted(by_ms)))
            print("  비례 기울기 %.5g per ms, 직선에서 벗어난 폭 %.2g"
                  % (k, dev))
            dead = sum(1 for t in near if t[6] == "dead")
            below = sum(1 for t in near if t[6] == "below")
            print("  바로 옆 등수 %d 쌍 가운데 %d 쌍은 어긋남이 0 이어도 찍는"
                  " 자리수만으로 안 갈리고, %d 쌍은 1 ms 미만에서 상한에"
                  " 덮인다" % (len(near), dead, below))
            if inside:
                print("  잰 범위 안에서 지워지는 쌍 %d, 그 어긋남 중앙 %.3g ms"
                      % (len(inside),
                         sorted(t[5] for t in inside)[len(inside) // 2]))
            print("  표 전체 폭을 지우는 어긋남: %.3g ms%s"
                  % (max(t[5] for t in tol),
                     " 초과" if any(t[6] == "above" for t in tol) else ""))
            for ms in (0.0,) + TAUS_MS:
                held = sum(1 for t in tol if exceeds(t[4], by_ms[ms]))
                print("  어긋남 %5.1f ms 에서 조건을 만족하는 쌍 %3d / %3d"
                      % (ms, held, len(tol)))
                summary.append({
                    "board": key, "metric": metric, "tau_ms": "%g" % ms,
                    "twoD": "%.6g" % by_ms[ms],
                    "twoD_source": source_of[ms],
                    "n_pairs": len(tol), "n_held": held,
                    "n_adjacent": len(near),
                    "n_adjacent_dead_at_zero": dead,
                    "n_adjacent_covered_below_1ms": below,
                    # 묶는 꼴이 확인된 순위표인지. 미확인이면 n_held 는
                    # 인증 수가 아니라 "평균 꼴이라고 보면 조건을 만족하는
                    # 쌍" 이다. 그 구분이 칸에 안 남으면 읽는 쪽이 둘을
                    # 같은 것으로 센다.
                    "aggregation": board["aggregation"],
                    # 본문이 상한·하한으로 읽는 열은 잰 값을 그대로 적는다.
                    # 여기서 깎으면 뒷단이 바깥으로 깎아도 이미 안쪽이다.
                    "display_term": repr(float(h)),
                    "col_min": repr(float(min(vals))),
                    "col_max": repr(float(max(vals))),
                    "span_tolerance_ms": repr(float(max(t[5] for t in tol))),
                    "span_tolerance_where":
                        "above" if any(t[6] == "above" for t in tol)
                        else "inside",
                    "linearity_max_rel_dev": repr(float(dev)),
                })
            for a, b, na, nb, g, ms, where in tol:
                rows.append({
                    "board": key, "metric": metric, "column": label,
                    "unit": UNITS[metric]["unit"],
                    "twoD_per_ms": "%.6g" % k,
                    "linearity_max_rel_dev": repr(float(dev)),
                    "rank_a": a, "method_a": na, "rank_b": b, "method_b": nb,
                    "display_term": repr(float(h)),
                    "gap_less_display": "%.6g" % g,
                    "tolerance_ms": "%.6g" % ms,
                    "tolerance_where": where,
                    "aggregation": board["aggregation"],
                    "source": board["source"],
                })

    if not rows:
        raise SystemExit("잴 것이 없다")
    for path, data in ((args.out, rows), (args.summary, summary)):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)
        print("\n%d 줄 -> %s" % (len(data), path))


if __name__ == "__main__":
    main()
