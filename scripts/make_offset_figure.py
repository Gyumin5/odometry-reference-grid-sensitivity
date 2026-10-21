#!/usr/bin/env python3
"""어긋남에 대한 상한과 그 아래 깔린 두 바닥을 한 장에 그린다.

본문이 말로 적은 것 셋을 한 눈에 둔다. 상한은 어긋남에 비례한다는 것, 그
비례선이 순위표가 찍는 한 칸을 어디에서 지나는가, 그리고 공개 행렬을 날로
읽을 때 생기는 수치 잔여가 그 선의 어느 높이에 걸리는가.

수는 전부 커밋된 CSV 에서 읽는다. 손으로 옮겨 적으면 측정을 다시 돌렸을 때
그림만 옛 값으로 남는다.
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402

LABELS = {
    "kitti": "KITTI", "kitti360": "KITTI-360", "boreas": "Boreas",
    "helipr": "HeLiPR", "nuscenes": "nuScenes", "pit30m": "Pit30M",
    "tumrgbd": "TUM RGB-D", "nclt": "NCLT", "tumvi": "TUM VI",
    "euroc": "EuRoC MAV",
}
MARKERS = {"kitti": "o", "kitti360": "s", "boreas": "^", "helipr": "v",
           "nuscenes": "D", "pit30m": "x", "tumrgbd": "+", "nclt": "*",
           "tumvi": "P", "euroc": "h"}


def read_table(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def series(rows, release, rule, column):
    """그 릴리스·규칙의 (어긋남 ms, 2D) 들. 표가 적는 D 를 2D 로 옮긴다."""
    out = []
    for r in rows:
        if r["release"] != release or r["rule"] != rule:
            continue
        if r["axis"] != "tau" or not r["param"].endswith("ms"):
            continue
        out.append((float(r["param"][:-2]), 2.0 * float(r[column])))
    return sorted(out)


def display_step(rows, board, metric):
    """그 순위표 칸이 찍는 한 칸. 여러 줄에 같은 값이 적혀 있다."""
    for r in rows:
        if r["board"] == board and r["metric"] == metric:
            return float(r["display_term"])
    return None


def worst_floor(rows, rule, releases):
    """그 규칙에서 가장 높은 수치 잔여. 공개 행렬을 날로 읽을 때의 값이다.

    그림이 그리는 릴리스만 센다. 설명문이 이 선을 "그려진 선들의 잔여 중 가장
    큰 것" 으로 읽는데, 선이 없는 릴리스가 최대를 쥐면 그림에 근거가 없는
    높이에 선이 그어진다.
    """
    vals = [float(r["floor_deg_per_100m"]) for r in rows
            if r["rule"] == rule and r["release"] in releases]
    if not vals:
        raise SystemExit("%s 규칙에서 그림이 그리는 릴리스의 잔여가 없다" % rule)
    return max(vals)


def panel(ax, sweep, rule, column, releases):
    for rel in releases:
        pts = series(sweep, rel, rule, column)
        if not pts:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                marker=MARKERS[rel], markersize=3.5, linewidth=1.0,
                label=LABELS[rel])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("reference offset $\\tau$ (ms)")
    ax.grid(True, which="major", linewidth=0.4, alpha=0.5)
    ax.grid(True, which="minor", linewidth=0.3, alpha=0.25)


def mark(ax, y, style, label, side="right", above=True):
    """가로선 하나와 그 이름.

    한 칸에 선이 둘이면 좌우로 갈라 붙인다. 두 선이 한 자릿수 안에 붙어
    있어서 위아래로 가르면 두 이름이 같은 틈에서 만나고, 같은 쪽에 두면
    글자가 겹친다.
    """
    ax.axhline(y, color="0.35", linestyle=style, linewidth=0.9)
    ax.text(0.99 if side == "right" else 0.01,
            y * (1.25 if above else 0.72), label, fontsize=7,
            color="0.3", ha=side,
            transform=ax.get_yaxis_transform(),
            va="bottom" if above else "top")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="results/next/protocol_sweep_table.csv")
    ap.add_argument("--floor",
                    default="results/next/orthonormality_floor.csv")
    ap.add_argument("--boards",
                    default="results/next/offset_tolerance_summary.csv")
    ap.add_argument("--out", default="paper_tim/fig_offset.pdf")
    args = ap.parse_args()

    sweep = read_table(args.sweep)
    floors = read_table(args.floor)
    boards = read_table(args.boards)
    rule = "seg"
    releases = [r for r in LABELS
                if series(sweep, r, rule, "D_trans_pct")]

    # 한 단 폭(3.4 인치)에 두 칸을 세로로 쌓는다. 두 단을 가로지르는 그림은
    # 그 자리에 다른 글이 못 와서 쪽을 더 먹는다.
    plt.rcParams.update({"font.size": 7.0})
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 2.05), sharex=True)
    panel(axes[0], sweep, rule, "D_trans_pct", releases)
    # 두 칸 다 2D 를 그린다는 것은 설명문이 말한다. 이름에 다시 적으면
    # 칸이 낮아졌을 때 두 세로 이름이 서로 겹친다.
    # 이름을 두 줄로 끊는다. 한 줄로 두면 글자의 세로 길이가 제 칸보다 길어져
    # 위아래 두 이름이 서로를 덮는다 — 칸 높이를 2.05 인치로 내렸을 때 실제로
    # 그렇게 찍혔다(원고 그림에서 "rotation (deg/100 m)" 이 "translation (%)"
    # 위로 올라갔다). 끊으면 어느 칸 높이에서도 제 칸 안에 들어간다.
    axes[0].set_ylabel("translation\n(%)")
    step = display_step(boards, "kitti", "translation")
    if step is not None:
        mark(axes[0], step, "--", "printed step")

    panel(axes[1], sweep, rule, "D_rot_deg_per_100m", releases)
    axes[1].set_ylabel("rotation\n(deg/100 m)")
    rstep = display_step(boards, "kitti", "rotation")
    if rstep is not None:
        # 순위표 회전 칸은 도/m 로 찍는다. 그림의 단위로 옮긴다.
        mark(axes[1], 100 * rstep, "--", "printed step")
    res = worst_floor(floors, rule, releases)
    # 두 선이 한 자릿수 안에 붙어 있다. 왼쪽은 선들이 낮게 깔려 글자가 그
    # 위에 얹히므로, 둘 다 오른쪽에 두되 위아래로 가른다 — 큰 어긋남 쪽은
    # 선들이 이 높이를 지나 올라가 있어서 빈 자리다.
    mark(axes[1], res, ":", "numerical residue", above=False)
    # 잔여선은 그려진 선들보다 낮아서 칸의 바닥에 붙는다. 그 아래로 한 칸을
    # 더 열어야 이름이 테두리와 가로축 눈금 위에 얹히지 않는다.
    axes[1].set_ylim(bottom=res / 3.0)

    # 세로로 쌓으면 가로축이 하나라 위 칸의 이름은 지운다.
    axes[0].set_xlabel("")
    # 이름표는 아래 칸의 왼쪽 위에 둔다. 위 칸은 선들이 왼쪽 아래에서
    # 오른쪽 위로 가로질러서 빈 구석이 없다 — 거기 두면 글자가 선과 표식
    # 위에 얹힌다. 아래 칸은 가장 높은 선조차 왼쪽에서 한참 낮게 시작해
    # 왼쪽 위가 비어 있다. 세 줄로 접어 그 빈 자리 안에 들어가게 한다.
    axes[1].legend(fontsize=5.8, loc="upper left", frameon=False,
                   ncol=3, handlelength=1.3, columnspacing=0.8,
                   labelspacing=0.2)
    fig.tight_layout(pad=0.4)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out)
    print("%s -> %s" % (", ".join(LABELS[r] for r in releases), args.out))


if __name__ == "__main__":
    main()
