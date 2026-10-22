#!/usr/bin/env python3
"""기준을 조금 다르게 읽으면 구간 상대 병진이 어디서 달라지는가를 한 장에 둔다.

본문은 이 변화를 가중평균 한 수로 적는다. 그 수가 어디서 나오는지는 식으로만
읽히는데, 읽는 쪽이 머릿속에 세우기 어렵다는 지적을 받았다. 그래서 한 구간
하나를 끝까지 따라가 그림으로 둔다.

세 칸이다.
  ⑴ 궤적 전체와 그 위에 얹힌 평가 구간 하나.
  ⑵ 구간 시작 자세를 확대한 것 — 두 기준 읽기가 위치를 얼마나 옮기고 그
     자세의 앞방향을 얼마나 돌리는가.
  ⑶ 두 상대 병진 벡터의 끝점 — 둘의 차이가 그 구간에서 점수에 들어가는 양이다.

가운데 칸이 이 그림의 요점이다. 끝점이 옮겨간 거리보다 벡터 차이가 훨씬 큰데,
그 차이는 시작 자세의 회전이 100 m 짜리 벡터를 다시 표현하면서 생긴다. 끝점만
보면 센티미터고, 점수에 들어가는 것은 미터다.

수는 전부 이 대본이 같이 내는 CSV 로 나가고 본문은 그 CSV 에서 만든 매크로로만
부른다. 그림 안에 글자로 박으면 측정을 다시 돌렸을 때 그림만 옛 값으로 남는다.
"""

import argparse
import csv
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kitti_segment_D import (gather, relative_time,  # noqa: E402
                            relative_translation, resample, segments)
from protocol_sweep import window  # noqa: E402

SEQUENCE = "00"
TAU_MS = 25.0
LENGTH = 100.0


def pair(release="kitti", sequence=SEQUENCE, tau_ms=TAU_MS, length=LENGTH):
    """두 기준 읽기와 그 위에서 잰 구간들.

    훑기와 같은 길로 읽는다 — 공통 창 안의 프레임에서 바꾸지 않은 규약(null)과
    어긋난 규약을 둘 다 구면보간으로 뜨고, 구간 집합은 null 위에서 고른다.
    한쪽만 원본 행렬로 읽으면 직교 이탈이 섞여 그림이 그 몫까지 보이게 된다.
    """
    for name, (t, rot, pos) in gather(release):
        if name != sequence:
            continue
        t = relative_time(np.asarray(t, dtype=float))
        rot = np.asarray(rot, dtype=float)
        pos = np.asarray(pos, dtype=float)
        dt = float(np.median(np.diff(t)))
        lo, hi = window(t, dt)
        tw = t[(t >= lo) & (t <= hi)]
        r0, p0 = resample(t, rot, pos, tw, "slerp")
        r1, p1 = resample(t, rot, pos, tw + tau_ms * 1e-3, "slerp")
        idx = [s for s in segments(p0) if s[2] == length]
        if not idx:
            raise SystemExit("%s %s 에 %g m 구간이 없다"
                             % (release, sequence, length))
        return (r0, p0), (r1, p1), idx
    raise SystemExit("%s 에 시퀀스 %s 가 없다" % (release, sequence))


def facts(null, off, idx):
    """그림이 그리고 본문이 부르는 수들."""
    (r0, p0), (r1, p1) = null, off
    a = relative_translation(r0, p0, idx)
    b = relative_translation(r1, p1, idx)
    d = np.linalg.norm(a - b, axis=1)
    k = int(np.argmax(d))
    i, j, L = idx[k]
    # 차이를 두 몫으로 가른다. 시작 자세만 어긋난 읽기로 바꾼 중간 벡터를
    # 거쳐 가면, 앞 몫이 자세 회전이고 뒤 몫이 끝점 이동이다. 그림 ⑶ 칸이
    # 같은 둘을 그린다.
    mid = r1[i].T @ (p0[j] - p0[i])
    return {
        "sequence": SEQUENCE,
        "tau_ms": repr(TAU_MS),
        "segment_length_m": repr(float(L)),
        "n_segments": len(idx),
        "shown_start_frame": i,
        "shown_end_frame": j,
        # 보인 구간은 차이가 가장 큰 것이다. 설명문이 그렇게 적는다 — 중앙값을
        # 같이 내보내 그 구간이 묶음의 어디쯤인지 읽는 쪽이 알 수 있게 한다.
        "shown_diff_m": repr(float(d[k])),
        "median_diff_m": repr(float(np.median(d))),
        "min_diff_m": repr(float(d.min())),
        "shown_null_len_m": repr(float(np.linalg.norm(a[k]))),
        "shown_off_len_m": repr(float(np.linalg.norm(b[k]))),
        "shown_rot_part_m": repr(float(np.linalg.norm(mid - a[k]))),
        "shown_end_part_m": repr(float(np.linalg.norm(b[k] - mid))),
        "shown_start_shift_m": repr(float(np.linalg.norm(p1[i] - p0[i]))),
        "shown_end_shift_m": repr(float(np.linalg.norm(p1[j] - p0[j]))),
        # 시작 자세가 돌아간 각. 끝점 이동만으로는 설명이 안 되는 몫이 여기서
        # 나온다 — 100 m 벡터를 다시 표현하기 때문이다.
        "shown_start_turn_deg": repr(float(np.degrees(np.arccos(np.clip(
            (np.trace(r0[i].T @ r1[i]) - 1.0) / 2.0, -1.0, 1.0))))),
    }, a, b, k


def heading(rot):
    """x-z 평면에 눕힌 앞방향. 키티 자세는 카메라 축이라 z 가 앞이다."""
    v = rot[:, 2]
    return np.array([v[0], v[2]])


def panel_track(ax, p0, idx, k):
    """궤적 전체, 평가 구간들의 시작점, 그리고 보이는 구간 하나.

    시작점을 다 찍는 이유는 평가가 궤적을 어떻게 덮는지가 이 칸의 일이기
    때문이다. 구간 하나만 굵게 그리면 한 단 폭에서 100 m 가 궤적의 몇
    픽셀이라 아무것도 안 보인다 — 실제로 처음 판이 그랬다.
    """
    i, j, _ = idx[k]
    # 길은 흐리게, 구간 시작점은 진하게. 시작점이 길 위에 놓이므로 같은
    # 진하기로 그리면 선에 묻혀 평가가 궤적을 어떻게 덮는지가 안 보인다.
    ax.plot(p0[:, 0], p0[:, 2], color="0.78", linewidth=0.7)
    starts = [s[0] for s in idx]
    ax.plot(p0[starts, 0], p0[starts, 2], linestyle="none", marker=".",
            markersize=1.1, color="C0", alpha=0.75)
    ax.plot(p0[i:j + 1, 0], p0[i:j + 1, 2], color="C3", linewidth=2.2)
    ax.plot([p0[i, 0]], [p0[i, 2]], marker="o", markersize=3.0, color="C0")
    ax.plot([p0[j, 0]], [p0[j, 2]], marker="s", markersize=3.0, color="C1")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.tick_params(labelsize=5.6)


def panel_ends(ax, null, off, idx, k):
    """구간의 두 끝을 확대한다. 각 끝은 자기 기준점을 원점으로 둔 상대 좌표다.

    끝 둘을 한 칸에 겹쳐 둔다 — 궤적에서 100 m 떨어진 두 자리라 같은 좌표로는
    한 칸에 못 담고, 칸을 둘로 쪼개면 세 칸이 넷이 되어 쪽을 더 먹는다.
    여기 보이는 것은 센티미터 단위의 이동이고, ⑶ 칸의 차이는 미터다. 그 차이를
    만드는 것은 점이 옮겨간 거리가 아니라 시작 자세가 돌아간 각이다.
    """
    (r0, p0), (r1, p1) = null, off
    i, j, _ = idx[k]
    # 각 끝을 자기 기준점에 맞춘 (원점, 옮겨간 점) 네 수.
    ends = []
    for m in (i, j):
        d = np.array([p1[m, 0] - p0[m, 0], p1[m, 2] - p0[m, 2]])
        ends.append(np.array([0.0, 0.0, d[0], d[1]]))
    # 두 끝의 기준점을 한 원점에 겹쳐 두고, 어긋난 읽기가 그 점을 어디로
    # 옮기는지만 화살 둘로 그린다. 끝마다 칸을 따로 두면 세 칸이 넷이 된다.
    ax.plot([0.0], [0.0], marker="o", markersize=3.0, color="C0", zorder=3)
    for pt, marker, label, va in ((ends[0], "o", "start", "top"),
                                  (ends[1], "s", "end", "top")):
        ax.annotate("", xy=(pt[2], pt[3]), xytext=(0.0, 0.0),
                    arrowprops=dict(arrowstyle="-|>", color="C3",
                                    linewidth=0.8, mutation_scale=5))
        ax.plot([pt[2]], [pt[3]], marker=marker, markersize=3.2, color="C3",
                zorder=3)
        # 이름은 칸 안에 붙인다. 이름표를 따로 두면 칸 셋 중 하나만 머리가
        # 높아져서 세 칸의 윗선이 안 맞는다.
        ax.annotate(label, xy=(pt[2], pt[3]), fontsize=5.4, color="0.25",
                    ha="center", va=va,
                    xytext=(0, 5 if va == "bottom" else -6),
                    textcoords="offset points")
    # 두 축을 같은 폭으로 열어 칸을 정사각으로 둔다. 자동 범위에 맡기면 세로
    # 폭이 가로의 몇 분의 일이라 칸이 납작해지고 이름이 테두리 밖으로 나간다.
    lim = 1.45 * max(float(np.max(np.abs(np.asarray(ends)[:, 2:]))), 1e-6)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    # 시작 자세의 두 방향. 길이는 칸에 맞춘 것이고 각만 뜻이 있다 — 두 선이
    # 겹쳐 보이는 것이 이 칸의 요점이다. 그 각이 이 자리에서 눈에 안 보일
    # 만큼 작은데도 ⑶ 칸의 차이는 미터다.
    for rot, style in ((r0[i], "-"), (r1[i], "--")):
        h = heading(rot)
        h = 0.72 * lim * h / np.linalg.norm(h)
        ax.annotate("", xy=(h[0], h[1]), xytext=(0.0, 0.0),
                    arrowprops=dict(arrowstyle="-|>", color="C2",
                                    linewidth=0.7, linestyle=style,
                                    mutation_scale=4))
    # 이름표는 둘이 아니라 하나다. 두 방향이 1 도 남짓 차이라 이 칸에서는
    # 겹쳐 보이고, 이름을 둘 달면 글자끼리만 부딪친다. 각은 설명문이 적는다.
    h = heading(r0[i])
    h = 0.72 * lim * h / np.linalg.norm(h)
    ax.annotate("$R_s,R'_s$", xy=(h[0], h[1]), fontsize=5.4, color="C2",
                ha="center", va="bottom", xytext=(0, 3),
                textcoords="offset points")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.tick_params(labelsize=5.6)


def panel_tips(ax, null, off, idx, a, b, k):
    """차이 벡터가 어디서 오는지 두 단계로 그린다.

    q = R_s^T (p_e - p_s) 를 원점에 두고, 시작 자세만 어긋난 읽기로 바꾼
    q* = R'_s^T (p_e - p_s) 를 거쳐 양 끝까지 바꾼 q' = R'_s^T (p'_e - p'_s)
    로 간다. q -> q* 가 시작 자세 회전의 몫이고 q* -> q' 가 끝점 이동의 몫이다.
    둘의 벡터 합이 점수에 들어가는 차이 q' - q 다.
    """
    (_, p0), (r1, _) = null, off
    i, j, _ = idx[k]
    u = a[k]
    q = np.zeros(3)
    qs = r1[i].T @ (p0[j] - p0[i]) - u
    qp = b[k] - u
    n = u / np.linalg.norm(u)
    # 세 점에만 이름을 단다. 화살에까지 이름을 달면 이 칸 크기에서 글자끼리
    # 부딪친다 — 어느 화살이 어느 단계인지는 색과 설명문이 가른다.
    for pt, color, marker, label, shift in (
            (q, "C0", "o", "$q$", (-8, -2)),
            (qs, "C2", "^", "$q^{*}$", (-13, 1)),
            (qp, "C3", "o", "$q'$", (6, -2))):
        ax.plot([pt[0]], [pt[2]], marker=marker, markersize=3.0, color=color,
                zorder=3)
        ax.annotate(label, xy=(pt[0], pt[2]), xytext=shift,
                    textcoords="offset points", fontsize=5.8, color=color)
    # 두 단계 화살과 그 합. 수는 안 적는다 — 측정이 바뀌면 그림만 옛 값으로
    # 남는다. 크기는 설명문이 매크로로 부른다.
    for tail, head, color in ((q, qs, "C2"), (qs, qp, "C3")):
        ax.annotate("", xy=(head[0], head[2]), xytext=(tail[0], tail[2]),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                    linewidth=0.8, mutation_scale=5))
    # 이름표가 테두리 밖으로 안 나가게 여백을 연다. 자동 범위는 점만 담아서
    # 글자가 잘린다.
    ax.margins(0.3)
    ax.annotate("", xy=(qp[0], qp[2]), xytext=(0.0, 0.0),
                arrowprops=dict(arrowstyle="-|>", color="0.25",
                                linewidth=0.9, mutation_scale=6,
                                linestyle=":"))
    ax.annotate("$\\Delta$", xy=(0.5 * qp[0], 0.5 * qp[2]),
                xytext=(-7, 3), textcoords="offset points",
                fontsize=6.6, color="0.2", ha="right", va="bottom")
    # null 벡터가 들어오는 방향. 끝점만 보면 어느 쪽이 구간의 앞인지 안 보인다.
    ax.annotate("", xy=(0.0, 0.0), xytext=tuple(-0.8 * n[[0, 2]]),
                arrowprops=dict(arrowstyle="-", color="0.75",
                                linewidth=0.8, linestyle="--"))
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.tick_params(labelsize=5.6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper_tim/fig_mechanism.pdf")
    ap.add_argument("--csv", default="results/next/mechanism_example.csv")
    ap.add_argument("--height", type=float, default=1.3)
    args = ap.parse_args()

    null, off, idx = pair()
    row, a, b, k = facts(null, off, idx)

    plt.rcParams.update({"font.size": 6.4})
    # 한 단 폭(3.4 인치)에 세 칸을 가로로 둔다. 쪽수가 한 줄 여유라 높이는
    # 읽히는 선에서 가장 낮게 잡는다.
    fig, axes = plt.subplots(1, 3, figsize=(3.4, args.height))
    panel_track(axes[0], null[1], idx, k)
    panel_ends(axes[1], null, off, idx, k)
    panel_tips(axes[2], null, off, idx, a, b, k)
    for ax, tag in zip(axes, ("(a)", "(b)", "(c)")):
        ax.set_title(tag, fontsize=6.2, pad=2.0)
        # 눈금은 칸마다 셋까지. 기본 개수로 두면 좁은 칸에서 눈금 글자가
        # 서로 붙어 읽히지 않는다 — 확대 칸에서 실제로 그랬다.
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
    fig.tight_layout(pad=0.25, w_pad=0.6)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out)

    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    print("%s, 구간 %s -> %s, 차이 %.4f m (중앙 %.4f), 시작 회전 %.4f 도"
          % (args.out, row["shown_start_frame"], row["shown_end_frame"],
             float(row["shown_diff_m"]), float(row["median_diff_m"]),
             float(row["shown_start_turn_deg"])))


if __name__ == "__main__":
    main()
