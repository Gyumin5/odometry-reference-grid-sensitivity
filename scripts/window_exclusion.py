#!/usr/bin/env python3
"""공통 창이 공식 전체 집합에서 무엇을 덜어내고 점수를 얼마나 옮기나.

재표집 비교는 공통 창 안에서만 한다. 창은 반 표본 위상과 반 주기 솎기가 둘 다
공개 표본 안에 들도록 꼬리를 조금 자른다. 그러면 공식 평가가 쓰는 구간 집합의
일부만 남고, 거기서 두 가지를 물을 수 있다.

  - 몇 개가 빠지나. 구간 수로 센다.
  - 빠진 만큼 점수가 어디로 움직이나.

둘째는 기준만으로는 못 묻는다 — 점수는 추정이 있어야 생긴다. 그래서 키티에서는
공식 프로그램에 합성 추정을 넣고 전체 집합과 창 집합에서 각각 돌려 그 차이를
직접 잰다. 어긋남은 안 넣는다. 여기서 보는 것은 어긋남의 영향이 아니라 창이
남기는 편향이다.

점수가 거의 안 움직이는 자리에서는 상대차 대신 절대차를 적는다. 작은 수를 작은
수로 나누면 보고된 비가 분모의 잡음을 따라가고, 그 비가 몇 퍼센트인지는 그
자리에서 아무것도 말해 주지 않는다. 같은 줄에 찍는 한 칸 대비를 같이 적는다 —
그게 그 차이가 표에 보이는지를 가르는 유일한 기준이다.
"""

import argparse
import csv
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kitti_official_binary_check import (build, drift_poses,  # noqa: E402
                                         run_official, write_poses)
from kitti_segment_D import (LENGTHS, RELEASES, STEP,  # noqa: E402
                             common_window, euclidean, exact, gather,
                             geodesic, load_poses, proper, relative,
                             relative_time, relative_translation, segments)
from protocol_sweep import window as sweep_window  # noqa: E402
from protocol_sweep import window_segments, window_segments_original  # noqa: E402,E501

DISPLAY_STEP_DEG_PER_M = 1e-4
DISPLAY_STEP_PCT = 1e-2

def short(s):
    """화면에 찍을 때만 깎는다. 파일에 적히는 수는 깎지 않는다."""
    return "%.3g" % float(s) if s else "-"


def counts(release, lengths, step):
    """그 릴리스에서 전체 집합과 두 창 집합의 구간 수.

    창이 둘인 것은 실험이 둘이기 때문이다. 주표는 공개 표본 첫 점부터 읽고
    꼬리만 자르지만, 규약 훑기는 홀수 솎기를 포함해서 원본이 t[1] 부터
    시작한다. 둘을 한 수로 적으면 어느 실험의 제외량인지가 사라진다.
    """
    full = main = swp = 0
    for _, (t, _, pos) in gather(release):
        # 시각 표현을 재는 쪽에 맞춘다. 재는 쪽(kitti_segment_D.measure 와
        # protocol_sweep.measure)이 첫 점을 빼고 창을 잡으므로, 여기서 안 빼면
        # 유닉스 기원 릴리스에서 창이 한두 프레임 다르게 잡혀 제외량이 다른
        # 구간 집합에서 세어진다.
        t = relative_time(t)
        full += len(segments(pos, lengths=lengths, step=step))
        keep = common_window(t)
        main += len(segments(pos[:keep], lengths=lengths, step=step))
        # 훑기가 실제로 건너뛰는 시퀀스는 0 으로 센다. 거기 구간을 살려
        # 두면 훑기가 안 읽은 것을 읽은 것처럼 적는 셈이다. 조건은
        # protocol_sweep.measure 의 것을 그대로 쓴다.
        if len(t) < 8:
            continue
        dt = float(np.median(np.diff(t)))
        lo, hi = sweep_window(t, dt)
        if not (hi > lo):
            continue
        m = (t >= lo) & (t <= hi)
        swp += len(window_segments(pos, m, lengths=lengths, step=step))
    return full, main, swp


def shift_rows(args):
    """릴리스마다 두 규칙의 구간 수와 두 창이 덜어낸 비율."""
    rows = []
    for release in RELEASES:
        for tag, lengths, step in (("seg", LENGTHS, STEP),
                                   ("rpe1m", (1.0,), 1)):
            full, main, swp = counts(release, lengths, step)
            if full == 0:
                continue
            rows.append({
                "release": release, "rule": tag,
                "n_segments_full": full,
                "n_segments_window": main,
                "n_excluded": full - main,
                # 잰 값을 그대로 적는다. 깎는 것은 본문에 실을 때 방향을
                # 정해 한 번만 한다.
                "excluded_pct": repr(100.0 * (full - main) / full),
                "n_segments_sweep_window": swp,
                "n_excluded_sweep": full - swp,
                "excluded_sweep_pct": repr(100.0 * (full - swp) / full),
            })
            print("[%s %s] 전체 %d  주표 창 %d (%.4g %%)  훑기 창 %d (%.4g %%)"
                  % (release, tag, full, main,
                     100.0 * (full - main) / full, swp,
                     100.0 * (full - swp) / full))
    return rows


def per_segment(gt_rot, gt_pos, est_rot, est_pos, idx):
    """구간마다 길이로 나눈 오차 둘. 공식 지표가 한 자루에 담는 항이 이것이다."""
    L = np.asarray([p[2] for p in idx], dtype=float)
    rot = np.degrees(geodesic(relative(gt_rot, idx),
                              relative(est_rot, idx))) / L
    trans = 100.0 * euclidean(
        relative_translation(gt_rot, gt_pos, idx),
        relative_translation(est_rot, est_pos, idx)) / L
    return rot, trans


def sets_for(t, pos, lengths, step):
    """그 시퀀스의 세 구간 집합. 색인은 전부 원본 기준이다.

    시각 표현은 재는 쪽과 같다(relative_time). counts 와 같은 까닭이다.
    """
    t = relative_time(t)
    out = {"full": segments(pos, lengths=lengths, step=step)}
    keep = common_window(t)
    mask = np.zeros(len(t), dtype=bool)
    mask[:keep] = True
    out["main"] = window_segments_original(pos, mask, lengths, step)
    out["sweep"] = []
    if len(t) >= 8:
        dt = float(np.median(np.diff(t)))
        lo, hi = sweep_window(t, dt)
        if hi > lo:
            out["sweep"] = window_segments_original(
                pos, (t >= lo) & (t <= hi), lengths, step)
    return out


def pooled_from(seqs, lengths, step, eps):
    """시퀀스 묶음에서 전체 집합과 두 창 집합의 점수. 어긋남은 안 넣는다.

    공식 집계 규칙 그대로 구간을 한 자루에 담아 평균한다 — 시퀀스마다 평균을
    내고 그 평균을 다시 평균하지 않는다. 둘은 시퀀스 길이가 다르면 다른 수다.
    추정은 공개 기준에 프레임마다 같은 각을 얹은 합성이고 위치는 기준
    그대로다 — 회전이 틀어지면 구간 이동 벡터도 같이 틀어져서 두 칸이 함께
    움직인다.
    """
    bags = {k: ([], []) for k in ("full", "main", "sweep")}
    for t, rot, pos in seqs:
        est = drift_poses(rot, eps)
        for key, idx in sets_for(t, pos, lengths, step).items():
            if not idx:
                continue
            r, p = per_segment(rot, pos, est, pos, idx)
            bags[key][0].append(r)
            bags[key][1].append(p)
    out = {}
    for key, (rs, ps) in bags.items():
        if not rs:
            out[key] = None
            continue
        out[key] = (float(np.mean(np.concatenate(rs))),
                    float(np.mean(np.concatenate(ps))),
                    int(sum(len(a) for a in rs)))
    return out


def pooled_bias(release, lengths, step, eps):
    """그 릴리스의 시퀀스를 전부 읽어 pooled_from 에 넘긴다.

    공식 프로그램이 없는 릴리스에서도 집합 교체의 효과를 같은 꼴로 잴 수 있다.
    """
    seqs = (v for _, v in gather(release))
    return pooled_from(seqs, lengths, step, eps)


def bias_all_rows(args):
    """일곱 릴리스 전부에서 창이 점수를 어디로 옮기나.

    키티는 공식 프로그램이 같은 양을 찍어 주지만 나머지 여섯에는 그런 구현이
    없다. 그래서 공식과 같은 꼴을 배정밀도로 직접 계산해 열세 줄을 채운다.
    공식 출력과의 대조는 키티 한 줄에서 따로 한다 — 그 줄이 이 계산이 공식과
    같은 양을 내는지 보증하는 자리다.
    """
    eps = float(args.pooled_eps)
    rows = []
    for release in RELEASES:
        for tag, lengths, step in (("seg", LENGTHS, STEP),
                                   ("rpe1m", (1.0,), 1)):
            got = pooled_bias(release, lengths, step, eps)
            if got["full"] is None:
                continue
            rot_f, trans_f, n_f = got["full"]
            row = {"release": release, "rule": tag,
                   "eps_rad_per_frame": repr(float(eps)),
                   "n_segments_full": n_f,
                   "rot_full_deg_per_m": "%.6g" % rot_f,
                   "trans_full_pct": "%.6g" % trans_f}
            for key in ("main", "sweep"):
                if got[key] is None:
                    row.update({"n_segments_%s" % key: 0,
                                "rot_%s_abs_diff" % key: "",
                                "rot_%s_over_display" % key: "",
                                "rot_%s_rel_pct" % key: "",
                                "trans_%s_abs_diff" % key: "",
                                "trans_%s_over_display" % key: "",
                                "trans_%s_rel_pct" % key: ""})
                    continue
                rot_w, trans_w, n_w = got[key]
                d_rot, d_trans = abs(rot_w - rot_f), abs(trans_w - trans_f)
                # 한 칸 대비와 자기 점수 대비를 같이 적는다. 한 칸은 키티
                # 순위표의 격자라 순위표가 없는 릴리스에서는 빌려 쓴 자다 —
                # 점수가 키티와 다른 자릿수인 릴리스에서는 그 자가 작아 보이는
                # 쪽으로도 커 보이는 쪽으로도 기울 수 있다. 자기 점수로 나눈
                # 비는 그 기울기가 없다.
                row.update({
                    "n_segments_%s" % key: n_w,
                    "rot_%s_abs_diff" % key: exact(d_rot),
                    "rot_%s_over_display" % key: exact(
                        d_rot / DISPLAY_STEP_DEG_PER_M),
                    "rot_%s_rel_pct" % key: (exact(100.0 * d_rot / rot_f)
                                             if rot_f else ""),
                    "trans_%s_abs_diff" % key: exact(d_trans),
                    "trans_%s_over_display" % key: exact(
                        d_trans / DISPLAY_STEP_PCT),
                    "trans_%s_rel_pct" % key: (
                        exact(100.0 * d_trans / trans_f)
                        if trans_f else ""),
                })
            rows.append(row)
            print("[%s %s] 전체 점수 회전 %.6g 병진 %.6g  주표 창 차 %s/%s"
                  "  훑기 창 차 %s/%s (한 칸 대비), 자기 점수 대비 %s/%s %%"
                  % (release, tag, rot_f, trans_f,
                     short(row["rot_main_over_display"]),
                     short(row["trans_main_over_display"]),
                     short(row["rot_sweep_over_display"]),
                     short(row["trans_sweep_over_display"]),
                     short(row["rot_main_rel_pct"]),
                     short(row["trans_main_rel_pct"])))
    return rows


def bias_rows(args):
    """키티에서 창이 공식 점수를 어디로 옮기나. 어긋남 없이 집합만 바꾼다."""
    work = tempfile.mkdtemp(prefix="window_exclusion_")
    binary, cut = build(args.devkit, work)
    print("\n공식 함수부 %d 줄을 그대로 빌드했다 -> %s" % (cut, binary))

    rows = []
    for seq in args.sequences.split(","):
        path = os.path.join(args.kitti_root, "poses", "%s.txt" % seq)
        t, rot_raw, pos = load_poses(path)
        # 기하 비교에 들어가는 입력은 눌러서 읽는다. 공식 대조와 같은 규약이고,
        # 공개 행렬의 비직교 자체를 보는 자리는 거기 영변위 탐침이다.
        rot = proper(rot_raw)
        keep = common_window(t)
        full_gt = os.path.join(work, "%s_gt_full.txt" % seq)
        win_gt = os.path.join(work, "%s_gt_win.txt" % seq)
        write_poses(full_gt, rot, pos)
        write_poses(win_gt, rot[:keep], pos[:keep])

        for eps in [float(v) for v in args.eps.split(",")]:
            est = drift_poses(rot, eps)
            full_est = os.path.join(work, "%s_est_full_%g.txt" % (seq, eps))
            win_est = os.path.join(work, "%s_est_win_%g.txt" % (seq, eps))
            write_poses(full_est, est, pos)
            write_poses(win_est, est[:keep], pos[:keep])
            nf, rf, tf = run_official(binary, full_gt, full_est)
            nw, rw, tw = run_official(binary, win_gt, win_est)
            rf, rw = np.degrees(rf), np.degrees(rw)
            tf, tw = 100.0 * tf, 100.0 * tw
            rows.append({
                "sequence": seq, "eps_rad_per_frame": repr(float(eps)),
                "n_frames_full": len(t), "n_frames_window": int(keep),
                "n_segments_full": nf, "n_segments_window": nw,
                "rot_full_deg_per_m": "%.6g" % rf,
                "rot_window_deg_per_m": "%.6g" % rw,
                "rot_abs_diff_deg_per_m": exact(abs(rw - rf)),
                "rot_diff_over_display": exact(
                    abs(rw - rf) / DISPLAY_STEP_DEG_PER_M),
                "trans_full_pct": "%.6g" % tf,
                "trans_window_pct": "%.6g" % tw,
                "trans_abs_diff_pct": exact(abs(tw - tf)),
                "trans_diff_over_display": exact(
                    abs(tw - tf) / DISPLAY_STEP_PCT),
            })
            print("[%s] eps %.1e  구간 %d -> %d  회전 차 %.4g (한 칸 %.3g)"
                  "  병진 차 %.4g (한 칸 %.3g)"
                  % (seq, eps, nf, nw, abs(rw - rf),
                     abs(rw - rf) / DISPLAY_STEP_DEG_PER_M, abs(tw - tf),
                     abs(tw - tf) / DISPLAY_STEP_PCT))
    return rows


def write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("%d 행 -> %s" % (len(rows), path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devkit", required=True, help="devkit/cpp 디렉토리")
    ap.add_argument("--kitti-root", default="/mnt/Data/velref/kitti/dataset")
    ap.add_argument("--sequences",
                    default="00,01,02,03,04,05,06,07,08,09,10")
    ap.add_argument("--eps", default="1.2e-05,1.6e-05,2.0e-05")
    ap.add_argument("--pooled-eps", default="1.6e-05",
                    help="일곱 릴리스 묶음 측정에 쓰는 프레임당 각")
    ap.add_argument("--out-counts",
                    default="results/next/window_exclusion_counts.csv")
    ap.add_argument("--out-bias",
                    default="results/next/window_exclusion_bias.csv")
    ap.add_argument("--out-pooled",
                    default="results/next/window_exclusion_pooled.csv")
    args = ap.parse_args()

    write(args.out_counts, shift_rows(args))
    pooled = bias_all_rows(args)
    write(args.out_pooled, pooled)
    rows = bias_rows(args)
    write(args.out_bias, rows)
    print("\n찍는 한 칸보다 크게 옮긴 칸: 회전 %d, 병진 %d / %d."
          % (len([r for r in rows
                  if float(r["rot_diff_over_display"]) > 1.0]),
             len([r for r in rows
                  if float(r["trans_diff_over_display"]) > 1.0]),
             len(rows)))
    over = [r for r in pooled
            for k in ("main", "sweep")
            if r["rot_%s_over_display" % k]
            and (float(r["rot_%s_over_display" % k]) > 1.0
                 or float(r["trans_%s_over_display" % k]) > 1.0)]
    print("일곱 릴리스 묶음: 두 창 중 한 쪽이라도 한 칸보다 크게 옮긴 줄 %d / %d."
          % (len(set(id(r) for r in over)), len(pooled)))


if __name__ == "__main__":
    main()
