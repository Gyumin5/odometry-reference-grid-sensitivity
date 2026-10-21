#!/usr/bin/env python3
"""순위표에 실릴 만한 크기의 추정에서 기준을 밀면 공식 점수가 얼마나 움직이나.

여기서 닫으려는 반론 하나가 있다. "순위표 점수도 같은 단정밀도 프로그램이
만든 것이니 그 점수 자체가 어긋남에 둔하고, 그래서 상한이 커도 실제 표는 안
움직인다." 이 말이 맞으면 논문의 결론이 뒤집힌다.

맞는지는 논증이 아니라 한 번 돌려 보면 갈린다. 추정을 순위표에 실린 회전오차
크기(0.0009~0.0022 도/m)로 만들어 놓고, 그 추정을 고정한 채 기준만 1·5 밀리초
민다. 그리고 공식 프로그램이 찍는 두 수의 차이를 본다.

움직인 양을 두 갈래로 나눠 적는다. 공식 프로그램은 기준을 밀면 구간을 다시
고른다 — 누적 이동거리로 끝 프레임을 잡기 때문에 시작·끝 색인이 바뀐다. 그
차이에는 시간 효과와 구간 재선택 효과가 같이 들어 있고, 상한은 구간 집합이
고정일 때만 선다. 그래서 둘을 갈라 놓는다.

  - 고정 구간에서 움직인 양. 구간 목록을 민 쪽에서 다시 고르지 않고 원래
    목록을 그대로 쓴다. 상한이 묶는 것은 이 양이다.
  - 공식 프로그램이 실제로 찍는 두 수의 차이. 여기에는 재선택이 섞인다.
  - 같은 조건에서 잰 D (상한)와 순위표가 찍는 한 칸.

표시값이 바뀌는지는 이동량을 한 칸과 견줘서 판정하지 않는다. 한 칸보다 작은
이동도 반올림 경계를 넘으면 찍히는 자리수가 바뀐다. 그래서 두 점수를 순위표
자리수로 실제로 반올림해 문자열이 달라지는지 직접 본다.

회전과 병진을 같이 본다. 순위표 칸이 둘이고 상한도 둘이라, 한쪽만 재면 다른
칸에 대해서는 아무 말도 못 한다. 손잡이는 하나다 — 프레임당 쌓는 각 하나가
두 칸을 같이 움직인다. 회전이 틀어지면 구간 이동 벡터의 방향도 같이 틀어져서
병진오차가 따라 올라가기 때문이고, 키티 00 에서 프레임당 1e-05 라디안이
회전 0.00076 도/m 와 병진 1.31 퍼센트를 같이 만든다. 위치에 배율을 따로
거는 것은 쓰지 않는다 — 구간 현의 길이가 경로 길이보다 훨씬 짧아서 배율
0.005 가 병진을 0.06 퍼센트밖에 못 올렸다. 그래서 각 하나로 두 칸을 순위표
범위(회전 0.0009~0.0022 도/m, 병진 0.53~2.72 퍼센트) 안에 같이 넣는다.
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
from kitti_segment_D import (common_window, euclidean, exact,  # noqa: E402
                             geodesic, load_poses, proper, relative,
                             relative_translation, resample, segment_weights,
                             segments)

# 순위표가 찍는 한 칸과 그 자리수. kitti_rank_check 가 자료로 들고 있는 것과
# 같아야 한다 — 회전은 네 자리(도/m), 병진은 두 자리(퍼센트)다.
DISPLAY_STEP_DEG_PER_M = 1e-4
DISPLAY_STEP_PCT = 1e-2
DISPLAY_DECIMALS_DEG_PER_M = 4
DISPLAY_DECIMALS_PCT = 2


def shown(value, decimals):
    """순위표가 그 수를 찍을 때 남는 문자열."""
    return "%.*f" % (decimals, value)


def bound(rot, pos, shifted, shifted_pos, idx):
    """그 구간 목록에서의 D 둘(도/미터, 퍼센트). 추정과 무관한 양이다."""
    w = segment_weights(idx)
    rot_d = float(np.degrees(np.dot(
        w, geodesic(relative(rot, idx), relative(shifted, idx)))))
    trans_d = 100.0 * float(np.dot(
        w, euclidean(relative_translation(rot, pos, idx),
                     relative_translation(shifted, shifted_pos, idx))))
    return rot_d, trans_d


def over_bound(moved, bound_value, what):
    """이동량을 상한으로 나눈 비. 상한이 0 인 자리를 0 으로 적지 않는다.

    상한이 0 이면 두 경우가 갈린다. 이동도 0 이면 비가 정의되지 않으니 빈
    칸으로 둔다 — 0 으로 적으면 "상한의 0 퍼센트" 라는, 묻지도 않은 비교의
    답이 가장 안전한 값으로 표에 들어간다. 이동이 0 이 아니면 그건 상한 안이
    아니라 상한을 깬 것이고, 그 자리를 수로 적을 길은 없다.
    """
    if bound_value != 0.0:
        return exact(moved / bound_value)
    if moved == 0.0:
        return ""
    raise SystemExit("%s: 상한이 0 인데 고정 구간 이동량이 %g 다 — 정리가 "
                     "깨졌거나 구간 목록이 두 쪽에서 다르다" % (what, moved))


def fixed_score(gt_rot, gt_pos, est_rot, est_pos, idx):
    """구간 목록을 고정한 채 공식 지표와 같은 꼴로 점수를 낸다.

    공식 프로그램은 기준을 밀면 구간을 다시 고르므로 그 출력의 차이에는 시간
    효과와 재선택 효과가 섞인다. 상한이 묶는 것은 구간이 고정일 때의 변화라,
    그 양을 따로 내려면 여기처럼 같은 목록을 양쪽에 쓴다. 꼴은 공식과 같다 —
    회전은 상대자세 사이의 측지각, 병진은 구간 이동 벡터 사이의 유클리드
    거리이고 둘 다 1/(N*L) 로 가중평균한다.
    """
    w = segment_weights(idx)
    rot_e = float(np.degrees(np.dot(
        w, geodesic(relative(gt_rot, idx), relative(est_rot, idx)))))
    trans_e = 100.0 * float(np.dot(
        w, euclidean(relative_translation(gt_rot, gt_pos, idx),
                     relative_translation(est_rot, est_pos, idx))))
    return rot_e, trans_e


def same_segments(a, b):
    """두 구간 목록이 시작·끝·요구길이까지 같은가."""
    return len(a) == len(b) and all(x == y for x, y in zip(a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devkit", required=True, help="devkit/cpp 디렉토리")
    ap.add_argument("--kitti-root", default="/mnt/Data/velref/kitti/dataset")
    ap.add_argument("--sequences",
                    default="00,01,02,03,04,05,06,07,08,09,10",
                    help="공개 기준 자세가 있는 키티 시퀀스 전부")
    ap.add_argument("--eps", default="1.2e-05,1.6e-05,2.0e-05",
                    help="프레임당 쌓는 각(라디안). 추정의 크기를 정한다")
    ap.add_argument("--offsets-ms", default="1,5")
    ap.add_argument("--out", default="results/next/method_scale_probe.csv")
    args = ap.parse_args()

    work = tempfile.mkdtemp(prefix="method_scale_")
    binary, cut = build(args.devkit, work)
    print("공식 함수부 %d 줄을 그대로 빌드했다 -> %s" % (cut, binary))

    rows = []
    for seq in args.sequences.split(","):
        path = os.path.join(args.kitti_root, "poses", "%s.txt" % seq)
        t_full, rot_raw, pos_full = load_poses(path)
        # 기하 비교에 들어가는 입력은 전부 눌러서 읽는다. 재표집은 쿼터니언을
        # 거쳐 반드시 직교한 자세를 돌려주므로, 한쪽만 공개 행렬 그대로 두면
        # 아무것도 안 바꾼 규약에서도 둘이 갈린다. 공개 행렬의 비직교 자체를
        # 보는 자리는 orthonormality_floor.py 와 공식 대조의 영변위 탐침이다.
        rot_full = proper(rot_raw)
        keep = common_window(t_full)
        t, rot, pos = t_full[:keep], rot_full[:keep], pos_full[:keep]
        idx = segments(pos)

        gt_path = os.path.join(work, "%s_gt.txt" % seq)
        write_poses(gt_path, rot, pos)

        for ms in [float(v) for v in args.offsets_ms.split(",")]:
            sq, sp = resample(t_full, rot_full, pos_full, t + 1e-3 * ms,
                              "slerp")
            gt_shift = os.path.join(work, "%s_gt_%gms.txt" % (seq, ms))
            write_poses(gt_shift, sq, sp)
            rot_d, trans_d = bound(rot, pos, sq, sp, idx)
            kept = same_segments(idx, segments(sp))

            for eps in [float(v) for v in args.eps.split(",")]:
                est_rot = drift_poses(rot, eps)
                est = os.path.join(work, "%s_est_%g.txt" % (seq, eps))
                write_poses(est, est_rot, pos)
                _, r0, t0 = run_official(binary, gt_path, est)
                _, r1, t1 = run_official(binary, gt_shift, est)
                r0, r1 = np.degrees(r0), np.degrees(r1)
                t0, t1 = 100.0 * t0, 100.0 * t1
                dr, dt = abs(r1 - r0), abs(t1 - t0)
                fr0, ft0 = fixed_score(rot, pos, est_rot, pos, idx)
                fr1, ft1 = fixed_score(sq, sp, est_rot, pos, idx)
                fdr, fdt = abs(fr1 - fr0), abs(ft1 - ft0)
                rot_shown = shown(r0, DISPLAY_DECIMALS_DEG_PER_M)
                rot_shown_shift = shown(r1, DISPLAY_DECIMALS_DEG_PER_M)
                trans_shown = shown(t0, DISPLAY_DECIMALS_PCT)
                trans_shown_shift = shown(t1, DISPLAY_DECIMALS_PCT)
                rows.append({
                    "sequence": seq, "offset_ms": "%g" % ms,
                    "eps_rad_per_frame": repr(float(eps)),
                    "segments_kept": "1" if kept else "0",
                    # 본문이 이 열의 최대·최소를 상한·하한으로 부른다.
                    # 저장 단계에서 깎으면 그 범위가 잰 값을 안 품는다.
                    "rot_score_deg_per_m": repr(float(r0)),
                    "rot_score_shifted_deg_per_m": "%.6g" % r1,
                    "rot_moved_deg_per_m": "%.4g" % dr,
                    "rot_fixed_moved_deg_per_m": "%.4g" % fdr,
                    "rot_bound_deg_per_m": "%.4g" % rot_d,
                    # 상한 대비와 한 칸 대비는 1 과 견주려고 적는 수다.
                    # 깎아서 적으면 1.002 가 "1" 이 되어 상한을 넘은 칸이
                    # 안 넘은 칸으로 세어진다 — 위반을 반올림이 지운다.
                    "rot_fixed_moved_over_bound": over_bound(
                        fdr, rot_d, "%s %g ms 회전" % (seq, ms)),
                    "rot_moved_over_display": exact(
                        dr / DISPLAY_STEP_DEG_PER_M),
                    "rot_shown": rot_shown,
                    "rot_shown_shifted": rot_shown_shift,
                    "rot_shown_changed": (
                        "1" if rot_shown != rot_shown_shift else "0"),
                    "trans_score_pct": repr(float(t0)),
                    "trans_score_shifted_pct": "%.6g" % t1,
                    "trans_moved_pct": "%.4g" % dt,
                    "trans_fixed_moved_pct": "%.4g" % fdt,
                    "trans_bound_pct": "%.4g" % trans_d,
                    "trans_fixed_moved_over_bound": over_bound(
                        fdt, trans_d, "%s %g ms 병진" % (seq, ms)),
                    "trans_moved_over_display": exact(
                        dt / DISPLAY_STEP_PCT),
                    "trans_shown": trans_shown,
                    "trans_shown_shifted": trans_shown_shift,
                    "trans_shown_changed": (
                        "1" if trans_shown != trans_shown_shift else "0"),
                })
                print("[%s] %g ms  eps %.1e  회전 %s -> %s  병진 %s -> %s"
                      "  고정구간 %s  구간유지 %s"
                      % (seq, ms, eps, rot_shown, rot_shown_shift,
                         trans_shown, trans_shown_shift,
                         "회전 %.3g·병진 %.3g 배 상한" % (
                             fdr / rot_d if rot_d else 0.0,
                             fdt / trans_d if trans_d else 0.0),
                         "예" if kept else "아니오"))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\n%d 행 -> %s" % (len(rows), args.out))
    print("기준을 밀어도 구간 목록이 그대로인 칸: %d / %d."
          % (len([r for r in rows if r["segments_kept"] == "1"]), len(rows)))
    for axis in ("rot", "trans"):
        over = [r for r in rows
                if float(r["%s_fixed_moved_over_bound" % axis]) > 1.0]
        big = [r for r in rows
               if float(r["%s_moved_over_display" % axis]) > 1.0]
        changed = [r for r in rows if r["%s_shown_changed" % axis] == "1"]
        print("%s: 고정 구간에서 상한을 넘은 칸 %d, 한 칸보다 크게 움직인 칸 %d,"
              " 찍히는 값이 바뀐 칸 %d / %d."
              % (axis, len(over), len(big), len(changed), len(rows)))


if __name__ == "__main__":
    main()
