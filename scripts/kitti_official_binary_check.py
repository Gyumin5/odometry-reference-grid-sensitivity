#!/usr/bin/env python3
"""공식 평가 이진파일이 찍는 수와 우리 D 가 같은지 끝에서 끝까지 맞댄다.

kitti_official_crosscheck.py 는 구간 고르기만 맞댔다. 여기서는 공식 C++ 코드를
실제로 빌드해 돌리고, 그 프로그램이 찍는 회전오차 평균과 우리가 잰 D 를 맞댄다.

맞대는 방식이 이 측정의 전부다. 공식 프로그램에 기준으로 공개된 자세 파일을,
추정으로 그 파일을 반 표본 민 자리에서 다시 읽은 궤적을 준다. 그러면 공식
프로그램이 재는 구간 오차가 글자 그대로 d(Q, Q') 이고, 그 요약이 우리 D 다.
두 수가 같으면 "공개 파일을 다른 규약으로 읽은 것만으로 공식 지표가 이만큼
움직인다" 가 우리 식이 아니라 공식 프로그램의 출력으로 적힌다.

준비물은 공식 개발도구다.

    https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_odometry.zip
    sha256 c4703456de60cdeaaec672381fbe169155ba4c2fcded102335b42cb07ff76758

풀어서 cpp 디렉토리 경로를 --devkit 으로 준다. 그 안의 evaluate_odometry.cpp 는
main 이 같이 배포되는 mail.h 와 안 맞아 그대로는 빌드가 안 되므로, main 이
시작하는 줄 앞까지만 잘라 사본을 만들고 그 사본을 끌어다 쓴다. 자른 앞부분은
원본 바이트 그대로다.
"""

import argparse
import csv
import glob
import os
import subprocess
import sys
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kitti_segment_D import (  # noqa: E402
    common_window, euclidean, geodesic, load_poses, measure, proper, relative,
    relative_translation, resample, segment_weights, segments,
)

DRIVER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "official_mean.cpp")
DRIFT_EPS = 1e-5   # 프레임마다 쌓는 각(라디안). D 가 사는 자리와 같은 크기다.
# 순위표를 읽는 자리의 어긋남. 반 표본(50 ms)보다 훨씬 작아서 공통 창 안에 든다.
OFFSETS_MS = (1.0, 5.0)


def truncate_before_main(src, dst):
    """main 이 시작하는 줄 앞까지만 옮긴다. 옮긴 부분은 바이트 그대로다."""
    with open(src, "rb") as f:
        lines = f.readlines()
    cut = None
    for i, line in enumerate(lines):
        if line.startswith(b"int32_t main") or line.startswith(b"int main"):
            cut = i
            break
    if cut is None:
        raise SystemExit("원본에서 main 을 못 찾았다: %s" % src)
    with open(dst, "wb") as f:
        f.writelines(lines[:cut])
    return cut


def build(devkit, workdir):
    funcs = os.path.join(workdir, "devkit_funcs.cpp")
    cut = truncate_before_main(os.path.join(devkit, "evaluate_odometry.cpp"),
                               funcs)
    binary = os.path.join(workdir, "official_mean")
    cmd = ["g++", "-O2", "-w", "-o", binary, DRIVER,
           os.path.join(devkit, "matrix.cpp"), "-I", devkit,
           "-DDEVKIT_FUNCS=\"%s\"" % funcs]
    subprocess.run(cmd, check=True)
    return binary, cut


def drift_poses(rot, eps, axis=(0.0, 1.0, 0.0)):
    """프레임 k 를 몸체축 기준으로 k*eps 만큼 더 돌린 궤적.

    크기를 마음대로 정할 수 있는 깨끗한 흔들림이다. 구간 오차의 닫힌 꼴을
    쓰지는 않는다 — 몸체축에 얹으면 상대회전이 켤레로 섞여서 (j-i)*eps 가
    아니다. 기준값은 아래 drift_reference 가 배정밀도로 직접 잰다.
    """
    extra = Rotation.from_rotvec(
        np.outer(np.arange(len(rot)) * eps, np.asarray(axis, dtype=float))
    ).as_matrix()
    return np.einsum("nab,nbc->nac", rot, extra)


def drift_reference(rot, pos, eps):
    """그 입력에 대한 공식 요약을 배정밀도로 직접 잰 값(라디안/미터)."""
    idx = segments(pos)
    q = relative(rot, idx)
    qd = relative(drift_poses(rot, eps), idx)
    return float(np.dot(segment_weights(idx), geodesic(q, qd)))


def single_precision_angle(err):
    """공식 rotationError 를 C 의 형변환 자리까지 그대로 흉내낸다.

        float a = pose_error.val[0][0];        // 배정밀도 -> 단정밀도
        float d = 0.5*(a+b+c-1.0);             // a+b+c 는 단정밀도 덧셈 둘
        return acos(max(min(d,1.0f),-1.0f));   // 반환형이 float

    각이 작을수록 이게 크게 번진다. 대각합이 1 에 붙어 있는데 acos 의 기울기가
    1/각 이라, 1 근처의 단정밀도 간격(약 6e-8)이 각에 반비례해 증폭된다.

    다만 이것만으로는 공식 프로그램과의 차이가 다 설명되지 않는다. 실측에서
    이 흉내는 우리 배정밀도 값에서 0.15 퍼센트쯤밖에 안 움직이는데 공식
    프로그램은 1.2 퍼센트까지 낮게 찍는다. 그래서 이 함수는 원인이 아니라
    "원인의 일부" 로만 적는다. 크기를 가르는 것은 아래 answer-known 탐침이다.
    """
    err = np.asarray(err)
    a = err[:, 0, 0].astype(np.float32)
    b = err[:, 1, 1].astype(np.float32)
    c = err[:, 2, 2].astype(np.float32)
    s = (a + b).astype(np.float32) + c           # 단정밀도 덧셈 둘
    d = (0.5 * (s.astype(np.float64) - 1.0)).astype(np.float32)
    ang = np.arccos(np.clip(d.astype(np.float64), -1.0, 1.0))
    return ang.astype(np.float32)


def single_precision_mean(ang, lengths):
    """saveStats 의 합산: float 변수에 차례로 더하고 float 개수로 나눈다."""
    acc = np.float32(0.0)
    for a, L in zip(ang, lengths):
        acc = np.float32(acc + np.float32(a / np.float32(L)))
    return float(acc / np.float32(len(ang)))


def emulated_D(t, rot, pos, shift=None):
    """우리 구간·우리 보간에 공식의 단정밀도 산술만 얹은 D(라디안/미터).

    shift 를 주면 그 초만큼 민 격자에서 잰다. 안 주면 반 표본 위상이다.
    밀리초 자리에서도 흉내가 공식 출력의 부족분을 설명하는지 보려면 같은
    어긋남에서 재야 한다 — 한 점에서만 맞으면 그 점의 우연일 수 있다.
    """
    dt = float(np.median(np.diff(t)))
    keep = common_window(t, dt)
    t_src, rot_src, pos_src = t, rot, pos
    t, rot, pos = t[:keep], rot[:keep], pos[:keep]
    idx = segments(pos)
    q = relative(rot, idx)
    step = 0.5 * dt if shift is None else shift
    moved = resample(t_src, rot_src, pos_src, t + step, "slerp")[0]
    qp = relative(moved, idx)
    err = np.einsum("nba,nbc->nac", qp, q)
    return single_precision_mean(single_precision_angle(err),
                                 [p[2] for p in idx])


def write_poses(path, rot, pos):
    """공식 프로그램에 줄 자세 파일. 배정밀도를 그대로 왕복시킨다.

    열세 자리로 적으면 공식 프로그램이 읽는 수가 우리가 잰 수와 마지막
    네 자리만큼 다르고, 그 차이가 두 출력의 차이에 섞인다. 그러면 남은
    차이가 공식 산술 때문인지 우리가 적으면서 깎은 것 때문인지 가를 수
    없다. 열일곱 자리는 배정밀도를 정확히 되살린다.
    """
    with open(path, "w") as f:
        for r, p in zip(rot, pos):
            f.write(" ".join("%.17g" % v for v in
                             [r[0, 0], r[0, 1], r[0, 2], p[0],
                              r[1, 0], r[1, 1], r[1, 2], p[1],
                              r[2, 0], r[2, 1], r[2, 2], p[2]]) + "\n")


def run_official_raw(binary, gt_path, est_path):
    """구동부 출력 다섯 칸.

    (구간 수, 회전 평균, 병진 평균, 같은 둘을 단정밀도로 누산한 값).
    뒤 둘은 공식 saveStats 와 같은 누산형이라, 우리 구동부가 배정밀도로
    바꾼 자리가 얼마를 바꾸는지 추정 없이 읽힌다.
    """
    out = subprocess.run([binary, gt_path, est_path], check=True,
                         capture_output=True, text=True).stdout.split()
    return (int(out[0]),) + tuple(float(v) for v in out[1:5])


def run_official(binary, gt_path, est_path):
    return run_official_raw(binary, gt_path, est_path)[:3]


def sweep(binary, work, rot, pos, seq, out_path):
    """흔들림 크기를 바꿔 가며 공식 프로그램의 바닥을 잰다.

    각이 작아질수록 공식 쪽이 아래로 처진다면, 그 벤치마크는 그 크기 아래의
    회전오차를 가를 수 없다는 뜻이다. D 가 사는 자리가 바로 그 근처다.
    """
    gt_path = os.path.join(work, "sweep_gt.txt")
    est_path = os.path.join(work, "sweep_est.txt")
    write_poses(gt_path, rot, pos)
    rows = []
    print("\n[%s] 흔들림 크기별 바닥" % seq)
    for k in range(-8, -1):
        eps = 10.0 ** k
        write_poses(est_path, drift_poses(rot, eps), pos)
        got = run_official(binary, gt_path, est_path)[1]
        want = drift_reference(rot, pos, eps)
        rel = abs(got - want) / want if want > 0 else 0.0
        rows.append({
            "sequence": seq, "eps_rad_per_frame": repr(float(eps)),
            "reference_deg_per_m": repr(float(np.degrees(want))),
            "official_deg_per_m": repr(float(np.degrees(got))),
            "relative_error": repr(float(rel)),
        })
        print("  eps %.0e  기준 %.6g  공식 %.6g 도/m  상대오차 %.3g"
              % (eps, np.degrees(want), np.degrees(got), rel))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("%d 행 -> %s" % (len(rows), out_path))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devkit", required=True, help="devkit/cpp 디렉토리")
    ap.add_argument("--kitti-root", default="/mnt/Data/velref/kitti/dataset")
    ap.add_argument("--out", default="results/next/kitti_official_binary.csv")
    ap.add_argument("--sweep-sequence", default="00")
    ap.add_argument("--sweep-out",
                    default="results/next/kitti_official_floor.csv")
    args = ap.parse_args()

    work = tempfile.mkdtemp(prefix="official_mean_")
    binary, cut = build(args.devkit, work)
    print("공식 함수부 %d 줄을 그대로 빌드했다 -> %s" % (cut, binary))

    rows = []
    for p in sorted(glob.glob(os.path.join(args.kitti_root, "poses", "*.txt"))):
        seq = os.path.basename(p)[:-4]
        t_full, rot_raw, pos_full = load_poses(p)
        # 공개 행렬은 정확히 직교하지 않는다. 그대로 쓰면 우리 쪽 relative 가
        # 전치를 역행렬로 쓰는 자리에서 그 이탈이 섞이고, 공식 쪽은 진짜
        # 역행렬을 쓰므로 둘의 차이가 산술 차이가 아니라 입력 차이가 된다.
        # 주 대조는 양쪽 다 눌러서 돌리고, 안 누른 경우는 아래 영변위
        # 탐침으로 따로 적는다.
        rot_full = proper(rot_raw)
        dt = float(np.median(np.diff(t_full)))
        # 공식 프로그램에 넘기는 두 파일도 주표와 같은 평가 창에서 만든다.
        # 끝을 잘라 붙인 파일을 넘기면 공식 쪽만 마지막 프레임의 어긋남이
        # 0 인 궤적을 받아서, 상대차가 구현 차이가 아니라 창 차이가 된다.
        keep = common_window(t_full, dt)
        t, rot, pos = t_full[:keep], rot_full[:keep], pos_full[:keep]
        rq, pq = resample(t_full, rot_full, pos_full, t + 0.5 * dt, "slerp")

        gt_path = os.path.join(work, "%s_gt.txt" % seq)
        est_path = os.path.join(work, "%s_half.txt" % seq)
        write_poses(gt_path, rot, pos)
        write_poses(est_path, rq, pq)

        # 답을 아는 입력 둘과 크기를 정한 흔들림 하나. 같은 궤적이면 0 이고,
        # 앞에서 같은 각만큼 더 돌려도 상대회전은 안 변하니 0 이다. 둘 다
        # 공식 정의에서 글자 그대로 따라 나오므로 근사가 아니다.
        same_path = os.path.join(work, "%s_same.txt" % seq)
        rigid_path = os.path.join(work, "%s_rigid.txt" % seq)
        drift_path = os.path.join(work, "%s_drift.txt" % seq)
        write_poses(same_path, rot, pos)
        write_poses(rigid_path, np.einsum(
            "ab,nbc->nac",
            Rotation.from_rotvec([0.0, 1e-3, 0.0]).as_matrix(), rot), pos)
        write_poses(drift_path, drift_poses(rot, DRIFT_EPS), pos)
        # 영변위 탐침. 시각을 하나도 안 밀고 재표집만 한 번 거친 궤적을
        # 추정으로 준다. 눌러서 읽으면 남는 것이 없어야 하고, 안 누르고
        # 읽으면 공개 행렬의 비직교 이탈만 남는다. 두 수의 차이가 공식
        # 대조에서 "산술 차이" 로 읽힐 뻔한 입력 오염의 크기다.
        null_prj = resample(t_full, rot_full, pos_full, t, "slerp")
        null_raw = resample(t_full, rot_raw, pos_full, t, "slerp")
        gt_raw_path = os.path.join(work, "%s_gt_raw.txt" % seq)
        null_prj_path = os.path.join(work, "%s_null_prj.txt" % seq)
        null_raw_path = os.path.join(work, "%s_null_raw.txt" % seq)
        write_poses(gt_raw_path, rot_raw[:keep], pos)
        write_poses(null_prj_path, null_prj[0], null_prj[1])
        write_poses(null_raw_path, null_raw[0], null_raw[1])
        r_null_prj = run_official(binary, gt_path, null_prj_path)[1]
        r_null_raw = run_official(binary, gt_raw_path, null_raw_path)[1]

        r_same = run_official(binary, gt_path, same_path)[1]
        r_rigid = run_official(binary, gt_path, rigid_path)[1]
        r_drift = run_official(binary, gt_path, drift_path)[1]
        want = drift_reference(rot, pos, DRIFT_EPS)
        rel_drift = abs(r_drift - want) / want

        n_off, r_off, t_off, r_off_f, t_off_f = run_official_raw(
            binary, gt_path, est_path)
        # 누산형만 바꾼 두 값의 차. 상대차와 찍는 한 칸 대비를 같이 낸다 —
        # 상대차만으로는 그 차이가 표에 보이는지를 말할 수 없다.
        accum = {
            "accum_double_rot_deg_per_m": repr(float(np.degrees(r_off))),
            "accum_float_rot_deg_per_m": repr(float(np.degrees(r_off_f))),
            "accum_double_trans_pct": repr(float(100.0 * t_off)),
            "accum_float_trans_pct": repr(float(100.0 * t_off_f)),
            "accum_rel_diff_rot": repr(
                float(abs(r_off_f - r_off) / r_off if r_off > 0 else 0.0)),
            "accum_rel_diff_trans": repr(
                float(abs(t_off_f - t_off) / t_off if t_off > 0 else 0.0)),
            "accum_diff_over_display_rot": repr(float(
                abs(np.degrees(r_off_f) - np.degrees(r_off)) / 1e-4)),
            "accum_diff_over_display_trans": repr(float(
                abs(100.0 * t_off_f - 100.0 * t_off) / 1e-2)),
        }
        # 창은 measure 와 emulated_D 가 스스로 잡는다. 이미 좁힌 배열을
        # 넘기면 두 번 좁혀서 구간 수가 공식 쪽과 안 맞는다.
        ours, n_ours = measure(t_full, rot_full, pos_full)
        d_ours = ours["phase_half"]
        d_emu = emulated_D(t_full, rot_full, pos_full)
        rel = abs(r_off - d_ours) / d_ours if d_ours > 0 else 0.0
        rel_emu = abs(r_off - d_emu) / d_emu if d_emu > 0 else 0.0

        # 반 표본 위상 한 점에서만 맞대면, 두 구현이 다른 규약에서도 같이
        # 가는지는 안 보인다. 순위표를 읽는 자리인 밀리초 어긋남에서도 맞댄다.
        extra = {}
        for ms in OFFSETS_MS:
            tag = ("%g" % ms).replace(".", "p")
            eq, ep = resample(t_full, rot_full, pos_full,
                              t + 1e-3 * ms, "slerp")
            path_ms = os.path.join(work, "%s_off%s.txt" % (seq, tag))
            write_poses(path_ms, eq, ep)
            _, r_ms, t_ms = run_official(binary, gt_path, path_ms)
            idx_ms = segments(pos)
            w_ms = segment_weights(idx_ms)
            d_ms = float(np.dot(w_ms, geodesic(relative(rot, idx_ms),
                                               relative(eq, idx_ms))))
            # 병진도 같이 맞댄다. 순위표 칸이 둘이라 회전만 대조하면 다른
            # 칸에서 두 구현이 갈리는지는 아무 말도 못 한다.
            dt_ms = float(np.dot(
                w_ms, euclidean(relative_translation(rot, pos, idx_ms),
                                relative_translation(eq, ep, idx_ms))))
            emu_ms = emulated_D(t_full, rot_full, pos_full, shift=1e-3 * ms)
            # 본문이 이 상대차들의 최댓값을 "최대 몇 퍼센트" 로 부른다.
            # 저장 단계에서 깎지 않고 잰 값을 그대로 적는다.
            extra["D_ours_%sms_deg_per_m" % tag] = repr(
                float(np.degrees(d_ms)))
            extra["D_official_%sms_deg_per_m" % tag] = repr(
                float(np.degrees(r_ms)))
            extra["D_emulated_%sms_deg_per_m" % tag] = repr(
                float(np.degrees(emu_ms)))
            extra["relative_difference_%sms" % tag] = repr(
                float(abs(r_ms - d_ms) / d_ms if d_ms > 0 else 0.0))
            extra["relative_difference_emulated_%sms" % tag] = repr(
                float(abs(r_ms - emu_ms) / emu_ms if emu_ms > 0 else 0.0))
            extra["D_ours_trans_%sms_pct" % tag] = repr(float(100.0 * dt_ms))
            extra["D_official_trans_%sms_pct" % tag] = repr(float(100.0 * t_ms))
            extra["relative_difference_trans_%sms" % tag] = repr(
                float(abs(t_ms - dt_ms) / dt_ms if dt_ms > 0 else 0.0))

        rows.append({
            "sequence": seq,
            "n_segments_ours": n_ours, "n_segments_official": n_off,
            # 잰 값을 그대로 적는다. 본문이 이 열들에서 "최대 이만큼" 을
            # 뽑아 상한으로 부르므로, 저장 단계에서 먼저 깎으면 뒷단이
            # 바깥으로 깎아도 이미 잰 값보다 안쪽이다.
            "D_ours_deg_per_m": repr(float(np.degrees(d_ours))),
            "D_official_deg_per_m": repr(float(np.degrees(r_off))),
            "D_single_precision_emulated_deg_per_m":
                repr(float(np.degrees(d_emu))),
            "relative_difference": repr(float(rel)),
            "relative_difference_vs_emulated": repr(float(rel_emu)),
            "probe_identical_rad_per_m": repr(float(r_same)),
            "null_official_projected_rad_per_m": repr(float(r_null_prj)),
            "null_official_raw_rad_per_m": repr(float(r_null_raw)),
            "probe_global_rotation_rad_per_m": repr(float(r_rigid)),
            "probe_drift_relative_error": repr(float(rel_drift)),
            **accum,
            **extra,
        })
        print("[%s] 구간 우리 %5d 공식 %5d  D 우리 %.6g 공식 %.6g 흉내 %.6g "
              "도/m  상대차 %.3g  흉내와의 상대차 %.3g  탐침(같음 %.1g, 전역회전"
              " %.1g, 점증 상대오차 %.3g)"
              % (seq, n_ours, n_off, np.degrees(d_ours), np.degrees(r_off),
                 np.degrees(d_emu), rel, rel_emu, r_same, r_rigid, rel_drift))

    if not rows:
        print("잴 것이 없다")
        return
    worst = max(float(r["relative_difference"]) for r in rows)
    worst_emu = max(float(r["relative_difference_vs_emulated"]) for r in rows)
    worst_same = max(float(r["probe_identical_rad_per_m"]) for r in rows)
    worst_rigid = max(float(r["probe_global_rotation_rad_per_m"]) for r in rows)
    worst_drift = max(float(r["probe_drift_relative_error"]) for r in rows)
    print("\n시퀀스 %d 개에서 공식 프로그램 출력과 우리 D 의 상대차는 최대 %.3g,"
          " 단정밀도를 흉내낸 값과의 상대차는 최대 %.3g 다."
          % (len(rows), worst, worst_emu))
    print("누산형만 바꾼 차: 회전 상대차 최대 %.3g (찍는 한 칸의 %.3g 배),"
          " 병진 상대차 최대 %.3g (한 칸의 %.3g 배)."
          % (max(float(r["accum_rel_diff_rot"]) for r in rows),
             max(float(r["accum_diff_over_display_rot"]) for r in rows),
             max(float(r["accum_rel_diff_trans"]) for r in rows),
             max(float(r["accum_diff_over_display_trans"]) for r in rows)))
    print("답을 아는 입력: 같은 궤적 최대 %.3g, 전역 회전 최대 %.3g, 프레임마다"
          " %.0e 씩 쌓은 궤적의 상대오차 최대 %.3g."
          % (worst_same, worst_rigid, DRIFT_EPS, worst_drift))
    for ms in OFFSETS_MS:
        tag = ("%g" % ms).replace(".", "p")
        print("어긋남 %g ms: 회전 상대차 최대 %.3g, 병진 최대 %.3g,"
              " 단정밀도 흉내와의 회전 상대차 최대 %.3g."
              % (ms,
                 max(float(r["relative_difference_%sms" % tag])
                     for r in rows),
                 max(float(r["relative_difference_trans_%sms" % tag])
                     for r in rows),
                 max(float(r["relative_difference_emulated_%sms" % tag])
                     for r in rows)))


    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("%d 행 -> %s" % (len(rows), args.out))

    sp = os.path.join(args.kitti_root, "poses", "%s.txt" % args.sweep_sequence)
    if os.path.exists(sp):
        _, srot_raw, spos = load_poses(sp)
        # 바닥 훑기도 눌러서 읽는다. 이 파일 안의 다른 기하 비교와 같은
        # 규약이고, 공개 행렬의 비직교를 보는 자리는 위 영변위 탐침이다.
        srot = proper(srot_raw)
        sweep(binary, work, srot, spos, args.sweep_sequence,
              args.sweep_out)


if __name__ == "__main__":
    main()
