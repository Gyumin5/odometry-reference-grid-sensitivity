#!/usr/bin/env python3
"""공식 평가 코드와 우리 구간 선택이 같은 구간을 고르는지 맞댄다.

kitti_segment_D.py 는 공식 지표의 단위로 D 를 적는다. 그 수가 쓸모 있으려면
구간을 공식과 같은 규칙으로 골라야 한다 — 끝 프레임이 한 칸만 달라도 그 구간의
상대회전이 달라지고, 길이로 나누는 정규화도 같이 어긋난다.

맞대는 상대는 공식 개발도구의 C++ 원본이다.

    https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_odometry.zip
    sha256 c4703456de60cdeaaec672381fbe169155ba4c2fcded102335b42cb07ff76758
    cpp/evaluate_odometry.cpp
    sha256 4dab8a5a770f0ded7eb49bb5e8cc3e1bd243feb1f4475ebcb826ee7ff7aa51a7

아래 official_segments 는 그 파일의 trajectoryDistances +
lastFrameFromSegmentLength + calcSequenceErrors 바깥 루프를 글자 그대로 옮긴
것이다. 벡터화하지 않았다 — 빠르게 만들려고 고쳐 쓰면 맞대는 의미가 없다.

원본이 float(단정밀도)로 거리를 누적하므로 dtype 을 받는다. 공식 이진파일과
같은 조건은 float32 이고, 우리 구현은 float64 다. 둘을 따로 세어 그 차이가
규칙 차이인지 누적 반올림인지 가른다.

맞대는 것은 개수가 아니라 목록이다. 구간 수가 같아도 시작·끝이 다를 수 있어서
개수만 비교하면 일치를 과장한다. --check 는 (시작, 끝, 요구길이) 세쌍을 그대로
맞대고 어긋난 수를 따로 적는다.
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kitti_segment_D import (  # noqa: E402
    common_window, geodesic, load_poses, proper, relative, resample,
    segment_weights, segments,
)

# evaluate_odometry.cpp 의 lengths[] 와 step_size 그대로.
OFFICIAL_LENGTHS = (100, 200, 300, 400, 500, 600, 700, 800)
OFFICIAL_STEP = 10


def official_distances(pos, dtype="float64"):
    """trajectoryDistances: 프레임 사이 직선거리를 차례로 더한다.

    산술 순서를 원본에 맞춘다. Matrix 의 원소는 double 이므로(matrix.h 46줄
    typedef double FLOAT) 뺄셈은 배정밀도에서 일어나고, 그 결과를 받는 변수가
    float 이라 차에서 한 번 잘린다. 좌표를 먼저 자르고 빼면 마지막 자리가
    달라져 누적 거리가 갈리고, 그 거리가 끝 프레임을 고르므로 구간 자체가
    달라진다.
    """
    cast = np.dtype(dtype).type
    pos = np.asarray(pos, dtype="float64")
    dist = [cast(0.0)]
    for i in range(1, len(pos)):
        dx = cast(pos[i - 1][0] - pos[i][0])
        dy = cast(pos[i - 1][1] - pos[i][1])
        dz = cast(pos[i - 1][2] - pos[i][2])
        dist.append(cast(dist[i - 1]
                         + np.sqrt(cast(dx * dx + dy * dy + dz * dz))))
    return dist


def official_last_frame(dist, first_frame, length):
    """lastFrameFromSegmentLength: 넘는 첫 프레임. 없으면 -1.

    문턱 덧셈도 누적 거리와 같은 정밀도에서 한다 — 원본의 dist 와 len 이 둘 다
    float 이라 그 합이 float 이다. 길이를 파이썬 수로 두면 덧셈이 배정밀도로
    올라가서 경계에 걸린 프레임의 판정이 뒤집힌다.
    """
    like = type(dist[0])
    length = like(length)
    for i in range(first_frame, len(dist)):
        if dist[i] > like(dist[first_frame] + length):
            return i
    return -1


def official_segments(pos, dtype="float64", lengths=OFFICIAL_LENGTHS,
                      step=OFFICIAL_STEP):
    """calcSequenceErrors 의 구간 고르기. (시작, 끝, 요구길이) 목록."""
    dist = official_distances(pos, dtype)
    out = []
    for first in range(0, len(dist), step):
        for length in lengths:
            last = official_last_frame(dist, first, length)
            if last == -1:
                continue
            out.append((first, last, float(length)))
    return out


def official_rotation_error(delta_gt, delta_est):
    """rotationError: 오차행렬 대각합으로 각을 뽑는다(라디안)."""
    err = np.asarray(delta_est).T @ np.asarray(delta_gt)
    d = 0.5 * (err[0, 0] + err[1, 1] + err[2, 2] - 1.0)
    return float(np.arccos(max(min(d, 1.0), -1.0)))


def official_mean_rotation(errors_per_segment, lengths):
    """공식 요약: r_err/len 을 전부 더해 개수로 나눈다(도/미터 아님, 라디안)."""
    v = [e / L for e, L in zip(errors_per_segment, lengths)]
    return float(sum(v) / len(v))


def phase_half_D(t, rot, idx, src):
    """주어진 구간 목록으로 phase_half 규약의 D(라디안/미터).

    t·rot 은 평가 창이고 src 는 재표집 원본(공개 표본 전체)이다. 끝을 잘라
    붙이지 않으므로 주표와 같은 수가 나온다.
    """
    t = np.asarray(t, dtype=float)
    dt = float(np.median(np.diff(t)))
    q = relative(rot, idx)
    half = resample(src[0], src[1], src[2], t + 0.5 * dt, "slerp")[0]
    return float(np.dot(segment_weights(idx), geodesic(q, relative(half, idx))))


def compare_sequence(t, rot, pos):
    """우리 목록과 공식 목록을 맞댄다.

    세는 것이 둘이다. 하나는 구간이 같은가, 하나는 달라진 구간이 보고하는 수를
    움직이는가다. 앞의 것만 세면 "몇 개 다르다" 에서 멈추고 그게 큰지 작은지를
    못 말한다.
    """
    # 구간 고르기는 공개 파일의 위치를 그대로 쓴다 — 공식 프로그램이 읽는 것이
    # 그 숫자다. D 는 다르다. 주표의 D 는 회전을 가장 가까운 회전으로 눌러
    # 읽으므로, 여기서도 눌러 읽지 않으면 같은 이름의 두 수가 다른 입력에서
    # 나온다. 누름은 회전만 건드려서 구간 목록은 안 바뀐다.
    rot = proper(rot)
    src = (np.asarray(t, dtype=float), rot, pos)
    keep = common_window(t)
    t, rot, pos = src[0][:keep], rot[:keep], pos[:keep]
    ours = segments(pos)
    o64 = official_segments(pos, "float64")
    o32 = official_segments(pos, "float32")

    def differing(a_list, b_list):
        """세쌍을 자리마다 맞댄다. 길이가 다르면 남는 쪽도 어긋남으로 센다 —
        zip 은 짧은 쪽에서 멈춰서 꼬리의 불일치를 통째로 빠뜨린다."""
        n = max(len(a_list), len(b_list))
        out = []
        for i in range(n):
            a = a_list[i] if i < len(a_list) else None
            b = b_list[i] if i < len(b_list) else None
            if a != b:
                out.append((a, b))
        return out

    d64 = differing(ours, o64)
    d32 = differing(ours, o32)
    shift = max((abs(a[1] - b[1]) for a, b in d32 if a and b), default=0)
    d_ours = phase_half_D(t, rot, ours, src)
    d_off32 = phase_half_D(t, rot, o32, src)
    rel = abs(d_off32 - d_ours) / d_ours if d_ours > 0 else 0.0
    return len(ours), len(o64), len(o32), len(d64), len(d32), shift, \
        d_ours, d_off32, rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti-root", default="/mnt/Data/velref/kitti/dataset")
    ap.add_argument("--out", default="results/next/kitti_official_crosscheck.csv")
    args = ap.parse_args()

    rows = []
    for p in sorted(glob.glob(os.path.join(args.kitti_root, "poses", "*.txt"))):
        seq = os.path.basename(p)[:-4]
        t, rot, pos = load_poses(p)
        (n_ours, n64, n32, d64, d32, shift,
         d_ours, d_off32, rel) = compare_sequence(t, rot, pos)
        rows.append({
            "sequence": seq, "n_segments_ours": n_ours,
            "n_segments_official_f64": n64, "n_segments_official_f32": n32,
            "differing_f64": d64, "differing_f32": d32,
            "max_end_frame_shift_f32": shift,
            "D_phase_half_deg_per_m": "%.5g" % np.degrees(d_ours),
            "D_phase_half_official_f32_deg_per_m": "%.5g" % np.degrees(d_off32),
            # 매크로 생성기가 이 열을 읽어 "최대" 자리에 쓴다. 여기서 깎으면
            # 그 자리가 잰 값보다 작아져 상한이 안 선다 — 잰 값 그대로 적는다.
            "relative_change": repr(float(rel)),
        })
        print("[%s] 구간 %5d  배정밀도 공식과 다른 쌍 %3d  단정밀도 공식과 %3d "
              "(끝 프레임 최대 차 %d)  D 상대변화 %.3g"
              % (seq, n_ours, d64, d32, shift, rel))

    if not rows:
        print("맞댈 시퀀스가 없다")
        return
    tot = sum(r["n_segments_ours"] for r in rows)
    t64 = sum(r["differing_f64"] for r in rows)
    t32 = sum(r["differing_f32"] for r in rows)
    worst = max(float(r["relative_change"]) for r in rows)
    print("\n구간 %d 개 가운데 배정밀도 공식과 다른 것 %d 개, 단정밀도 공식과 "
          "다른 것 %d 개. 단정밀도 구간으로 바꿔 재도 D 의 상대변화는 최대 "
          "%.3g 다." % (tot, t64, t32, worst))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("%d 행 -> %s" % (len(rows), args.out))


if __name__ == "__main__":
    main()
