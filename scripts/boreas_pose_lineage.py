#!/usr/bin/env python3
"""보레아스의 센서별 기준 자세가 한 해에서 나온 것인지 가른다.

보레아스는 같은 주행의 기준 자세를 라이다용·레이다용·카메라용으로 따로 공개한다.
세 파일의 열 이름이 같고 시각만 다르다. 문서에는 이것이 한 해를 센서 시각으로
옮겨 담은 것인지, 센서마다 따로 만든 해인지가 안 적혀 있다. 그 답이 이 릴리스를
처리 단계 비교축으로 쓸 수 있는지를 가른다 — 한 해를 옮긴 것이면 센서별 차이는
보간 방식만 재는 셈이라 못 쓴다.

가르는 법은 조밀한 원본과 맞대는 것이다. 같은 주행의 사후처리 해(200 Hz,
applanix/gps_post_process.csv)를 센서 시각으로 보간해 센서 파일과 맞대면, 한 해에서
뽑은 것이라면 남는 것은 고정된 센서 장착 회전 하나뿐이다. 그 고정분을 빼고 남는
잔차를 "같은 해를 반으로 솎았다가 되돌렸을 때의 보간 오차" 와 비교한다. 잔차가
그 눈금 아래면 한 해에서 뽑은 것이고, 위면 따로 만든 해다.

회전 규약은 공식 devkit 에서 가져왔다(utiasASRL/pyboreas,
pyboreas/utils/utils.py). 거기 yawPitchRollToRot 가

    C_enu_sensor = roll(r) @ pitch(p) @ yaw(y)

이고 roll·pitch·yaw 가 각각 표준 능동회전의 전치 꼴이다. 규약을 잘못 잡으면
멀쩡히 같은 해인데도 잔차가 도 단위로 나온다 — 실제로 scipy 기본 "xyz" 오일러로
읽었을 때 라이다 잔차 중앙값이 0.97 도였다.

실행:
    PYTHONPATH=src python3 scripts/boreas_pose_lineage.py
"""

import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kitti_segment_D import ypr_to_rot_boreas as ypr_to_rot  # noqa: E402

ROOT = "/mnt/Data/velref/boreas"
SEQUENCES = ("boreas-2020-11-26-13-58", "boreas-2021-06-17-17-52",
             "boreas-2021-09-02-11-42")
SENSORS = ("lidar", "radar", "camera")
DENSE = "gps_post_process.csv"
OUT = "results/next/boreas_pose_lineage.csv"


def load_poses(path):
    """(시각[초], 회전, 위치) 를 돌려준다.

    열 순서는 GPSTime, easting, northing, altitude, ..., roll, pitch, heading.
    조밀한 쪽은 시각이 초이고 센서 쪽은 마이크로초라 단위를 맞춘다.
    """
    a = np.loadtxt(path, delimiter=",", skiprows=1,
                   usecols=(0, 1, 2, 3, 7, 8, 9), ndmin=2)
    t = a[:, 0]
    if t[0] > 1e12:
        t = t / 1e6
    rot = Rotation.from_matrix(ypr_to_rot(a[:, 6], a[:, 5], a[:, 4]))
    return t, rot, a[:, 1:4]


def interpolation_yardstick(t, rot):
    """같은 해를 반으로 솎았다가 되돌렸을 때의 각오차(도).

    "한 해에서 뽑았다면 이 정도까지는 어긋난다" 는 눈금이다. 센서 파일의
    잔차를 이것과 맞대야 크기에 뜻이 생긴다.
    """
    mid = 0.5 * (t[:-1] + t[1:])
    coarse = Slerp(t[::2], rot[::2])
    keep = mid <= t[::2][-1]
    err = (coarse(mid[keep]).inv() * Slerp(t, rot)(mid[keep])).magnitude()
    return np.degrees(np.median(err)), np.degrees(np.max(err))


def lineage(t_dense, rot_dense, pos_dense, t_s, rot_s, pos_s):
    """센서 파일이 조밀한 해에서 나왔는지 재는 값들."""
    keep = (t_s >= t_dense[0]) & (t_s <= t_dense[-1])
    t_s, rot_s, pos_s = t_s[keep], rot_s[keep], pos_s[keep]
    at = Slerp(t_dense, rot_dense)(t_s)
    rel = at.inv() * rot_s
    offset = rel.mean()
    residual = np.degrees((offset.inv() * rel).magnitude())

    # 장착 위치는 몸체 좌표에서 고정이어야 한다. 조밀한 해의 자세로 되돌린다.
    at_pos = np.stack([np.interp(t_s, t_dense, pos_dense[:, k])
                       for k in range(3)], axis=1)
    lever = np.einsum("nba,nb->na", at.as_matrix(), pos_s - at_pos)
    return {
        "n": len(t_s),
        "offset_deg": float(np.degrees(offset.magnitude())),
        "residual_median_deg": float(np.median(residual)),
        "residual_p95_deg": float(np.percentile(residual, 95)),
        "residual_max_deg": float(np.max(residual)),
        "lever_norm_m": float(np.linalg.norm(lever.mean(axis=0))),
        "lever_spread_m": float(np.max(np.std(lever, axis=0))),
    }


def main():
    rows = []
    for seq in SEQUENCES:
        d = os.path.join(ROOT, seq, "applanix")
        if not os.path.exists(os.path.join(d, DENSE)):
            print("%s: 조밀한 해가 없어 건너뛴다" % seq)
            continue
        t_d, rot_d, pos_d = load_poses(os.path.join(d, DENSE))
        mark_med, mark_max = interpolation_yardstick(t_d, rot_d)
        print("[%s] 조밀한 해 %d 점, 눈금 중앙 %.3e 도 최대 %.3e 도"
              % (seq, len(t_d), mark_med, mark_max))
        for s in SENSORS:
            p = os.path.join(d, "%s_poses.csv" % s)
            if not os.path.exists(p):
                print("  %-7s 파일 없음" % s)
                continue
            t_s, rot_s, pos_s = load_poses(p)
            r = lineage(t_d, rot_d, pos_d, t_s, rot_s, pos_s)
            r.update(sequence=seq, sensor=s, yardstick_median_deg=mark_med,
                     yardstick_max_deg=mark_max,
                     below_yardstick=int(r["residual_median_deg"] < mark_med))
            rows.append(r)
            print("  %-7s n=%5d  고정 회전 %8.4f 도  잔차 중앙 %.3e  95%% %.3e"
                  "  최대 %.3e  장착 %.3f m (퍼짐 %.4f m)"
                  % (s, r["n"], r["offset_deg"], r["residual_median_deg"],
                     r["residual_p95_deg"], r["residual_max_deg"],
                     r["lever_norm_m"], r["lever_spread_m"]))

    if not rows:
        print("잴 것이 없다")
        return
    worst = max(r["residual_median_deg"] / r["yardstick_median_deg"]
                for r in rows)
    print("\n센서 파일 %d 개 전부 잔차 중앙값이 눈금의 %.3f 배 이하다."
          % (len(rows), worst) if all(r["below_yardstick"] for r in rows)
          else "\n눈금을 넘는 센서 파일이 있다(최대 %.3f 배)." % worst)

    cols = ["sequence", "sensor", "n", "offset_deg", "residual_median_deg",
            "residual_p95_deg", "residual_max_deg", "yardstick_median_deg",
            "yardstick_max_deg", "below_yardstick", "lever_norm_m",
            "lever_spread_m"]
    with open(OUT, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join("%s" % r[c] if isinstance(r[c], (str, int))
                             else "%.6g" % r[c] for c in cols) + "\n")
    print("%d 행 -> %s" % (len(rows), OUT))


if __name__ == "__main__":
    main()
