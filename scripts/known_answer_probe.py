#!/usr/bin/env python3
"""답이 닫힌 꼴로 나오는 궤적에 측정 경로를 통째로 걸어 본다.

공식 이진파일과의 대조는 두 구현이 서로 같은지만 말한다. 둘 다 같은 쪽으로
틀렸으면 그 대조는 통과한다. 여기서는 답을 먼저 알고 재는 쪽을 맞댄다.

고른 궤적 둘은 보간 오차가 0 이 되게 짰다. 위치가 시간에 대해 1 차라서 선형
보간이 정확하고, 각속도가 일정해서 구면 선형보간이 정확하다. 그래서 남는
차이는 전부 측정 경로의 것이다.

  나사  진행 방향 축으로 도는 등속 직선. 시간을 통째로 미는 것이 궤적 전체를
        강체로 옮기는 것과 같아서, 몸체 기준 상대자세가 하나도 안 변한다.
        회전·병진 이동량이 둘 다 정확히 0 이다.

  비틀림  같은 등속 직선인데 도는 축이 진행 방향에 수직이다. 상대회전은 그대로
        0 이고, 상대병진만 어긋남 tau 에 대해 2 sin(omega tau / 2) 배로
        돌아간다. 구간마다 |dp| * 2 sin(omega tau / 2) 이고 가중치는 공식과
        같은 1/(N L) 이다.

반 주기 솎기는 두 궤적 모두에서 0 이다 — 솎은 격자에서 되살려도 1 차와 등각속도는
그대로 복원되기 때문이다.
"""

import argparse
import csv
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kitti_segment_D import (  # noqa: E402
    LENGTHS, STEP, common_window, measure, segments,
)

RATE_HZ = 10.0
SPEED_MPS = 10.0
OMEGA_RAD_PER_S = 0.05
DURATION_S = 300.0

# 각가속이 일정한 궤적의 설정. 답이 0 이 아닌 회전 칸을 만드는 자리다.
SPINUP_ACCEL_RAD_PER_S2 = 0.002
# 한 구간이 세 프레임을 건넌다. 홀수라서 반 주기 솎기에서 두 끝의 홀짝이
# 반드시 갈리고, 그래야 그 규약의 답도 0 이 아니다. 짝수로 건너면 두 끝이
# 같은 양만큼 밀려 차가 0 이 되어, 솎기 경로를 꺼도 통과하는 칸이 된다.
SPINUP_LENGTHS = (2.5,)
SPINUP_STEP = 1


def trajectory(axis, n=None, rate=RATE_HZ, speed=SPEED_MPS,
               omega=OMEGA_RAD_PER_S, duration=DURATION_S):
    """등속 직선 + 등각속도. axis 가 (1,0,0) 이면 나사, (0,0,1) 이면 비틀림."""
    if n is None:
        n = int(duration * rate) + 1
    t = np.arange(n, dtype=float) / rate
    pos = np.zeros((n, 3))
    pos[:, 0] = speed * t
    rot = Rotation.from_rotvec(
        np.outer(omega * t, np.asarray(axis, dtype=float))).as_matrix()
    return t, rot, pos


def twist_reference(t, pos, tau, omega=OMEGA_RAD_PER_S,
                    lengths=LENGTHS, step=STEP):
    """비틀림 궤적에서 어긋남 tau 의 병진 이동량(무차원)을 닫힌 꼴로."""
    keep = common_window(t)
    idx = segments(pos[:keep], lengths, step)
    if not idx:
        return None, 0
    turn = 2.0 * abs(np.sin(0.5 * omega * tau))
    acc = 0.0
    for i, j, L in idx:
        acc += float(np.linalg.norm(pos[j] - pos[i])) * turn / L
    return acc / len(idx), len(idx)


def spinup(axis=(0.0, 0.0, 1.0), rate=RATE_HZ, speed=SPEED_MPS,
           accel=SPINUP_ACCEL_RAD_PER_S2, duration=DURATION_S):
    """등속 직선 + 각가속이 일정한 회전. 각이 시간의 2 차다.

    나사·비틀림 두 궤적은 각속도가 일정해서 회전 쪽 답이 어느 규약에서나
    0 이다. 재는 경로를 통째로 꺼도 0 이 나오므로, 그 칸들만으로는 회전 경로가
    도는지를 못 가른다. 각을 2 차로 만들면 그 답이 0 이 아니면서도 닫힌 꼴로
    남는다 — 구면 선형보간은 축이 고정일 때 각을 선형으로 잇고, 2 차 함수의
    선형 보간이 놓치는 양은 정확히 계산된다.
    """
    n = int(duration * rate) + 1
    t = np.arange(n, dtype=float) / rate
    pos = np.zeros((n, 3))
    pos[:, 0] = speed * t
    theta = 0.5 * accel * t * t
    rot = Rotation.from_rotvec(
        np.outer(theta, np.asarray(axis, dtype=float))).as_matrix()
    return t, rot, pos


def spinup_reference(idx, rate=RATE_HZ, accel=SPINUP_ACCEL_RAD_PER_S2):
    """그 구간 목록에서 두 규약의 회전 이동량(라디안/미터)을 닫힌 꼴로.

    각 theta(t) = a t^2 / 2 를 쓴다. 보간된 각에서 원래 각을 뺀 양이
      반 표본 위상  theta'_k - theta_k = a dt (t_k + dt/2) / 2
      반 주기 솎기  홀수 번째만 a dt^2 / 2, 짝수 번째는 0
    이고, 구간 이동량은 두 끝의 그 차이를 뺀 절댓값이다. 앞은 t_j - t_i 에만
    기대고 뒤는 두 끝의 홀짝에만 기댄다.
    """
    dt = 1.0 / rate
    out = {}
    for name in ("phase_half", "decim2"):
        acc = 0.0
        for i, j, L in idx:
            if name == "phase_half":
                ang = 0.5 * accel * dt * (j - i) * dt
            else:
                ang = 0.5 * accel * dt * dt * abs((j % 2) - (i % 2))
            acc += ang / L
        out[name] = acc / len(idx)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/next/known_answer_probe.csv")
    args = ap.parse_args()

    rows = []
    for name, axis in (("screw", (1.0, 0.0, 0.0)),
                       ("twist", (0.0, 0.0, 1.0))):
        t, rot, pos = trajectory(axis)
        tau = 0.5 / RATE_HZ
        got, nseg = measure(t, rot, pos)
        if name == "screw":
            want_rot, want_trans = 0.0, 0.0
        else:
            want_rot = 0.0
            want_trans, _ = twist_reference(t, pos, tau)
            if want_trans is None:
                raise SystemExit("비틀림 궤적에서 구간이 하나도 안 선다")
        for field, want in (("phase_half", want_rot),
                            ("t_phase_half", want_trans),
                            ("decim2", 0.0), ("t_decim2", 0.0)):
            value = float(got[field])
            rows.append({
                "trajectory": name, "quantity": field,
                "segments": nseg,
                # 잰 값을 그대로 적는다. 본문이 이 열에서 "최대 이만큼"
                # 을 뽑으므로 저장 단계에서 깎으면 그 상한이 안쪽이 된다.
                "expected": repr(float(want)),
                "measured": repr(value),
                "absolute_difference": repr(abs(value - want)),
                "relative_difference":
                    repr(abs(value - want) / want) if want > 0 else "",
            })
            print("[%-5s %-12s] 기대 %.12g  실측 %.12g  차 %.3g"
                  % (name, field, want, value, abs(value - want)))

    # 각가속 궤적. 회전 두 칸만 적는다 — 병진 쪽은 닫힌 꼴을 안 냈고, 안 낸
    # 칸을 0 으로 채우면 "안 움직였다" 가 재지 않은 채로 표에 들어간다.
    t, rot, pos = spinup()
    got, nseg = measure(t, rot, pos, lengths=SPINUP_LENGTHS, step=SPINUP_STEP)
    if got is None:
        raise SystemExit("각가속 궤적에서 구간이 하나도 안 선다")
    keep = common_window(t)
    want = spinup_reference(segments(pos[:keep], SPINUP_LENGTHS, SPINUP_STEP))
    for field in ("phase_half", "decim2"):
        value = float(got[field])
        rows.append({
            "trajectory": "spinup", "quantity": field, "segments": nseg,
            "expected": repr(float(want[field])),
            "measured": repr(value),
            "absolute_difference": repr(abs(value - want[field])),
            "relative_difference":
                repr(abs(value - want[field]) / want[field]),
        })
        print("[%-5s %-12s] 기대 %.12g  실측 %.12g  차 %.3g"
              % ("spinup", field, want[field], value,
                 abs(value - want[field])))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 답이 0 인 칸은 수로 가른다. 글자로 견주면 적는 꼴이 바뀔 때마다 조용히
    # 빈 묶음이 된다 — "0" 이 "0.0" 으로 적히자 이 자리가 그렇게 비었다.
    worst_zero = max(float(r["absolute_difference"]) for r in rows
                     if float(r["expected"]) == 0.0)
    nonzero = [r for r in rows if float(r["expected"]) != 0.0]
    print("\n답이 0 인 칸 %d 개의 최대 절대차 %.3g."
          % (len(rows) - len(nonzero), worst_zero))
    for r in nonzero:
        print("답이 0 이 아닌 칸: %s %s 상대차 %s"
              % (r["trajectory"], r["quantity"], r["relative_difference"]))
    print("%d 행 -> %s" % (len(rows), args.out))


if __name__ == "__main__":
    main()
