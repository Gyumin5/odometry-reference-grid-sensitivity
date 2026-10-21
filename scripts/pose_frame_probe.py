#!/usr/bin/env python3
"""공개 자세가 본체를 세계로 보내는 쪽인지, 그 반대쪽인지 릴리스마다 가린다.

왜 필요한가. 회전 행렬을 전치해서 읽어도 회전 쪽 수치는 거의 안 바뀐다 — 모든
자세를 한꺼번에 전치하면 상대 회전이 켤레로 옮겨갈 뿐이라 측지각이 그대로다.
드러나는 곳은 구간 상대 병진 R_s^T (p_e - p_s) 하나뿐이다. 그래서 전치 실수는
회전 표를 아무리 들여다봐도 안 보이고, 병진 표의 한 줄만 조용히 틀린다.
실제로 보레아스가 그랬다.

어떻게 가리는가. 차가 앞으로 갈 때 본체 좌표에서 본 변위 방향은 주행 내내
한 자리에 머문다. 장치가 돌아 달려 있거나 전진축이 x 가 아니어도 그 자리가
다른 상수일 뿐이고, 상수는 우리 측정에 안 섞인다(같은 회전을 양쪽에 곱하면
차의 크기가 안 변한다). 반대로 전치해서 읽으면 그 방향이 방위를 따라 돌아서
수십 도로 벌어진다. 그래서 판정은 방향의 값이 아니라 흩어짐으로 하고, 어느
축이 앞인지 모르니 평면 방위가 아니라 삼차원 방향으로 잰다.

가릴 수 없는 자리도 있다. 손에 들고 찍은 실내 궤적은 본체 좌표에서 본 변위
방향이 원래부터 흩어진다 — 그런 줄은 어느 쪽이 맞다고 못 하고 미정으로 둔다.
못 가린다고 틀린 것은 아니다. 틀린 것은 전치 쪽이 더 모이는 줄이다.

쓰임. results/next/pose_frame_probe.csv 로 떨어뜨린다. 규약은 시퀀스가 아니라
릴리스의 성질이니 판정도 릴리스 단위로 읽는다 — 가린 줄의 다수가 전치 쪽이면
그 릴리스의 적재기를 다시 본다. 한 줄만 반대로 나오는 것은 규약이 아니라 그
시퀀스의 움직임이 가를 만하지 않았다는 뜻이다.
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kitti_segment_D import RELEASES, gather  # noqa: E402

# 이 폭 아래로 모이면 "한 자리에 머문다" 로 본다. 실측에서 바르게 읽은 차량
# 궤적은 전부 이 아래이고 전치하면 수십 도로 벌어진다.
TIGHT_DEG = 20.0
# 두 읽기 중 한쪽이 다른 쪽의 이 비율보다 좁아야 갈렸다고 본다. 둘 다 넓거나
# 둘이 비슷하면 그 시퀀스로는 못 가린다 — 손에 들고 찍은 실내 궤적이 그렇다.
RATIO = 2.0
# 회전이 거의 없는 짧은 직선 주행은 전치해도 안 벌어진다. 가르려면 주행 중
# 방위가 이만큼은 돌아야 한다.
TURN_DEG = 30.0


def fast_mask(pos, lag, keep=0.10):
    """빠른 쪽 keep 만 고른다. 서 있는 구간은 변위가 잡음이라 방향이 없다."""
    d = pos[lag:] - pos[:-lag]
    sp = np.linalg.norm(d, axis=1)
    if len(sp) == 0:
        return None, None
    m = sp > np.quantile(sp, 1.0 - keep)
    if m.sum() < 50:
        return None, None
    return m, d[m] / sp[m, None]


def direction_spread(rot, pos, lag, keep=0.10):
    """본체 좌표에서 본 변위 방향이 평균 방향에서 벌어진 각(도)의 75 분위.

    평면 방위가 아니라 삼차원 방향으로 잰다 — 전진축이 x 인지 z 인지 릴리스마다
    달라서(키티는 카메라 좌표라 z 가 앞이다) 평면에 눕히면 뜻이 없어진다.
    빠른 쪽 keep 만 쓴다. 서 있는 구간은 변위가 잡음이라 방향이 없다.
    """
    m, u = fast_mask(pos, lag, keep)
    if m is None:
        return None
    body = np.einsum("nji,nj->ni", rot[:-lag][m], u)
    mean = body.mean(axis=0)
    n = np.linalg.norm(mean)
    if n < 1e-12:
        return 180.0
    mean = mean / n
    cos = np.clip(body @ mean, -1.0, 1.0)
    return float(np.degrees(np.percentile(np.arccos(cos), 75)))


def turn_coverage(rot):
    """쓴 표본들 사이에서 자세가 얼마나 돌았는지(도).

    궤적 전체가 아니라 실제로 방향을 잰 표본만 본다. 둘을 섞으면 안 된다 —
    긴 주행에서 빠른 쪽만 고르면 그 표본들이 한 직선 구간에 몰리는 일이 잦고,
    그러면 전체로는 90 도를 돌았는데 쓴 표본은 안 돈 상태가 된다. 안 돈
    표본만으로는 전치해도 방향이 안 벌어져서 두 읽기가 똑같이 좁게 나온다.
    """
    tr = np.einsum("nij,kj->nik", rot, rot[0]).trace(axis1=1, axis2=2)
    ang = np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))
    return float(ang.max())


def probe(release, lag_seconds=1.0):
    rows = []
    for seq, (t, rot, pos) in gather(release):
        if len(t) < 200:
            continue
        dt = float(np.median(np.diff(t)))
        lag = max(1, int(round(lag_seconds / dt))) if dt > 0 else 1
        if lag >= len(t) - 50:
            continue
        got = direction_spread(rot, pos, lag)
        flipped = direction_spread(np.transpose(rot, (0, 2, 1)), pos, lag)
        m, _ = fast_mask(pos, lag)
        if got is None or flipped is None or m is None:
            continue
        turn = turn_coverage(rot[:-lag][m])
        if turn < TURN_DEG:
            verdict = "indecisive"
        elif got < TIGHT_DEG and flipped > RATIO * got:
            verdict = "as_read"
        elif flipped < TIGHT_DEG and got > RATIO * flipped:
            verdict = "transposed"
        else:
            verdict = "indecisive"
        rows.append({
            "release": release,
            "sequence": seq,
            "lag_frames": lag,
            "turn_deg": "%.4g" % turn,
            "as_read_spread_deg": "%.4g" % got,
            "transposed_spread_deg": "%.4g" % flipped,
            "verdict": verdict,
        })
    return rows


FIELDS = ("release", "sequence", "lag_frames", "turn_deg",
          "as_read_spread_deg", "transposed_spread_deg", "verdict")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/next/pose_frame_probe.csv")
    ap.add_argument("--releases", nargs="*", default=list(RELEASES))
    args = ap.parse_args()

    rows = []
    for rel in args.releases:
        got = probe(rel)
        rows.extend(got)
        bad = [r for r in got if r["verdict"] == "transposed"]
        und = [r for r in got if r["verdict"] == "indecisive"]
        ok = len(got) - len(bad) - len(und)
        # 규약은 시퀀스가 아니라 릴리스의 성질이다. 그래서 한 줄이 반대로
        # 나왔다고 규약이 틀린 게 아니고, 가린 줄의 다수가 반대일 때만 그렇다.
        if ok + len(bad) == 0:
            call = "못 가림"
        elif len(bad) > ok:
            call = "전치 의심"
        else:
            call = "읽은 대로"
        print("%-10s 시퀀스 %4d  읽은 대로 %4d  전치 %d  미정 %d  ->  %s"
              % (rel, len(got), ok, len(bad), len(und), call))
        for r in bad:
            print("   %s 읽은 대로 %s 도, 전치 %s 도"
                  % (r["sequence"], r["as_read_spread_deg"],
                     r["transposed_spread_deg"]))

    if not rows:
        raise SystemExit("잴 것이 없다")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(FIELDS))
        w.writeheader()
        w.writerows(rows)
    bad = len([r for r in rows if r["verdict"] == "transposed"])
    und = len([r for r in rows if r["verdict"] == "indecisive"])
    print("\n%d 줄, 전치 %d, 미정 %d -> %s"
          % (len(rows), bad, und, args.out))


if __name__ == "__main__":
    main()
