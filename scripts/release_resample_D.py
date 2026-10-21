#!/usr/bin/env python3
"""공개된 기준 궤적마다 재표집 이동량 D 를 잰다.

평가는 기준을 추정기의 시각으로 옮겨 놓고 맞대는데, 그 옮기는 방식은 평가
코드마다 다르다. 같은 파일을 쓰고도 보간을 무엇으로 하느냐에 따라 상대회전이
움직이고, 그만큼 점수가 움직인다. 그 이동량이 D 다.

    D = sum_k w_k * d(Q_k, Q'_k),  Q 는 기준 규약, Q' 는 대안 규약

scripts/resample_sensitivity.py 가 |E' - E| <= D 를 확인했으므로, 두 방법의
평균 각거리 점수 차가 2D 보다 크면 보간 규약을 어떻게 잡든 순위가 안 뒤집힌다.
거꾸로 점수 차가 2D 보다 작으면 그 비교는 기준 파일이 떠받치지 못한다.

규약은 재기 전에 고정한다.
  기준       공개 파일의 이웃 표본 상대회전 Q_k = R_k^T R_{k+1}. 보간이 안 끼는
             유일한 값이라 이걸 기준으로 둔다.
  phase_half 공개 시각을 반 표본 민 자리에서 구면 선형보간(slerp). 추정기 시각이
             공개 시각과 맞아떨어지지 않을 때의 최악 위치이고, 한 표집 구간
             안에서 상대회전이 얼마나 흔들리는지를 준다.
  decim2     절반 주기로 솎은 뒤 공개 시각으로 slerp 해 되돌린다. 같은 해를 더
             느리게 실었으면 무엇을 잃었을지를 그 파일 안에서 본다.
  interp     추정기가 기준보다 느린 0.7 배 주기로 돈다고 두고, 그 격자에서
             slerp 와 가장 가까운 표본 집기를 맞댄다. 평가 코드가 고르는
             보간 방식 하나가 점수를 얼마나 미는지다. 여기만 기준이 공개
             증분이 아니라 같은 격자의 slerp 다.
  가중       균등. 쌍은 이웃 쌍.

뺀 것 하나. 공개 시각을 통째로 민 자리에서 "가장 가까운 표본" 을 집는 규약은
재 봐야 0 이다. 어떤 표본을 집든 집어오는 것은 공개된 표본 그 자체라 상대회전이
글자 그대로 공개 증분과 같아진다(실측 1e-6 밀리도 이하). 시각 라벨만 바뀌고
값은 안 바뀐다는 뜻이라, 움직이는 것은 보간과 주기뿐이다.

D 는 라디안이라 작은 수다. 보기 좋으라고 밀리도(1e-3 도)로 같이 적는다.
"""

import argparse
import csv
import os

import numpy as np

from reference_resolution_audit import SYNTHETIC_TIME, gather
from resample_sensitivity import (
    displacement, relative, resample, weights,
)

MAX_SAMPLES = 200000


def pairs_of(n):
    return np.stack([np.arange(n - 1), np.arange(1, n)], axis=1)


PROTOCOLS = ("phase_half", "decim2", "interp")
EST_RATE = 0.7  # 추정기 주기 / 기준 주기. 공개 격자와 안 맞아떨어지게 고른 값.


def protocol_pair(t, rot, dt, name):
    """규약 하나가 비교하는 상대회전 두 묶음 (Q, Q')."""
    if name == "interp":
        # 공개 격자와 약분이 안 되는 독립 격자. 여기서만 기준이 같은 격자의
        # slerp 다 — 공개 증분과는 쌍 자체가 다르기 때문이다.
        tq = np.arange(t[0], t[-1], dt / EST_RATE)
        pr = pairs_of(len(tq))
        return (relative(resample(t, rot, tq, "slerp"), pr),
                relative(resample(t, rot, tq, "nearest"), pr))

    # 끝 쪽 한 표본은 민 자리가 구간 밖이라 끝값으로 잘린다. 그 잘림이 상대회전을
    # 0 으로 만들어 평균을 끌어내리므로 쌍에서 뺀다.
    pr = pairs_of(len(t) - 1)
    q = relative(rot, pr)
    if name == "phase_half":
        return q, relative(resample(t, rot, t + 0.5 * dt, "slerp"), pr)
    if name == "decim2":
        return q, relative(resample(t[::2], rot[::2], t, "slerp"), pr)
    raise ValueError("모르는 규약: %s" % name)


def measure(t, rot, protocols=PROTOCOLS):
    """한 파일의 D 들. (dict, n_pairs, fs) 반환."""
    t = np.asarray(t, dtype=float)
    if len(t) > MAX_SAMPLES:
        t, rot = t[:MAX_SAMPLES], rot[:MAX_SAMPLES]
    if len(t) < 64:
        return None, 0, float("nan")
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        return None, 0, float("nan")
    fs = 1.0 / dt

    rot = np.asarray(rot, dtype=float)
    out, npairs = {}, 0
    for name in protocols:
        q, qp = protocol_pair(t, rot, dt, name)
        out[name] = displacement(q, qp, weights(len(q)))
        npairs = max(npairs, len(q))

    # D 만 적으면 큰지 작은지를 말할 수 없다. 그 파일에서 몸체가 이웃 표본 사이에
    # 실제로 얼마나 도는지를 같이 적어 둔다 — 규약이 가르는 몫이 그중 얼마인지가
    # 읽히는 숫자다.
    q = relative(rot, pairs_of(len(t) - 1))
    tr = np.trace(q, axis1=1, axis2=2)
    out["increment"] = float(np.median(np.arccos(np.clip((tr - 1) / 2, -1.0, 1.0))))
    return out, npairs, fs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--helipr-root", default="/mnt/Data/velref/helipr")
    ap.add_argument("--boreas-root", default="data/boreas")
    ap.add_argument("--kitti-root", default="/mnt/Data/velref/kitti/dataset")
    ap.add_argument("--kitti360-root", default="/mnt/Data/velref/kitti360")
    ap.add_argument("--pit30m-root", default="/mnt/Data/velref/pit30m")
    ap.add_argument("--nuscenes-can", default="/mnt/Data/velref/nuscenes/can_bus")
    ap.add_argument("--nuscenes-limit", type=int, default=0)
    ap.add_argument("--out", default="results/next/release_resample_D.csv")
    args = ap.parse_args()

    rows = []
    for release, name, path, loader in gather(args):
        try:
            t, rot, _ = loader(path)
        except Exception as e:
            print("[%s] %s 읽기 실패: %s" % (release, name, e))
            continue
        try:
            d, npairs, fs = measure(t, np.asarray(rot, dtype=float))
        except Exception as e:
            print("[%s] %s 재표집 실패: %s" % (release, name, e))
            continue
        if d is None:
            continue
        row = {
            "release": release, "sequence": name, "n_pairs": npairs,
            "fs_hz": round(fs, 3),
            # 시각을 프레임 번호로 지어낸 릴리스는 표본 번호 해석이다. 반 표본은
            # 여전히 뜻이 있지만 초 단위 해석은 명목 주기에 기댄 값이다.
            "time_source": "표본번호" if release in SYNTHETIC_TIME else "시각",
        }
        for k, v in d.items():
            row["D_%s_rad" % k] = "%.6g" % v
            row["D_%s_mdeg" % k] = "%.4g" % (np.degrees(v) * 1000.0)
        inc = d.get("increment", 0.0)
        row["frac_of_increment"] = ("%.4g" % (d["phase_half"] / inc)) if inc else ""
        rows.append(row)
        print("[%s] %-24s n=%6d fs=%7.2f  D(phase_half)=%.4g mdeg"
              % (release, name, npairs, fs, np.degrees(d["phase_half"]) * 1000.0))

    if not rows:
        print("잴 것이 없다")
        return
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print()
    by = {}
    for r in rows:
        by.setdefault(r["release"], []).append(r)
    head = "%-15s %5s %-24s %-24s %-24s %-22s %s" % (
        "릴리스", "n", "D phase_half(mdeg)", "D decim2(mdeg)", "D interp(mdeg)",
        "이웃증분(mdeg)", "증분대비")
    print(head)
    print("-" * len(head))
    for rel, rs in sorted(by.items()):
        cells = []
        for k in PROTOCOLS:
            v = sorted(float(r["D_%s_mdeg" % k]) for r in rs)
            med = v[len(v) // 2]
            cells.append("%.3g (%.3g~%.3g)" % (med, v[0], v[-1]))
        inc = sorted(float(r["D_increment_mdeg"]) for r in rs)
        fr = sorted(float(r["frac_of_increment"]) for r in rs
                    if r["frac_of_increment"])
        print("%-15s %5d %-24s %-24s %-24s %-22s %s"
              % (rel, len(rs), *cells,
                 "%.3g (%.3g~%.3g)" % (inc[len(inc) // 2], inc[0], inc[-1]),
                 "%.3f" % fr[len(fr) // 2] if fr else "-"))
    print("\n%d 행 -> %s" % (len(rows), args.out))


if __name__ == "__main__":
    main()
