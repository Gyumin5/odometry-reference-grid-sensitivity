#!/usr/bin/env python3
"""공개 자세 행렬이 직교에서 벗어난 폭과 그것이 점수에 남기는 바닥을 잰다.

릴리스 열 중 둘(키티·키티-360)만 자세를 3x4 행렬로 싣고 여덟은 회전을
쿼터니언이나 오일러 각으로 싣는다. 행렬로
실린 쪽은 그 3x3 칸이 정확히 직교하지 않는다 — 소수 자리를 잘라서 싣기
때문이다. 보간은 쿼터니언을 거치므로 돌아온 자세는 반드시 직교한다. 그래서
기준 쪽만 원본 행렬을 쓰면, 아무것도 안 바꾼 규약에서도 두 쪽이 다르다.

재는 것 둘.
  ⑴ 직교 이탈폭 max |R^T R - I|
  ⑵ 그 이탈이 구간 상대회전에 남기는 D — 원본으로 잰 상대회전과 가장 가까운
     회전으로 눌러 잰 상대회전 사이의 측지각을 공식 가중치로 평균한 값

측지각은 항등 근처에서 각도를 이탈폭의 제곱근으로 키운다. 그래서 1e-7 짜리
이탈이 1 m 간격 규칙에서는 도/100 m 단위로 올라온다.
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kitti_segment_D import (LENGTHS, RELEASES, STEP,  # noqa: E402
                             common_window, gather, geodesic, proper,
                             relative, relative_time, segment_weights,
                             segments)

RULES = (("seg", LENGTHS, STEP), ("rpe1m", (1.0,), 1))


def deviation(rot):
    """max |R^T R - I|. 행렬이 회전이면 0 이다."""
    g = np.einsum("nba,nbc->nac", rot, rot) - np.eye(3)
    return float(np.max(np.abs(g)))


def floor_of(t, rot, pos, lengths, step):
    """원본과 눌린 것 사이의 D (라디안/미터) 와 구간 수.

    본문은 이 바닥을 같은 릴리스·같은 규칙의 상한과 나란히 읽는다. 그러려면
    두 쪽이 같은 구간 집합 위에서 서야 한다 — measure 는 시각에서 첫 점을 뺀
    뒤 공통 창까지만 재므로 여기서도 같은 길로 자른다. 안 자르면 바닥은 궤적
    전체를, 비교 상대는 창 안만 세고, 그 차이가 바닥과 상한의 비에 섞인다.
    """
    t = relative_time(np.asarray(t, dtype=float))
    rot = np.asarray(rot, dtype=float)
    pos = np.asarray(pos, dtype=float)
    keep = common_window(t)
    rot, pos = rot[:keep], pos[:keep]
    if len(pos) < 2:
        return None, 0
    idx = segments(pos, lengths, step)
    if not idx:
        return None, 0
    w = segment_weights(idx)
    raw = relative(np.asarray(rot, dtype=float), idx)
    fixed = relative(proper(rot), idx)
    return float(np.dot(w, geodesic(raw, fixed))), len(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",
                    default="results/next/orthonormality_floor.csv")
    args = ap.parse_args()

    rows = []
    for release in RELEASES:
        devs, acc = [], {name: ([], []) for name, _, _ in RULES}
        for _, (t, rot, pos) in gather(release, project=False):
            rot = np.asarray(rot, dtype=float)
            # 이탈폭은 공개 파일 전체에 대한 말이라 창으로 안 자른다. 창 안
            # 최대는 전체 최대보다 작을 수만 있어서, 자르면 공개 행렬이 실제보다
            # 직교에 가깝다고 적게 된다. 바닥 쪽은 비교 상대와 같은 창을 쓴다.
            devs.append(deviation(rot))
            for name, lengths, step in RULES:
                d, n = floor_of(t, rot, pos, lengths, step)
                if d is None:
                    continue
                acc[name][0].append(d * n)
                acc[name][1].append(n)
        if not devs:
            print("[%s] 자료 없음" % release)
            continue
        for name, _, _ in RULES:
            tot, cnt = acc[name]
            if not cnt:
                continue
            rows.append({
                "release": release,
                "rule": name,
                "n_sequences": len(devs),
                "n_segments": int(sum(cnt)),
                # 잰 값을 그대로 적는다. 본문이 이 열에서 상한을 뽑으므로
                # 여기서 먼저 깎으면 바깥으로 깎는 그 뒷단이 할 일이 없어진다.
                "ortho_dev_max": repr(float(max(devs))),
                "ortho_dev_median": repr(float(np.median(devs))),
                "floor_deg_per_100m":
                    repr(100.0 * float(np.degrees(sum(tot) / sum(cnt)))),
            })
            print("[%s %s] 이탈 최대 %.3g, 바닥 %.4g 도/100m"
                  % (release, name, max(devs),
                     100.0 * np.degrees(sum(tot) / sum(cnt))))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\n%d 줄 -> %s" % (len(rows), args.out))


if __name__ == "__main__":
    main()
