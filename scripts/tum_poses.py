#!/usr/bin/env python3
"""튬 꼴 기준 궤적 파일을 읽는다 — timestamp tx ty tz qx qy qz qw.

주행 릴리스 여섯은 공개 주기가 10~100 Hz 안에 있고 전부 차량이다. 동시추정
위치추정 쪽 릴리스는 손에 들거나 날리는 것이라 움직임이 다르고, 기준 자세를
동작포착으로 100~200 Hz 에 싣는다. 같은 상한을 그쪽에서도 재려면 그 꼴을
읽어야 한다.

동작포착은 표식을 놓치는 구간이 있어서 시각이 군데군데 빈다. 그 구멍을 넘겨
보간하면 실제로는 없는 자세를 지어내므로, 중앙 간격의 다섯 배를 넘는 자리에서
궤적을 끊어 따로 센다. 끊고 남은 토막이 너무 짧으면 버린다 — 구간을 하나도
못 만드는 토막은 어차피 안 쓰인다.
"""

import numpy as np
from scipy.spatial.transform import Rotation

GAP = 5.0
MIN_FRAMES = 32


def read_tum(path):
    """파일 한 장의 (시각, 쿼터니언, 위치). 주석과 모자란 줄은 건너뛴다."""
    t, q, p = [], [], []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            v = line.split()
            if len(v) != 8:
                continue
            t.append(float(v[0]))
            p.append([float(v[1]), float(v[2]), float(v[3])])
            q.append([float(v[4]), float(v[5]), float(v[6]), float(v[7])])
    return (np.asarray(t), np.asarray(q).reshape(-1, 4),
            np.asarray(p).reshape(-1, 3))


def pieces(t, q, p, gap=GAP, min_frames=MIN_FRAMES):
    """구멍에서 끊은 토막들의 (시각, 회전행렬, 위치)."""
    if len(t) < 2:
        return
    order = np.argsort(t, kind="stable")
    t, q, p = t[order], q[order], p[order]
    # 같은 시각이 두 번 적힌 줄이 있다. 그대로 두면 재표집이 거절한다 —
    # 보간은 시각이 엄격히 늘어야 한다. 먼저 적힌 줄만 남긴다.
    keep = np.r_[True, np.diff(t) > 0]
    t, q, p = t[keep], q[keep], p[keep]
    if len(t) < 2:
        return
    d = np.diff(t)
    med = float(np.median(d))
    cut = np.flatnonzero(d > gap * med) + 1
    for a, b in zip(np.r_[0, cut], np.r_[cut, len(t)]):
        if b - a < min_frames:
            continue
        rot = Rotation.from_quat(q[a:b]).as_matrix()
        yield t[a:b], rot, p[a:b]


