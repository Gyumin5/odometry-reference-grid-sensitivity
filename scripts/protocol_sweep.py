#!/usr/bin/env python3
"""규약 하나하나에 기대지 않도록 위상과 주기를 훑고, 효과를 갈라 잰다.

kitti_segment_D.py 는 규약 셋을 한 점씩 쟀다. 반 표본 이동 하나, 짝수 솎기
하나, 주기비 0.7 하나다. 그 셋에는 세 군데 빈틈이 있다.

  ⑴ 위상이 한 점이다. 반 표본이 늘 최악이라는 보장이 없다.
  ⑵ 솎기 위상이 짝수 한쪽이다. 홀수로 솎으면 다른 값이 나올 수 있다.
  ⑶ 주기를 바꾼 것과 보간 방식을 바꾼 것이 한 수에 섞여 있었다.

여기서는 셋을 가른다. 그리고 모든 변형이 같은 구간 집합을 쓴다 — 질의 시각이
전부 원본 시각 범위 안에 들어오도록 공통 유효 창을 먼저 잡고, 그 창의 자세로
구간을 한 번만 고른다. 끝을 잘라 붙이는(clip) 처리를 안 쓰므로 경계에서
길이 0 짜리 구간이나 중복 끝점이 생기지 않는다.

재는 양은 그대로 D = sum w d(Q,Q') 다. 상한이 서는 조건(구간·대응·가중치·
추정 상대자세 고정)이 변형마다 지켜진다.
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kitti_segment_D import (LENGTHS, PENDING, RELEASES, STEP,  # noqa: E402
                             euclidean, exact,
                             gather, geodesic, relative, relative_time,
                             relative_translation, resample, segment_weights,
                             segments)

SHIFTS = (0.1, 0.25, 0.5, 0.75, 0.9)
RATIOS = (0.5, 0.7, 0.9)
# 표본 비율로 적은 이동은 릴리스를 가로질러 못 읽는다 — 100 Hz 릴리스의 반
# 표본은 5 밀리초고 10 Hz 릴리스의 반 표본은 50 밀리초라, 같은 "반 표본" 이
# 열 배 다른 시각 어긋남이다. 그래서 절대 단위로도 같이 잰다.
TAUS_MS = (1.0, 2.0, 5.0, 10.0, 25.0, 50.0)


def coarse_grid(t, dt, ratio):
    """주기를 ratio 배로 낮춘 격자. 원본 끝을 넘는 점은 만들지 않는다."""
    span = float(t[-1] - t[0])
    stepsize = dt / ratio
    n = int(np.floor(span / stepsize))
    return t[0] + stepsize * np.arange(n + 1)


def window(t, dt, shifts=SHIFTS, ratios=RATIOS, taus_ms=TAUS_MS):
    """모든 변형의 질의 시각이 자기 원본 범위 안에 드는 공통 창 (시작, 끝)."""
    lo = float(t[1])            # 홀수 솎기의 원본이 t[1] 부터 시작한다.
    hi = float(t[-1]) - max(max(shifts) * dt, max(taus_ms) * 1e-3)
    hi = min(hi, float(t[::2][-1]), float(t[1::2][-1]))
    for r in ratios:
        hi = min(hi, float(coarse_grid(t, dt, r)[-1]))
    return lo, hi


def variants(t, rot, pos, dt, lo, hi, shifts=SHIFTS, ratios=RATIOS,
             taus_ms=TAUS_MS):
    """창 안 프레임의 (변형 이름, 대안 자세, 대안 위치) 들."""
    m = (t >= lo) & (t <= hi)
    tw = t[m]

    # 아무것도 안 바꾼 규약. 0 이 나와야 맞지만 실제로는 0 이 아니다 —
    # 공개 자세 행렬이 정확히 직교하지 않아서 회전으로 바꾸는 순간 가장
    # 가까운 회전으로 눌리고, 측지각이 항등 근처에서 그 차이를 제곱근으로
    # 키운다. 그 바닥보다 작은 D 는 규약 몫이라고 읽으면 안 된다.
    yield ("null",) + resample(t, rot, pos, tw, "slerp")

    for s in shifts:
        yield ("shift_%g" % s,) + resample(t, rot, pos, tw + s * dt, "slerp")

    for ms in taus_ms:
        yield ("tau_%gms" % ms,) + resample(t, rot, pos, tw + ms * 1e-3,
                                            "slerp")

    for name, sl in (("decim2_even", slice(None, None, 2)),
                     ("decim2_odd", slice(1, None, 2))):
        yield (name,) + resample(t[sl], rot[sl], pos[sl], tw, "slerp")

    for r in ratios:
        tc = coarse_grid(t, dt, r)
        cr, cp = resample(t, rot, pos, tc, "slerp")
        sr, sp = resample(tc, cr, cp, tw, "slerp")
        nr, npz = resample(tc, cr, cp, tw, "nearest")
        yield ("rate_%g_slerp" % r, sr, sp)
        yield ("rate_%g_nearest" % r, nr, npz)
        # 보간 방식만 바꾼 몫. 주기는 양쪽이 같으므로 이 값이 방식의 몫이다.
        yield ("method_%g" % r, nr, npz, sr, sp)


def window_segments_original(pos, m, lengths=LENGTHS, step=STEP):
    """창 안에 온전히 드는 구간. 색인은 원본 그대로 둔다.

    창 안의 자세만 떼어 넘기면 구간 선택이 창의 첫 프레임부터 다시 세어진다.
    이 창은 홀수 솎기 때문에 원본 색인 1 에서 시작하므로, 그렇게 고르면
    100~800 m 규칙의 시작점이 원본 0, 10, 20... 에서 1, 11, 21... 로 밀려
    공식 집합과 한 점도 겹치지 않는 집합이 된다. 원본에서 고른 뒤 시작점과
    끝점이 둘 다 창 안에 드는 것만 남기면 집합이 공식 집합의 부분집합이 되고,
    창이 덜어낸 수가 곧 제외 수가 된다.
    """
    hit = np.flatnonzero(m)
    if len(hit) == 0:
        return []
    off, last = int(hit[0]), int(hit[-1])
    return [(i, j, L)
            for (i, j, L) in segments(pos, lengths=lengths, step=step)
            if off <= i and j <= last]


def window_segments(pos, m, lengths=LENGTHS, step=STEP):
    """같은 집합을 창 기준 색인으로 옮긴 것. 창 안의 배열에 바로 쓴다."""
    hit = np.flatnonzero(m)
    if len(hit) == 0:
        return []
    off = int(hit[0])
    return [(i - off, j - off, L)
            for (i, j, L) in window_segments_original(pos, m, lengths, step)]


def measure(t, rot, pos, raw=None, lengths=LENGTHS, step=STEP,
            subtract_origin=True):
    """변형마다 D. (dict, 구간 수) 반환. 구간은 공통 창에서 한 번만 고른다.

    시각에서 첫 점을 먼저 뺀다. 까닭은 kitti_segment_D.measure 와 같고, 여기서
    더 크게 걸린다 — 이 훑기가 만드는 변형에는 절대 단위 어긋남(1~50 밀리초)이
    있고, 유닉스 기원 크기에서 1 밀리초를 더하면 실제로 더해지는 양이 2.4e-07
    초 칸에 맞춰 떨어진다. 그 치우침이 그대로 D 에 실린다.

    subtract_origin=False 는 그 뺄셈 없이 공개 시각 그대로 도는 대조 경로다.
    쓰는 데가 하나뿐이다 — grid_origin_probe 가 "원점을 빼는 것이 수를 가르나"
    를 물으려면 안 뺀 판이 있어야 하는데, 측정 쪽에서 늘 빼 버리면 두 판이
    같은 것을 재서 차이가 0 으로 나온다. 측정에는 쓰지 않는다.
    """
    t = np.asarray(t, dtype=float)
    rot = np.asarray(rot, dtype=float)
    pos = np.asarray(pos, dtype=float)
    if len(t) < 8:
        return None, 0
    if subtract_origin:
        t = relative_time(t)
    dt = float(np.median(np.diff(t)))
    lo, hi = window(t, dt)
    if not (hi > lo):
        return None, 0
    m = (t >= lo) & (t <= hi)
    rw, pw = rot[m], pos[m]
    idx = window_segments(pos, m, lengths, step)
    if not idx:
        return None, 0
    w = segment_weights(idx)
    L = np.asarray([p[2] for p in idx], dtype=float)

    out = {}
    for v in variants(t, rot, pos, dt, lo, hi):
        if len(v) == 3:
            name, ar, ap = v
            br, bp = rw, pw
        else:
            name, ar, ap, br, bp = v
        g = geodesic(relative(br, idx), relative(ar, idx))
        gt = euclidean(relative_translation(br, bp, idx),
                       relative_translation(ar, ap, idx))
        out[name] = float(np.dot(w, g))
        out["t_" + name] = float(np.dot(w, gt))
        if raw is not None:
            raw[name] = (g, gt, L)
    return out, len(idx)


def pool(raws):
    """구간을 한 자루에 담아 낸 변형별 D. 시퀀스별 값을 다시 평균하지 않는다."""
    names = set()
    for r in raws:
        names |= set(r)
    out = {}
    for name in sorted(names):
        g = np.concatenate([r[name][0] for r in raws if name in r])
        gt = np.concatenate([r[name][1] for r in raws if name in r])
        L = np.concatenate([r[name][2] for r in raws if name in r])
        out[name] = float(np.mean(g / L))
        out["t_" + name] = float(np.mean(gt / L))
    return out


def as_row(seq, d, nframes, nseg):
    """파일로 나가는 한 줄. D 는 깎지 않는다.

    이 줄을 되읽는 쪽이 둘이다 — offset_tolerance 는 2D 가 순위표 간격과
    같아지는 어긋남을 이 점들 사이에서 되짚고, protocol_sweep_table 은 변형들을
    D 로 줄 세운다. 다섯 자리로 깎으면 되짚는 점이 상대 5e-06 만큼 움직이고
    간발의 차로 갈리던 순서가 동률이 된다. 화면에 찍는 수는 따로 깎는다.
    """
    row = {"sequence": seq, "n_frames": nframes, "n_segments": nseg}
    for k, v in d.items():
        if k.startswith("t_"):
            row["D_%s_pct" % k[2:]] = exact(100.0 * v)
            continue
        row["D_%s_deg_per_100m" % k] = exact(100.0 * np.degrees(v))
    return row


def main():
    ap = argparse.ArgumentParser()
    # 표에 아직 없는 릴리스도 따로 훑어 볼 수 있다. 묶음 표는 RELEASES 만 돈다.
    ap.add_argument("--release", choices=RELEASES + PENDING, default="kitti")
    ap.add_argument("--root", default="",
                    help="비우면 RELEASE_ROOTS 의 값")
    ap.add_argument("--rpe1m", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    lengths, step = ((1.0,), 1) if args.rpe1m else (LENGTHS, STEP)
    tag = "rpe1m" if args.rpe1m else "segment"
    out = args.out or ("results/next/%s_%s_sweep.csv" % (args.release, tag))

    rows, raws, frames, segs = [], [], 0, 0
    for seq, (t, rot, pos) in gather(args.release, args.root):
        r = {}
        d, nseg = measure(t, rot, pos, raw=r, lengths=lengths, step=step)
        if d is None:
            print("[%s] 구간 없음" % seq)
            continue
        raws.append(r)
        frames += len(t)
        segs += nseg
        rows.append(as_row(seq, d, len(t), nseg))
        print("[%s] 구간 %6d  shift_0.5=%.4g %%, decim2_even=%.4g %%, "
              "decim2_odd=%.4g %%" % (seq, nseg, 100.0 * d["t_shift_0.5"],
                                      100.0 * d["t_decim2_even"],
                                      100.0 * d["t_decim2_odd"]))
    if not rows:
        print("잴 것이 없다")
        return

    pooled = as_row("ALL", pool(raws), frames, segs)
    rows.append(pooled)
    print("\n%-16s %14s %14s" % ("변형", "D 회전(도/100m)", "D 병진(%)"))
    for k in sorted(pooled):
        if k.startswith("D_") and k.endswith("_deg_per_100m"):
            name = k[2:-13]
            print("%-16s %14s %14s"
                  % (name, pooled[k], pooled.get("D_%s_pct" % name, "")))

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[-1].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\n%d 행 -> %s" % (len(rows), out))


if __name__ == "__main__":
    main()
