#!/usr/bin/env python3
"""공개된 기준 궤적 파일 하나만 읽고 그 기준 자신의 분해능을 잰다.

다른 어떤 알고리즘도 돌리지 않는다. 점군도 열지 않고 도플러도 안 쓴다. 읽는 것은
릴리스가 평가 기준으로 배포한 자세 시계열 하나뿐이다.

재는 것 넷.
  잡음 바닥 N  - 회전 증분 Q_k = R(t_k)^T R(t_k+1) 에서 각속도 w_k = Log(Q_k)/dt 를 얻고,
                 차량 운동 대역(f_c 아래)을 제외한 위쪽 띠의 스펙트럼 밀도를 적분한 제곱근.
                 단위 rad/s. 기준 자신이 떨고 있는 폭이다.
  유효 대역폭   - 스펙트럼이 그 바닥보다 뚜렷이 높은 가장 높은 주파수. 평활을 세게 건 해는
                 이 값이 낮고, 그만큼 실제 운동을 못 담는다.
  시각 흔들림   - 표본 간격의 중앙값 대비 표준편차.
  값 양자화     - 저장된 값의 0 아닌 최소 간격.

영향으로 바꾸는 식.
  e_ang = N * h          적분 구간 h 동안 쌓이는 각도 오차 (rad)
  e_pos = v * N * h      그 각도가 속도 v 에서 만드는 위치 오차 (m)
이 공개본으로 매긴 순위는 e_pos 보다 작은 점수차를 가를 수 없다.
"""

import argparse
import csv
import glob
import json
import math
import os

import numpy as np


# ---------------------------------------------------------------- 자세 읽기

def quat_to_rot(q):
    """xyzw 사원수 배열 (N,4) 를 회전행렬 (N,3,3) 으로."""
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    r = np.empty((len(q), 3, 3))
    r[:, 0, 0] = 1 - 2 * (y * y + z * z)
    r[:, 0, 1] = 2 * (x * y - z * w)
    r[:, 0, 2] = 2 * (x * z + y * w)
    r[:, 1, 0] = 2 * (x * y + z * w)
    r[:, 1, 1] = 1 - 2 * (x * x + z * z)
    r[:, 1, 2] = 2 * (y * z - x * w)
    r[:, 2, 0] = 2 * (x * z - y * w)
    r[:, 2, 1] = 2 * (y * z + x * w)
    r[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return r


def ypr_to_rot(yaw, pitch, roll):
    """Z-Y-X 내재 회전. 라디안."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    r = np.empty((len(yaw), 3, 3))
    r[:, 0, 0] = cy * cp
    r[:, 0, 1] = cy * sp * sr - sy * cr
    r[:, 0, 2] = cy * sp * cr + sy * sr
    r[:, 1, 0] = sy * cp
    r[:, 1, 1] = sy * sp * sr + cy * cr
    r[:, 1, 2] = sy * sp * cr - cy * sr
    r[:, 2, 0] = -sp
    r[:, 2, 1] = cp * sr
    r[:, 2, 2] = cp * cr
    return r


def so3_log(r):
    """회전행렬 묶음 (N,3,3) 의 로그. (N,3) 회전벡터."""
    tr = np.clip((r[:, 0, 0] + r[:, 1, 1] + r[:, 2, 2] - 1) / 2, -1.0, 1.0)
    theta = np.arccos(tr)
    v = np.stack([r[:, 2, 1] - r[:, 1, 2],
                  r[:, 0, 2] - r[:, 2, 0],
                  r[:, 1, 0] - r[:, 0, 1]], axis=1)
    small = theta < 1e-8
    scale = np.where(small, 0.5, theta / (2 * np.sin(np.where(small, 1.0, theta))))
    return v * scale[:, None]


# ---------------------------------------------------------------- 파일 형식

def load_tum(path):
    """timestamp x y z qx qy qz qw. 헬리퍼 LiDAR_GT 가 이 형식이고 시각은 나노초다."""
    t, q = [], []
    with open(path) as f:
        for line in f:
            c = line.split()
            if len(c) < 8:
                continue
            t.append(float(c[0]) * 1e-9)
            q.append([float(c[4]), float(c[5]), float(c[6]), float(c[7])])
    return np.array(t), quat_to_rot(q), q


def load_boreas_lidar_poses(path):
    """GPSTime(us), easting, northing, altitude, vel..., roll, pitch, heading, angvel..."""
    t, ypr = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            t.append(float(row["GPSTime"]) * 1e-6)
            ypr.append([float(row["heading"]), float(row["pitch"]), float(row["roll"])])
    a = np.array(ypr)
    return np.array(t), ypr_to_rot(a[:, 0], a[:, 1], a[:, 2]), a.tolist()


def load_nuscenes_pose(path):
    """차량 통신망 확장팩 scene-*_pose.json. orientation 은 wxyz 순이다."""
    with open(path) as f:
        rows = json.load(f)
    t = np.array([r["utime"] * 1e-6 for r in rows])
    o = np.array([r["orientation"] for r in rows])
    q = np.stack([o[:, 1], o[:, 2], o[:, 3], o[:, 0]], axis=1)
    return t, quat_to_rot(q), q.tolist()


def load_kitti_odometry(path):
    """키티 주행거리계 공개 자세. poses/NN.txt 는 3x4 행렬이고 시각은
    sequences/NN/times.txt 에 따로 있다."""
    seq = os.path.basename(path).replace(".txt", "")
    root = os.path.dirname(os.path.dirname(path))
    times = os.path.join(root, "sequences", seq, "times.txt")
    rot = []
    with open(path) as f:
        for line in f:
            c = [float(v) for v in line.split()]
            if len(c) != 12:
                continue
            rot.append([c[0:3], c[4:7], c[8:11]])
    t = np.array([float(x) for x in open(times)]) if os.path.exists(times) \
        else np.arange(len(rot)) * 0.1
    n = min(len(t), len(rot))
    r = np.array(rot[:n])
    return t[:n], r, r.reshape(n, 9).tolist()


# 공개본이 시각을 안 싣고 프레임 번호만 주는 릴리스. 시각 관련 값은 안 낸다.
SYNTHETIC_TIME = {"kitti360"}

# 전진 방향 축. 위치 오차에는 진행 방향에 수직인 두 축만 기여하므로 어느 축이 앞인지가
# 결과를 가른다. 키티 주행거리계의 poses 는 카메라 좌표계(cam0)라 z 가 앞이고, 나머지는
# 차체·관성 좌표계라 x 가 앞이다.
FORWARD_AXIS = {"kitti-odometry": "z"}

# 대역 기울기가 이 값을 넘으면 백색 각속도 모형을 안 쓴다. 0.5 는 백색(0)과
# 미분된 독립 자세오차(+2) 사이의 가운데보다 백색 쪽에 붙여 둔 선이다 — 가운데에
# 두면 기울기 1 짜리 어중간한 계열까지 백색으로 환산한다.
SLOPE_TOL = 0.5


def load_kitti360_poses(path):
    """키티-360 poses.txt 는 프레임 번호 + 3x4 행렬이다. 시각 파일이 따로 없고
    프레임은 10 Hz 라 번호에 0.1 s 를 곱한다. 번호가 건너뛴 구간(98% 는 연속)은
    분석 쪽의 구간 끊기가 알아서 가른다."""
    idx, rot = [], []
    with open(path) as f:
        for line in f:
            c = [float(v) for v in line.split()]
            if len(c) != 13:
                continue
            idx.append(c[0])
            rot.append([c[1:4], c[5:8], c[9:12]])
    r = np.array(rot)
    return np.array(idx) * 0.1, r, r.reshape(len(r), 9).tolist()


def load_pit30m_poses(path):
    """핏서티엠 all_poses.npz 의 continuous 자세. valid 가 선 구간만 쓴다."""
    d = np.load(path, allow_pickle=True)["data"]
    ok = d["continuous"]["valid"] & d["poses_and_differentials_valid"]
    d = d[ok]
    t = d["capture_time"].astype(float)
    c = d["continuous"]
    # 각도는 도 단위로 실려 있다. 공개 문서에 단위 표기가 없어 값의 크기로 확인했다 —
    # yaw 가 116~143 으로 나오고 라디안이면 한 바퀴를 열여덟 번 돈 값이 된다.
    y, p_, r = (np.deg2rad(c[k].astype(float)) for k in ("yaw", "pitch", "roll"))
    return (t, ypr_to_rot(y, p_, r),
            np.stack([c["yaw"], c["pitch"], c["roll"]], axis=1).tolist())


def load_boreas_gps_post(path, stride=1):
    """보레아스 applanix/gps_post_process.csv — 같은 릴리스의 고속 자세 해.

    라이다 프레임에 맞춰 뽑은 lidar_poses.csv 와 달리 원래 해의 주기를 그대로 싣는다.
    같은 릴리스 안에서 유효 대역폭이 표집률에 어떻게 묶이는지를 보여 준다.
    """
    t, ypr = [], []
    with open(path) as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i % stride:
                continue
            t.append(float(row["GPSTime"]))
            ypr.append([float(row["heading"]), float(row["pitch"]), float(row["roll"])])
    a = np.array(ypr)
    return np.array(t), ypr_to_rot(a[:, 0], a[:, 1], a[:, 2]), a.tolist()


# ---------------------------------------------------------------- 스펙트럼

def welch_psd(x, fs, nseg):
    """한 성분의 단측 파워 스펙트럼 밀도. 한 창, 50% 겹침, 해닝."""
    empty = np.empty(0)
    nseg = min(nseg, len(x))
    if nseg < 16:
        return empty, empty
    step = nseg // 2
    win = np.hanning(nseg)
    scale = 1.0 / (fs * (win ** 2).sum())
    acc = np.zeros(nseg // 2 + 1)
    cnt = 0
    for start in range(0, len(x) - nseg + 1, step):
        seg = x[start:start + nseg]
        seg = seg - seg.mean()
        spec = np.abs(np.fft.rfft(seg * win)) ** 2 * scale
        spec[1:-1] *= 2
        acc = acc + spec
        cnt += 1
    if not cnt:
        return empty, empty
    return np.fft.rfftfreq(nseg, 1.0 / fs), acc / cnt


def welch_psd_segments(segs, k, fs, nseg):
    """토막마다 스펙트럼을 내고 창 수로 가중해 평균한다. 토막을 이어붙이지 않는다."""
    usable = [s for s in segs if len(s) >= nseg]
    if not usable:
        # 토막이 다 짧으면 가장 긴 것 하나에 맞춰 창을 줄인다.
        longest = max(segs, key=len)
        nseg = len(longest)
        usable = [longest]
    acc, total = None, 0
    for s in usable:
        f, p = welch_psd(s[:, k], fs, nseg)
        if len(f) == 0:
            continue
        nwin = (len(s) - nseg) // (nseg // 2) + 1
        acc = p * nwin if acc is None else acc + p * nwin
        total += nwin
    if acc is None:
        return np.empty(0), np.empty(0)
    return np.fft.rfftfreq(nseg, 1.0 / fs), acc / total


def quantum(column):
    """한 열의 저장 분해능 상한을 돌려준다. 잴 수 없으면 None.

    상한으로 적는 이유는 그것이 공개 파일에서 실제로 확인되는 전부이기 때문이다.
    서로 다른 두 값의 최소 간격은 참 격자의 정수배이므로 격자의 상한이다. 그 정수가
    1 인지(= 이 값이 곧 격자인지)는 확인할 수 없다 — 키티-360 회전행렬 아홉 칸은
    전부 소수 열 자리인데 실제로 잡히는 최소 간격은 열마다 1e-10, 9e-10, 2.5e-9,
    1.73e-8 로 다르다. 한 칸 차이로 떨어진 이웃이 그 열에 있었느냐에 달려 있다.

    정수배를 가려내 참 격자를 복원하려는 시도는 배정밀도에서 성립하지 않는다.
    크기 1 쯤 되는 값의 표현 오차가 2e-16 이라 1e-10 격자의 백만분의 이에 해당하는데,
    간격이 격자의 백만 배쯤 되는 자리에서는 그 오차가 격자 한 칸을 넘는다. 그래서
    "이 간격이 격자의 몇 배인가" 자체가 자료에 안 남아 있다.

    상한으로 충분한 이유는 이 값을 쓰는 데가 하나이기 때문이다 — 저장 분해능이
    만드는 잡음이 측정된 밀도에 비해 무시할 만한가. 상한이 작으면 그 답은 예다.
    """
    uniq = np.unique(np.asarray(column, dtype=float))
    uniq = uniq[np.isfinite(uniq)]
    if len(uniq) < 16:
        return None
    d = np.diff(uniq)
    d = d[d > 0]
    if len(d) < 8:
        return None
    q = float(d.min())
    scale = float(np.abs(uniq).max())
    if q <= 0 or scale <= 0:
        return None
    # 배정밀도 표현 오차와 구분이 안 되는 간격은 "상한을 못 쟀다" 로 둔다. 그 값을
    # 적으면 저장 분해능이 아니라 부동소수점 해상도를 보고하는 것이 된다.
    return None if q < 64 * float(np.finfo(float).eps) * scale else q


def quantum_report(raw):
    """열마다 따로 재고 그중 가장 큰 상한을 적는다.

    자세를 어떻게 싣는지는 릴리스마다 달라서(회전행렬 아홉 칸, 요·피치·롤 세 칸)
    한 줄의 값들이 같은 단위가 아니다. 다 섞어 하나로 재면 단위가 다른 열이 한
    숫자로 뭉개진다. 열마다 재고, 릴리스 하나의 저장 분해능으로는 가장 거친 열의
    상한을 적는다 — 여러 열로 자세 하나를 만들면 가장 거친 열이 결과를 지배한다.
    """
    rows = [r if isinstance(r, (list, tuple)) else [r] for r in raw]
    if not rows:
        return ""
    qs = [q for q in (quantum(c) for c in zip(*rows)) if q is not None]
    if not qs:
        return "못 잼"
    return "<=%.3g" % max(qs)


# ---------------------------------------------------------------- 본 계산

def rate_segments(t, rot):
    """자세 열에서 이어진 각속도 토막들을 뽑는다. (segs, dt_med, fs, kept, n_inc).

    audit 와 묶음 분석이 같은 토막을 봐야 한다. 토막 나누는 규칙이 두 벌이 되면
    같은 파일을 놓고 낸 두 수치가 왜 다른지를 설명할 수 없다.
    """
    dt = np.diff(t)
    dt_med = float(np.median(dt))
    fs = 1.0 / dt_med

    # 회전 증분에서 각속도. 간격이 고르지 않아도 각 증분을 제 간격으로 나눈다.
    q_inc = np.einsum("nji,njk->nik", rot[:-1], rot[1:])
    w = so3_log(q_inc) / dt[:, None]

    # 끊긴 자리를 건너뛰며 보간하면 없는 신호를 만든다. 간격이 중앙값의 세 배를 넘는
    # 곳에서 끊고 이어진 토막만 쓴다. 핏서티엠처럼 구간이 띄엄띄엄한 공개본이 여기 걸린다.
    # 공백을 건너뛰는 증분 자체도 버린다. 그 증분은 공백 양 끝을 잇는 회전이라
    # 어느 토막에도 속하지 않는다. 앞 토막에 남겨 두면 보간이 공백 안으로 들어가
    # 없는 표본을 만든다 — 그래서 원본보다 표본이 많아지는 일이 생겼다.
    tc = t[:-1] + dt / 2
    cut = np.where(dt > 3 * dt_med)[0]
    starts = np.concatenate([[0], cut + 1])
    ends = np.concatenate([cut, [len(tc)]])
    segs = []
    for a, b in zip(starts, ends):
        if b - a < 64:
            continue
        ts, ws = tc[a:b], w[a:b]
        # 격자 표본 수를 원래 표본 수로 묶는다. 보간은 고르게 다시 놓는 것이지
        # 자료를 늘리는 것이 아니다.
        ngrid = min(b - a, int((ts[-1] - ts[0]) / dt_med) + 1)
        grid = ts[0] + dt_med * np.arange(ngrid)
        segs.append(np.stack([np.interp(grid, ts, ws[:, k]) for k in range(3)], axis=1))
    kept = sum(len(s) for s in segs)
    return segs, dt_med, fs, kept, len(tc)


def audit(t, rot, name, horizon, speed, band_lo=2.0, band_hi=4.0,
          win_s=16.0, forward="x"):
    """한 공개 파일의 자세 열에서 회전 분해능과 관련된 양을 읽는다.

    비교 대역(band_lo, band_hi)과 창 길이(win_s)는 초 단위로 공통이다. 표본 수로
    창을 정하면 10 Hz 공개본과 200 Hz 공개본이 서로 다른 주파수 해상도와 서로 다른
    대역을 보게 되어, 릴리스 사이의 값 차이에 표집률이 섞여 들어간다.
    """
    n = len(t)
    if n < 64:
        return None
    dt = np.diff(t)
    segs, dt_med, fs, kept, n_inc = rate_segments(t, rot)
    f_ny = fs / 2
    nseg = max(16, int(round(win_s * fs)))
    if not segs:
        return None
    wg = np.concatenate(segs, axis=0)

    if band_lo >= f_ny:
        return {"sequence": name, "n": n,
                "note": "표집률이 낮아 비교 대역이 나이퀴스트 위에 있다"}

    inlier = dt[dt <= 3 * dt_med]
    out = {
        "sequence": name,
        "n": n,
        "duration_s": round(float(t[-1] - t[0]), 3),
        "fs_hz": round(fs, 4),
        "nyquist_hz": round(f_ny, 4),
        "n_segments": len(segs),
        "kept_frac": round(kept / max(n_inc, 1), 4),
        "gap_max_s": round(float(dt.max()), 4),
        "jitter_rms_ms": round(float(np.std(inlier) * 1000), 4),
        "jitter_p99_ms": round(float(np.percentile(np.abs(inlier - dt_med), 99) * 1000), 4),
        "win_s": win_s,
        "nseg": nseg,
    }

    spectra, f_effs, dead = {}, [], []
    freqs = np.empty(0)
    for k, axis in enumerate("xyz"):
        if np.allclose(wg[:, k], 0.0):
            # 고정된 축. 평면으로 배포한 기준이 여기 걸린다 — 잡음이 아니라 내용이 없다.
            dead.append(axis)
            out["S0_%s" % axis] = "0"
            continue
        f, s = welch_psd_segments(segs, k, fs, nseg)
        if len(f) == 0:
            continue
        freqs = f
        spectra[axis] = s

    out["flat_axes"] = "".join(dead)
    if not spectra or len(freqs) == 0:
        return out

    band = (freqs >= band_lo) & (freqs <= min(band_hi, f_ny))
    if band.sum() < 2:
        out["note"] = "비교 대역에 들어오는 주파수 칸이 모자라다"
        return out
    out["band_lo_hz"], out["band_hi_hz"] = band_lo, round(min(band_hi, f_ny), 4)
    out["band_bins"] = int(band.sum())

    # 축을 따로 중앙값 내고 더하면 몸체를 상수 회전만 시켜도 총계가 바뀐다. 중앙값은
    # 선형이 아니기 때문이다. 세 축을 주파수마다 먼저 더한 뒤(스펙트럼 행렬의 대각합,
    # 이것은 좌표 회전에 불변이다) 그 합의 중앙값을 쓴다.
    s_tot = np.zeros(len(freqs))
    for axis, s in spectra.items():
        s_tot = s_tot + s
        floor_axis = float(np.median(s[band]))
        out["S0_%s" % axis] = "%.4g" % floor_axis
        above = np.where(s > 10 * floor_axis)[0]
        f_eff = float(freqs[above[-1]]) if len(above) else 0.0
        f_effs.append(f_eff)
        out["f_eff_%s_hz" % axis] = round(f_eff, 4)
    s0_tot = float(np.median(s_tot[band]))
    out["S0_total_rad2ps2perHz"] = "%.4g" % s0_tot
    # 유효 대역은 환산의 산물이 아니라 그 파일이 무엇을 싣고 있는지의 성질이다.
    # 기울기 문 뒤에 두면 환산이 거부된 릴리스에서 통째로 빈칸이 된다 — 바로 그
    # 릴리스들이 "이 대역 위로는 아무것도 말해 주지 않는다" 를 보여야 하는 쪽이다.
    out["f_eff_min_hz"] = round(min(f_effs), 4) if f_effs else 0.0

    # 대역 안의 기울기. 이 한 숫자가 환산을 열고 닫는다.
    #
    # 적분해서 각도 임의보행으로 바꾸는 환산은 이 대역의 밀도가 백색 각속도 오차일
    # 때만 성립한다. 자세에 독립적인(적분되지 않은) 오차가 실려 있으면 그것을 미분한
    # 스펙트럼은 4 sin^2(pi f dt)/dt^2 를 타고 올라가 f^2 에 가까운 기울기를 낸다.
    # 그 경우 지금 대역의 높이를 h 초 구간으로 끌어내리는 것은 근거가 없다 —
    # 숫자를 내는 대신 식별 불가라고 말한다. 평활된 궤적은 반대로 기울기가 음수로
    # 떨어진다.
    lf, ls = np.log(freqs[band]), np.log(np.maximum(s_tot[band], 1e-300))
    slope = float(np.polyfit(lf, ls, 1)[0]) if band.sum() >= 4 else float("nan")
    out["band_slope"] = round(slope, 3)
    # 기울기 하나로는 섞인 것을 못 가른다. 평활로 내려오는 성분과 양자화·독립 자세
    # 오차로 올라가는 성분이 같은 대역에 겹치면 합이 U 자가 되고 그 직선 기울기는
    # 0 을 지난다 — 백색과 구분이 안 된다. 굽힘을 같이 적어 그 경우를 보이게 한다.
    out["band_curv"] = (round(float(np.polyfit(lf, ls, 2)[0]), 3)
                        if band.sum() >= 5 else "")
    if not math.isfinite(slope) or abs(slope) > SLOPE_TOL:
        out["model"] = "f^2 경사" if slope > 0 else "기울어짐"
        out["note"] = ("대역 기울기 %.2f 라 백색 각속도 모형이 안 선다. "
                       "적분 환산은 식별 불가." % slope)
        return out
    out["model"] = "백색"

    arw = math.sqrt(s0_tot / 2)                     # rad/sqrt(s)
    e_ang = arw * math.sqrt(horizon)                # rad

    # 위치에는 진행 방향에 수직인 두 축만 기여한다. 전진축은 릴리스마다 다르다 —
    # 키티 주행거리계 자세는 카메라 좌표계라 z 가 앞이다. 박아 두면 넣을 축과 뺄 축이
    # 뒤바뀐다.
    perp = [a for a in "xyz" if a != forward]
    s_perp = np.zeros(len(freqs))
    for a in perp:
        if a in spectra:
            s_perp = s_perp + spectra[a]
    s0_perp = float(np.median(s_perp[band]))
    arw_perp = math.sqrt(s0_perp / 2)

    out["forward_axis"] = forward
    out["ARW_rad_per_sqrt_s"] = "%.4g" % arw
    out["ARW_deg_per_sqrt_hr"] = "%.4g" % (math.degrees(arw) * 60)
    out["ARW_perp_rad_per_sqrt_s"] = "%.4g" % arw_perp
    out["e_ang_deg"] = "%.4g" % math.degrees(e_ang)
    # 각도 오차를 속도에 곱하면 미터가 아니라 초당 미터다. 등속 직진에서 방향 오차를
    # 구간 h 동안 적분해야 위치가 된다 — 백색 각속도 오차의 이중 적분이라
    # 표준편차가 v*ARW*h^(3/2)/sqrt(3) 다.
    out["e_pos_m"] = "%.4g" % (speed * arw_perp * horizon ** 1.5 / math.sqrt(3))
    out["horizon_s"] = horizon
    out["speed_mps"] = speed
    return out


# ---------------------------------------------------------------- 입력 모으기

def gather(args):
    jobs = []
    if args.helipr_root and os.path.isdir(args.helipr_root):
        for seq in sorted(os.listdir(args.helipr_root)):
            for gt in sorted(glob.glob(os.path.join(args.helipr_root, seq, "LiDAR_GT", "*_gt.txt"))):
                sensor = os.path.basename(gt).replace("_gt.txt", "")
                if sensor.startswith("global_"):
                    # 전역 좌표판은 같은 궤적을 상수 회전만큼 돌린 것이라 회전 증분이
                    # 같다. 두 번 세지 않는다 — 두 판이 같은 값을 주는 것 자체가
                    # 이 측정이 전역 자세에 안 묶인다는 확인이다.
                    continue
                jobs.append(("helipr", "%s/%s" % (seq, sensor), gt, load_tum))
    if args.boreas_root and os.path.isdir(args.boreas_root):
        for seq in sorted(os.listdir(args.boreas_root)):
            p = os.path.join(args.boreas_root, seq, "applanix", "lidar_poses.csv")
            if os.path.exists(p):
                jobs.append(("boreas", "%s/lidar_poses" % seq, p, load_boreas_lidar_poses))
            g = os.path.join(args.boreas_root, seq, "applanix", "gps_post_process.csv")
            if os.path.exists(g):
                jobs.append(("boreas-hirate", "%s/gps_post_process" % seq, g,
                             load_boreas_gps_post))
    if args.kitti_root and os.path.isdir(args.kitti_root):
        for p in sorted(glob.glob(os.path.join(args.kitti_root, "poses", "*.txt"))):
            jobs.append(("kitti-odometry", os.path.basename(p)[:-4], p, load_kitti_odometry))
    if args.kitti360_root and os.path.isdir(args.kitti360_root):
        for p in sorted(glob.glob(os.path.join(args.kitti360_root, "*_sync", "poses.txt"))):
            name = os.path.basename(os.path.dirname(p)).replace("2013_05_28_drive_", "")
            jobs.append(("kitti360", name.replace("_sync", ""), p, load_kitti360_poses))
    if args.pit30m_root and os.path.isdir(args.pit30m_root):
        for log in sorted(os.listdir(args.pit30m_root)):
            p = os.path.join(args.pit30m_root, log, "all_poses.npz")
            if os.path.exists(p) and os.path.getsize(p) > 0:
                jobs.append(("pit30m", log[:8], p, load_pit30m_poses))
    if args.nuscenes_can and os.path.isdir(args.nuscenes_can):
        paths = sorted(glob.glob(os.path.join(args.nuscenes_can, "scene-*_pose.json")))
        if args.nuscenes_limit:
            paths = paths[:args.nuscenes_limit]
        for p in paths:
            jobs.append(("nuscenes", os.path.basename(p).split("_")[0], p, load_nuscenes_pose))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--helipr-root", default="/mnt/Data/velref/helipr")
    ap.add_argument("--boreas-root", default="/mnt/Data/velref/boreas-imu")
    ap.add_argument("--kitti-root", default="/mnt/Data/velref/kitti/dataset")
    ap.add_argument("--kitti360-root", default="/mnt/Data/velref/kitti360")
    ap.add_argument("--pit30m-root", default="/mnt/Data/velref/pit30m")
    ap.add_argument("--nuscenes-can", default="/mnt/Data/velref/nuscenes/can_bus")
    ap.add_argument("--nuscenes-limit", type=int, default=0,
                    help="장면 수 제한. 0 이면 전부.")
    ap.add_argument("--band", default="2,4",
                    help="비교 대역(Hz). 릴리스마다 같은 절대 대역을 본다.")
    ap.add_argument("--horizon", type=float, default=1.0, help="적분 구간 h(s)")
    ap.add_argument("--speed", type=float, default=10.0, help="영향 환산에 쓰는 속도 v(m/s)")
    ap.add_argument("--win-s", type=float, default=16.0,
                    help="스펙트럼 한 창의 길이(s). 표본 수가 아니라 초로 맞춘다.")
    ap.add_argument("--out", default="results/next/reference_resolution.csv")
    args = ap.parse_args()
    band_lo, band_hi = (float(v) for v in args.band.split(","))

    rows = []
    for release, name, path, loader in gather(args):
        try:
            t, rot, raw = loader(path)
        except Exception as e:
            print("[%s] %s 읽기 실패: %s" % (release, name, e))
            continue
        row = audit(t, rot, name, args.horizon, args.speed, band_lo, band_hi,
                    args.win_s, FORWARD_AXIS.get(release, "x"))
        if row is None:
            continue
        row["release"] = release
        if release in SYNTHETIC_TIME:
            # 시각을 프레임 번호로 지어낸 릴리스는 시계 흔들림을 잴 대상이 없다.
            # 비워 두지 않으면 번호가 건너뛴 자리가 시계 잡음으로 읽힌다.
            row["jitter_rms_ms"] = ""
            row["jitter_p99_ms"] = ""
        row["quantum_attitude"] = quantum_report(raw)
        rows.append(row)

    if not rows:
        print("잴 것이 없다")
        return

    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("%d 건 -> %s" % (len(rows), args.out))

    by_release = {}
    for r in rows:
        by_release.setdefault(r["release"], []).append(r)
    for rel, rs in sorted(by_release.items()):
        got = [r for r in rs if "ARW_rad_per_sqrt_s" in r]
        if not got:
            print("%-14s 잴 수 있는 것 없음 (%d 건)" % (rel, len(rs)))
            continue
        aa = sorted(float(r["ARW_deg_per_sqrt_hr"]) for r in got)
        ee = sorted(float(r["e_pos_m"]) for r in got)
        ff = sorted(r["f_eff_min_hz"] for r in got)
        jj = sorted(r["jitter_rms_ms"] for r in got if r["jitter_rms_ms"] != "")
        mid = len(aa) // 2
        flat = {r["flat_axes"] for r in got if r["flat_axes"]}
        print("%-14s n=%-4d fs %6.1f Hz  임의보행 %8.3f deg/sqrt(h)  유효대역 %5.2f Hz  "
              "시각흔들림 %s  e_pos %7.4f m  고정축 %s"
              % (rel, len(got), got[0]["fs_hz"], aa[mid], ff[mid],
                 "%6.3f ms" % jj[len(jj) // 2] if jj else "   공개 안 함",
                 ee[mid], ",".join(sorted(flat)) or "없음"))


if __name__ == "__main__":
    main()
