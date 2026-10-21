#!/usr/bin/env python3
"""키티 주행거리계 공식 지표의 단위로 재표집 이동량 D 를 쓴다.

release_resample_D.py 는 이웃 쌍의 평균 각거리로 D 를 쟀다. 순위 보존을 말하려면
사람들이 실제로 보고하는 수와 같은 단위여야 한다. 키티 주행거리계 지표는 이렇다.

    100, 200, ..., 800 m 구간마다  d(A_ij, Q_ij) / L  을 평균하고 도/미터로 적는다

이것도 각거리의 가중 평균이다 — 가중치가 w_k = 1/(N*L_k) 일 뿐이다. 그래서
|E' - E| <= sum w_k d(Q_k, Q'_k) 가 그대로 선다. 오른쪽을 그 단위로 재 둔다.

같은 표의 병진오차 칸도 같은 꼴이다. 공식 t_err 은 두 구간 이동 벡터 사이의
유클리드 거리라 역시 거리함수이고, 가중치도 1/(N*L_k) 로 같다. 그래서 같은
상한이 그대로 서고, 회전과 병진을 같은 구간에서 나란히 낸다.

그러면 읽는 사람이 자기 표를 들고 와서 바로 쓴다. 같은 시퀀스에서 두 방법의
차이가 2D 보다 크면 그 순위가 보간 규약에 흔들리지 않는다고 인증할 수 있고,
작으면 이 상한만으로는 인증하지 못한다(뒤집힌다는 뜻이 아니다 — 두 점수가
같은 방향으로 밀려 상쇄될 수 있고 그 쌍별 변화는 재지 않았다).

규약은 release_resample_D.py 와 같다. phase_half 와 decim2 는 프레임 구조를
그대로 두고 기준값만 바꾸므로 구간도 그대로다. interp 는 격자 자체가 달라
그 격자에서 구간을 다시 고른다.
"""

import argparse
import csv
import glob
import hashlib
import json
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tum_poses import pieces, read_tum  # noqa: E402

LENGTHS = (100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0)
STEP = 10  # 공식 평가가 쓰는 시작 프레임 간격.
EST_RATE = 0.7


def exact(x):
    """파일에 적어 두었다가 되읽어 견줄 수는 깎지 않고 적는다.

    세 자리로 깎으면 1.002 가 "1" 로 적히고, 그 줄을 다시 읽어 1 과 견주는
    쪽은 넘은 줄을 안 넘은 줄로 센다. 반대쪽도 같다 — 0.9996 도 "1" 이 된다.
    문턱 바로 위아래가 그 수를 적는 이유인데 깎는 자리가 정확히 거기다.
    상한을 넘었는지 세는 칸이 특히 그렇다 — 반올림이 위반을 지운다.

    화면에 찍는 수는 이 규칙 대상이 아니다. 되읽지 않으므로 깎아도 된다.
    """
    return repr(float(x))


def relative_time(t):
    """첫 표본을 뺀 시각. 재는 쪽과 그것을 설명하는 탐침이 같이 쓴다.

    뺄셈은 가까운 두 배정밀도 사이라 정확하다. 유닉스 기원 크기(1.7e9 초)에서는
    배정밀도 한 칸이 2.4e-07 초라 창 경계와 격자점이 그 칸 위로 눌리는데, 빼고
    나면 경과 시간 크기의 칸(1e-13 초 대)으로 내려간다. 같은 시퀀스라도 두 표현
    에서 창이 한두 프레임 다르게 잡히므로, 한쪽은 빼고 한쪽은 안 빼면 구간 집합이
    갈린다 — 그러면 묶음 수와 그 수를 설명하는 탐침이 다른 것을 센다.
    """
    t = np.asarray(t, dtype=float)
    return t - t[0]


def load_poses(path):
    """poses/NN.txt 의 3x4 행렬과 sequences/NN/times.txt 의 시각."""
    seq = os.path.basename(path)[:-4]
    root = os.path.dirname(os.path.dirname(path))
    rot, pos = [], []
    with open(path) as f:
        for line in f:
            c = [float(v) for v in line.split()]
            if len(c) != 12:
                continue
            rot.append([c[0:3], c[4:7], c[8:11]])
            pos.append([c[3], c[7], c[11]])
    tf = os.path.join(root, "sequences", seq, "times.txt")
    n = len(rot)
    t = np.array([float(x) for x in open(tf)])[:n] if os.path.exists(tf) \
        else np.arange(n) * 0.1
    return t[:n], np.array(rot[:n]), np.array(pos[:n])


def kitti360_timestamps(path):
    """그 드라이브의 oxts/timestamps.txt 를 초로 읽는다. 없으면 None.

    한 줄이 한 프레임이고 줄 번호가 poses.txt 의 프레임 번호다. 적힌 꼴은
    "2013-05-28 08:46:11.906941184" 로 나노초까지다. 나노초 정수를 그대로
    부동소수로 옮기면 1.37e18 이라 배정도가 못 담으므로, 첫 점을 빼고 나서
    초로 바꾼 뒤 그 첫 점의 초를 더한다.
    """
    p = os.path.join(os.path.dirname(path), "oxts", "timestamps.txt")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        raw = [line.strip() for line in f if line.strip()]
    ns = np.array(raw, dtype="datetime64[ns]")
    base = int(ns[0].astype("datetime64[s]").astype(np.int64))
    return float(base) + ((ns - np.datetime64(base, "s"))
                          / np.timedelta64(1, "s"))


def load_poses_kitti360(path, rate=10.0, stamped=False):
    """KITTI-360 data_poses 의 poses.txt — 첫 칸이 프레임 번호, 그 뒤 3x4.

    번호가 드문드문 빈다(한 칸 건너뛴 자리가 드라이브마다 2~5 퍼센트). 그래서
    행 번호가 아니라 프레임 번호를 공개 주기로 나눈다 — 행 번호로 쓰면 그
    구멍이 메워져서 실제로는 0.2 초 걸린 이동이 0.1 초짜리로 읽힌다.

    `stamped=True` 면 공식 oxts/timestamps.txt 의 시각을 쓴다. 기본값이 아닌
    이유는 그 시각이 자세의 간격을 안 설명하기 때문이다. 그 파일의 간격은
    0.100 초와 0.110 초를 오가는데, 자세 쪽 이동량은 그 두 자리에서 같다 —
    간격비가 1.10 인 이웃 쌍에서 이동량비 중앙값이 시각이 있는 아홉 드라이브
    전부 0.99988~1.00014 다(드라이브마다 904~15,813 쌍). 같은 속도로 달리는
    중이면 0.110 초 쪽이 10 퍼센트 더 가 있어야 한다. 그 축으로 읽으면 겉보기
    속도가 프레임마다 9.2~9.4 퍼센트씩 튀고 프레임 격자에서는 0.3~0.7 이다.

    이 시험이 정하는 것은 읽는 규약까지다. 자세가 배포 전에 등간격으로
    재표집됐는지, 절대 주기가 공개된 10 Hz 와 같은지는 이 자료로 안 갈린다.
    0008·0018 은 시각 파일이 없어 시험 대상이 아니다.
    """
    frame, rot, pos = [], [], []
    with open(path) as f:
        for line in f:
            c = [float(v) for v in line.split()]
            if len(c) != 13:
                continue
            frame.append(c[0])
            rot.append([c[1:4], c[5:8], c[9:12]])
            pos.append([c[4], c[8], c[12]])
    idx = np.asarray(frame, dtype=np.int64)
    t = idx.astype(float) / rate
    if stamped:
        stamps = kitti360_timestamps(path)
        if stamps is not None and idx.max() < len(stamps):
            t = stamps[idx]
    return t, np.array(rot), np.array(pos)


def ypr_to_rot_boreas(y, p, r):
    """공식 pyboreas 의 yawPitchRollToRot 을 벡터로 옮긴 것.

    그 구현은 roll(r) @ pitch(p) @ yaw(y) 이고 세 기본 행렬이 전부 흔히 쓰는
    능동 회전의 전치다. 그래서 결과는 scipy 의 from_euler("ZYX", [y,p,r]) 의
    전치가 된다. 둘을 바꿔 쓰면 회전 쪽은 안 드러난다 — 상대 회전의 측지각은
    전부 전치해도 그대로라서 그렇다. 드러나는 것은 구간 상대 병진
    R_s^T (p_e - p_s) 뿐이다.

    실제 파일로 가른다. 올바른 쪽이면 그 값의 방향이 주행 내내 한 자리에
    머물러야 한다(본체 좌표에서 본 전진 방향이니까). 보레아스 첫 시퀀스에서
    이 구성은 45.8 도에 사분위폭 1.2 도로 모이고, 전치하면 -41 도에서 130 도로
    벌어진다. 45.8 도는 라이다가 돌아 달린 것이라 상수이고 측정값에 안 섞인다.
    """
    cy, sy = np.cos(y), np.sin(y)
    cp, sp = np.cos(p), np.sin(p)
    cr, sr = np.cos(r), np.sin(r)
    z, o = np.zeros_like(y), np.ones_like(y)
    rr = np.stack([o, z, z, z, cr, sr, z, -sr, cr], -1).reshape(-1, 3, 3)
    pp = np.stack([cp, z, -sp, z, o, z, sp, z, cp], -1).reshape(-1, 3, 3)
    yy = np.stack([cy, sy, z, -sy, cy, z, z, z, o], -1).reshape(-1, 3, 3)
    return rr @ pp @ yy


def load_poses_boreas(path):
    """보레아스 applanix/lidar_poses.csv — 라이다 프레임 시각의 자세와 위치.

    이 릴리스를 공식 단위로 재는 까닭은 공개 순위표가 살아 있고 그 순위표가
    키티와 같은 구간 상대오차를 쓰기 때문이다. 자세는 방위·기울기·좌우기울기로,
    위치는 동·북·높이로 실린다. 회전은 공식 구성을 그대로 쓴다.
    """
    t, ypr, pos = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            t.append(float(row["GPSTime"]) * 1e-6)
            ypr.append([float(row["heading"]), float(row["pitch"]),
                        float(row["roll"])])
            pos.append([float(row["easting"]), float(row["northing"]),
                        float(row["altitude"])])
    a = np.asarray(ypr)
    rot = ypr_to_rot_boreas(a[:, 0], a[:, 1], a[:, 2])
    return np.asarray(t), rot, np.asarray(pos)


def load_poses_mav_csv(path):
    """에이에스엘 꼴 mav0/*/data.csv — 시각(나노초) x y z qw qx qy qz.

    쿼터니언 칸 순서가 실수 부분부터라 헬리퍼와 뒤집혀 있다. 머리글 줄은
    우물정자로 시작한다. 튬 VI 의 mocap0 와 유로크의 state_groundtruth 가
    같은 꼴이다 — 튬 VI 가 유로크의 배치를 그대로 받았다. 유로크 파일은
    뒤에 속도·편향 칸이 더 붙지만 앞 여덟 칸의 뜻이 같아 같은 적재기로
    읽는다.
    """
    a = np.loadtxt(path, delimiter=",", comments="#")
    quat = a[:, [5, 6, 7, 4]]
    return a[:, 0] * 1e-9, Rotation.from_quat(quat).as_matrix(), a[:, 1:4]


def load_poses_oxford(path):
    """옥스퍼드 로봇카 RTK 묶음의 <날짜>/rtk.csv.

    칸은 머리글이 적어 둔다 — 시각(마이크로초), 위도, 경도, 고도, 북쪽,
    동쪽, 아래, UTM 구역, 속도 셋, 좌우기울기, 앞뒤기울기, 방위. 위치는
    북·동·아래로, 자세는 라디안으로 읽고 NCLT 와 같은 Z-Y-X 내재 회전으로
    세운다.

    이어 붙이지 못하는 자리가 둘 있고 둘 다 그 파일 안에서 드러난다.

    하나는 UTM 구역이다. 한 주행이 30U 에서 30T 로 266 줄 갔다 돌아오는데
    (2015-08-18-14-07-43), 구역이 다르면 북·동 값의 원점이 달라서 그대로
    이으면 수백 킬로미터짜리 변위가 생긴다.

    다른 하나는 아직 안 닫혔다. 같은 줄에 실린 속도 칸으로 설명되지 않는
    걸음이 71 주행 중 18 에 있다 — 위치가 한 걸음에 천 미터씩 뛰는 동안
    속도는 걷는 속도 그대로다. 그 자리를 어떤 규약으로 읽을지 정하기 전까지
    이 릴리스는 RELEASE_ROOTS 에 안 넣는다. 잰 값은
    results/next/oxford_rtk_quality.csv 에 있다.
    """
    zone = np.genfromtxt(path, delimiter=",", skip_header=1, usecols=(7,),
                         dtype=str)
    a = np.genfromtxt(path, delimiter=",", skip_header=1,
                      usecols=(0, 4, 5, 6, 11, 12, 13, 8, 9, 10))
    a = np.atleast_2d(a)
    zone = np.atleast_1d(zone)
    empty = (np.zeros(0), np.zeros((0, 3, 3)), np.zeros((0, 3)))
    if a.size == 0 or len(a) < 2:
        return empty
    keep = ~np.isnan(a).any(axis=1)
    a, zone = a[keep], zone[keep]
    a, zone = longest_run(a, zone[1:] != zone[:-1]), None
    if len(a) < 2:
        return empty
    rot = Rotation.from_euler("ZYX", a[:, [6, 5, 4]]).as_matrix()
    return a[:, 0] * 1e-6, rot, a[:, 1:4]


def longest_run(a, broken):
    """이어지지 않는 자리에서 끊고 가장 긴 토막을 돌려준다.

    `broken[i]` 는 i 와 i+1 사이가 끊겼다는 뜻이라 길이가 len(a)-1 이다.
    """
    bounds = np.r_[0, np.flatnonzero(broken) + 1, len(a)]
    k = int(np.argmax(np.diff(bounds)))
    return a[bounds[k]:bounds[k + 1]]


def load_poses_nclt(path):
    """NCLT ground_truth/groundtruth_<날짜>.csv — 시각(마이크로초) x y z r p h.

    위치는 북·동·아래, 자세는 좌우기울기·앞뒤기울기·방위를 라디안으로 싣는다.
    Z-Y-X 내재 회전이라 순서를 뒤집어 넣는다. 보레아스와 달리 전치가 아니다 —
    같은 가름법(본체 좌표에서 본 전진 방향이 주행 내내 한 자리에 머무는가)으로
    재보면 이 구성이 -0.45 도에 사분위폭 4.2 도로 모이고 전치하면 -124 도에서
    136 도로 벌어진다.

    빈 칸이 -nan 으로 적힌 줄이 있다. 공개 파일이 그 시각의 자세를 모른다는
    뜻이므로 0 으로 메우지 않고 줄째로 뺀다 — 메우면 없는 자세가 구간 끝점에
    서서 그 구간의 변위를 통째로 지어낸다.
    """
    a = np.loadtxt(path, delimiter=",")
    a = a[~np.isnan(a).any(axis=1)]
    rot = Rotation.from_euler("ZYX", a[:, [6, 5, 4]]).as_matrix()
    return a[:, 0] * 1e-6, rot, a[:, 1:4]


def load_poses_helipr(path):
    """헬리퍼 LiDAR_GT 의 *_gt.txt — 시각(나노초) x y z qx qy qz qw."""
    t, quat, pos = [], [], []
    with open(path) as f:
        for line in f:
            c = line.split()
            if len(c) < 8:
                continue
            t.append(float(c[0]) * 1e-9)
            pos.append([float(c[1]), float(c[2]), float(c[3])])
            quat.append([float(c[4]), float(c[5]), float(c[6]), float(c[7])])
    rot = Rotation.from_quat(np.asarray(quat)).as_matrix()
    return np.asarray(t), rot, np.asarray(pos)


def load_poses_nuscenes(path):
    """누씬즈 차량통신망 확장팩 scene-*_pose.json.

    orientation 은 wxyz 순이고 pos 의 높이 칸은 전 장면에서 0 이다. 평면
    궤적이라 구간 길이는 수평 이동만으로 쌓인다.
    """
    with open(path) as f:
        rows = json.load(f)
    t = np.asarray([r["utime"] * 1e-6 for r in rows])
    o = np.asarray([r["orientation"] for r in rows])
    rot = Rotation.from_quat(np.stack([o[:, 1], o[:, 2], o[:, 3], o[:, 0]],
                                      axis=1)).as_matrix()
    return t, rot, np.asarray([r["pos"] for r in rows])


def load_poses_pit30m(path):
    """핏서티엠 all_poses.npz 의 continuous 자세. valid 가 선 구간만 쓴다.

    각도는 도 단위로 실려 있다. 묶음의 dtype 에도 공식 적재기
    (pit30m/data/log_reader.py)에도 단위 표기가 없어 같은 레코드의 독립된
    칸인 위치로 가렸다 — 달리는 동안 자세의 앞축과 이동 방향이 벌어진 각이
    도로 읽으면 기록 넷 전부에서 중앙값 0.36~0.51 도이고 라디안으로 읽으면
    89~93 도다(scripts/pit30m_angle_unit_probe.py,
    results/next/pit30m_angle_unit.csv). 궤적으로 가려지는 것은 앞축을
    세우는 yaw 와 pitch 뿐이라, roll 은 같은 레코드에 나란히 실렸다는 이유로
    같은 단위로 읽는다.
    """
    d = np.load(path, allow_pickle=True)["data"]
    d = d[d["continuous"]["valid"] & d["poses_and_differentials_valid"]]
    c = d["continuous"]
    ypr = np.deg2rad(np.stack([c["yaw"].astype(float),
                               c["pitch"].astype(float),
                               c["roll"].astype(float)], axis=1))
    pos = np.stack([c["x"].astype(float), c["y"].astype(float),
                    c["z"].astype(float)], axis=1)
    return (d["capture_time"].astype(float),
            Rotation.from_euler("ZYX", ypr).as_matrix(), pos)


def segments(pos, lengths=LENGTHS, step=STEP):
    """공식 규칙대로 (시작, 끝, 요구길이) 목록.

    evaluate_odometry.cpp 의 lastFrameFromSegmentLength 는 누적 이동거리가
    요구길이를 넘는 첫 프레임을 끝으로 잡는다 — 딱 닿는 자리는 안 받는다
    (`dist[i] > dist[first] + len`). 그래서 side="right" 다. 넘기 전에
    기록이 끝나면 그 구간은 버린다(공식은 -1 을 돌려주고 건너뛴다).
    """
    d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pos, axis=0),
                                                        axis=1))])
    out = []
    for i in range(0, len(d), step):
        for L in lengths:
            j = int(np.searchsorted(d, d[i] + L, side="right"))
            if j >= len(d):
                continue
            out.append((i, j, L))
    return out


def geodesic(a, b):
    tr = np.trace(np.einsum("nba,nbc->nac", a, b), axis1=1, axis2=2)
    return np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))


def relative(rot, idx):
    i = np.asarray([p[0] for p in idx])
    j = np.asarray([p[1] for p in idx])
    return np.einsum("nba,nbc->nac", rot[i], rot[j])


def relative_translation(rot, pos, idx):
    """공식 병진오차가 재는 양 — 시작 프레임에서 본 구간 이동 벡터.

    evaluate_odometry.cpp 는 pose_error = inv(delta_result) * delta_gt 를 만들고
    그 이동 성분의 길이를 t_err 로 쓴다. 그 성분은 R_r^T (t_g - t_r) 이고 회전은
    길이를 안 바꾸므로 t_err = ||t_g - t_r|| 다. 즉 점수는 두 구간 이동 벡터
    사이의 유클리드 거리이고, 각거리와 똑같이 거리함수다. 그래서 삼각부등식이
    그대로 서고 D 의 증명이 병진 칸에도 옮겨간다.
    """
    i = np.asarray([p[0] for p in idx])
    j = np.asarray([p[1] for p in idx])
    return np.einsum("nba,nb->na", rot[i], pos[j] - pos[i])


def euclidean(a, b):
    return np.linalg.norm(np.asarray(a) - np.asarray(b), axis=1)


def rms_displacement(g, idx):
    """제곱평균제곱근으로 적는 표에 맞는 D.

    논문 표가 늘 평균인 것은 아니다. 구간별 오차를 제곱평균제곱근으로 묶어 적는
    표가 있는데, 거기에는 평균 꼴 상한이 안 선다. 대신 같은 자리에 민코프스키
    부등식이 들어간다 — a_k = d(A_k,Q_k)/L_k 로 두면

        | ||a'||_2 - ||a||_2 |  <=  ||a' - a||_2  <=  || d(Q,Q')/L ||_2

    오른쪽이 이 함수다. 평균 꼴과 증명이 다르니 값도 따로 적는다.
    """
    L = np.asarray([p[2] for p in idx], dtype=float)
    return float(np.sqrt(np.mean((np.asarray(g) / L) ** 2)))


def segment_weights(idx):
    """w_k = 1/(N*L_k). 공식 지표가 구간마다 길이로 나눠 평균하기 때문이다."""
    L = np.asarray([p[2] for p in idx], dtype=float)
    return 1.0 / (len(idx) * L)


def resample(t, rot, pos, tq, method):
    r = Rotation.from_matrix(rot)
    if method == "slerp":
        rq = Slerp(t, r)(tq).as_matrix()
        pq = np.stack([np.interp(tq, t, pos[:, k]) for k in range(3)], axis=1)
        return rq, pq
    if method == "nearest":
        # 동률이면 앞 표본을 고른다. 이 자리는 실제로 밟힌다 — 성긴 격자가
        # 표본을 하나 걸러 고르면 남은 질의가 정확히 두 표본 한가운데에
        # 놓인다. 핏서티엠에서 규칙을 뒤집으면 묶음 D 가 상대 0.23 퍼센트까지
        # 달라지므로, 부등호 하나에 맡기고 안 적으면 복원 방법의 비용에
        # 이름 없는 인자가 하나 더 들어간다.
        #
        # 공통 원점을 미리 빼지는 않는다. 시각이 1e9 대라 두 간격의 끝자리가
        # 흔들리는 것은 맞지만, 그 흔들림은 t 를 그 크기로 저장한 순간 이미
        # 들어가 있어서 여기서 빼도 안 돌아온다. 정확히 표현되는 동률(0.25 초
        # 격자의 한가운데)은 원점을 1.7e9 까지 옮겨도 아래 규칙대로 앞 표본을
        # 고르고, 원점에 따라 답이 갈리는 것은 십진 0.2 초 격자처럼 애초에
        # 이진 동률이 아닌 입력뿐이다. 그런 입력은 원점을 바꾸면 수 자체가
        # 달라진 것이라 같은 질문이 아니다.
        j = np.clip(np.searchsorted(t, tq), 1, len(t) - 1)
        pick = np.where(tq - t[j - 1] <= t[j] - tq, j - 1, j)
        return rot[pick], pos[pick]
    raise ValueError("모르는 보간: %s" % method)


def common_window(t, dt=None):
    """규약을 바꿔도 외삽이 안 생기는 공통 창의 끝 색인. 배타적이다.

    부르는 쪽은 t[:keep] 로 자른다 — 돌려주는 색인의 프레임은 창 밖이다.

    반 표본 위상은 t+dt/2 를, 반 주기 솎기는 솎은 격자 위에서 t 를 묻는다.
    둘 다 공개 표본의 마지막 시각을 넘을 수 있는데, 넘는 점을 끝으로 잘라
    붙이면(clip) 그 프레임만 어긋남이 0 이 되거나 한 표본 통째로 어긋난다.
    어느 쪽이든 한 수 안에 규약이 다른 값이 섞인다 — 실측에서 반 주기 솎기의
    병진 이동량이 그 한 프레임 때문에 21 퍼센트 높게 나왔다.

    창을 좁히면 비교하는 두 쪽이 같은 구간 집합을 쓴다. 상한의 전제가
    요구하는 것이 그 동일성이다. 창은 평가 범위지 보간 범위가 아니다 —
    재표집의 원본은 공개 표본 전체로 남겨 둔다.
    """
    t = np.asarray(t, dtype=float)
    if dt is None:
        dt = float(np.median(np.diff(t)))
    last = min(t[-1] - 0.5 * dt, t[::2][-1])
    return int(np.searchsorted(t, last, side="right"))


def measure(t, rot, pos, raw=None, lengths=LENGTHS, step=STEP):
    """구간 쌍 기준 D 들. (dict, 구간 수) 반환.

    회전 쪽은 라디안/미터, 병진 쪽(`t_` 접두)은 무차원(미터/미터)이다. 공식 표가
    회전오차와 병진오차를 나란히 싣기 때문에 같은 구간·같은 가중치로 둘 다 낸다.

    raw 에 사전을 넘기면 규약마다 (회전 이동량, 병진 이동량, 구간길이) 세
    배열을 그대로 담아 준다. 시퀀스를 가로질러 하나로 묶을 때 쓴다 — 공식
    순위표는 시퀀스별 수를 다시 평균하는 게 아니라 구간을 통째로 모아 평균하는
    꼴이라, 시퀀스별 D 의 중앙값이 그 값과 같지 않다.

    구간 규칙(lengths·step)을 바꿔 부를 수 있다. 키티-360 공식 순위표가 쓰는
    상대오차는 1 m 떨어진 두 자세 사이에서 재므로 lengths=(1.0,), step=1 이다.
    같은 가중평균 꼴이라 상한의 증명은 그대로이고 구간 규칙만 다르다.

    맨 먼저 시각에서 첫 점을 뺀다. 일곱 중 다섯이 유닉스 기원 초로 시각을
    싣는데, 그 크기에서는 배정밀도 한 칸이 2.4e-07 초다. 여기서 만드는 시각은
    전부 그 칸 위로 눌린다 — 반 표본 위상도, 성긴 격자도, 밀리초 어긋남도
    그렇다. 1 밀리초를 더하면 실제로 더해지는 양이 1 밀리초가 아니라 그 칸에
    맞춰 떨어진 값이다. 빼는 것은 가까운 두 배정밀도의 뺄셈이라 정확하고,
    뺀 뒤에는 경과 시간 크기의 칸(1e-13 초 대)에서 돈다. 입력 자신이 이미
    그 칸 위에 있는 것은 이걸로 안 돌아온다 — 되돌릴 수 있는 쪽만 되돌린다.
    """
    t = np.asarray(t, dtype=float)
    rot = np.asarray(rot, dtype=float)
    pos = np.asarray(pos, dtype=float)
    t = relative_time(t)
    dt = float(np.median(np.diff(t)))
    t_src, rot_src, pos_src = t, rot, pos
    keep = common_window(t, dt)
    t, rot, pos = t[:keep], rot[:keep], pos[:keep]
    if len(t) < 2:
        return None, 0
    idx = segments(pos, lengths, step)
    if not idx:
        return None, 0
    w = segment_weights(idx)
    q = relative(rot, idx)
    qt = relative_translation(rot, pos, idx)

    out = {}

    def record(name, g, gt, jdx, wj):
        out[name] = float(np.dot(wj, g))
        out[name + "_rms"] = rms_displacement(g, jdx)
        out["t_" + name] = float(np.dot(wj, gt))
        out["t_" + name + "_rms"] = rms_displacement(gt, jdx)
        if raw is not None:
            raw[name] = (np.asarray(g), np.asarray(gt),
                         np.asarray([p[2] for p in jdx], dtype=float))

    hr, hp = resample(t_src, rot_src, pos_src, t + 0.5 * dt, "slerp")
    record("phase_half", geodesic(q, relative(hr, idx)),
           euclidean(qt, relative_translation(hr, hp, idx)), idx, w)

    dr, dp = resample(t_src[::2], rot_src[::2], pos_src[::2], t, "slerp")
    record("decim2", geodesic(q, relative(dr, idx)),
           euclidean(qt, relative_translation(dr, dp, idx)), idx, w)

    # 시각이 큰 수(위성항법 초 단위, 1e9 자리)로 실린 릴리스가 있다. 그런
    # 자리에서는 arange 가 끝점을 아주 조금 넘겨 짚고, 구면 선형보간이 범위
    # 밖이라며 거절한다. 넘는 점은 버린다 — 자르면 끝점이 겹쳐 길이 0 짜리
    # 구간이 생긴다.
    tq = np.arange(t[0], t[-1], dt / EST_RATE)
    tq = tq[tq <= t[-1]]
    rs, ps = resample(t, rot, pos, tq, "slerp")
    rn, pn = resample(t, rot, pos, tq, "nearest")
    jdx = segments(ps, lengths, step)
    if jdx:
        record("interp", geodesic(relative(rs, jdx), relative(rn, jdx)),
               euclidean(relative_translation(rs, ps, jdx),
                         relative_translation(rn, pn, jdx)),
               jdx, segment_weights(jdx))
    return out, len(idx)


def pool(raws):
    """시퀀스별 원시 배열들을 하나로 묶어 릴리스 전체의 D 를 낸다.

    공식 순위표는 시험 시퀀스 전체의 구간을 한 자루에 담아 평균한다. 시퀀스별
    D 를 다시 평균하면 구간 수가 적은 시퀀스가 과하게 세어지므로 값이 다르다.
    순위표의 한 칸과 맞대려면 이쪽이 맞는 꼴이다.
    """
    out = {}
    names = set()
    for r in raws:
        names |= set(r)
    for name in sorted(names):
        g = np.concatenate([r[name][0] for r in raws if name in r])
        gt = np.concatenate([r[name][1] for r in raws if name in r])
        L = np.concatenate([r[name][2] for r in raws if name in r])
        out[name] = float(np.mean(g / L))
        out[name + "_rms"] = float(np.sqrt(np.mean((g / L) ** 2)))
        out["t_" + name] = float(np.mean(gt / L))
        out["t_" + name + "_rms"] = float(np.sqrt(np.mean((gt / L) ** 2)))
    return out


def proper(rot):
    """공개 자세를 가장 가까운 회전으로 한 번 눌러 둔다.

    키티와 키티-360 은 자세를 3x4 행렬로 싣는데 그 3x3 칸이 정확히 직교하지
    않는다(각각 2.8e-07, 1.4e-10). 재표집은 쿼터니언을 거치므로 돌려받는
    자세는 반드시 직교한다. 기준 쪽만 원본 행렬을 쓰면 아무것도 안 바꾼
    규약에서도 두 쪽이 달라지고, 측지각이 항등 근처에서 그 차이를 제곱근으로
    키워서 키티 1 m 간격 규칙에서는 1.12 도/100 m 가 남는다 — 1 밀리초
    어긋남이 내는 값과 거의 같다. 양쪽을 같은 방식으로 읽으면 사라진다.

    이탈폭 자체와 그것이 남기는 바닥은 scripts/orthonormality_floor.py 가
    원본을 그대로 읽어 따로 잰다.

    특잇값 분해로 직접 짚는다. Rotation.from_matrix 왕복으로는 안 된다 —
    scipy 1.10 의 그 경로는 쿼터니언을 뽑아 길이만 맞추는 것이라 가장 가까운
    회전과 이탈폭의 세 배쯤 어긋난다(실측: 이탈 1e-07 에서 3.29e-07).
    프로베니우스 거리를 최소로 만드는 것은 U V^T 이고, 행렬식이 음이면 마지막
    특잇값 쪽 부호를 뒤집어 제대로 된 회전으로 만든다.
    """
    m = np.asarray(rot, dtype=float)
    u, _, vt = np.linalg.svd(m)
    d = np.sign(np.linalg.det(u @ vt))
    u = u.copy()
    u[..., :, -1] *= d[..., None] if np.ndim(d) else d
    return u @ vt


RELEASE_ROOTS = {
    "kitti": "/mnt/Data/velref/kitti/dataset",
    "kitti360": "/mnt/Data/velref/kitti360",
    "boreas": "/mnt/Data/velref/boreas",
    "helipr": "/mnt/Data/velref/helipr",
    "nuscenes": "/mnt/Data/velref/nuscenes/can_bus",
    "pit30m": "/mnt/Data/velref/pit30m",
    "tumrgbd": "/mnt/Data/velref/tumrgbd",
    "nclt": "/mnt/Data/velref/nclt",
    "tumvi": "/mnt/Data/velref/tumvi",
    "euroc": "/mnt/Data/velref/euroc",
}
EUROC_ROOT = RELEASE_ROOTS["euroc"]
# 옥스퍼드는 자료가 들어왔지만 표에 안 넣는다. 공개 위치가 정상 주기에서
# 튀는데 그 걸음을 거르려면 문턱을 우리가 골라야 하고, 그 선택 자체가
# 이 논문이 재는 대상이다.
OXFORD_ROOT = "/mnt/Data/velref/oxford/rtk_x"

RELEASES = tuple(RELEASE_ROOTS)
# 아직 표에 없지만 따로 재 볼 수는 있다. 묶음 표와 매크로는 RELEASES 만 돈다.
PENDING = ("oxford",)


def gather(release, root="", project=True):
    """(시퀀스 이름, 자세 세 묶음) 들. 릴리스마다 파일 꼴이 달라서 나눈다.

    뿌리 경로는 릴리스 하나당 하나다. 비우면 RELEASE_ROOTS 의 값을 쓴다 —
    릴리스를 더할 때 그 표에 한 줄만 넣으면 이 함수를 부르는 쪽은 안 고친다.

    project 를 끄면 공개 파일의 숫자를 그대로 준다. 직교 이탈을 재는 쪽만
    그렇게 읽는다.
    """
    for name, (t, rot, pos) in _gather_raw(release, root):
        yield name, ((t, proper(rot), pos) if project else (t, rot, pos))


def _digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_files(release, root=""):
    """그 릴리스에서 측정이 실제로 여는 (시퀀스 이름, 파일 경로들) 들.

    경로는 묶음으로 낸다. 첫째가 읽개에 건네는 자세 파일이고, 뒤는 그
    읽개가 같이 여는 보조 입력이다 — 키티의 시각 파일이 그것이다.

    고름이 여기 한곳에 있다. 폴더를 따로 훑어 목록을 만들면 그 목록은
    측정이 읽은 것과 조용히 갈린다 — 헬리퍼의 전역 좌표판처럼 측정이
    건너뛰는 파일이 목록에만 남는 꼴이다. 원본 고정표도 이 함수를 쓴다.
    """
    root = root or RELEASE_ROOTS.get(release, "")
    if release == "tumrgbd":
        # 같은 궤적이 두 이름으로 깔려 있다. fr1_xyz 는
        # rgbd_dataset_freiburg1_xyz 와 바이트까지 같은 사본이라, 그대로 읽으면
        # 그 한 주행이 집계에 두 번 들어가고 무게도 두 몫을 받는다. 내용 해시로
        # 가려내고 릴리스 자신의 이름(rgbd_dataset_ 로 시작하는 쪽)을 남긴다.
        # 이 릴리스 파일은 다 합쳐 수십 메가라 해시 값이 싸다.
        seen = {}
        for p in sorted(glob.glob(os.path.join(root, "*-groundtruth.txt"))):
            h = _digest(p)
            base = os.path.basename(p)[:-len("-groundtruth.txt")]
            if h not in seen or (base.startswith("rgbd_dataset_")
                                 and not seen[h][0].startswith(
                                     "rgbd_dataset_")):
                seen[h] = (base, p)
        for base, p in sorted(seen.values()):
            yield base, (p,)
        return
    if release == "kitti":
        # 이 릴리스만 자세 파일 하나에 시각 파일 하나가 붙는다 —
        # load_poses 가 sequences/NN/times.txt 를 같이 연다.
        for p in sorted(glob.glob(os.path.join(root, "poses", "*.txt"))):
            seq = os.path.basename(p)[:-4]
            tf = os.path.join(root, "sequences", seq, "times.txt")
            yield seq, (p, tf) if os.path.exists(tf) else (p,)
        return
    if release == "boreas":
        for p in sorted(glob.glob(os.path.join(root, "boreas-*", "applanix",
                                               "lidar_poses.csv"))):
            name = os.path.basename(os.path.dirname(os.path.dirname(p)))
            yield name.replace("boreas-", ""), (p,)
        return
    if release == "helipr":
        # 전역 좌표판은 같은 궤적을 상수 회전만큼 돌린 것이라 두 번 세지 않는다.
        for p in sorted(glob.glob(os.path.join(root, "*", "LiDAR_GT",
                                               "*_gt.txt"))):
            sensor = os.path.basename(p)[:-7]
            if sensor.startswith("global_"):
                continue
            seq = os.path.basename(os.path.dirname(os.path.dirname(p)))
            yield "%s/%s" % (seq, sensor), (p,)
        return
    if release == "nuscenes":
        for p in sorted(glob.glob(os.path.join(root, "scene-*_pose.json"))):
            yield os.path.basename(p).split("_")[0], (p,)
        return
    if release == "pit30m":
        for p in sorted(glob.glob(os.path.join(root, "*", "all_poses.npz"))):
            if os.path.getsize(p) == 0:
                continue
            yield os.path.basename(os.path.dirname(p))[:8], (p,)
        return
    if release == "nclt":
        for p in sorted(glob.glob(os.path.join(root, "groundtruth_*.csv"))):
            yield os.path.basename(p)[12:-4], (p,)
        return
    if release == "tumvi":
        for p in sorted(glob.glob(os.path.join(root, "room*.csv"))):
            yield os.path.basename(p)[:-4], (p,)
        return
    if release == "euroc":
        for p in sorted(glob.glob(os.path.join(root or EUROC_ROOT,
                                               "*.csv"))):
            yield os.path.basename(p)[:-4], (p,)
        return
    if release == "oxford":
        for p in sorted(glob.glob(os.path.join(root or OXFORD_ROOT,
                                               "*", "rtk.csv"))):
            yield os.path.basename(os.path.dirname(p)), (p,)
        return
    for p in sorted(glob.glob(os.path.join(root, "*_sync", "poses.txt"))):
        name = os.path.basename(os.path.dirname(p))
        yield name.split("_drive_")[1][:4], (p,)


LOADER = {
    "kitti": load_poses,
    "boreas": load_poses_boreas,
    "helipr": load_poses_helipr,
    "nuscenes": load_poses_nuscenes,
    "pit30m": load_poses_pit30m,
    "nclt": load_poses_nclt,
    "tumvi": load_poses_mav_csv,
    "euroc": load_poses_mav_csv,
    "oxford": load_poses_oxford,
}


def _gather_raw(release, root=""):
    if release == "tumrgbd":
        # 이 릴리스만 파일 하나가 토막 여럿이 된다 — 구멍에서 끊기 때문이다.
        for _name, paths in source_files(release, root):
            for k, piece in enumerate(pieces(*read_tum(paths[0]))):
                yield ("%s#%d" % (_name, k) if k else _name), piece
        return
    load = LOADER.get(release, load_poses_kitti360)
    for name, paths in source_files(release, root):
        got = load(paths[0])
        if release == "oxford" and len(got[0]) < 2:
            continue
        yield name, got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", choices=RELEASES + PENDING, default="kitti")
    ap.add_argument("--root", default="",
                    help="비우면 RELEASE_ROOTS 의 값")
    ap.add_argument("--rpe1m", action="store_true",
                    help="키티-360 공식 순위표의 구간 규칙 — 1 m 간격, 모든 "
                         "프레임이 시작점")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    lengths, step = ((1.0,), 1) if args.rpe1m else (LENGTHS, STEP)
    tag = "rpe1m" if args.rpe1m else "segment"
    out = args.out or ("results/next/%s_%s_D.csv" % (args.release, tag))

    def as_row(seq, d, nframes, nseg):
        # 깎지 않고 적는다. 이 표의 2D 는 kitti_rank_check 이 순위 간격과
        # 견주는 문턱으로 되읽는 수다 — 다섯 자리로 깎으면 문턱 바로 위아래
        # 쌍의 판정이 깎인 쪽을 따른다. 표시용 반올림은 본문 매크로를 만들 때
        # 한 번만 한다.
        row = {"sequence": seq, "n_frames": nframes, "n_segments": nseg}
        for k, v in d.items():
            if k.startswith("t_"):
                # 공식 표는 병진오차를 퍼센트로 적는다(미터/미터 × 100).
                row["D_%s_pct" % k] = exact(100.0 * v)
                row["twoD_%s_pct" % k] = exact(200.0 * v)
                continue
            row["D_%s_deg_per_m" % k] = exact(np.degrees(v))
            row["twoD_%s_deg_per_m" % k] = exact(2.0 * np.degrees(v))
            # 논문 표는 보통 100 m 당 도로 적는다. 읽는 쪽이 환산하다 틀리지
            # 않도록 그 단위도 같이 낸다.
            row["twoD_%s_deg_per_100m" % k] = exact(200.0 * np.degrees(v))
        return row

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
        print("[%s] 프레임 %5d 구간 %5d  2D(phase_half)=%.4g deg/100m, "
              "%.4g %%" % (seq, len(t), nseg,
                           200.0 * np.degrees(d["phase_half"]),
                           200.0 * d["t_phase_half"]))

    if not rows:
        print("잴 것이 없다")
        return
    # 공식 순위표의 한 칸과 맞댈 행. 시퀀스별 값을 다시 평균한 것이 아니라
    # 구간을 통째로 모아 평균한 것이다.
    pooled = as_row("ALL", pool(raws), frames, segs)
    rows.append(pooled)
    print("[ALL ] 프레임 %5d 구간 %5d  2D(phase_half)=%.4g deg/100m, %.4g %%"
          % (frames, segs,
             200.0 * float(pooled["D_phase_half_deg_per_m"]),
             2.0 * float(pooled["D_t_phase_half_pct"])))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 중앙과 범위는 시퀀스별 값에서만 낸다. 묶은 행을 섞으면 그 하나가
    # 시퀀스 하나처럼 세어진다.
    per_seq = [r for r in rows if r["sequence"] != "ALL"]
    for k in ("phase_half", "phase_half_rms", "decim2", "decim2_rms",
              "interp", "interp_rms"):
        v = sorted(float(r["D_%s_deg_per_m" % k]) for r in per_seq
                   if "D_%s_deg_per_m" % k in r)
        if v:
            m = v[len(v) // 2]
            print("%-15s D 중앙 %.4g deg/m (%.4g~%.4g), 2D 중앙 %.4g deg/m "
                  "= %.4g deg/100m, 묶음 2D %.4g deg/100m"
                  % (k, m, v[0], v[-1], 2 * m, 200 * m,
                     200.0 * float(pooled["D_%s_deg_per_m" % k])))
        u = sorted(float(r["D_t_%s_pct" % k]) for r in per_seq
                   if "D_t_%s_pct" % k in r)
        if u:
            m = u[len(u) // 2]
            print("%-15s 병진 D 중앙 %.4g %% (%.4g~%.4g), 2D 중앙 %.4g %%, "
                  "묶음 2D %.4g %%"
                  % (k, m, u[0], u[-1], 2 * m,
                     2.0 * float(pooled["D_t_%s_pct" % k])))
    print("\n%d 행 -> %s" % (len(rows), out))


if __name__ == "__main__":
    main()
