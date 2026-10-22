#!/usr/bin/env python3
"""쓰기로 한 공개 자료마다 무엇을 어디서 받아 어떻게 읽었는지 한 줄로 등록한다.

이 표가 없으면 "열 릴리스를 읽었다" 는 문장이 그 열이 무엇인지 아무 데도
안 적힌 채로 선다. 다시 돌리려는 사람이 알아야 하는 것은 데이터셋 이름이
아니라 그 안의 어느 산출물의 어느 파일을, 어떤 시각 단위로, 어떤 좌표계
규약으로 읽었나다. 세 가지가 다 릴리스마다 다르고 셋 중 하나만 어긋나도
수가 조용히 틀린다 — 보레아스 전치 결함(46ec12c)이 셋째 자리에서 났다.

나뉘는 칸 둘을 안 섞는다. 적힌 값은 그 자료의 문서에서 옮긴 것이고 잰 값은
우리가 파일에서 센 것이다. 적힌 쪽은 reconstruction_doc_audit.py 가 인용과
출처까지 들고 있고 여기서는 되풀이하지 않는다. 여기 수는 전부 잰 값이다.

이용 조건은 그 자료 쪽 문서에서 직접 읽은 문장만 적는다. 읽지 못한 것은
"미확인" 으로 둔다 — 보레아스·누씬즈·유로크·핏서티엠 넷은 안내가 자바스크립트로
그려져서 받아 읽히지 않는다. 짐작해 적으면 그게 근거로 읽힌다.

실행:
    PYTHONPATH=src python3 scripts/dataset_register.py
"""

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kitti_segment_D import RELEASE_ROOTS, exact, gather  # noqa: E402

OUT = "results/next/dataset_register.csv"

# 잰 값이 아닌 칸들. 파일에서 읽히지 않고 문서나 우리 결정에서 온다.
KNOWN = {
    "kitti": {
        "label": "KITTI 주행거리계",
        "product": "odometry 기준 자세",
        "file": "dataset/poses/<시퀀스>.txt, "
                "dataset/sequences/<시퀀스>/times.txt",
        "time_unit": "초(times.txt)",
        "time_source": "공식",
        "measured_axis": "공식 시각(times.txt). 없는 시퀀스는 행 번호 × 0.1 초",
        "terms": "published under the Creative Commons "
                 "Attribution-NonCommercial-ShareAlike 3.0 License",
        "terms_url": "https://www.cvlibs.net/datasets/kitti/",
        "decision": "채택",
        "decision_reason": "공식 순위표와 공식 평가 프로그램이 둘 다 있다",
    },
    "kitti360": {
        "label": "KITTI-360",
        "product": "data_poses 기준 자세 + oxts 시각",
        "file": "<드라이브>/poses.txt, <드라이브>/oxts/timestamps.txt",
        "time_unit": "프레임 번호 + 공식 시각(나노초 문자열, 아홉 드라이브)",
        "time_source": "공식(아홉), 없음(0008·0018)",
        "measured_axis": "프레임 격자(프레임 번호 ÷ 명목 주기 10 Hz). 공식 "
                         "시각은 따로 잰 한 사례로만 쓴다",
        "terms": "published under the Creative Commons "
                 "Attribution-NonCommercial-ShareAlike 3.0 License",
        "terms_url": "https://www.cvlibs.net/datasets/kitti-360/",
        "decision": "채택",
        "decision_reason": "공식 순위표가 있다(의미론 동시추정 위치추정)",
    },
    "boreas": {
        "label": "Boreas",
        "product": "applanix 라이다 시각 기준 자세",
        "file": "<시퀀스>/applanix/lidar_poses.csv",
        "time_unit": "마이크로초",
        "time_source": "공식",
        "measured_axis": "공개된 시각 그대로",
        "terms": "미확인(안내가 자바스크립트로 그려져 받아 읽히지 않는다)",
        "terms_url": "https://www.boreas.utias.utoronto.ca/",
        "decision": "채택(순위표는 제외)",
        "decision_reason": "계절이 다른 같은 경로를 되풀이한다. 순위표는 "
                           "운영자 지시로 비교에 안 쓴다",
    },
    "helipr": {
        "label": "HeLiPR",
        "product": "LiDAR_GT 기준 자세",
        "file": "<시퀀스>/LiDAR_GT/<센서>_gt.txt",
        "time_unit": "나노초",
        "time_source": "공식",
        "measured_axis": "공개된 시각 그대로",
        "terms": "distributed under the Creative Commons "
                 "Attribution-NonCommercial-ShareAlike 4.0 License",
        "terms_url": "https://github.com/minwoo0611/"
                     "HeLiPR-Pointcloud-Toolbox",
        "decision": "채택",
        "decision_reason": "복원 방식을 문서가 적어 둔 둘 중 하나다"
                           "(누적 비스플라인)",
    },
    "nuscenes": {
        "label": "nuScenes",
        "product": "차량통신망 확장 자세",
        "file": "can_bus/scene-<번호>_pose.json",
        "time_unit": "마이크로초",
        "time_source": "공식",
        "measured_axis": "공개된 시각 그대로",
        "terms": "미확인(안내가 자바스크립트로 그려져 받아 읽히지 않는다)",
        "terms_url": "https://www.nuscenes.org/terms-of-use",
        "decision": "채택",
        "decision_reason": "장면이 많아 짧은 궤적 쪽 분포를 채운다",
    },
    "pit30m": {
        "label": "Pit30M",
        "product": "all_poses 묶음",
        "file": "<로그>/all_poses.npz",
        "time_unit": "초(유닉스 기원)",
        "time_source": "공식",
        "measured_axis": "공개된 시각 그대로",
        "terms": "미확인(저장소의 MIT 는 도구 코드 쪽이라 자료에 그대로 "
                 "옮겨 적지 않는다)",
        "terms_url": "https://github.com/pit30m/pit30m",
        "decision": "채택",
        "decision_reason": "상한 표에서 가장 작은 칸이 여기서 나온다",
    },
    "tumrgbd": {
        "label": "TUM RGB-D",
        "product": "groundtruth.txt 동작 포착 궤적",
        "file": "<시퀀스>/groundtruth.txt",
        "time_unit": "초",
        "time_source": "공식",
        "measured_axis": "공개된 시각 그대로",
        "terms": "licensed under a Creative Commons 4.0 Attribution "
                 "License (CC BY 4.0)",
        "terms_url": "https://cvg.cit.tum.de/data/datasets/rgbd-dataset",
        "decision": "채택",
        "decision_reason": "손에 든 실내 궤적이라 차량과 다른 쪽 끝이다",
    },
    "nclt": {
        "label": "NCLT",
        "product": "ground_truth 동시추정 위치추정 궤적",
        "file": "groundtruth_<날짜>.csv",
        "time_unit": "마이크로초",
        "time_source": "공식",
        "measured_axis": "공개된 시각 그대로",
        "terms": "made available under the Open Database License",
        "terms_url": "https://robots.engin.umich.edu/nclt/",
        "decision": "채택",
        "decision_reason": "마디 사이를 주행거리계로 메운다고 적어 둔 "
                           "유일한 릴리스다",
    },
    "tumvi": {
        "label": "TUM VI",
        "product": "mocap0 동작 포착 궤적(방 안 시퀀스만)",
        "file": "dataset-room<N>_512_16/mav0/mocap0/data.csv",
        "time_unit": "나노초",
        "time_source": "공식",
        "measured_axis": "공개된 시각 그대로",
        "terms": "licensed under a Creative Commons 4.0 Attribution "
                 "License (CC BY 4.0)",
        "terms_url": "https://cvg.cit.tum.de/data/datasets/"
                     "visual-inertial-dataset",
        "decision": "채택(방 안 여섯만)",
        "decision_reason": "방 밖 시퀀스는 궤적 처음과 끝에만 기준이 있어 "
                           "가운데가 비고, 그 자리를 구간 규칙이 건너뛰며 "
                           "잇는다",
    },
    "euroc": {
        "label": "EuRoC MAV",
        "product": "state_groundtruth_estimate0",
        "file": "mav0/state_groundtruth_estimate0/data.csv",
        "time_unit": "나노초",
        "time_source": "공식",
        "measured_axis": "공개된 시각 그대로",
        # 안내 쪽은 자바스크립트로 그려져 안 읽힌다. 같은 식별자의 등록
        # 메타자료(api.datacite.org)에 권리 항목이 그대로 들어 있다.
        "terms": "In Copyright - Non-Commercial Use Permitted "
                 "(rightsstatements.org InC-NC/1.0)",
        "terms_url": "https://doi.org/10.3929/ethz-b-000690084",
        "decision": "채택(열한 시퀀스)",
        "decision_reason": "열한 시퀀스가 빠짐없이 읽히고 간격이 고르다. "
                           "100~800 m 구간은 MH_03_medium 하나에서만 서서 "
                           "그 줄만 값이 있고 나머지는 줄표다",
    },
}

# 표에 아직 없는 자리. 비워 두지 않고 왜 없는지와 다시 볼 조건을 적는다.
PENDING = (
    {
        "release": "oxford", "label": "Oxford RobotCar",
        "product": "RTK 기준 자세", "file": "<주행>/rtk.csv",
        "time_unit": "마이크로초", "time_source": "공식",
        "terms": "미확인(공식 개발도구만 Attribution-NonCommercial-"
                 "ShareAlike 4.0 International 로 적혀 있다. 자료 쪽 조건은 "
                 "내려받기 양식 뒤에 있어 안 읽었다)",
        "terms_url": "https://robotcar-dataset.robots.ox.ac.uk/",
        "decision": "안 들임",
        "decision_reason": "공개 위치가 정상 주기에서 튄다 — 속도 칸과 "
                           "어긋나는 걸음 13,409 개, 한 걸음 최대 2,889.9 m. "
                           "녹화가 끊긴 자리는 열 걸음뿐이고 잔차 분포에도 "
                           "틈이 없어 릴리스의 칸으로는 못 끊는다",
    },
)

# time_unit·time_source 는 그 릴리스가 무엇을 공개하는가이고 measured_axis 는
# 우리가 어느 축 위에서 쟀는가다. 둘이 같은 릴리스가 대부분이지만 키티-360 은
# 공식 시각이 있는데도 프레임 격자에서 재므로, 한 칸으로 적으면 되돌리려는
# 사람이 공식 시각으로 읽고 다른 수를 얻는다.
FIELDS = ("release", "label", "product", "file", "time_unit", "time_source",
          "measured_axis",
          "frame_convention", "sequences", "frames", "measured_hz",
          "measured_hz_min", "measured_hz_max", "span_hours", "local_path",
          "terms", "terms_url", "decision", "decision_reason")


def measured(release):
    """그 릴리스를 실제로 읽어 시퀀스 수·표본 수·주기·유효 구간을 센다."""
    n_seq = n_frame = 0
    span = 0.0
    dts = []
    # 세는 것은 시각뿐이라 회전을 정사영하지 않는다. 엔시엘티 1,900 만
    # 프레임에 특잇값 분해를 돌리면 몇 분이 그 자리에서 날아간다.
    for _name, (t, _rot, _pos) in gather(release, project=False):
        if len(t) < 2:
            continue
        n_seq += 1
        n_frame += len(t)
        span += float(t[-1] - t[0])
        dts.append(float(np.median(np.diff(t))))
    if not dts:
        return None
    # 중앙값 하나로 적으면 섞인 릴리스가 한 수로 보인다 — 튬 알지비디는
    # 시퀀스마다 100 Hz 와 300 Hz 가 섞여 있어서 중앙값만 적으면 100 Hz
    # 시퀀스가 표에서 사라진다. 양 끝을 같이 적는다.
    hz = [1.0 / d for d in dts]
    return (n_seq, n_frame, float(np.median(hz)), min(hz), max(hz),
            span / 3600.0)


def read_verdicts(path="results/next/pose_frame_probe.csv"):
    """규약 탐침이 릴리스마다 내린 판정. 없으면 빈 사전."""
    if not os.path.exists(path):
        return {}
    tally = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            d = tally.setdefault(r["release"], {})
            d[r["verdict"]] = d.get(r["verdict"], 0) + 1
    out = {}
    for rel, d in tally.items():
        ok, bad = d.get("as_read", 0), d.get("transposed", 0)
        # 전치 줄이 하나라도 있으면 통과시키지 않는다. 다수결로 읽으면
        # 섞인 릴리스가 조용히 넘어가는데, 규약은 릴리스의 성질이라
        # 섞여 나온 것 자체가 둘 중 하나가 틀렸다는 뜻이다.
        if bad:
            out[rel] = "전치 의심(줄 %d)" % bad
        elif ok == 0:
            out[rel] = "못 가림"
        else:
            out[rel] = "읽은 대로(가린 줄 %d)" % ok
    return out


def main():
    verdict = read_verdicts()
    rows = []
    for rel in RELEASE_ROOTS:
        base = dict(KNOWN[rel])
        base["release"] = rel
        base["local_path"] = RELEASE_ROOTS[rel]
        base["frame_convention"] = verdict.get(rel, "미측정")
        got = measured(rel)
        if got is None:
            base.update(sequences="", frames="", measured_hz="",
                        measured_hz_min="", measured_hz_max="",
                        span_hours="")
        else:
            n_seq, n_frame, hz, lo, hi, hours = got
            base.update(sequences=str(n_seq), frames=str(n_frame),
                        measured_hz=exact(hz), measured_hz_min=exact(lo),
                        measured_hz_max=exact(hi), span_hours=exact(hours))
        rows.append(base)
        print("%-9s %-14s 시퀀스 %4s  표본 %9s  %8s Hz (%8s~%8s)  "
              "%8s 시간  %s"
              % (rel, base["decision"], base["sequences"], base["frames"],
                 base["measured_hz"][:8], base["measured_hz_min"][:8],
                 base["measured_hz_max"][:8], base["span_hours"][:8],
                 base["frame_convention"]))
    for p in PENDING:
        row = dict(p)
        row.setdefault("measured_axis", "미측정")
        row.update(frame_convention="미측정", sequences="", frames="",
                   measured_hz="", measured_hz_min="", measured_hz_max="",
                   span_hours="",
                   local_path=RELEASE_ROOTS.get(p["release"], "미수신"))
        rows.append(row)
        print("%-9s %-14s %s" % (p["release"], p["decision"],
                                 p["decision_reason"][:40]))

    for r in rows:
        if set(r) != set(FIELDS):
            raise SystemExit("칸이 안 맞는다: %s" % r["release"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(FIELDS))
        w.writeheader()
        w.writerows(rows)
    print("\n%d 행 -> %s" % (len(rows), OUT))


if __name__ == "__main__":
    main()
