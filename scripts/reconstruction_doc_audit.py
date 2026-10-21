#!/usr/bin/env python3
"""릴리스 아홉이 자기 문서에 어떤 시간 복원 방식을 적어 뒀는지 모은다.

규약 훑기에서 방식 몫이 주기 몫의 열 배였다 — 키티 100~800 m 에서 주기를 반으로
줄인 값이 0.070 퍼센트인데 방식만 구면보간에서 최근접으로 바꾼 값이 0.682 다.
그러면 "5 Hz 로 재표집했다" 만 적은 평가는 큰 쪽을 안 적은 것이 된다. 그 주장을
세우려면 실제로 릴리스들이 무엇을 적어 두는지를 세야 한다.

여기 있는 것은 측정이 아니라 인용이다. 각 줄은 그 릴리스의 공개 문서·공식 코드에서
직접 읽은 문장 하나와 그 출처를 들고 있다. 추측은 넣지 않는다 — 문서가 말하지
않으면 "미기재" 로 적고, 우리가 잰 값은 measured_hz 칸에 따로 둔다. 둘을 같은
칸에 섞으면 우리가 잰 것이 그 릴리스가 적어 둔 것처럼 읽힌다.

방식 이름은 닫힌 목록으로 둔다. 문장마다 다르게 부르면 세는 순간 갈린다.
  nearest   가장 가까운 표본을 그대로 집는다
  bspline   누적 비스플라인 보간
  gpwnoa    가속도 백색잡음 사전분포를 둔 연속시간 궤적 보간
  odom      마디 사이를 주행거리계로 메운다
  index     시각을 안 쓰고 번호로 맞춘다
  none      문서가 방식을 적지 않는다

쓰임은 둘이다. results/next/reconstruction_doc_audit.csv 로 떨어뜨리고,
make_paper_macros.py 가 그 파일에서 본문 매크로를 만든다.
"""

import argparse
import csv

METHODS = ("nearest", "bspline", "gpwnoa", "odom", "index", "none")

# (release, 아래 칸들). quote 는 출처의 원문 그대로다 — 옮기지 않는다.
AUDIT = (
    {
        "release": "kitti",
        "label": "KITTI",
        "source_hz": "100",
        "published_hz": "10",
        "measured_hz": "9.64",
        "gt_source": "OXTS RT3003 GPS/IMU, RTK float/integer",
        "method": "nearest",
        "stated_error_ms": "5",
        "where": "Geiger et al. 2013 IJRR, Sec. Synchronization",
        "quote": "as it provides updates at 100 Hz, we collect the "
                 "information with the closest timestamp to the laser "
                 "scanner timestamp for a particular frame, resulting in a "
                 "worst-case time difference of 5 ms",
        "url": "https://www.cvlibs.net/publications/Geiger2013IJRR.pdf",
        "eval_method": "index",
        "eval_tolerance_ms": "",
        "eval_quote": "poses_gt.size()==0 || "
                      "poses_result.size()!=poses_gt.size()",
        "eval_url": "https://s3.eu-central-1.amazonaws.com/avg-kitti/"
                    "devkit_odometry.zip",
    },
    {
        "release": "kitti360",
        "label": "KITTI-360",
        "source_hz": "미기재",
        "published_hz": "미기재",
        "measured_hz": "10.00",
        "gt_source": "structure-from-motion, GPS·LiDAR 로 정칙화",
        "method": "none",
        "stated_error_ms": "",
        "where": "KITTI-360 documentation, Ego Motion",
        "quote": "We obtain accurate georegistered vehicle poses using "
                 "structure-from-motion based on all feature matches while "
                 "regularizing against the GPS location and the LiDAR scans.",
        "url": "https://www.cvlibs.net/datasets/kitti-360/documentation.php",
        "eval_method": "none",
        "eval_tolerance_ms": "",
        "eval_quote": "the frame indices may not be continuous, as this "
                      "list only provides pose at a given frame",
        "eval_url": "https://www.cvlibs.net/datasets/kitti-360/"
                    "documentation.php",
    },
    {
        "release": "boreas",
        "label": "Boreas",
        "source_hz": "미기재",
        "published_hz": "미기재",
        "measured_hz": "9.64",
        "gt_source": "post-processed Applanix POS LV GNSS",
        "method": "none",
        "stated_error_ms": "",
        "where": "pyboreas README",
        "quote": "camera, lidar, and radar measurements are not synchronous "
                 "... each sensor message has an accurate timestamp and pose "
                 "instead",
        "url": "https://github.com/utiasASRL/pyboreas",
        "eval_method": "gpwnoa",
        "eval_tolerance_ms": "",
        "eval_quote": "Runs a steam optimization with locked poses and "
                      "outputs poses queried at query_times ... # WNOA Qc "
                      "diagonal",
        "eval_url": "https://github.com/utiasASRL/pyboreas/blob/master/"
                    "pyboreas/utils/odometry.py",
    },
    {
        "release": "helipr",
        "label": "HeLiPR",
        "source_hz": "50",
        "published_hz": "10",
        "measured_hz": "10.00",
        "gt_source": "RTK GPS 와 관성항법",
        "method": "bspline",
        "stated_error_ms": "",
        "where": "Jung et al. 2024, Sec. Ground truth",
        "quote": "we calculate the location of LiDARs at specific timestamp "
                 "using B-spline interpolation",
        "url": "https://arxiv.org/abs/2309.14590",
        "eval_method": "bspline",
        "eval_tolerance_ms": "",
        "eval_quote": "we calculate the location of LiDARs at specific "
                      "timestamp using B-spline interpolation",
        "eval_url": "https://arxiv.org/abs/2309.14590",
    },
    {
        "release": "nuscenes",
        "label": "nuScenes",
        "source_hz": "미기재",
        "published_hz": "50",
        "measured_hz": "49.98",
        "gt_source": "라이다 지도 정합 위치추정, x-y 평면 2차원",
        "method": "none",
        "stated_error_ms": "",
        # 출처는 상한 표가 실제로 읽는 쪽이어야 한다. 표는 차량통신망 확장의
        # scene-*_pose.json 을 읽으므로 그 묶음의 README 가 정본이고, 거기
        # 주기가 적혀 있다. 본 묶음 ego_pose 의 설명(schema_nuscenes.md)을
        # 출처로 두면 우리가 안 읽는 흐름의 문서로 읽는 흐름을 평가하게 된다.
        "where": "nuscenes-devkit can_bus README, pose",
        "quote": "The current pose of the ego vehicle, sampled at 50Hz.",
        "url": "https://github.com/nutonomy/nuscenes-devkit/blob/master/"
               "python-sdk/nuscenes/can_bus/README.md",
        "eval_method": "none",
        "eval_tolerance_ms": "",
        "eval_quote": "",
        "eval_url": "https://github.com/nutonomy/nuscenes-devkit/blob/"
                    "master/docs/schema_nuscenes.md",
    },
    {
        "release": "pit30m",
        "label": "Pit30M",
        "source_hz": "100",
        "published_hz": "100",
        "measured_hz": "100.00",
        "gt_source": "사후처리 위치추정 (문서에 방식 미기재)",
        "method": "none",
        "stated_error_ms": "",
        "where": "pit30m log_reader.py",
        "quote": "Returns the internal raw pose array, which needs manual "
                 "association with other data types. 100Hz.",
        "url": "https://github.com/pit30m/pit30m/blob/main/pit30m/data/"
               "log_reader.py",
        "eval_method": "nearest",
        "eval_tolerance_ms": "10",
        "eval_quote": "Interpolate poses linearly. Poses are 100Hz so "
                      "nearest-neighbor lookups can be at most 10-ish ms off "
                      "which can cause 0.33m of error for 33 mps driving",
        "eval_url": "https://github.com/pit30m/pit30m/blob/main/pit30m/"
                    "indexing.py",
    },
    {
        "release": "tumrgbd",
        "label": "TUM RGB-D",
        "source_hz": "100",
        "published_hz": "100",
        # 한 수로 못 적는다. 공개 페이지는 100 Hz 라고 적는데 실제 표본
        # 간격의 중앙값이 freiburg1·freiburg3 는 10 ms 고 freiburg2 는
        # 3.3 ms 다(223 시퀀스 중 173 개). 한 수로 적으면 표본 비율로
        # 쓴 이동을 밀리초로 바꿀 때 세 배 틀린다.
        "measured_hz": "100.00 / 300.00",
        "gt_source": "고속 추적 카메라 여덟 대의 동작 포착",
        "method": "none",
        "stated_error_ms": "",
        "where": "TUM RGB-D dataset page",
        "quote": "with eight high-speed tracking cameras (100 Hz)",
        "url": "https://cvg.cit.tum.de/data/datasets/rgbd-dataset",
        "eval_method": "nearest",
        "eval_tolerance_ms": "20",
        "eval_quote": "maximally allowed time difference for matching "
                      "entries (default: 0.02)",
        "eval_url": "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/tools",
    },
    {
        "release": "nclt",
        "label": "NCLT",
        "source_hz": "미기재",
        # 공개 페이지가 적은 수는 "대략 100 Hz" 인데 실제 표본 간격의 중앙값이
        # 6.75 ms 라 148 Hz 다. 적어 둔 수를 그대로 믿고 표본 비율을
        # 밀리초로 옮기면 1.5 배 틀린다. 그래서 두 칸을 따로 둔다.
        "published_hz": "100 (대략)",
        "measured_hz": "148.17",
        "gt_source": "동시추정 위치추정 그래프, 마디 사이는 주행거리계",
        "method": "odom",
        "stated_error_ms": "",
        "where": "NCLT dataset page, Changelog 2019-03-24",
        "quote": "generated using a SLAM graph, and then interpolation "
                 "using odometry",
        "url": "https://robots.engin.umich.edu/nclt/",
        "eval_method": "none",
        "eval_tolerance_ms": "",
        "eval_quote": "",
        "eval_url": "https://robots.engin.umich.edu/nclt/",
    },
    {
        "release": "tumvi",
        "label": "TUM VI",
        "source_hz": "120",
        "published_hz": "120",
        "measured_hz": "120.00",
        "gt_source": "동작 포착 장치, 방 안 시퀀스만 궤적 전체",
        # 동작 포착은 자기 시계로 찍은 것을 그대로 싣는다. 적어 둔 것은
        # 복원 방식이 아니라 시계를 맞춘 방식이라 method 는 none 이다.
        "method": "none",
        "stated_error_ms": "",
        "where": "TUM VI dataset page, Ground truth",
        "quote": "time alignment with ground truth data is performed by "
                 "aligning gyro measurements and rotational velocities",
        "url": "https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset",
        "eval_method": "none",
        "eval_tolerance_ms": "",
        "eval_quote": "",
        "eval_url": "https://cvg.cit.tum.de/data/datasets/"
                    "visual-inertial-dataset",
    },
    {
        "release": "euroc",
        "label": "EuRoC MAV",
        "source_hz": "미기재",
        "published_hz": "미기재",
        "measured_hz": "200.01",
        "gt_source": "비콘 동작 포착(6자유도 자세), 기계실은 라이카 MS50 "
                     "레이저 추적기(3차원 위치)",
        # 공개 페이지는 기준 궤적이 맞춰져 있다고만 적고 무엇으로 메웠는지는
        # 적지 않는다. 관성 장치가 200 Hz 라는 것은 적혀 있지만 그것은
        # 기준 궤적을 만든 장치가 아니다. 그래서 method 는 none 이다.
        "method": "none",
        "stated_error_ms": "",
        "where": "EuRoC MAV dataset page, Available Data",
        "quote": "Spatio-temporally aligned ground truth",
        "url": "https://projects.asl.ethz.ch/datasets/euroc-mav/",
        "eval_method": "none",
        "eval_tolerance_ms": "",
        "eval_quote": "",
        "eval_url": "https://projects.asl.ethz.ch/datasets/euroc-mav/",
    },
)

FIELDS = ("release", "label", "source_hz", "published_hz", "measured_hz",
          "gt_source", "method", "stated_error_ms", "where", "quote", "url",
          "eval_method", "eval_tolerance_ms", "eval_quote", "eval_url")


def check(rows=AUDIT):
    """적어 둔 것이 셀 수 있는 모양인지 본다. 어기면 그 자리에서 멈춘다."""
    seen = set()
    for r in rows:
        key = r["release"]
        if key in seen:
            raise ValueError("릴리스가 두 번 적혔다: %s" % key)
        seen.add(key)
        for name in ("method", "eval_method"):
            if r[name] not in METHODS:
                raise ValueError("%s: 목록에 없는 방식 %r" % (key, r[name]))
        if set(r) != set(FIELDS):
            raise ValueError("%s: 칸이 안 맞는다" % key)
        # 방식을 적었다고 세려면 인용과 출처가 둘 다 있어야 한다.
        if r["method"] != "none" and not (r["quote"] and r["url"]):
            raise ValueError("%s: 방식을 적었는데 인용이 없다" % key)
        if r["eval_method"] != "none" and not r["eval_quote"]:
            raise ValueError("%s: 평가 방식을 적었는데 인용이 없다" % key)
        # 허용 폭은 최근접으로 맞출 때만 뜻이 있다. 보간하는 쪽에 적으면
        # 그 수가 보간 오차처럼 읽힌다.
        if r["eval_tolerance_ms"] and r["eval_method"] != "nearest":
            raise ValueError("%s: 최근접이 아닌데 허용 폭이 적혔다" % key)
    return len(rows)


def tally(rows=AUDIT):
    """세는 값들. 본문이 부르는 수는 전부 여기서 나온다."""
    named = [r for r in rows if r["method"] != "none"]
    eval_named = [r for r in rows if r["eval_method"] != "none"]
    rate = [r for r in rows if r["published_hz"] != "미기재"]
    return {
        "n_releases": len(rows),
        "n_method_named": len(named),
        "n_method_silent": len(rows) - len(named),
        "n_eval_named": len(eval_named),
        "n_rate_stated": len(rate),
        "n_rate_silent": len(rows) - len(rate),
        "n_nearest": len([r for r in rows
                          if "nearest" in (r["method"], r["eval_method"])]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",
                    default="results/next/reconstruction_doc_audit.csv")
    args = ap.parse_args()

    check()
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(FIELDS))
        w.writeheader()
        w.writerows(AUDIT)

    t = tally()
    print("릴리스 %d 가운데" % t["n_releases"])
    print("  기준 자세를 만든 복원 방식을 적은 것 %d, 안 적은 것 %d"
          % (t["n_method_named"], t["n_method_silent"]))
    print("  공개 자세의 주기를 적은 것 %d, 안 적은 것 %d"
          % (t["n_rate_stated"], t["n_rate_silent"]))
    print("  어느 한쪽에서 최근접을 쓰는 것 %d" % t["n_nearest"])
    print()
    for r in AUDIT:
        print("%-10s 생성 %-8s 평가 %-8s  (잰 주기 %s Hz)"
              % (r["label"], r["method"], r["eval_method"], r["measured_hz"]))
    print("\n%d 줄 -> %s" % (len(AUDIT), args.out))


if __name__ == "__main__":
    main()
