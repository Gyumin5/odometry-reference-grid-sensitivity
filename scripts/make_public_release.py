#!/usr/bin/env python3
"""논문과 같이 공개할 나무를 만든다 — 대본·검사·결과 표·읽는 법.

원자료는 안 담는다. 열 릴리스는 각자의 공식 주소에서 받아야 하고 그 조건도
거기 있다. 담는 것은 그 자료를 읽어 상한을 내는 대본과, 우리가 낸 결과 표와,
같은 수를 다시 만드는 명령이다.

무엇을 담을지는 손으로 적지 않고 두 규칙으로 고른다. 손 목록은 대본이 늘 때마다
어긋나고, 어긋난 쪽을 나중에 사실로 읽는다.

  대본 = 검사가 부르는 것 + 재생성 사슬이 부르는 것, 그리고 그것들이 다시
         부르는 것까지 닫는다. 옛 원고 때 쓰고 지금은 안 도는 대본은 검사가
         안 부르므로 저절로 빠진다.
  결과 = 담긴 대본이 이름을 적어 둔 표, 그리고 릴리스마다 이름이 만들어지는
         상한·훑기 표.

검사 중 velref 묶음을 부르는 둘은 안 담는다. 그 묶음은 앞 원고의 추정기
라이브러리라 여기 담기는 대본 중 아무도 안 부르고, 같이 담으면 받는 쪽이
이 논문과 무관한 코드를 받는다.

실행:
    python3 scripts/make_public_release.py --out /tmp/pose-reading-release
"""

import argparse
import os
import re
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TESTS = os.path.join(ROOT, "tests")
RESULTS = os.path.join(ROOT, "results", "next")

# 재생성 사슬(state/regen10-20261020/run.sh)이 부르는 것들, 그리고 사슬에 마디가
# 없어 손으로 도는 것들(복원 방식 대조). 검사가 안 부르는 자리가 있어도 결과 표를
# 만드는 쪽이므로 같이 담는다. 뒤 다섯은 심사 2 회차 지적을 닫은 측정이라, 받는
# 쪽이 그 수를 다시 내려면 있어야 한다.
CHAIN = ("kitti_segment_D", "release_bound_table", "protocol_sweep",
         "protocol_sweep_table", "orthonormality_floor", "pose_frame_probe",
         "tie_rule_probe", "window_exclusion", "offset_tolerance",
         "make_offset_figure", "dataset_register", "segment_set_stability",
         "kitti_official_binary_check", "nuscenes_pose_sources",
         "oxford_rtk_quality", "make_paper_macros", "kitti_rank_check",
         "kitti360_time_axis", "kitti_time_axis", "make_public_release",
         "source_manifest", "time_gap_audit", "board_sequence_sensitivity",
         "official_boundary_case", "reconstruction_doc_audit",
         "gap_policy_audit", "rule_ratio_common")

IMPORT = re.compile(r"^\s*(?:from|import)\s+([A-Za-z_]\w*)", re.M)
PER_RELEASE = re.compile(r"^[a-z0-9]+_(?:segment|rpe1m)_(?:D|sweep)\.csv$")


def script_names(d):
    return {f[:-3] for f in os.listdir(d) if f.endswith(".py")}


def test_names(tests_dir=TESTS):
    """담을 검사 파일들. 나무 밖을 보는 것은 뺀다.

    둘이다. velref 묶음을 부르는 검사와, 원고 파일(paper_tim)을 열어 본문과
    매크로를 맞춰 보는 검사. 원고는 이 나무에 안 담으므로 그 검사는 받는
    쪽에서 파일 없음으로 깨진다.
    """
    out = []
    for f in sorted(os.listdir(tests_dir)):
        if not f.endswith(".py"):
            continue
        with open(os.path.join(tests_dir, f)) as fh:
            s = fh.read()
        if "velref" not in s and "paper_tim" not in s:
            out.append(f)
    return out


def manifest(scripts_dir=HERE, tests_dir=TESTS):
    """담을 대본 이름들. 검사와 사슬에서 시작해 의존을 닫는다."""
    names = script_names(scripts_dir)
    want = set(CHAIN) & names
    for f in test_names(tests_dir):
        with open(os.path.join(tests_dir, f)) as fh:
            want |= names & set(IMPORT.findall(fh.read()))
    seen = set()
    while want - seen:
        n = (want - seen).pop()
        seen.add(n)
        with open(os.path.join(scripts_dir, n + ".py")) as fh:
            want |= names & set(IMPORT.findall(fh.read()))
    return sorted(want)


def support_names(scripts, scripts_dir=HERE):
    """담을 대본 곁파일들 — 담긴 대본이 이름을 적어 둔 파이썬 아닌 파일.

    공식 프로그램 대조는 C++ 한 조각을 그 자리에서 컴파일해 돌린다. 이름만
    적혀 있고 import 로는 안 걸려서, 대본만 담으면 받는 쪽에서 그 대조가
    파일 없음으로 깨진다. 결과 표를 고르는 규칙과 같은 꼴로 고른다 — 손
    목록을 따로 두면 곁파일이 늘 때마다 어긋난다.
    """
    text = ""
    for n in scripts:
        with open(os.path.join(scripts_dir, n + ".py")) as fh:
            text += fh.read()
    return [f for f in sorted(os.listdir(scripts_dir))
            if not f.endswith(".py") and f in text]


def result_names(scripts, scripts_dir=HERE, results_dir=RESULTS):
    """담을 표 이름들. 대본이 이름을 적어 둔 것과 릴리스별로 만들어지는 것."""
    text = ""
    for n in scripts:
        with open(os.path.join(scripts_dir, n + ".py")) as fh:
            text += fh.read()
    out = []
    for f in sorted(os.listdir(results_dir)):
        if not f.endswith(".csv"):
            continue
        if f in text or PER_RELEASE.match(f):
            out.append(f)
    return out


README = """# Reading a public reference trajectory changes the score

Code and result tables for the paper. The datasets themselves are not
here: each release has its own download page and its own terms, and the
scripts read them where you put them.

## What this computes

For one public reference trajectory, resampling it under a different
reading convention moves every segment by some distance. The weighted
sum of those distances, D, bounds how far a weighted-mean pose-distance
score can move for a fixed estimate. The scripts measure D on
{nrel} releases under the two segment rules that official leaderboards
use, and compare it against the gaps between leaderboard entries.

## Layout

    scripts/       measurement and table code ({nscript} files)
    tests/         the test suite ({ntest} files); run it first
    results/next/  the tables this code produced ({ncsv} files)

## Running the tests

    python3 -m pytest tests/ -q

The tests need no dataset. They build small synthetic trajectories and
check the pieces the measurements rely on.

## Reproducing a table

Point the release roots in `scripts/kitti_segment_D.py` at your copies
of the data, then

    python3 scripts/kitti_segment_D.py --release kitti
    python3 scripts/kitti_segment_D.py --release kitti --rpe1m
    python3 scripts/release_bound_table.py

`results/next/dataset_register.csv` lists, for every release, the file that
was read, the time it publishes, the axis we measured on, the rate we
measured, and the terms page. The published time and the measured axis
are separate columns: on KITTI-360 they differ.

## Requirements

Python 3, numpy, scipy, matplotlib.
"""


def write_readme(out, scripts, csvs, ntest, nrel, nsupport=0):
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write(README.format(nscript=len(scripts) + nsupport,
                              ncsv=len(csvs), ntest=ntest, nrel=nrel))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    scripts = manifest()
    csvs = result_names(scripts)
    # 표는 results/next 그대로 둔다. 대본과 검사가 그 경로를 글자로 들고
    # 있어서 옮기면 받는 쪽에서 파일 없음으로 깨진다.
    for sub in ("scripts", "tests", os.path.join("results", "next")):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    for n in scripts:
        shutil.copy2(os.path.join(HERE, n + ".py"),
                     os.path.join(args.out, "scripts", n + ".py"))
    support = support_names(scripts)
    for f in support:
        shutil.copy2(os.path.join(HERE, f),
                     os.path.join(args.out, "scripts", f))
    tests = test_names()
    for f in tests:
        shutil.copy2(os.path.join(TESTS, f), os.path.join(args.out, "tests", f))
    for f in csvs:
        shutil.copy2(os.path.join(RESULTS, f),
                     os.path.join(args.out, "results", "next", f))
    shutil.copy2(os.path.join(ROOT, "LICENSE"),
                 os.path.join(args.out, "LICENSE"))

    import csv as _csv
    with open(os.path.join(RESULTS, "release_bound_table.csv")) as f:
        nrel = len({r["release"] for r in _csv.DictReader(f)})
    write_readme(args.out, scripts, csvs, len(tests), nrel, len(support))

    print("대본 %d(곁파일 %d), 검사 %d, 표 %d, 릴리스 %d -> %s"
          % (len(scripts), len(support), len(tests), len(csvs), nrel,
             args.out))
    print("안 담은 대본 %d 개는 검사도 사슬도 안 부른다"
          % (len(script_names(HERE)) - len(scripts)))


if __name__ == "__main__":
    main()
