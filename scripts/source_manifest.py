#!/usr/bin/env python3
"""측정이 실제로 읽은 기준 궤적 파일을 해시까지 적어 고정한다.

폴더를 훑어 파일을 찾는 절차만으로는 같은 원본을 읽었다는 것이 안 선다.
같은 이름으로 다른 판이 깔려 있어도 대본은 조용히 돌고 수만 바뀐다.
그래서 릴리스마다 읽은 파일의 이름·크기·sha256 을 적어 둔다. 맞춰 보는
것은 `--check` 다 — 표를 새로 쓰지 않고 지금 디스크와 견줘 없어진 파일,
크기가 다른 파일, 해시가 다른 파일을 찍고 어긋난 것이 있으면 1 로 끝난다.
표를 만들기만 하면 재실행 때 같은 원본을 읽었다는 것이 안 선다.

읽은 파일 목록은 측정기에서 그대로 가져온다 — kitti_segment_D.source_files
가 상한을 내는 쪽이 실제로 여는 경로를 내놓는 유일한 자리다. 폴더를 따로
훑어 목록을 만들면 그 목록이 조용히 갈린다. 실제로 헬리퍼의 전역 좌표판
파일은 측정이 건너뛰는데 유리글에는 걸려서, 읽지도 않은 파일이 "읽은
파일" 로 적혀 있었다.

측정기는 자세 파일만 내지 않는다. 읽개가 같이 여는 보조 입력까지 묶어
내므로 키티의 시각 파일도 "읽은 파일" 로 선다. 유리글 표는 그래서 보조로만
남는다 — 상한 측정이 안 여는 파일(키티-360 의 시각 파일 같은 것)을 같이
고정하기 위한 것이고, 줄마다 read_by_bound 칸으로 어느 쪽인지 적는다.
"""

import argparse
import csv
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kitti_segment_D import RELEASES, source_files  # noqa: E402

# 릴리스 -> (뿌리 아래 찾을 유리글들). 기준 궤적과 그 시간축만 담는다.
PATTERNS = {
    "kitti": ("poses/*.txt", "sequences/*/times.txt"),
    "kitti360": ("*/poses.txt", "*/oxts/timestamps.txt"),
    "boreas": ("*/applanix/lidar_poses.csv",),
    "helipr": ("*/LiDAR_GT/*_gt.txt",),
    "nuscenes": ("scene-*_pose.json",),
    "pit30m": ("*/all_poses.npz",),
    "tumrgbd": ("*-groundtruth.txt",),
    "nclt": ("groundtruth_*.csv",),
    "tumvi": ("room*.csv",),
    "euroc": ("*.csv",),
}


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def roots(register):
    out = {}
    with open(register) as f:
        for r in csv.DictReader(f):
            p = r["local_path"]
            if os.path.isdir(p):
                out[r["release"]] = p
    return out


def check(table, found):
    """적힌 해시와 지금 디스크를 맞춘다. 어긋난 줄 수를 종료 코드로 낸다.

    표를 만드는 것만으로는 재실행 때 같은 원본을 읽었다는 것이 안 선다 —
    맞춰 보는 쪽이 있어야 한다. 없어진 파일, 크기가 달라진 파일, 해시가
    달라진 파일, 그리고 같은 바이트를 두 번 읽는 자리를 한 줄씩 찍는다.
    """
    bad = 0
    seen = 0
    with open(table) as f:
        rows = list(csv.DictReader(f))

    # 적힌 "읽는 파일" 이 지금 읽개가 여는 것과 같은가. 해시만 맞추면 이
    # 자리가 빈다 — 측정이 새 파일을 열기 시작해도 적힌 줄은 전부 그대로라
    # 조용히 통과한다. 세 경우를 가른다. 읽는 파일의 내용이 바뀐 것은 아래
    # 해시 대조가 잡고, 읽개가 새로 여는데 표에 없는 것과 표에는 있는데 안
    # 여는 것은 여기서 잡는다. 폴더에 안 읽는 파일이 하나 늘어난 것은
    # 어긋남이 아니다 — 그건 측정이 읽는 입력을 안 바꾼다.
    for release in sorted({r["release"] for r in rows}):
        root = found.get(release)
        # 읽개가 모르는 이름이면 "안 연다" 가 아니라 "물어볼 데가 없다" 다.
        if root is None or release not in RELEASES:
            continue
        live = {os.path.relpath(os.path.realpath(p), root)
                for _n, ps in source_files(release, root) for p in ps}
        noted = {r["path"] for r in rows
                 if r["release"] == release and r["read_by_bound"] == "yes"}
        for p in sorted(live - noted):
            print("%-9s %s 측정이 여는데 표에 없다" % (release, p))
            bad += 1
        for p in sorted(noted - live):
            print("%-9s %s 표에는 읽는다고 적혔는데 측정이 안 연다"
                  % (release, p))
            bad += 1

    # 같은 바이트가 두 이름으로 깔려 있으면 그 궤적이 집계에 두 번 들어가고
    # 무게도 두 몫을 받는다. 튬 알지비디의 fr1_xyz 가 그랬다 — 수는 멀쩡해
    # 보이고 시퀀스 하나가 조용히 두 번 세어진다. 상한이 여는 파일만 본다.
    dup = {}
    for r in rows:
        if r["read_by_bound"] == "yes":
            dup.setdefault((r["release"], r["sha256"]), []).append(r["path"])
    for (release, _h), paths in sorted(dup.items()):
        if len(paths) > 1:
            print("%-9s 같은 바이트를 두 번 읽는다: %s"
                  % (release, ", ".join(sorted(paths))))
            bad += 1

    for r in rows:
        root = found.get(r["release"])
        if root is None:
            print("%-9s %s 뿌리가 없다 — 못 맞춘다"
                  % (r["release"], r["path"]))
            bad += 1
            continue
        p = os.path.join(root, r["path"])
        seen += 1
        if not os.path.exists(p):
            print("%-9s %s 없어졌다" % (r["release"], r["path"]))
            bad += 1
            continue
        if os.path.getsize(p) != int(r["bytes"]):
            print("%-9s %s 크기가 다르다 (%d -> %d)"
                  % (r["release"], r["path"], int(r["bytes"]),
                     os.path.getsize(p)))
            bad += 1
            continue
        if digest(p) != r["sha256"]:
            print("%-9s %s 해시가 다르다" % (r["release"], r["path"]))
            bad += 1
    print("맞춰 본 파일 %d, 어긋난 것 %d" % (seen, bad))
    return 1 if bad else 0


def main():
    import glob

    ap = argparse.ArgumentParser()
    ap.add_argument("--register",
                    default="results/next/dataset_register.csv")
    ap.add_argument("--out", default="results/next/source_manifest.csv")
    ap.add_argument("--check", action="store_true",
                    help="표를 새로 쓰지 않고 적힌 해시와 지금 파일을 맞춰 본다")
    args = ap.parse_args()

    if args.check:
        raise SystemExit(check(args.out, roots(args.register)))

    found = roots(args.register)
    rows = []
    for release in sorted(PATTERNS):
        root = found.get(release)
        if root is None:
            print("%-9s 뿌리가 없다 — 건너뛴다" % release)
            continue
        read = {os.path.realpath(p)
                for _n, ps in source_files(release, root) for p in ps}
        paths = []
        for pat in PATTERNS[release]:
            paths += glob.glob(os.path.join(root, pat))
        paths = sorted(set(paths) | read)
        if not paths:
            raise SystemExit("%s: 유리글이 하나도 안 맞았다 — 규칙을 고쳐라"
                             % release)
        if not read:
            raise SystemExit("%s: 측정이 여는 파일이 하나도 없다" % release)
        total = 0
        for p in paths:
            n = os.path.getsize(p)
            total += n
            rows.append({"release": release,
                         "path": os.path.relpath(p, root),
                         "bytes": n, "sha256": digest(p),
                         "read_by_bound":
                             "yes" if os.path.realpath(p) in read else "no"})
        print("%-9s 파일 %4d (상한이 여는 것 %d)  %8.1f MB"
              % (release, len(paths), len(read), total / 1e6))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["release", "path", "bytes",
                                          "sha256", "read_by_bound"])
        w.writeheader()
        w.writerows(rows)
    print("%d 행 -> %s" % (len(rows), args.out))


if __name__ == "__main__":
    main()
