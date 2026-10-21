#!/usr/bin/env python3
"""공백 처리가 채점 구간과 D 에 얼마나 들어가는지 원본 시간축부터 따라간다.

time_gap_audit.py 는 적재가 끝난 자료를 읽는다. 그래서 적재 단계에서 이미
끊기고 버려진 것은 못 본다 — 튬 알지비디를 읽는 쪽은 중앙 간격의 다섯 배를
넘는 자리에서 궤적을 끊고, 32 프레임보다 짧은 토막을 버린다. 공백을 센
결과만 내면 그 릴리스에서는 "공백이 없다" 가 "공백을 미리 지웠다" 라는 뜻이
된다.

여기서는 둘을 낸다.

  적재 단계 — 파일에서 읽은 표본 수와 적재가 남긴 표본 수, 끊긴 토막 수와
    버려진 토막 수, 버려진 시간. 끊지 않는 릴리스는 두 수가 같다.
  채점 단계 — 같은 공백 규칙(중앙 간격의 다섯 배)을 적재된 시간축에 씌워,
    공백을 건너뛰는 채점 구간이 몇 개이고 가중치를 얼마나 들고 있으며 그
    구간을 빼면 D 가 얼마나 달라지는지.

D 는 한 규약에서만 낸다 — 기준 시각을 1 밀리초 옮긴 자리다. 공백을 건너뛰는
구간이 D 에 얼마나 들어가는지를 가르는 데는 규약 하나면 되고, 여섯 규약을
다 돌면 가장 무거운 릴리스에서 한 시간이 넘는다.

실행:
    PYTHONPATH=src python3 scripts/gap_policy_audit.py [--release <이름>]
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from kitti_segment_D import (  # noqa: E402
    LENGTHS, RELEASES, STEP, common_window, euclidean, gather, geodesic,
    relative, relative_time, relative_translation, resample, segment_weights,
    segments, source_files,
)
from tum_poses import GAP, MIN_FRAMES, pieces, read_tum  # noqa: E402

TAU_S = 1e-3
SPLIT_ROOT = {"tumrgbd": "/mnt/Data/velref/tumrgbd"}


def gap_index(t, factor=GAP):
    """공백 뒤 첫 표본의 색인들. 중앙 간격의 factor 배를 넘는 자리."""
    if len(t) < 3:
        return np.zeros(0, dtype=int)
    d = np.diff(t)
    med = float(np.median(d))
    if med <= 0.0:
        return np.zeros(0, dtype=int)
    return np.flatnonzero(d > factor * med) + 1


def crossing(g, idx):
    """구간마다 그 안에 공백이 드는지. g 는 공백 뒤 첫 표본의 색인들.

    구간은 양 끝 표본을 다 쓴다 — relative() 가 rot[a] 와 rot[b] 를 함께
    읽는다. 공백은 g-1 과 g 사이에 있으므로, 그 공백이 구간 안에 드는
    조건은 a <= g-1 이면서 g <= b, 곧 a < g <= b 다.

    a <= g < b 로 쓰면 양쪽으로 틀린다. 공백 바로 뒤에서 시작하는 구간
    (g == a)은 공백을 안 건너뛰는데 건너뛴다고 찍히고, 공백 바로 뒤에서
    끝나는 구간(g == b)은 실제로 건너뛰는데 안 찍힌다.
    """
    g = np.asarray(g)
    return np.array([bool(np.any((g > a) & (g <= b))) for a, b, _ in idx])


def support_start(t, query):
    """질의 시각마다 보간이 왼쪽으로 쓰는 표본의 색인.

    t[j] <= query < t[j+1] 인 j 다. 보간은 그 j 와 j+1 두 표본만 읽으므로,
    이 색인 하나가 "그 끝점이 어느 두 표본 사이에서 왔나" 를 다 말한다.
    """
    t = np.asarray(t, dtype=float)
    return np.searchsorted(t, np.asarray(query, dtype=float),
                           side="right") - 1


def endpoint_touched(g, idx, support):
    """구간마다 끝점 보간이 공백 위에서 이뤄지는지. g 는 공백 뒤 첫 표본.

    공백이 점수에 닿는 길은 둘이고 이 함수가 그중 하나를 고른다.

    구간을 밀린 시각에서 다시 읽을 때 끝점 자세는 그 시각을 사이에 둔 두
    표본에서 보간해 온다. support[e] 가 그 왼쪽 표본이므로 짝은
    (support[e], support[e]+1) 이고, 공백은 g-1 과 g 사이에 있으니 그 짝이
    공백 위에 놓이는 조건은 g == support[e]+1 이다.

    색인을 e+1 로 못박지 않는 이유가 있다. 어긋남이 그 자리 표본 간격보다
    크면 표적 시각이 다음 표본을 지나쳐서 보간이 더 뒤의 짝을 읽는다.
    간격은 릴리스 평균이 아니라 자리마다 다르므로, 중앙 간격이 어긋남보다
    훨씬 커도 짧은 자리 하나가 그렇게 어긋날 수 있다. 실제 지지 표본으로
    판정하면 그 경우와 평범한 경우가 같은 식으로 처리된다.

    crossing() 과 겹치지 않는 자리가 있다. 구간 끝 b 바로 뒤가 공백이면
    공백은 구간 안에 없지만 끝점 자세는 그 공백을 건너 보간된다 — 끝점을
    고정했다고 보는 구간의 점수가 그래서 바뀐다. 반대로 구간 한가운데
    깊숙한 공백은 끝점을 안 건드리고 공개 경로 길이를 거쳐 구간이 어디서
    잘리는지만 바꾼다.
    """
    g = np.asarray(g)
    support = np.asarray(support)
    return np.array([bool(np.any((g == support[a] + 1)
                                 | (g == support[b] + 1)))
                     for a, b, _ in idx])


def class_score(w, err, mask):
    """그 묶음 안에서 무게를 다시 맞춘 D. 묶음이 비면 nan.

    이 수는 D 전체와 같은 척도가 아니다. 무게가 w_k = 1/(N L_k) 라 D 는
    길이로 나눈 뒤 평균한 값인데, 여기서는 묶음 안에서 무게를 다시 1 로
    맞추므로 "그 묶음의 구간 하나가 평균 얼마나 움직였나" 가 된다.
    두 길의 크기를 견주려면 class_share 를 쓴다.
    """
    if not mask.any():
        return float("nan")
    ww = w[mask]
    return float(np.dot(ww / ww.sum(), err[mask]))


def class_share(w, err, mask):
    """그 묶음이 D 전체에 넣는 몫. 묶음들을 더하면 D 가 된다.

    D = sum_k w_k d_k 라서 묶음별 합이 그대로 분해가 된다. 무게를 다시
    맞추지 않는 것이 요점이다 — 다시 맞추면 묶음마다 분모가 달라져서
    "어느 길이 더 큰가" 를 물을 수 없다.
    """
    if not mask.any():
        return 0.0
    return float(np.dot(w[mask], err[mask]))


def load_stage(release):
    """적재 단계에서 끊기고 버려진 양. 끊지 않는 릴리스는 전부 0 이다."""
    root = SPLIT_ROOT.get(release)
    if not root:
        return None
    rows = []
    # 폴더를 따로 훑지 않는다. 측정이 여는 목록을 그대로 받는다 — 이
    # 릴리스는 같은 궤적이 두 이름으로 깔려 있어서, 훑으면 그 주행이 적재
    # 집계에도 두 번 든다.
    for seq, paths in source_files(release, root):
        t, q, p = read_tum(paths[0])
        kept = list(pieces(t, q, p))
        # 끊긴 자리 수는 토막 수에서 하나를 뺀 값이 아니다. 짧아서 버려진
        # 토막도 끊긴 자리를 만들기 때문이다. 원본에서 직접 센다.
        ts = np.sort(np.asarray(t, dtype=float), kind="stable")
        if len(ts) > 1:
            ts = ts[np.r_[True, np.diff(ts) > 0]]
        cuts = len(gap_index(ts))
        # 끊으면 토막이 cuts + 1 개 난다 — 표본이 하나라도 있을 때만.
        # 빈 파일은 끊을 것도 날 것도 없어서 0 이다. 이 릴리스에 그런
        # 파일이 하나 있다(보정판 촬영분). cuts + 1 로 세면 있지도 않은
        # 토막을 버렸다고 적는다.
        made = cuts + 1 if len(t) else 0
        rows.append({
            "release": release,
            "sequence": seq,
            "samples_file": len(t),
            "samples_kept": sum(len(x[0]) for x in kept),
            "cuts": cuts,
            "pieces_kept": len(kept),
            "pieces_made": made,
            "pieces_dropped": made - len(kept),
            "min_frames": MIN_FRAMES,
            "seconds_file": repr(float(ts[-1] - ts[0]) if len(ts) > 1
                                 else 0.0),  # 빈 파일은 0 으로 남긴다
            "seconds_kept": repr(sum(float(x[0][-1] - x[0][0])
                                     for x in kept if len(x[0]) > 1)),
        })
    return rows


def score_stage(release, lengths, step, rule):
    """채점 구간 가운데 공백을 건너뛰는 것들의 수·무게·D 기여."""
    rows = []
    for name, (t, rot, pos) in gather(release):
        t = relative_time(np.asarray(t, dtype=float))
        rot = np.asarray(rot, dtype=float)
        pos = np.asarray(pos, dtype=float)
        if len(t) < 3:
            continue
        dt = float(np.median(np.diff(t)))
        keep = common_window(t, dt)
        ts, rs, ps = t[:keep], rot[:keep], pos[:keep]
        if len(ts) < 2:
            continue
        idx = segments(ps, lengths, step)
        if not idx:
            continue
        # 공백은 보간이 읽는 축에서 찾는다. 창으로 자른 축에서 찾으면 창
        # 끝 바로 뒤의 공백이 목록에 없는데, 마지막 끝점의 보간은 그 공백을
        # 건너서 온다 — 그 자리가 통째로 "공백과 무관" 으로 찍힌다. 중앙
        # 간격도 같은 이유로 전체 축에서 잡는다. 규칙 자체가 시퀀스의
        # 중앙 간격으로 적혀 있어서, 창 안에서만 재면 규칙이 창마다 달라진다.
        g = gap_index(t)
        w = segment_weights(idx)
        cross = crossing(g, idx)
        mr, mp = resample(t, rot, pos, ts + TAU_S, "slerp")
        er = geodesic(relative(rs, idx), relative(mr, idx))
        et = euclidean(relative_translation(rs, ps, idx),
                       relative_translation(mr, mp, idx))
        d_all = float(np.dot(w, er))
        dt_all = float(np.dot(w, et))
        if cross.all():
            d_free = dt_free = float("nan")
        else:
            wf = segment_weights([p for p, c in zip(idx, cross) if not c])
            d_free = float(np.dot(wf, er[~cross]))
            dt_free = float(np.dot(wf, et[~cross]))
        # 공백이 닿는 두 길을 갈라 센다. 끝점 보간이 공백 위에서 이뤄지는
        # 구간과, 공백이 구간 안에 있으면서 끝점은 안 건드리는 구간이다.
        # 둘을 한 수로 묶으면 "구간이 어디서 잘리나" 와 "끝점 자세가
        # 바뀌나" 가 구분이 안 된다.
        # 지지 표본은 밀린 시각을 원본 시간축에서 찾아 정한다. 공백 검출과
        # 같은 축이라야 한다 — 다른 축에서 재면 색인이 어긋나 엉뚱한 구간이
        # 공백에 닿는다고 찍힌다.
        sup = support_start(t, ts + TAU_S)
        # 둘을 따로 센다. 창 너머를 읽는 끝점은 창의 마지막 표본 하나뿐이라
        # 어느 시퀀스나 1 이 나오고 그것이 정상이다 — 창은 질의가 공개 표본
        # 안에 들도록 잡은 것이지 보간이 창 안쪽만 읽으라는 것이 아니다.
        # 파일 너머를 읽는 끝점은 외삽이라 0 이어야 한다.
        past_window = int(np.sum(sup + 1 > keep - 1))
        past_file = int(np.sum(sup + 1 > len(t) - 1))
        touch = endpoint_touched(g, idx, sup)
        inner = cross & ~touch
        rows.append({
            "release": release, "rule": rule, "sequence": name,
            "segments": len(idx), "segments_crossing": int(cross.sum()),
            "weight_crossing": repr(float(w[cross].sum() / w.sum())),
            "endpoints_support_past_window": past_window,
            "endpoints_support_past_file": past_file,
            "segments_endpoint_touched": int(touch.sum()),
            "weight_endpoint_touched": repr(float(w[touch].sum() / w.sum())),
            "segments_inside_only": int(inner.sum()),
            "weight_inside_only": repr(float(w[inner].sum() / w.sum())),
            "D_rot_all": repr(d_all), "D_rot_gapfree": repr(d_free),
            "D_trans_all": repr(dt_all), "D_trans_gapfree": repr(dt_free),
            "D_rot_endpoint_touched": repr(class_score(w, er, touch)),
            "D_trans_endpoint_touched": repr(class_score(w, et, touch)),
            "D_rot_inside_only": repr(class_score(w, er, inner)),
            "D_trans_inside_only": repr(class_score(w, et, inner)),
            # 아래 넷이 두 길의 크기다. 셋을 더하면 D 전체가 된다 —
            # 끝점이 닿는 길, 공백이 안에만 있는 길, 공백을 안 건드리는 나머지.
            "share_rot_endpoint_touched": repr(class_share(w, er, touch)),
            "share_trans_endpoint_touched": repr(class_share(w, et, touch)),
            "share_rot_inside_only": repr(class_share(w, er, inner)),
            "share_trans_inside_only": repr(class_share(w, et, inner)),
        })
    return rows


def refuse_to_shrink(path, rows):
    """이미 있는 표보다 적은 릴리스로 덮으려 하면 멈춘다.

    릴리스를 골라 돌리면 그 릴리스만 적힌다. 이 표는 생성기가 읽어 본문
    수를 내므로, 모자란 표로 덮으면 열 릴리스를 센 수가 조용히 한 릴리스를
    센 수가 된다. 실제로 그랬다 — 한 릴리스만 다시 재려고 돌렸더니 1802
    줄이 55 줄이 됐고, 표 자체는 멀쩡해 보였다. 일부러 좁혀 재는 중이면
    다른 출력 경로를 준다.
    """
    if not os.path.exists(path):
        return
    with open(path) as f:
        had = {r["release"] for r in csv.DictReader(f)}
    missing = had - {r["release"] for r in rows}
    if missing:
        raise SystemExit(
            "%s 에 이미 있는 릴리스를 덮어 지운다: %s."
            " 좁혀 재는 중이면 다른 출력 경로를 줘라"
            % (path, ", ".join(sorted(missing))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", nargs="*", default=list(RELEASES))
    ap.add_argument("--out-load",
                    default="results/next/gap_policy_load.csv")
    ap.add_argument("--out-score",
                    default="results/next/gap_policy_score.csv")
    args = ap.parse_args()

    load_rows, score_rows = [], []
    for release in args.release:
        got = load_stage(release)
        if got:
            load_rows += got
            dropped = sum(r["pieces_dropped"] for r in got)
            lost = sum(r["samples_file"] - r["samples_kept"] for r in got)
            print("%-9s 적재: 파일 %d 표본 중 %d 버림, 버린 토막 %d"
                  % (release, sum(r["samples_file"] for r in got), lost,
                     dropped))
        # 두 규칙을 다 돈다. 궤적이 짧은 릴리스는 100~800 m 규칙에서 구간이
        # 한 개도 안 서서, 그 규칙만 보면 공백 영향이 "없다" 로 읽힌다.
        for lengths, step, rule in ((None, None, "seg"), ((1.0,), 1, "rpe1m")):
            kw = {} if lengths is None else {"lengths": lengths, "step": step}
            rows = score_stage(release, kw.get("lengths", LENGTHS),
                               kw.get("step", STEP), rule)
            score_rows += rows
            if rows:
                n = sum(r["segments"] for r in rows)
                c = sum(r["segments_crossing"] for r in rows)
                worst = max(float(r["weight_crossing"]) for r in rows)
                print("%-9s 채점(%s): 구간 %d 중 공백을 건너뛰는 것 %d,"
                      " 한 시퀀스 최대 무게 몫 %.4g"
                      % (release, rule, n, c, worst))

    for path, rows in ((args.out_load, load_rows),
                       (args.out_score, score_rows)):
        if not rows:
            continue
        refuse_to_shrink(path, rows)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print("%d 행 -> %s" % (len(rows), path))


if __name__ == "__main__":
    main()
