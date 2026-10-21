#!/usr/bin/env python3
"""공식 순위표의 칸 차이를 2D 와 맞대 어느 비교가 조건을 채우는지 센다.

충족은 인증이 아니다. 순위는 공개 안 된 시험 구간에서 매겨지는데 우리가 잰 D 는
공개 구간의 것이라, 한쪽에서 잰 상한이 다른 쪽에서 매긴 순서를 인증하지 못한다.
여기 적는 것은 공개 구간의 D 를 대입했을 때 충분조건이 서는 쌍의 수다.

kitti_segment_D.py 가 잰 것은 상한의 크기다. 그 수가 쓸모 있으려면 사람들이
실제로 겨루는 차이와 맞대 봐야 한다.

맞대는 상대는 그 데이터셋이 직접 운영하는 공식 순위표뿐이다. 다른 논문의
결과표는 쓰지 않는다 — 논문마다 어느 방법을 싣느냐로 "겨루는 폭" 이 달라지고,
그러면 결론이 표를 고른 사람에게 달린다. 공식 순위표는 그 데이터셋이 정한
평가 코드로 매긴 하나뿐인 목록이라 고를 여지가 없다.

받을 수 있는 순위표의 조건은 하나다. 점수가 자세오차의 가중 평균이어야 한다 —
상한이 그 꼴에만 서기 때문이다. 그래서 문턱 안에 든 질의의 비율(옥스퍼드
장기 위치추정), 장소인식의 곡선아래면적·최상위적중률(헬리퍼), 강체 정렬을
먼저 거는 절대오차(키티-360 의 첫 칸)는 대상이 아니다.

그 꼴이 확인된 순위표에서만 판정을 낸다. 키티는 공식 개발도구가 구간 오차를
더해 개수로 나누는 것을 코드에서 확인했다. 키티-360 의 상대오차 칸은 어느 꼴로
묶는지가 어디에도 안 적혀 있어서, 수는 같이 내되 인증 칸을 미확인으로 적는다 —
제곱평균제곱근 꼴이면 같은 D 로도 순위가 갈릴 수 있다.

판정은 한 줄이다. 두 항목의 차이가 2D 와 표시값 반올림 폭의 합보다 크면 그
순위가 기준 보간 규약에 흔들리지 않는다. 그 2D 는 잰 규약 중 가장 큰 것이다 —
"규약에 흔들리지 않는다" 는 규약 하나가 아니라 잰 규약 전부에 대한 말이라,
한 규약으로 문턱을 세우면 다른 규약이 더 큰 값을 낼 때 그 문장이 거짓이 된다. 작으면 이 상한만으로는 그렇게 말하지
못한다 — 뒤집힌다는 뜻이 아니다. 두 점수가 같은 방향으로 밀려 상쇄되는 몫이
있고, 그 크기는 pair_cancellation.py 가 따로 잰다.

반올림 폭을 빼먹으면 안 된다. 순위표는 점수를 자리수를 잘라 찍으므로 적힌 수와
실제 점수가 그 칸 폭의 절반까지 다를 수 있다. 두 항목이 반대쪽으로 잘렸으면
실제 간격이 적힌 간격보다 폭 하나만큼 좁다. 그래서 문턱은 2D 가 아니라
2D + (h_A+h_B)/2 이고, 한 칸 안에서는 두 폭이 같아 2D + h 다. 키티 병진은
소수 둘째 자리까지 찍어 h 가 0.01 인데 2D 가 0.0172 라, 이 항을 빼면 문턱이
예순 퍼센트 모자란다.

순위는 정답이 비공개인 시험 구간에서 매기고 우리가 D 를 잰 것은 공개된 쪽이다.
같은 수집·같은 후처리라 릴리스 성질로 보되, 구간별 대응이 아니라는 것을 자료와
출력에 남긴다. 그래서 상한은 시퀀스별 값이 아니라 구간을 통째로 묶은 행(ALL)
에서 가져온다 — 공식 점수가 구간을 모아 평균하는 꼴과 같다.
"""

import argparse
import csv
import itertools
import os

# 단위와 CSV 칸. scale 은 CSV 값에 곱해 순위표의 단위로 적은 2D 를 만든다.
#   회전: CSV 가 도/미터, 순위표도 도/미터 -> ×2
#   병진: CSV 가 이미 퍼센트 -> ×2
METRICS = {
    "rotation": {"unit": "도/미터", "key": "D_%s_deg_per_m", "scale": 2.0},
    "translation": {"unit": "%", "key": "D_t_%s_pct", "scale": 2.0},
}

# 상한은 가중 평균 꼴 점수에만 선다. 제곱 꼴로 묶는 점수는 반례로 범위 밖에
# 고정돼 있으므로, 어느 꼴로 묶는지 확인되지 않은 순위표에서는 판정을 내지
# 않고 "미확인" 으로 적는다. 전에는 안 적힌 자리에서 평균 꼴을 가정했는데,
# 그건 인증을 더 쉽게 내주는 쪽이라 틀렸을 때 그 문장이 거짓이 된다.
PROTOCOLS = ("phase_half", "decim2", "interp")

# 묶은 행의 이름. kitti_segment_D.py 가 이 이름으로 적는다.
POOLED = "ALL"

LEADERBOARDS = {
    "kitti": {
        "source": "KITTI Visual Odometry / SLAM Evaluation 2012 공식 순위표"
                  " (cvlibs.net/datasets/kitti/eval_odometry.php), 2026-10-19"
                  " 조회, 상위 30",
        "csv": "results/next/kitti_segment_D.csv",
        # 묶는 꼴을 공식 개발도구에서 직접 확인했다. evaluate_odometry.cpp 의
        # saveSequenceErrors 다음 묶음이 구간 오차를 더해 개수로 나눈다
        # ("compute sum of t_err, r_err" 다음 "t_err/num, r_err/num").
        # 가중 평균 꼴이라 이 상한이 그대로 선다.
        "aggregation": "mean",
        # "our evaluation computes translational and rotational errors for
        # all possible subsequences of length (100, ..., 800) meters."
        # 병진은 퍼센트, 회전은 도/미터로 적힌다.
        "measured": "공개 시퀀스 00~10 (순위는 비공개 정답의 11~21 에서 매긴다)",
        # 셋째 값은 그 칸이 찍는 소수 자리수다. 반올림 폭 h = 10^-자리수.
        "columns": (("translation", "병진오차", 2),
                    ("rotation", "회전오차", 4)),
        "entries": [
            ("SOFT2", {"translation": 0.53, "rotation": 0.0009}),
            ("V-LOAM", {"translation": 0.54, "rotation": 0.0013}),
            ("LOAM", {"translation": 0.55, "rotation": 0.0013}),
            ("TVL-SLAM+", {"translation": 0.56, "rotation": 0.0015}),
            ("Traj-LIO", {"translation": 0.57, "rotation": 0.0015}),
            ("CT-ICP2", {"translation": 0.58, "rotation": 0.0012}),
            ("Traj-LO", {"translation": 0.58, "rotation": 0.0014}),
            ("AKF-LIO", {"translation": 0.59, "rotation": 0.0014}),
            ("GLIM", {"translation": 0.59, "rotation": 0.0015}),
            ("universal-slam", {"translation": 0.59, "rotation": 0.0014}),
            ("CT-ICP", {"translation": 0.59, "rotation": 0.0014}),
            ("SDV-LOAM", {"translation": 0.60, "rotation": 0.0015}),
            ("MagneticPillars++", {"translation": 0.60, "rotation": 0.0018}),
            ("CELLmap", {"translation": 0.61, "rotation": 0.0017}),
            ("KISS-ICP", {"translation": 0.61, "rotation": 0.0017}),
            ("MOLA-LO", {"translation": 0.62, "rotation": 0.0017}),
            ("SiMpLE", {"translation": 0.62, "rotation": 0.0015}),
            ("SLAMLab", {"translation": 0.63, "rotation": 0.0013}),
            ("PIN-SLAM", {"translation": 0.64, "rotation": 0.0015}),
            ("filter-reg", {"translation": 0.65, "rotation": 0.0016}),
            ("SOFT-SLAM", {"translation": 0.65, "rotation": 0.0014}),
            ("MULLS", {"translation": 0.65, "rotation": 0.0019}),
            ("retro-slam", {"translation": 0.66, "rotation": 0.0016}),
            ("VGMM-LO", {"translation": 0.66, "rotation": 0.0015}),
            ("ELO", {"translation": 0.68, "rotation": 0.0021}),
            ("IMLS-SLAM", {"translation": 0.69, "rotation": 0.0018}),
            ("MC2SLAM", {"translation": 0.69, "rotation": 0.0016}),
            ("Onion-LO", {"translation": 0.69, "rotation": 0.0017}),
            ("ISC-LOAM", {"translation": 0.72, "rotation": 0.0022}),
            ("FLOAM", {"translation": 0.72, "rotation": 0.0022}),
        ],
    },
    "kitti360": {
        "source": "KITTI-360 Semantic SLAM Localization 공식 순위표"
                  " (cvlibs.net/datasets/kitti-360/"
                  "leaderboard_semantic_slam.php?task=trajectory),"
                  " 2026-10-19 조회, 전체",
        "csv": "results/next/kitti360_rpe1m_D.csv",
        # 구간 상대오차를 어느 꼴로 묶어 한 칸에 적는지가 공개 문서에도
        # 개발도구에도 없다. 평균 꼴이면 이 상한이 서고 제곱평균제곱근 꼴이면
        # 안 선다 — 큰 구간이 제곱으로 더 무거워져 같은 D 로도 순위가 갈릴 수
        # 있다. 확인 전까지 판정을 내지 않는다.
        "aggregation": "unconfirmed",
        # "We adopt the standard Absolute Pose Error (APE) and Relative Pose
        #  Error (RPE) as metrics ... The RPE is evaluated between two frames
        #  with a distance of 1 meter."
        # 절대오차 칸은 안 쓴다 — 강체 정렬을 먼저 걸므로 이 상한의 범위
        # 밖이다(반례로 고정돼 있다). 상대오차 칸만 같은 가중평균 꼴이고,
        # 구간 길이가 100~800 m 가 아니라 1 m 다.
        "measured": "공개 드라이브 아홉 (순위는 비공개 정답에서 매긴다),"
                    " 1 m 간격 구간 87,074 개",
        "columns": (("translation", "상대오차(1 m 간격)", 2),),
        "entries": [
            ("SOFT2", {"translation": 0.84}),
            ("CT-ICP2", {"translation": 1.00}),
            ("ORB-SLAM2", {"translation": 2.03}),
            ("SUMA++", {"translation": 2.72}),
        ],
    },
}


def pooled_two_d(rows, metric, protocols=PROTOCOLS):
    """묶은 행에서 그 순위표의 단위로 적은 2D 와 그것을 낸 규약 이름.

    규약을 하나 골라 쓰지 않는다. 본문이 이 판정을 "기준 보간 규약에 흔들리지
    않는다" 로 읽는데, 그건 규약 하나가 아니라 잰 규약 전부에 대한 말이다.
    한 규약으로만 재면 그 규약이 가장 큰 값을 내는 쪽인지가 가정으로 남는다 —
    아니면 문턱이 모자라고, 맞아도 맞는지를 자료가 안 보여 준다. 그래서
    여기서 최대를 잡고 어느 규약이 그 최대인지 같이 돌려준다.
    """
    m = METRICS[metric]
    for r in rows:
        if r["sequence"] != POOLED:
            continue
        vals = [(m["scale"] * float(r[m["key"] % p]), p) for p in protocols
                if (m["key"] % p) in r]
        if not vals:
            raise KeyError("묶은 행에 %s 규약 칸이 하나도 없다"
                           % ", ".join(protocols))
        return max(vals)
    raise KeyError("묶은 행 %s 가 없다 — kitti_segment_D.py 를 다시 돌려라"
                   % POOLED)


def display_step(decimals):
    """그 칸이 찍는 한 칸의 폭. 소수 둘째 자리까지면 0.01 이다."""
    return 10.0 ** (-decimals)


def display_term(decimals_a, decimals_b=None):
    """문턱에 더하는 표시값 항 (h_A+h_B)/2. 한 칸 안에서는 h 와 같다."""
    if decimals_b is None:
        decimals_b = decimals_a
    return 0.5 * (display_step(decimals_a) + display_step(decimals_b))


def rounds_to_display(board, metric, decimals):
    """적어 둔 점수가 그 자리수로 찍힌 수와 같은가.

    자리수를 손으로 적으므로 틀리면 문턱이 조용히 틀린다. 적힌 값보다 자리수를
    크게 잡으면 반올림 항이 작아져 인증이 공짜로 늘어난다.
    """
    return all(abs(v - round(v, decimals)) < 1e-12
               for _, v in scores(board, metric))


def scores(board, metric):
    """그 칸의 (이름, 점수)를 좋은 쪽부터. 둘 다 작을수록 좋은 점수다.

    칸마다 따로 세운다. 순위표가 한 칸으로 줄을 세워도 다른 칸의 순서는 그와
    다르므로, "몇 등 떨어진 자리" 를 물으려면 그 칸 안의 등수여야 한다.
    """
    out = [(name, cols[metric]) for name, cols in board["entries"]
           if metric in cols]
    return sorted(out, key=lambda e: e[1])


def exceeds(gap, limit):
    """그 차이가 문턱을 넘나. 마지막 비트로 넘기지 않는다.

    충분조건이므로 애매한 자리는 "못 가린다" 쪽으로 둔다. 적힌 수를 빼서
    만든 차이와 측정으로 만든 문턱은 둘 다 마지막 비트가 흔들리는데, 그
    흔들림이 판정을 가르면 같은 거리의 두 쌍이 갈린다 — 표시 격자에서
    정확히 같은 거리인 쌍이 하나는 통과하고 하나는 떨어진 것이 그 자리였다.
    상대 여유를 두면 그 폭 안의 차이는 전부 "못 가린다" 로 떨어진다.

    이 비교는 순위표를 쓰는 모든 판정이 같이 쓴다. 두 스크립트가 각자
    부등호를 쓰면 같은 충분조건이 자리마다 다른 답을 낸다.
    """
    return gap > limit * (1.0 + 1e-9)


def verdicts(board, metric, limit):
    """모든 두 항목 쌍에 대해 (등수a, 등수b, 이름a, 이름b, 차이, 조건 충족).

    쌍을 고르지 않는다. 인접한 두 등수만 보면 "1 등과 2 등은 못 가린다" 까지고,
    전수를 보면 "몇 등 떨어진 자리까지 못 가리나" 가 같이 나온다.

    충족은 인증이 아니다. 순위표는 공개 안 된 시험 구간에서 매기는데 우리가
    잰 D 는 공개 구간의 것이라, 그 D 를 대입해 조건이 선다는 말까지다 —
    원고 제7절이 적은 그대로다(심사 26 회차 필수 7).
    """
    s = scores(board, metric)
    out = []
    for (ia, (na, va)), (ib, (nb, vb)) in itertools.combinations(
            enumerate(s), 2):
        gap = abs(va - vb)
        out.append((ia + 1, ib + 1, na, nb, gap, exceeds(gap, limit)))
    return out


def first_certified_distance(recs):
    """몇 등 차이부터 조건을 채우는 쌍이 하나라도 생기나. 없으면 None."""
    best = None
    for ia, ib, _, _, _, ok in recs:
        if ok and (best is None or ib - ia < best):
            best = ib - ia
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/next/kitti_rank_check.csv")
    args = ap.parse_args()

    out_rows = []
    for key in sorted(LEADERBOARDS):
        board = LEADERBOARDS[key]
        with open(board["csv"]) as f:
            rows = list(csv.DictReader(f))
        print("\n출처: %s" % board["source"])
        print("상한을 잰 곳: %s" % board["measured"])
        certifies = board["aggregation"] == "mean"
        if not certifies:
            print("묶는 꼴 미확인 — 판정을 내지 않는다. 아래 수는 평균 꼴이라고"
                  " 가정했을 때의 것이고 충족 칸은 미확인으로 적는다.")
        for metric, label, decimals in board["columns"]:
            if not rounds_to_display(board, metric, decimals):
                raise SystemExit("%s %s: 적어 둔 점수가 소수 %d 자리로 안 찍힌다"
                                 % (key, metric, decimals))
            lim, worst_protocol = pooled_two_d(rows, metric)
            h = display_term(decimals)
            recs = verdicts(board, metric, lim + h)
            held = sum(1 for r in recs if r[5])
            s = scores(board, metric)
            span = max(v for _, v in s) - min(v for _, v in s)
            near = [r for r in recs if r[1] - r[0] == 1]
            near_held = sum(1 for r in near if r[5])
            far = first_certified_distance(recs)
            print("\n칸: %s (%s), 규약 %s 중 가장 큰 %s"
                  % (label, METRICS[metric]["unit"],
                     "·".join(PROTOCOLS), worst_protocol))
            print("  2D = %.4g, 표시값 항 = %.4g, 문턱 = %.4g"
                  % (lim, h, lim + h))
            print("  순위표가 퍼진 폭 = %.4g (2D 의 %.2f 배)"
                  % (span, span / lim))
            word = "조건 충족" if certifies else "조건 충족(묶는 꼴 미확인)"
            print("  모든 쌍      %s %3d / %3d (%.1f%%)"
                  % (word, held, len(recs), 100.0 * held / len(recs)))
            print("  바로 옆 등수  %s %3d / %3d"
                  % (word, near_held, len(near)))
            print("  처음 생기는 등수 차이: %s"
                  % ("없음" if far is None else "%d 등" % far))
            for ia, ib, na, nb, gap, ok in recs:
                out_rows.append({
                    "board": key, "metric": metric, "column": label,
                    "unit": METRICS[metric]["unit"],
                    "protocols": "|".join(PROTOCOLS),
                    "protocol_worst": worst_protocol,
                    "rank_a": ia, "method_a": na,
                    "rank_b": ib, "method_b": nb,
                    "rank_distance": ib - ia,
                    # 잰 값을 그대로 적는다. 본문이 이 열들에서 상한·하한을
                    # 뽑으므로 저장 단계에서 먼저 깎으면 그 보장이 깨진다.
                    "gap": repr(float(gap)), "twoD": repr(float(lim)),
                    "display_term": repr(float(h)),
                    "threshold": repr(float(lim + h)),
                    # 묶는 꼴이 확인된 순위표에서만 판정을 적는다. 미확인인
                    # 자리에 0 을 적으면 "못 가린다" 로, 1 을 적으면 "가린다"
                    # 로 읽히는데 둘 다 자료가 받쳐 주지 않는다.
                    #
                    # 이름이 condition_met 인 이유는 이게 인증이 아니기
                    # 때문이다. 공개 구간에서 잰 D 를 대입해 충분조건이
                    # 선다는 말이고, 순위는 공개 안 된 시험 구간에서 매긴다.
                    "aggregation": board["aggregation"],
                    "condition_met": ("1" if ok else "0") if certifies
                                     else "unconfirmed",
                    "source": board["source"],
                })

    if not out_rows:
        return
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print("\n%d 행 -> %s" % (len(out_rows), args.out))


if __name__ == "__main__":
    main()
