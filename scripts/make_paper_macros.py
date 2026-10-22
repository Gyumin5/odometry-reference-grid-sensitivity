#!/usr/bin/env python3
"""측정 CSV 에서 원고가 쓸 매크로와 표 본문을 만든다.

원고에 수를 손으로 적지 않는다. 손으로 적으면 측정을 다시 돌렸을 때 본문이
조용히 옛 값으로 남는다. 여기서 만든 파일만 \\input 하고, 본문에서는 매크로
이름으로만 부른다.

만드는 것 둘.
  numbers.tex      매크로. \\DkittiSegTrans 처럼 릴리스·규칙·칸으로 이름이 선다.
  table_bounds.tex 릴리스마다 한 줄씩, 그 릴리스에 서는 구간 규칙 칸들.
"""

import argparse
import csv
import json
import math
import os
import re
import sys
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from offset_tolerance import slope  # noqa: E402

NAMES = {
    "kitti": "Kitti", "kitti360": "KittiThreeSixty", "boreas": "Boreas",
    "helipr": "Helipr", "nuscenes": "Nuscenes", "pit30m": "PitThirtyM",
    "tumrgbd": "TumRgbd", "nclt": "Nclt", "tumvi": "TumVi",
    "euroc": "Euroc",
}
RULES = {"seg": "Seg", "rpe1m": "Rpe"}
# 공개 행렬이 이미 직교라고 볼 이탈폭. 쿼터니언으로 싣는 릴리스는 1e-15 쪽이고
# 행렬로 싣는 둘은 1e-10 위다 — 그 사이 다섯 자릿수가 비어 있어 경계가 선다.
EXACT_DEV = 1e-12
LABELS = {
    "kitti": "KITTI", "kitti360": "KITTI-360", "boreas": "Boreas",
    "helipr": "HeLiPR", "nuscenes": "nuScenes", "pit30m": "Pit30M",
    "tumrgbd": "TUM RGB-D", "nclt": "NCLT", "tumvi": "TUM VI",
    "euroc": "EuRoC MAV",
}
COLS = (("twoD_rot_deg_per_100m", "Rot"), ("twoD_trans_pct", "Trans"),
        ("n_segments", "Segments"))

# 규약 훑기에서 본문이 부르는 변형. 이름의 숫자는 글자로 바꿔 적는다.
SWEEP = {
    ("shift", "0.1"): "ShiftLo", ("shift", "0.9"): "ShiftHi",
    ("decim2", "even"): "DecimEven", ("decim2", "odd"): "DecimOdd",
    ("rate", "0.5_slerp"): "RateSlerp",
    ("rate", "0.5_nearest"): "RateNearest",
    ("method", "0.5"): "MethodOnly",
    ("tau", "1ms"): "TauOne", ("tau", "5ms"): "TauFive",
    ("tau", "10ms"): "TauTen", ("tau", "25ms"): "TauTwentyFive",
    ("tau", "50ms"): "TauFifty",
}
SWEEP_COLS = (("D_rot_deg_per_100m", "Rot"), ("D_trans_pct", "Trans"))

# 밀리초 환산에서 본문이 부르는 칸.
BOARDS = {("kitti", "translation"): "BoardKittiTrans",
          ("kitti", "rotation"): "BoardKittiRot",
          ("kitti360", "translation"): "BoardKittiThreeSixty"}
TAU_NAMES = {"0": "Zero", "1": "One", "2": "Two", "5": "Five", "10": "Ten",
             "25": "TwentyFive", "50": "Fifty"}


def read_table(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def rounded_outward(value, fmt, up):
    """본문이 "최대 …" / "최소 …" 로 부르는 수를 바깥쪽으로 깎는다.

    가장 가까운 쪽으로 깎으면 부등호가 원자료를 안 품는다. 잰 최댓값이
    0.511334 인데 "최대 0.51" 이라고 적으면 그 문장이 실제로 잰 수를 배제하고,
    "최소 88.96" 을 "최소 89" 로 적으면 재지 않은 구간까지 주장한다. 위쪽
    경계는 올리고 아래쪽 경계는 내리면 두 경우가 다 닫힌다.

    자리수는 쓰는 쪽 서식에서 읽는다 — "%.2f" 면 소수 둘째 자리, "%.3g" 면
    유효숫자 셋이다.
    """
    x = Decimal(repr(float(value)))
    kind, digits = fmt[-1], int(fmt[2:-1])
    if kind == "f":
        step = Decimal(1).scaleb(-digits)
    elif kind == "g":
        step = (Decimal(1).scaleb(x.adjusted() - (digits - 1)) if x
                else Decimal(1))
    else:
        raise ValueError("깎을 자리수를 못 읽는 서식: %s" % fmt)
    rounding = ROUND_CEILING if up else ROUND_FLOOR
    q = x.quantize(step, rounding=rounding)
    # 읽을 만한 크기면 지수 표기를 쓰지 않는다. "%.2g" 는 100 을 1e+02 로
    # 찍는데 본문에서 그 자리는 퍼센트라 "1e+02 %" 가 그대로 조판된다. 서식으로
    # 되돌리지 않고 q 를 그대로 적는 것은, 되돌리면 가장 가까운 쪽으로 다시
    # 깎여 방금 바깥으로 민 자리가 제자리로 돌아갈 수 있기 때문이다. 끝의 0 은
    # 턴다 — 품는 범위는 그대로이고(0.920 과 0.92 는 같은 수다) 본문에서 이웃한
    # 수와 자리수가 어긋나 보이는 것만 없어진다. 더 작은 수는 지수 표기가 읽기
    # 쉬워서 그대로 둔다 — 1.44e-10 을 0 열 개로 적지 않는다.
    if q and abs(q) >= Decimal("0.0001"):
        s = format(q, "f")
        return s.rstrip("0").rstrip(".") if "." in s else s
    return fmt % float(q)


def at_most(value, fmt="%.3g"):
    """"최대 …" 자리. 잰 값을 품도록 위로 깎는다."""
    return rounded_outward(value, fmt, True)


def at_least(value, fmt="%.3g"):
    """"최소 …" 자리. 잰 값을 품도록 아래로 깎는다."""
    return rounded_outward(value, fmt, False)


def shown(value, fmt="%.5g"):
    """본문에 찍을 자리수로 깎는다. 깎는 자리는 여기 한 곳뿐이다.

    결과 표는 깎지 않은 수를 싣는다 — 문턱과 견주는 쪽이 그 수를 되읽기
    때문이다. 그래서 읽기 좋은 자리수로 줄이는 일은 본문을 만들 때 한다.
    구간 수처럼 수가 아닌 칸은 그대로 돌려준다.
    """
    try:
        return fmt % float(value)
    except (TypeError, ValueError):
        return value


EXPONENT = re.compile(r"^(-?[0-9]+(?:\.[0-9]+)?)e([-+])0*([0-9]+)$")


def tex_number(text):
    """지수 꼴 하나를 조판 꼴로 바꾼다. 1.6e-05 -> \\ensuremath{1.6\\times10^{-5}}.

    파이썬 서식이 내는 e 꼴은 프로그램의 표기지 지면의 표기가 아니다. 같은
    원고가 바로 옆 문장에서 10^{-7} 을 제대로 쓰고 있어서, 섞여 나오면 두
    수가 다른 종류처럼 보인다. 바꾸는 자리는 파일에 적는 순간 한 곳뿐이다 —
    셈과 결과표는 그대로 e 꼴을 쓰고, 되읽는 쪽은 plain_number 로 되돌린다.
    수가 아닌 칸(이름·날짜·줄표)은 손대지 않는다.
    """
    m = EXPONENT.match(str(text))
    if not m:
        return text
    mant, sign, exp = m.groups()
    return "\\ensuremath{%s\\times10^{%s%s}}" % (
        mant, "-" if sign == "-" else "", exp)


def plain_number(text):
    """tex_number 의 역. 조판 꼴을 되읽어 수로 쓸 글자로 돌린다."""
    m = re.match(r"^\\ensuremath\{(-?[0-9.]+)\\times10\^\{(-?[0-9]+)\}\}$",
                 str(text))
    return "%se%s" % (m.group(1), m.group(2)) if m else text


def macros(rows):
    """(이름, 값) 들. 이름에는 숫자와 밑줄을 쓰지 않는다 — 라텍 매크로 규칙."""
    out = []
    for r in rows:
        rel, rule = r["release"], r["rule"]
        if rel not in NAMES or rule not in RULES:
            continue
        for key, tag in COLS:
            value = r[key] if key == "n_segments" else shown(r[key])
            out.append(("D%s%s%s" % (NAMES[rel], RULES[rule], tag), value))
    return out


def per_sequence_macros(rows):
    """묶음 값 하나가 접어 버리는 시퀀스별 폭.

    공식 집계 규칙대로 구간을 한 자루에 담으면 릴리스마다 한 수가 나오는데,
    그 수만 적으면 어느 폭을 가운데로 접은 것인지 읽는 쪽이 알 수 없다.
    가장 넓게 벌어진 칸을 배율로 적는다.
    """
    out = []
    for key_lo, key_hi, tag in (
            ("twoD_rot_seq_min_deg_per_100m",
             "twoD_rot_seq_max_deg_per_100m", "Rot"),
            ("twoD_trans_seq_min_pct", "twoD_trans_seq_max_pct", "Trans")):
        best = None
        for r in rows:
            if r["release"] not in LABELS or not r.get(key_lo):
                continue
            lo, hi = float(r[key_lo]), float(r[key_hi])
            if lo <= 0:
                continue
            if best is None or hi / lo > best[0]:
                best = (hi / lo, r)
        if best is None:
            continue
        ratio, r = best
        out += [("PerSeq%sRatio" % tag, "%.3g" % ratio),
                ("PerSeq%sRelease" % tag, LABELS[r["release"]]),
                ("PerSeq%sRule" % tag,
                 "100--800\\,m" if r["rule"] == "seg" else "1\\,m"),
                ("PerSeq%sMin" % tag, at_least(r[key_lo], "%.5g")),
                ("PerSeq%sMax" % tag, at_most(r[key_hi], "%.5g")),
                ("PerSeq%sSequences" % tag, r["n_sequences"])]
    return out


def segment_rule_population_macros(rows, register):
    """각 구간 규칙이 등록 시퀀스의 일부에서만 서는 릴리스들.

    상한 표는 그 규칙이 실제로 선 시퀀스만 센다. 등록 수를 모르면 그 수가
    릴리스 전체의 성질처럼 읽히는데, 한 시퀀스에서만 선 값과 열한 개를
    모은 값은 같은 줄에 적혀도 같은 종류가 아니다.

    두 규칙을 한 줄에 같이 센다. 긴 규칙만 세면 짧은 규칙은 읽은 것
    전부에서 선다는 말이 되는데, 1 m 규칙도 공통 창 안에서 누적 이동이
    1 m 를 못 넘는 시퀀스를 떨군다(뉴신 909/979, 튬 알지비디 74/221).
    규칙마다 목록을 따로 내지 않는 것은 자리 때문이다 — 릴리스 이름이
    두 번 적히면 표 설명이 한 줄 늘고 그만큼 본문을 덜어내야 한다.
    """
    reg = {r["release"]: int(r["sequences"]) for r in register
           if r.get("sequences")}

    def have(rel, rule):
        # 구간이 하나도 안 서는 릴리스도 이 목록에 든다. 그 규칙의 줄이
        # 있는 것만 돌면 0 개인 릴리스는 들어올 길이 없어서, "일부에서만
        # 선다" 는 목록이 "전혀 안 선다" 는 쪽을 조용히 빠뜨린다.
        hit = [r for r in rows if r["release"] == rel and r["rule"] == rule]
        return int(hit[0]["n_sequences"]) if hit else 0

    short = []
    for rel in NAMES:
        if not any(r["release"] == rel for r in rows):
            continue
        total = reg.get(rel)
        if total is None:
            continue
        coarse, fine = have(rel, "seg"), have(rel, "rpe1m")
        if coarse < total or fine < total:
            short.append("%s %d/%d of %d"
                         % (LABELS[rel], coarse, fine, total))
    if not short:
        return []
    return [("RuleShortList", ", ".join(short)),
            ("RuleShortCount", str(len(short)))]


def sweep_population_macros(rows):
    """상한 표와 훑기 표의 시퀀스 수가 갈리는 자리.

    훑기는 밀린 변형이 자료 밖으로 안 나가게 양 끝을 자르고 구간을 짓기
    때문에, 짧은 시퀀스는 구간이 하나도 안 남아 표에서 빠진다. 두 표를
    나란히 읽는 쪽에는 그 차이가 보이지 않으므로 가장 크게 갈리는 자리를
    본문이 짚는다. 묶음 행은 시퀀스로 세지 않는다 — 양쪽 다 그렇다.
    """
    best = None
    for r in rows:
        if r["release"] not in LABELS or not r.get("n_sequences_sweep"):
            continue
        a, b = int(r["n_sequences"]), int(r["n_sequences_sweep"])
        if a > b and (best is None or a - b > best[0]):
            best = (a - b, r, a, b)
    if best is None:
        return []
    _, r, a, b = best
    return [("PopDropRelease", LABELS[r["release"]]),
            ("PopDropRule",
             "100--800\\,m" if r["rule"] == "seg" else "1\\,m"),
            ("PopDropBound", str(a)),
            ("PopDropSweep", str(b))]


WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
         7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven"}
# 바퀴를 안 쓴 실내 릴리스 — 손에 들고 찍은 둘과 실내를 나는 유로크다.
# 나머지는 바퀴 달린 대 위에 있다.
INDOOR = ("tumrgbd", "tumvi", "euroc")


def word(n):
    """본문이 글자로 세는 자리. 열한을 넘으면 숫자로 떨어뜨린다."""
    return WORDS.get(n, str(n))


def release_count_macros(rows):
    """상한 표에 선 릴리스 수. 본문이 "일곱" 을 글자로 들고 있으면 릴리스를
    더할 때마다 여섯 자리가 조용히 틀린다 — 표가 세는 수를 그대로 쓴다."""
    rel = {r["release"] for r in rows}
    hand = len([r for r in rel if r in INDOOR])
    return [("NReleases", word(len(rel))), ("NReleasesNum", str(len(rel))),
            ("NIndoor", word(hand)), ("NVehicle", word(len(rel) - hand))]


def wheeled_indoor_macros(load_rows):
    """실내 릴리스 안에 섞인 바퀴 시퀀스 수.

    튬 알지비디는 릴리스 전체를 손에 들고 찍은 것으로 적으면 틀린다 —
    공개 묶음에 파이어니어 로봇 위에서 찍은 범주가 따로 있고 우리가 읽는
    집합에도 들어 있다. 본문이 플랫폼을 가르는 자리에서 그 수를 글자로
    박으면 등록부가 바뀌어도 문장이 안 따라온다.

    세는 단위는 토막이 아니라 공개 시퀀스다. 적재 표는 파일 하나에 줄
    하나라 그 단위가 바로 공개 시퀀스다. 채점 표에서 세면 단위가 토막이고
    모집단도 1 m 구간이 선 토막으로 좁아져서, 파이어니어 시퀀스 하나가
    구간을 못 세우면 본문의 "공개된 시퀀스" 가 조용히 줄어든다.
    """
    base = {r["sequence"] for r in load_rows if "pioneer" in r["sequence"]}
    return [("RegTumRgbdWheeled", word(len(base)))]


def crosscheck_macros(rows):
    """공식 구간 목록과 우리 목록이 얼마나 같은지.

    세는 단위를 개수에서 목록으로 올린다. 구간 수가 같아도 시작·끝이 다를
    수 있어서 "개수가 맞았다" 는 일치의 증거로 약하다. 맞댄 것은 (시작, 끝,
    요구길이) 세쌍이고, 배정밀도와 단정밀도를 따로 적어 어긋남이 규칙
    차이인지 공식 누적의 반올림인지 읽는 쪽이 가를 수 있게 한다.

    시퀀스 수도 이 표에서 낸다. 본문이 그 수를 이 대조의 범위로 적는데,
    공식 이진파일 탐침 표에서 세면(\\OfficialSequences) 두 표의 시퀀스 집합이
    갈리는 순간 문장만 조용히 틀린다. 지금은 둘 다 열하나다.
    """
    total = sum(int(r["n_segments_ours"]) for r in rows)
    d64 = sum(int(r["differing_f64"]) for r in rows)
    d32 = sum(int(r["differing_f32"]) for r in rows)
    shift = max(int(r["max_end_frame_shift_f32"]) for r in rows)
    rel = max(float(r["relative_change"]) for r in rows)
    return [("CrossSequences", str(len(rows))),
            ("CrossSegments", "{:,}".format(total)),
            ("CrossDifferingF", str(d64)),
            ("CrossDifferingSingle", str(d32)),
            ("CrossEndShift", word(shift)),
            # 본문이 "at most" 로 쓰는 자리라 바깥쪽으로 올린다.
            ("CrossRelMax", at_most(rel, "%.1g"))]


def register_count_macros(register):
    """릴리스마다 우리가 읽은 시퀀스 수.

    본문이 "이 릴리스는 일부만 읽었다" 고 적는 자리에 쓴다. 그 수를 글자로
    박으면 등록부가 바뀌어도 문장이 안 따라오고, 읽는 쪽은 표와 글 중 어느
    쪽이 지금 참인지 모른다.
    """
    # 이름은 NAMES 에서만 가져온다. 릴리스 이름을 그대로 다듬어 쓰면 숫자가
    # 섞인 제어어(\RegSeqsKitti360)가 나오는데, 텍스는 제어어 이름에 숫자를
    # 못 넣어서 그 줄을 통째로 버리고 남은 글자를 본문에 찍는다. 조판은
    # 성공으로 끝나고 첫 쪽이 그 찌꺼기만 있는 빈 쪽이 된다.
    out = []
    for r in register:
        if not r.get("sequences"):
            continue
        if r["release"] not in NAMES:
            continue
        out.append(("RegSeqs%s" % NAMES[r["release"]],
                    group(int(r["sequences"]))))
    return out


def rate_group_macros(rows, register, sweep=None):
    """표본율이 1 m 규칙 병진 상한을 어디까지 가르는가.

    sweep 을 주면 같은 두 묶음을 공통 절대 이동(1 ms)에서도 재서 범위를
    같이 낸다. 반표본 위상은 느린 릴리스에 더 긴 시간 이동을 주므로, 그
    조건에서 갈린 것을 표본율만의 효과로 읽으면 안 된다 — 그 대조를 본문이
    적으려면 두 범위가 매크로로 있어야 한다.

    빠른 쪽과 느린 쪽을 가르는 경계를 글자로 박지 않는다. 잰 주기를 늘어놓고
    이웃한 두 값의 배율이 가장 큰 자리에서 끊는다 — 릴리스를 더해도 경계가
    자료를 따라가고, 50 같은 수를 본문에 박아 두는 일이 없다.

    묶음 안에서 주기가 순서를 안 정한다고는 적지 않는다. 쌍 몇 개가 어긋난다는
    것은 그 묶음 안에서 주기에 예측력이 없다는 뜻이 아니다. 어긋난 쌍을 묶음
    별로 세어 그대로 내놓는다.

    묶음 둘이 안 겹치는 것이 이 문단의 전제다. 깨지면 매크로를 하나도 안 내서
    본문이 빌드에서 멈춘다 — 전제가 깨진 채 문장만 남는 쪽이 더 나쁘다.
    """
    hz = {r["release"]: float(r["measured_hz"]) for r in register
          if r.get("measured_hz")}
    trans = {r["release"]: float(r["twoD_trans_pct"]) for r in rows
             if r["rule"] == "rpe1m"}
    rot = {r["release"]: float(r["twoD_rot_deg_per_100m"]) for r in rows
           if r["rule"] == "rpe1m"}
    sel = sorted((hz[k], k) for k in trans if k in hz)
    if len(sel) < 3:
        return []
    cut = max(range(1, len(sel)), key=lambda i: sel[i][0] / sel[i - 1][0])
    slow = [k for _, k in sel[:cut]]
    fast = [k for _, k in sel[cut:]]
    if min(trans[k] for k in slow) <= max(trans[k] for k in fast):
        return []
    top = max(rot, key=lambda k: rot[k])

    def order(left, right):
        """주기로 매긴 순서와 상한으로 매긴 순서가 어긋나는 쌍을 센다.

        주기가 같은 쌍은 어느 쪽이 더 빠른지가 없어 분모에서도 뺀다.
        """
        seen, tot, inv, tie = set(), 0, 0, 0
        for a in left:
            for b in right:
                if a == b or (b, a) in seen:
                    continue
                seen.add((a, b))
                if hz[a] == hz[b]:
                    tie += 1
                elif (hz[a] > hz[b]) == (trans[a] > trans[b]):
                    tot, inv = tot + 1, inv + 1
                else:
                    tot += 1
        return tot, inv, tie

    slow_tot, slow_inv, slow_tie = order(slow, slow)
    fast_tot, fast_inv, fast_tie = order(fast, fast)
    cross_tot, cross_inv, cross_tie = order(slow, fast)
    return [
        ("RateSlowPairs", str(slow_tot)),
        ("RateSlowInv", str(slow_inv)),
        ("RateFastPairs", str(fast_tot)),
        ("RateFastInv", str(fast_inv)),
        ("RateCrossPairs", str(cross_tot)),
        ("RateCrossInv", str(cross_inv)),
        ("RateTiedPairs", str(slow_tie + fast_tie + cross_tie)),
        ("RateSlowCount", word(len(slow))),
        ("RateFastCount", word(len(fast))),
        # 범위의 양 끝은 바깥으로 깎는다. 맨 반올림이면 잰 값이 적은 범위
        # 밖으로 나간다 — 49.98 Hz 를 50 으로 찍어 "50 부터" 라고 적으면
        # 그 릴리스가 범위 밖이 된다.
        ("RateSlowHzMin", at_least(min(hz[k] for k in slow), "%.3g")),
        ("RateSlowHzMax", at_most(max(hz[k] for k in slow), "%.3g")),
        ("RateFastHzMin", at_least(min(hz[k] for k in fast), "%.3g")),
        ("RateFastHzMax", at_most(max(hz[k] for k in fast), "%.3g")),
        ("RateSlowTransMin", at_least(min(trans[k] for k in slow), "%.2g")),
        ("RateFastTransMax", at_most(max(trans[k] for k in fast), "%.2g")),
        ("RateRotTopRelease", LABELS[top]),
        ("RateRotTopHz", shown(hz[top], "%.3g")),
        ("RateRotTopValue", shown(rot[top])),
    ] + ([("AxisKittiHz", shown(hz["kitti"], "%.3g"))] if "kitti" in hz
         else []) + common_offset_range(sweep, slow, fast)


def common_offset_range(sweep, slow, fast, param="1ms"):
    """같은 두 묶음을 공통 절대 이동에서 잰 범위. 한쪽이라도 비면 안 낸다.

    훑기 표는 D 로 적히고 상한 표는 2D 로 적힌다. 이 범위는 본문에서 반 표본
    범위 바로 옆에 서고 그쪽이 2D 라, 여기서 2 를 곱해 같은 척도로 낸다.
    척도가 다른 두 범위를 나란히 두면 "겹친다·안 겹친다" 가 두 배만큼 틀린다.
    """
    if not sweep:
        return []
    at = {r["release"]: 2.0 * float(r["D_trans_pct"]) for r in sweep
          if r["rule"] == "rpe1m" and r["axis"] == "tau"
          and r["param"] == param}
    out = []
    for group, tag in ((slow, "Slow"), (fast, "Fast")):
        vals = [at[k] for k in group if k in at]
        if len(vals) != len(group):
            return []
        out += [("RateTauOne%sMin" % tag, at_least(min(vals), "%.2g")),
                ("RateTauOne%sMax" % tag, at_most(max(vals), "%.2g"))]
    return out


def table_min_macros(rows):
    """표에서 가장 작은 칸. 계측 바닥과 맞댈 때 손으로 고르지 않으려고 낸다."""
    out = []
    for key, tag in (("twoD_rot_deg_per_100m", "Rot"),
                     ("twoD_trans_pct", "Trans")):
        vals = [(float(r[key]), r) for r in rows if r["release"] in LABELS]
        if not vals:
            continue
        v, r = min(vals, key=lambda p: p[0])
        # 본문이 "가장 작은 칸이 바닥을 넘는다" 로 읽는 자리라 아래로 깎는다.
        # 적힌 수가 잰 값보다 작으면 그 문장이 약해질 뿐이지만, 위로 깎으면
        # 바닥을 안 넘는 칸을 넘는다고 적게 된다.
        out += [("TableMin%s" % tag, at_least(v, "%.5g")),
                ("TableMin%sRelease" % tag, LABELS[r["release"]])]
    # 초록이 부르는 두 범위. 릴리스 이름을 손으로 골라 적으면 표에 줄이 늘 때
    # 어긋난다 — 튬 브이아이가 들어온 뒤 100~800 m 하한이 한 릴리스 낡은 채로
    # 남아 있었다.
    for rule, tag in (("seg", "Seg"), ("rpe1m", "Rpe")):
        vals = [float(r["twoD_trans_pct"]) for r in rows
                if r["release"] in LABELS and r["rule"] == rule]
        if not vals:
            continue
        # 초록에 찍히는 자리라 두 자리로 줄이되 잰 범위를 품게 바깥으로
        # 깎는다 — 안쪽으로 깎으면 실제보다 좁은 범위를 적게 된다.
        out += [("Table%sTransMin" % tag, at_least(min(vals), "%.2g")),
                ("Table%sTransMax" % tag, at_most(max(vals), "%.2g"))]
    return out


def ratio_macros(rows):
    """규칙을 바꿨을 때와 릴리스를 바꿨을 때 병진 상한이 움직이는 배율.

    규칙 배율은 두 규칙이 다 있는 릴리스에서만 선다 — 한쪽이 없는 릴리스를
    끼우면 "모든 릴리스에서" 라는 문장이 거짓이 된다.
    """
    by = {(r["release"], r["rule"]): float(r["twoD_trans_pct"]) for r in rows}
    pairs = [(by[(rel, "rpe1m")] / by[(rel, "seg")], rel) for rel in NAMES
             if (rel, "seg") in by and (rel, "rpe1m") in by]
    ratios = [v for v, _ in pairs]
    # 바깥으로 깎는다. 본문이 표 설명 자리에 그 규약을 적어 두었는데 이
    # 자리만 맨 반올림이라, 잰 값 77.0 이 77 로 나와 본문이 표보다 좁아졌다.
    out = [("RatioRuleMin", at_least(min(ratios), "%.2g")),
           ("RatioRuleMax", at_most(max(ratios), "%.2g")),
           # 어느 릴리스에서 나온 배율인지 적는다. 공백 감사가 그 릴리스를
           # 공백 지배로 표시하면 이 수도 같이 읽어야 한다.
           ("RatioRuleMaxRelease", LABELS[max(pairs)[1]]),
           ("RatioRuleReleases", str(len(ratios)))]
    for rule, tag in RULES.items():
        vals = [v for (_, ru), v in by.items() if ru == rule]
        # 이 자리는 범위의 끝이 아니라 잰 배율 하나다. 바깥으로 깎으면
        # 77.0 이 78 로 올라가는데, 본문은 이 수를 규칙 배율의 상한과
        # 견주므로 올린 수끼리는 어느 쪽이 큰지 못 가린다. 맨 반올림으로
        # 적어야 읽는 쪽이 지면의 두 수만으로 그 비교를 확인한다.
        out.append(("RatioRelease%s" % tag,
                    str(shown(max(vals) / min(vals), "%.2g"))))
    return out


def rule_ratio_common_macros(rows):
    """같은 시퀀스 위에서 다시 잰 규칙 배율.

    규칙을 바꾸면 그 규칙이 설 수 있는 시퀀스도 같이 바뀐다 — 짧은 주행은
    1 m 쪽에만 들어온다. 그래서 묶음 값을 바로 나눈 배율에는 구간 길이의
    효과와 모집단 변화가 섞인다. 둘이 갈리는 폭을 본문이 적을 수 있게
    공통 시퀀스 위의 배율 범위와, 가장 크게 움직인 릴리스를 같이 낸다.
    """
    if not rows:
        return []
    common = [float(r["ratio_common"]) for r in rows]
    worst = max(rows, key=lambda r: abs(float(r["ratio_common"])
                                        / float(r["ratio_all"]) - 1.0))
    return [("RatioCommonMin", at_least(min(common), "%.2g")),
            ("RatioCommonMax", at_most(max(common), "%.2g")),
            ("RatioCommonShiftRelease", LABELS[worst["release"]]),
            ("RatioCommonShiftAll", "%.3g" % float(worst["ratio_all"])),
            ("RatioCommonShiftCommon", "%.3g" % float(worst["ratio_common"])),
            ("RatioCommonShiftSeg", worst["n_seg_rule"]),
            ("RatioCommonShiftRpe", worst["n_rpe_rule"])]


def sweep_macros(rows):
    """규약 훑기 묶음표의 (이름, 값) 들. 본문이 부르는 변형만 꺼낸다."""
    out = []
    for r in rows:
        rel, rule = r["release"], r["rule"]
        tag = SWEEP.get((r["axis"], r["param"]))
        if rel not in NAMES or rule not in RULES or tag is None:
            continue
        for key, unit in SWEEP_COLS:
            out.append(("S%s%s%s%s" % (NAMES[rel], RULES[rule], tag, unit),
                        "%.3g" % float(r[key])))
    return (out + phase_ratio_macros(rows) + decim_ratio_macros(rows)
            + offset_linearity_macros(rows)
            # 그림은 공식 구간 규칙 한 쪽만 그린다. 그 그림을 설명하는 글이
            # 두 규칙을 합쳐 센 이름·폭을 쓰면, 그림에 선이 없는 릴리스가
            # 설명에 이름을 올린다.
            + offset_linearity_macros(rows, rule="seg",
                                      prefix="OffsetLinearSeg")
            + method_over_rate_macros(rows))


def public_commit_macros(out_dir):
    """공개 저장소의 어느 커밋을 가리키나. 본문이 주소만 적지 않게 한다.

    분기 주소는 그 뒤의 커밋까지 가리켜서, 심사가 읽은 나무와 독자가 여는
    나무가 달라진다(31 회차 권고 1). 커밋은 공개 나무를 밀 때 같이 고치는
    파일 하나에 둔다 — 본문에 손으로 적으면 미루고 잊는다.
    """
    path = os.path.join(out_dir, "public_commit.txt")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [("PublicCommit", f.read().strip())]


def moving_threshold():
    """키티-360 시간축 진단이 쌍을 고르는 이동량 문턱. 그 코드의 상수다."""
    from kitti360_time_axis import MOVING_M

    return float(MOVING_M)


def fast_threshold():
    """핏서티엠 각도 단위 탐침이 걸음을 고르는 속력 문턱. 그 코드의 상수다."""
    from pit30m_angle_unit_probe import FAST

    return float(FAST)


def sweep_ratio_macros():
    """주기를 낮추는 비율 집합. 훑기 코드의 상수를 그대로 읽는다.

    본문이 "주기를 낮춘 쪽" 의 최댓값끼리 맞대 비교하는데, 그 최댓값이 어느
    집합 위에서의 최댓값인지가 적혀 있지 않으면 읽는 쪽이 연속 훑기로 읽는다.
    유한한 셋이다 — 손으로 옮겨 적으면 상수를 바꿀 때 본문만 옛 값으로 남는다.
    """
    from protocol_sweep import RATIOS

    v = ["$%g$" % r for r in RATIOS]
    return [("SweepRatios", ", ".join(v[:-1]) + " and " + v[-1])]


def method_over_rate_macros(rows):
    """복원 방식의 몫이 주기의 몫보다 큰 줄이 몇인가.

    본문 결론이 "가장 큰 몫은 주기가 아니라 복원 방식" 이라고 적는다. 그
    말은 키티 한 줄에서만 보였는데 결론은 릴리스 전체를 가리킨다. 두 축을
    줄마다 맞대 세어, 몇 줄에서 그런지를 수로 적게 한다. 맞대는 것은 같은
    주기에서 방식만 바꾼 몫과, 방식을 구면보간으로 고정한 채 주기만 줄인
    몫이다 — 최근접 쪽은 두 몫이 겹쳐 있어 비교 상대가 못 된다.
    """
    by = {}
    for r in rows:
        if r["axis"] not in ("method", "rate"):
            continue
        if r["axis"] == "rate" and not r["param"].endswith("_slerp"):
            continue
        v = r.get("D_trans_pct")
        if not v:
            continue
        cur = by.setdefault((r["release"], r["rule"]), {})
        cur[r["axis"]] = max(cur.get(r["axis"], 0.0), float(v))
    both = [v for v in by.values() if "method" in v and "rate" in v]
    if not both:
        return []
    return [("MethodOverRateRows",
             str(sum(1 for v in both if v["method"] > v["rate"]))),
            ("MethodOverRateTotal", str(len(both)))]


def decim_ratio_macros(rows, release="kitti"):
    """두 솎기 위상이 몇 배 갈리나. 규칙마다 따로 낸다.

    "쉰일곱 배" 처럼 본문에 적어 두면 구간 집합이 바뀌어도 그 말이 남는다.
    실제로 한 번 남았다 — 구간 시작점이 공식 격자에서 한 칸 밀려 있던 동안은
    두 위상의 값이 서로 바뀐 자리에 있었고, 본문의 배율은 그대로였다.
    """
    by = {}
    for r in rows:
        if r["release"] != release or r["axis"] != "decim2":
            continue
        by.setdefault(r["rule"], {})[r["param"]] = r
    out = []
    for rule, tag in (("seg", "Seg"), ("rpe1m", "Rpe")):
        pair = by.get(rule, {})
        if set(pair) != {"even", "odd"}:
            continue
        for key, unit in SWEEP_COLS:
            a, b = float(pair["even"][key]), float(pair["odd"][key])
            lo, hi = min(a, b), max(a, b)
            if lo <= 0:
                continue
            out.append(("DecimRatio%s%s" % (tag, unit), "%.3g" % (hi / lo)))
    return out


def phase_ratio_macros(rows):
    """위상 0.1 에서 0.9 로 옮겼을 때 D 가 몇 배가 되나.

    "아홉 배" 라고 본문에 적어 두면 선형성이 깨지는 릴리스가 생겨도 그 문장은
    그대로 남는다. 잰 범위를 적는다.
    """
    by = {}
    for r in rows:
        if r["axis"] != "shift" or r["param"] not in ("0.1", "0.9"):
            continue
        by.setdefault((r["release"], r["rule"]), {})[r["param"]] = r
    ratios = []
    for pair in by.values():
        if len(pair) != 2:
            continue
        for key in ("D_rot_deg_per_100m", "D_trans_pct"):
            lo = float(pair["0.1"][key])
            if lo > 0:
                ratios.append(float(pair["0.9"][key]) / lo)
    if not ratios:
        return []
    return [("PhaseRatioMin", at_least(min(ratios))),
            ("PhaseRatioMax", at_most(max(ratios)))]


def offset_linearity_macros(rows, rule=None, prefix="OffsetLinear"):
    """절대 어긋남에 비례하는가를 릴리스마다 따로 적는다.

    rule 을 주면 그 구간 규칙만 센다. 그림을 설명하는 글이 쓰는 쪽이다 —
    그림은 규칙 하나만 그리므로, 합쳐 센 이름을 그 글에 쓰면 선이 없는
    릴리스가 설명에 들어간다.

    순위표가 있는 두 칸에서 비례가 선다고 일곱 릴리스 전부가 그렇다고 적으면
    안 된다. 그 문장은 한 줄로 쓰기 쉬운 만큼 실제로 틀리는 자리가 생긴다 —
    같은 릴리스 안에서 표본 간격이 하나가 아니면 밀리초와 위상이 서로 옮겨
    가지 않는다. 비례가 서는 릴리스와 안 서는 릴리스를 나눠 세고, 안 서는
    쪽은 이름과 벗어난 폭을 그대로 낸다.

    벗어난 폭은 원점을 지나는 직선에서의 상대 편차이고, 같은 계산을 순위표
    칸에도 쓴다(offset_tolerance.slope).
    """
    by = {}
    for r in rows:
        if r["axis"] != "tau" or not r["param"].endswith("ms"):
            continue
        if rule is not None and r["rule"] != rule:
            continue
        ms = float(r["param"][:-2])
        for key, axis in (("D_rot_deg_per_100m", "rot"),
                          ("D_trans_pct", "trans")):
            by.setdefault((r["release"], r["rule"], axis), []).append(
                (ms, float(r[key])))
    devs = {}
    for k, pts in by.items():
        if len(pts) < 2:
            continue
        devs[k] = slope(sorted(pts))[1]
    if not devs:
        return []
    # 경계는 1 퍼센트다. 측정 자체의 자리수가 다섯이라 그보다 한참 위이고,
    # 아래 무리와 위 무리가 실제로 그 사이에서 갈린다.
    tight = sorted({k[0] for k, d in devs.items() if d <= 0.01}
                   - {k[0] for k, d in devs.items() if d > 0.01})
    loose = sorted({k[0] for k, d in devs.items() if d > 0.01})
    # 본문은 "그 폭 안으로" 와 "그 폭까지 간다" 로 읽는다. 둘 다 잰 수가
    # 적힌 수 안에 남도록 바깥으로 깎는다.
    out = [(prefix + "Tight", ", ".join(LABELS[r] for r in tight)),
           (prefix + "TightCount", str(len(tight))),
           (prefix + "TightMaxPct", at_most(100.0 * max(
               [d for k, d in devs.items() if k[0] in tight] or [0.0]))),
           (prefix + "Loose", ", ".join(LABELS[r] for r in loose)),
           (prefix + "LooseCount", str(len(loose)))]
    if loose:
        hot = max(((d, k) for k, d in devs.items() if k[0] in loose),
                  key=lambda p: p[0])
        out += [(prefix + "LooseWorst", LABELS[hot[1][0]]),
                (prefix + "LooseMaxPct", at_most(100.0 * hot[0]))]
    return out


def board_macros(rows):
    """밀리초 환산 요약의 (이름, 값) 들.

    어긋남마다 조건을 만족하는 쌍 수, 그리고 칸마다 한 번씩 쌍 수·인접 쌍 수·
    1 ms 미만에서 덮이는 인접 쌍 수·표 전체 폭을 지우는 어긋남.
    """
    out, seen = [], set()
    for r in rows:
        tag = BOARDS.get((r["board"], r["metric"]))
        ms = TAU_NAMES.get(r["tau_ms"])
        if tag is None or ms is None:
            continue
        # 묶는 꼴이 확인된 순위표의 수만 본문으로 내보낸다. 미확인인 자리의
        # 쌍 수는 "평균 꼴이라고 보면" 이라는 가정이 붙은 수라, 매크로로
        # 나가면 본문이 그 가정을 안 적은 채 인증처럼 쓴다.
        if r["aggregation"] != "mean":
            continue
        out.append(("%sHeld%s" % (tag, ms), r["n_held"]))
        if tag in seen:
            continue
        seen.add(tag)
        out += [("%sPairs" % tag, r["n_pairs"]),
                ("%sAdjacent" % tag, r["n_adjacent"]),
                # 본문이 "이웃 쌍 수 + 1" 을 적으면 읽는 쪽이 그 덧셈을 해야
                # 한다. 항목 수를 그대로 낸다.
                ("%sEntries" % tag, str(int(r["n_adjacent"]) + 1)),
                # 두 칸을 나눠 낸다. 어긋남이 0 이어도 찍는 자리수만으로 안
                # 갈리는 쌍과, 시간 때문에 1 ms 미만에서 덮이는 쌍은 다른
                # 것을 재는 수다. 한 칸에 더해 놓으면 시간 민감도가 그만큼
                # 부풀어 보인다.
                ("%sAdjacentDead" % tag, r["n_adjacent_dead_at_zero"]),
                ("%sAdjacentCovered" % tag,
                 r["n_adjacent_covered_below_1ms"]),
                # 아래 넷은 결과표가 잰 값을 그대로 들고 있어서 여기서
                # 자리수를 줄인다. 저장 단계에서 줄이면 본문이 상한으로
                # 부르는 자리의 바깥 깎기가 이미 줄어든 수 위에서 돌아
                # 보장이 앞단에서 깨진다.
                # 칸의 양 끝이라 본문이 범위로 읽는다. 바깥으로 깎아야 적은
                # 범위가 순위표에 실제로 있는 수를 품는다.
                ("%sMin" % tag, at_least(r["col_min"])),
                ("%sMax" % tag, at_most(r["col_max"])),
                # 훑은 범위를 넘어선 칸은 그 경계를 값처럼 적지 않는다.
                # 50 ms 까지 재고 거기서도 안 지워진 칸에 "50" 만 적으면
                # 읽는 쪽은 거기서 지워진다고 읽는다 — 반대 뜻이다.
                ("%sSpanMs" % tag,
                 ("$>$%s" if r["span_tolerance_where"] == "above" else "%s")
                 % shown(r["span_tolerance_ms"], "%.4g")),
                ("%sDisplay" % tag, shown(r["display_term"], "%.4g")),
                # 비례에서 벗어난 폭의 최댓값이라 본문이 상한으로 읽는다.
                # 백분율로 낸다 — 본문의 바로 앞 문장이 같은 종류의 이탈을
                # 백분율로 적어서, 여기만 무차원 비로 두면 두 수가 만 배
                # 어긋나 보인다.
                ("%sLinearity" % tag,
                 at_most(100.0 * float(r["linearity_max_rel_dev"])))]
    return out


def median(values):
    s = sorted(values)
    n = len(s)
    if not n:
        raise ValueError("빈 목록의 중앙값은 없다")
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def cancellation_macros(rows):
    """상한이 얼마나 보수적인가의 (이름, 값) 들.

    자료는 조건 하나에 시퀀스 여러 줄이다. 본문이 부르는 것은 조건 단위라
    시퀀스를 먼저 중앙값으로 접고 그 다음에 조건끼리 센다. 줄 단위로 세면
    같은 조건의 꼬리가 조건 수처럼 보인다.
    """
    by = {}
    for r in rows:
        key = (r["score_deg_per_100m"], r["gap_deg_per_100m"], r["rho"])
        by.setdefault(key, []).append(r)
    worst = [max(float(x["flip_rate"]) for x in v) for v in by.values()]
    frac = [median([float(x["frac_of_twoD_median"]) for x in v])
            for v in by.values()]
    flip = [median([float(x["flip_rate"]) for x in v]) for v in by.values()]
    # 접기 전의 두 수. 조건 범위만 내면 그것이 시행 전체의 범위처럼 읽힌다 —
    # 한 시퀀스의 중앙값과 한 시행의 최댓값이 그 범위 위로 올라간다.
    seq_frac = max(float(r["frac_of_twoD_median"]) for r in rows)
    draw_frac = max(float(r["shift_max"]) / float(r["twoD_deg_per_100m"])
                    for r in rows if float(r["twoD_deg_per_100m"]) > 0)
    # 조건 격자의 축을 그대로 적는다. 조건 수만 적으면 그 수가 무엇을
    # 곱한 것인지가 본문에서 사라진다.
    def axis(col):
        return ", ".join(sorted(set(r[col] for r in rows), key=float))

    return [("CancelConditions", str(len(by))),
            ("CancelScoreList", axis("score_deg_per_100m")),
            ("CancelGapList", axis("gap_deg_per_100m")),
            ("CancelRhoList", axis("rho")),
            ("CancelFracSeqMax", at_most(100 * seq_frac, "%.3g")),
            ("CancelFracDrawMax", at_most(100 * draw_frac, "%.3g")),
            ("CancelFracMin", at_least(100 * min(frac), "%.2g")),
            ("CancelFracMax", at_most(100 * max(frac), "%.2g")),
            ("CancelFlipNone", str(sum(1 for x in flip if x == 0))),
            ("CancelFlipMax", at_most(100 * max(flip), "%.2g")),
            # 중앙값만 내면 시퀀스 소수에서만 뒤집히는 조건이 0 으로 접힌다.
            # 접히기 전의 수를 같이 내야 본문이 그 꼬리를 숨기지 않는다.
            ("CancelFlipAnyConditions",
             str(sum(1 for x in worst if x > 0))),
            ("CancelFlipWorst", at_most(100 * max(worst), "%.3g"))]


def official_macros(rows):
    """공식 이진파일을 직접 두드려 본 결과의 (이름, 값) 들.

    둘을 가른다. 답을 아는 입력(같은 궤적·전역 회전·점증 드리프트)에 대한
    탐침은 공식 구현 자체의 성질이고, 우리 이식과의 상대차는 두 구현이
    얼마나 벌어지나다. 한 수로 묶으면 어느 쪽 이야기인지 사라진다.
    """
    rel = [float(r["relative_difference"]) for r in rows]
    drift = [float(r["probe_drift_relative_error"]) for r in rows]
    same = [float(r["probe_identical_rad_per_m"]) for r in rows]
    rigid = [float(r["probe_global_rotation_rad_per_m"]) for r in rows]
    out = []
    # 밀리초 어긋남에서의 재대조는 나중에 더해졌다. 열이 없으면 그 매크로만
    # 안 만든다 — 없는 값을 0 으로 채우면 "둘이 완전히 같다" 로 읽힌다.
    for ms, tag in (("1", "OneMs"), ("5", "FiveMs")):
        key = "relative_difference_%sms" % ms
        if rows and key in rows[0]:
            v = 100 * max(float(r[key]) for r in rows)
            # 백분율이 두 자리를 넘으면 지수꼴이 본문에 그대로 실린다.
            out.append(("OfficialSpread%s" % tag,
                        at_most(v, "%.0f" if v >= 10 else "%.2g")))
        # 같은 어긋남에서 병진 칸과 단정밀도 흉내도 같이 맞댄다. 회전만
        # 대조하면 다른 칸에서 두 구현이 갈리는지, 회전의 어긋남이 산술
        # 때문인지 아무 말도 못 한다.
        for col, name, fmt in (("relative_difference_trans_%sms" % ms,
                                "OfficialTransSpread%s", "%.2g"),
                               ("relative_difference_emulated_%sms" % ms,
                                "OfficialEmulatedSpread%s", "%.3g")):
            if rows and col in rows[0]:
                # 둘 다 그 열의 최댓값이고 본문이 "최대 … 퍼센트" 로 읽는다.
                # 위로 깎지 않으면 실측 0.00295 가 0.0029 로 찍혀 그 문장이
                # 자기가 잰 가장 큰 어긋남을 품지 못한다.
                out.append((name % tag, at_most(
                    100 * max(float(r[col]) for r in rows), fmt)))
    # 누산형만 바꾼 차. 상대차와 찍는 한 칸 대비를 둘 다 낸다 — 상대차만으로는
    # 그 차이가 표에 보이는지를 말할 수 없다.
    for col, name, fmt in (
            ("accum_rel_diff_rot", "AccumRelRot", "%.2g"),
            ("accum_rel_diff_trans", "AccumRelTrans", "%.2g"),
            ("accum_diff_over_display_rot", "AccumDisplayRot", "%.2g"),
            ("accum_diff_over_display_trans", "AccumDisplayTrans", "%.2g")):
        if rows and col in rows[0]:
            # 본문이 "최대 …" 로 부른다. 위로 깎아 잰 값을 품게 한다.
            out.append((name, at_most(max(float(r[col]) for r in rows), fmt)))
    if rows and "relative_difference_vs_emulated" in rows[0]:
        out.append(("OfficialEmulatedMax", at_most(100 * max(
            float(r["relative_difference_vs_emulated"])
            for r in rows), "%.2g")))
    # 영변위 탐침. 누른 입력과 안 누른 입력을 갈라 적는다 — 한 수로 묶으면
    # 비직교 이탈이 남긴 것이 산술 차이로 읽힌다.
    for key, tag in (("null_official_projected_rad_per_m", "NullProjected"),
                     ("null_official_raw_rad_per_m", "NullRaw")):
        if rows and key in rows[0]:
            # 시퀀스를 통틀어 가장 큰 잔차다. 본문이 그 수 하나로 "탐침이
            # 이만큼 돌려준다" 를 말하므로 상한으로 읽힌다 — 위로 깎는다.
            out.append(("Official%s" % tag,
                        at_most(max(float(r[key]) for r in rows), "%.3g")))
    return out + [
            ("OfficialSpreadMin", at_least(100 * min(rel), "%.2g")),
            ("OfficialSpreadMax", at_most(100 * max(rel), "%.2g")),
            ("OfficialSequences", str(len(rows))),
            ("OfficialProbeIdentical", at_most(max(same), "%.3g")),
            ("OfficialProbeRigid", at_most(max(rigid), "%.3g")),
            ("OfficialDriftMedian", "%.2g" % (100 * median(drift))),
            ("OfficialDriftMax", at_most(100 * max(drift), "%.2g")),
            # 그 탐침이 추정에 실은 프레임당 각. 본문이 이 값을 적어야
            # 재현이 선다 — 공식 프로그램의 상대오차가 각의 크기에 강하게
            # 걸려서(이 자료에서 1e-7 이면 100 %, 1e-5 면 1.2 %) 다른 값을
            # 넣으면 본문 수가 안 나온다. 코드 상수에서 바로 낸다.
            ("OfficialDriftEps", "%g" % drift_eps())]


def drift_eps():
    """공식 비교의 드리프트 탐침과 경계 사례가 쓰는 프레임당 각.

    경계 사례는 그 값을 기본값으로 가져다 쓰므로 출처가 하나다. 숫자를
    본문에 손으로 적으면 상수가 바뀔 때 문장만 낡는다.
    """
    from kitti_official_binary_check import DRIFT_EPS

    return DRIFT_EPS


# 시작 프레임 분해. 두 위상을 "시작 프레임을 품는 쪽" 과 "안 품는 쪽" 으로
# 부른다 — 짝·홀은 그 자료에서만 뜻이 있는 이름이다.
DECOMP = (("start_pos_err_m", "StartPos"), ("start_rot_err_deg", "StartRot"),
          ("end_pos_err_m", "EndPos"), ("D_trans_pct", "Trans"),
          ("D_trans_pct_end_even", "TransEndEven"),
          ("D_trans_pct_end_odd", "TransEndOdd"),
          ("D_rot_deg_per_100m", "Rot"))


def decomp_macros(rows):
    """구간 시작 자세 분해의 (이름, 값) 들."""
    out, trans = [], {}
    for r in rows:
        side = ("Kept" if r["start_parity_retained"] == "1" else "Lost")
        trans[side] = float(r["D_trans_pct"])
        for key, tag in DECOMP:
            out.append(("Decomp%s%s" % (side, tag), "%.3g" % float(r[key])))
    # 끝점 몫이 시작 자세 몫에 비해 얼마나 작은가. 시작을 그대로 쓰는
    # 위상에서는 남는 것이 끝점뿐이라 그 비가 끝점 몫의 크기다.
    if trans.get("Lost"):
        out.append(("DecompEndpointRatio",
                    "%d" % round(trans["Lost"] / trans["Kept"])))
    return out


def floor_macros(rows):
    """직교 이탈과 그 바닥의 (이름, 값) 들. 행렬로 싣는 릴리스만 부른다.

    FloorExactWorst 는 각을 그대로 싣는 릴리스 — 공개 행렬이 이미 직교인 쪽 —
    의 바닥 중 가장 큰 것이다. 그 집합을 먼저 고르고 그 안에서 최대를 잡는다.
    거르는 줄을 0 으로 바꿔 최대에 섞으면 집합이 비었을 때 0 이 그대로 답이
    되어, "바닥이 없다" 가 재지 않은 채로 본문에 적힌다.
    """
    out, seen = [], set()
    exact = [float(r["floor_deg_per_100m"]) for r in rows
             if float(r["ortho_dev_max"]) < EXACT_DEV]
    if not exact:
        sys.exit("직교 이탈이 %g 아래인 줄이 하나도 없다 — FloorExactWorst "
                 "는 그 집합의 최대라서 잴 수 없다" % EXACT_DEV)
    for r in rows:
        rel, rule = r["release"], r["rule"]
        if rel not in NAMES or rule not in RULES:
            continue
        out.append(("Floor%s%s" % (NAMES[rel], RULES[rule]),
                    "%.3g" % float(r["floor_deg_per_100m"])))
        if rel not in seen:
            seen.add(rel)
            # 본문이 "가장 큰 이탈은 이만큼" 으로 읽는 자리다. 맨 반올림이면
            # 잰 최대가 적은 수보다 커질 수 있어 바깥으로 깎는다.
            out.append(("Dev%s" % NAMES[rel],
                        at_most(float(r["ortho_dev_max"]))))
    out.append(("FloorExactWorst", at_most(max(exact))))
    return out


def known_macros(rows):
    """답을 아는 궤적 점검의 (이름, 값) 들.

    단위를 본문과 맞춘다 — 회전은 도/100 m, 병진은 퍼센트다. 답이 0 인 칸과
    0 이 아닌 칸을 따로 적는다. 0 인 칸은 절대 잔차로, 0 이 아닌 칸은 상대차로
    읽어야 뜻이 선다.
    """
    zero = [r for r in rows if float(r["expected"]) == 0.0]
    rot = [float(r["absolute_difference"]) for r in zero
           if not r["quantity"].startswith("t_")]
    trans = [float(r["absolute_difference"]) for r in zero
             if r["quantity"].startswith("t_")]
    rel = [float(r["relative_difference"]) for r in rows
           if r["relative_difference"]]
    return [("KnownZeroRot",
             at_most(max(rot) * 180.0 / math.pi * 100.0, "%.2g")),
            ("KnownZeroTrans", at_most(max(trans) * 100.0, "%.2g")),
            ("KnownNonzeroRel", at_most(max(rel), "%.2g")),
            # 0 이 아닌 칸의 수를 같이 적는다. 그 수가 하나면 "답이 0 인 칸만
            # 통과하는 구현" 을 한 축에서만 거른 것이고, 본문이 그걸 전부로
            # 읽으면 안 된다.
            ("KnownNonzeroCells", str(len(rel))),
            ("KnownZeroCells", str(len(zero))),
            ("KnownCells", str(len(rows)))]


def official_floor_macros(rows):
    """공식 이진파일의 수치 바닥을 흔들림 크기별로 적은 (이름, 값) 들.

    한 수로 요약하지 않는다. 이 프로그램의 오차는 재는 양의 크기에 따라
    달라서, 큰 쪽에서 맞는다는 사실과 작은 쪽에서 통째로 사라진다는 사실이
    둘 다 필요하다 — 전자는 순위표 값에, 후자는 밀리초 어긋남에 해당한다.
    """
    vanish = [r for r in rows if float(r["official_deg_per_m"]) == 0.0]
    big = [r for r in rows if float(r["eps_rad_per_frame"]) >= 1e-3]
    mid = [r for r in rows if float(r["eps_rad_per_frame"]) == 1e-6]
    near = [r for r in rows if float(r["eps_rad_per_frame"]) == 1e-5]
    exp = int(round(-math.log10(max(float(r["eps_rad_per_frame"])
                                    for r in vanish))))
    # 어느 시퀀스에서 쟀는지가 그 수의 범위다. 탐침은 한 시퀀스에 한 꼴의
    # 흔들림만 넣는다 — 본문이 그 조건을 적을 수 있게 이름을 같이 낸다.
    seqs = sorted({r["sequence"] for r in rows})
    return [("OfficialFloorZeroExp", str(exp)),
            ("OfficialFloorSequence", seqs[0] if len(seqs) == 1 else
             "%s--%s" % (seqs[0], seqs[-1])),
            # 넷 다 그 구간에서 가장 큰 값이고 본문이 그 크기로 읽는다.
            # 최댓값이라 위로 깎는다.
            ("OfficialFloorMid", at_most(100 * max(
                float(r["relative_error"]) for r in mid), "%.2g")),
            ("OfficialFloorBoardValue", at_most(max(
                float(r["official_deg_per_m"]) for r in near), "%.3g")),
            ("OfficialFloorBoard", at_most(100 * max(
                float(r["relative_error"]) for r in near), "%.2g")),
            ("OfficialFloorLarge", at_most(max(
                float(r["relative_error"]) for r in big), "%.2g"))]


def method_scale_macros(rows):
    """순위표 크기의 추정에서 기준을 밀었을 때의 (이름, 값) 들.

    세 수를 따로 적는다. 상한 대비는 그 상한이 얼마나 느슨한가이고, 찍는 한 칸
    대비는 움직임의 크기이고, 찍히는 값이 바뀐 칸은 표가 실제로 달라지는가다.
    셋을 한 수로 합치면 어느 물음에도 답이 안 된다 — 특히 한 칸보다 작은
    움직임도 반올림 경계를 넘으면 찍히는 값을 바꾼다.

    상한 대비는 고정 구간에서 잰 변화로만 낸다. 공식 프로그램은 기준을 밀면
    구간을 다시 고르므로 그 출력의 차이에는 재선택이 섞이고, 정리가 요구하는
    전제가 그 비교에는 안 선다.
    """
    # 드리프트 값은 시퀀스마다 고르지 않고 같은 목록을 전부에 적용한다.
    # 본문이 "순위표 칸에 맞춰 시퀀스마다 골랐다" 고 적으면 생성 절차가
    # 자료와 어긋난다 — 그래서 개수와 양 끝을 자료에서 뽑아 낸다.
    eps = sorted({float(r["eps_rad_per_frame"]) for r in rows})
    out = [("MoveCells", str(len(rows))),
           ("MoveSequences", str(len({r["sequence"] for r in rows}))),
           ("MoveEpsCount", word(len(eps))),
           ("MoveEpsMin", at_least(eps[0], "%.2g")),
           ("MoveEpsMax", at_most(eps[-1], "%.2g")),
           ("MoveSegmentsKeptCount", str(len(
               [r for r in rows if r["segments_kept"] == "1"])))]
    # 구간 선택은 기준 경로만 보고 돌아가므로 추정의 드리프트와 무관하다.
    # 칸 수(시퀀스 x 드리프트 x 어긋남)로 세면 같은 선택이 드리프트 수만큼
    # 거듭 세어져 독립 사례가 부풀려진다. 조건은 (시퀀스, 어긋남) 이다.
    sel = {}
    for r in rows:
        sel.setdefault((r["sequence"], r["offset_ms"]), set()).add(
            r["segments_kept"])
    mixed = sorted(k for k, v in sel.items() if len(v) > 1)
    if mixed:
        sys.exit("구간 선택이 드리프트에 따라 갈린 조건이 있다: %s" % (mixed,))
    out += [("MoveSelectionConditions", str(len(sel))),
            ("MoveSelectionKeptConditions", str(len(
                [k for k, v in sel.items() if v == {"1"}])))]
    # 상한을 넘은 칸은 축을 가리지 않고 센다. 한 축만 세면 다른 축에서 깨진
    # 것이 0 으로 적힌다. 상한이 0 이라 비가 안 서는 칸은 빈 칸으로 오고,
    # 그 칸은 "넘었다" 로도 "안 넘었다" 로도 안 센다 — 센 칸 수를 따로 적어
    # 두 수가 같은 모집단을 말하는지 본문에서 대조할 수 있게 한다.
    ratios = [(r.get("rot_fixed_moved_over_bound", ""),
               r.get("trans_fixed_moved_over_bound", "")) for r in rows]
    rated = [(a, b) for a, b in ratios if a != "" and b != ""]
    out.append(("MoveOverBoundCount", str(len(
        [1 for a, b in rated if float(a) > 1.0 or float(b) > 1.0]))))
    out.append(("MoveOverBoundRatedCells", str(len(rated))))
    for axis, tag, col in (("rot", "", "rot_score_deg_per_m"),
                           ("trans", "Trans", "trans_score_pct")):
        scores = [float(r[col]) for r in rows]
        rated_axis = [float(r["%s_fixed_moved_over_bound" % axis])
                      for r in rows
                      if r.get("%s_fixed_moved_over_bound" % axis, "") != ""]
        if not rated_axis:
            sys.exit("%s 축에서 상한 대비를 낼 수 있는 칸이 하나도 없다" % axis)
        out += [("Move%sScoreMin" % tag, at_least(min(scores), "%.4f")),
                ("Move%sScoreMax" % tag, at_most(max(scores), "%.4f")),
                ("Move%sOverBoundMax" % tag,
                 at_most(100 * max(rated_axis), "%.2g")),
                # 찍는 한 칸을 넘은 칸은 세어서 적는다. 최댓값만 적으면
                # "한 칸을 넘는 일이 있다" 가 몇 번인지가 안 남는다.
                ("Move%sOverDisplayCount" % tag, str(len(
                    [r for r in rows
                     if float(r["%s_moved_over_display" % axis]) > 1.0]))),
                # 찍히는 값이 실제로 바뀐 칸. 위 수와 다른 물음이다.
                ("Move%sShownChangedCount" % tag, str(len(
                    [r for r in rows
                     if r["%s_shown_changed" % axis] == "1"])))]
        # 안 잰 어긋남은 매크로를 안 만든다. 0 으로 채우면 "안 움직였다" 로
        # 읽힌다.
        for ms, when in ((1.0, "One"), (5.0, "Five")):
            sub = [r for r in rows if float(r["offset_ms"]) == ms]
            if sub:
                out += [("Move%s%sOverDisplayMax" % (tag, when),
                         at_most(max(float(r["%s_moved_over_display" % axis])
                                     for r in sub), "%.2g")),
                        ("Move%s%sShownChangedCount" % (tag, when), str(len(
                            [r for r in sub
                             if r["%s_shown_changed" % axis] == "1"])))]
    for ms, when in ((1.0, "One"), (5.0, "Five")):
        sub = [r for r in rows if float(r["offset_ms"]) == ms]
        if sub:
            # 이름에 Any 를 넣는다. 뺀 이름은 위 축별 고리가 회전 자리에
            # 빈 꼬리표로 이미 쓰고 있어서 조용히 덮어쓴다.
            # 세는 단위는 시퀀스·드리프트 조건이고 표시 점수 칸이 아니다 —
            # 한 줄이 회전과 병진 두 칸을 들고 있어서 칸 수는 이 수의 두
            # 배다(1 ms 에서 조건 2/33 대 칸 2/66, 5 ms 에서 15/33 대 18/66).
            # 이름의 Cells 를 믿고 본문에 "칸" 이라고 적으면 단위가 갈린다.
            out += [("Move%sCells" % when, str(len(sub))),
                    ("Move%sAnyShownChangedCount" % when, str(len(
                        [r for r in sub
                         if r["rot_shown_changed"] == "1"
                         or r["trans_shown_changed"] == "1"])))]
    return out


def bound_check_macros(rows):
    """적대 조건 훑기의 (이름, 값) 들.

    사례 수를 본문에 손으로 적으면 훑기를 넓혔을 때 그 수만 옛 값으로 남는다.
    """
    # 빈 칸은 D 가 0 에 붙어 비를 못 내는 줄이다. 0 으로 읽으면 가장 느슨한
    # 조건으로 세어진다.
    tight = [float(r["tightness"]) for r in rows if r["tightness"] != ""]
    return [("BoundCases", str(len(rows))),
            ("BoundViolations", str(len(
                [r for r in rows if int(r["violated"])]))),
            ("BoundTightnessMax", at_most(max(tight), "%.4g"))]


def window_counts_macros(rows):
    """두 공통 창이 공식 전체 집합에서 덜어낸 구간의 (이름, 값) 들.

    창마다 따로 적는다. 주표와 규약 훑기는 변형이 달라 창이 다르고, 한 수로
    묶으면 어느 실험의 제외량인지가 사라진다.
    """
    pct = [float(r["excluded_pct"]) for r in rows]
    # 두 최대가 어느 줄에 있는지를 같이 낸다. 본문이 "가장 짧은 릴리스에서
    # 가장 큰 비율이 빠진다" 처럼 까닭을 적으면 그 문장은 측정을 다시 돌려
    # 줄이 바뀌어도 그대로 남는다. 줄 이름을 자료에서 가져오면 안 남는다.
    top_pct = max(rows, key=lambda r: float(r["excluded_pct"]))
    top_cnt = max(rows, key=lambda r: int(r["n_excluded"]))
    # 본문이 "A 와 B 사이" 로 읽는 두 끝이라 바깥쪽으로 깎는다. 가장 가까운
    # 쪽으로 깎으면 그 범위가 실제로 잰 줄을 배제할 수 있다.
    out = [("WindowExcludedPctMax", at_most(max(pct))),
           ("WindowExcludedPctMin", at_least(min(pct))),
           ("WindowExcludedMax", str(max(
               int(r["n_excluded"]) for r in rows))),
           ("WindowExcludedPctMaxRelease", LABELS[top_pct["release"]]),
           ("WindowExcludedPctMaxRule",
            "100--800\\,m" if top_pct["rule"] == "seg" else "1\\,m"),
           ("WindowExcludedMaxRelease", LABELS[top_cnt["release"]]),
           ("WindowExcludedMaxRule",
            "100--800\\,m" if top_cnt["rule"] == "seg" else "1\\,m")]
    # 훑기 창 열은 나중에 더해졌다. 없으면 그 매크로만 안 만든다.
    if rows and "excluded_sweep_pct" in rows[0]:
        swp = [float(r["excluded_sweep_pct"]) for r in rows]
        out += [("SweepWindowExcludedPctMax", at_most(max(swp))),
                ("SweepWindowExcludedPctMin", at_least(min(swp))),
                ("SweepWindowExcludedMax", str(max(
                    int(r["n_excluded_sweep"]) for r in rows)))]
    return out


def window_bias_macros(rows):
    """창이 공식 점수를 옮긴 양. 거의 0 인 자리라 절대차로 적는다.

    상대차로 적으면 분모가 그 시퀀스의 점수라 같은 편향이 시퀀스마다 다른
    수로 보인다. 표에 보이는지를 가르는 것은 찍는 한 칸 대비 하나다.
    """
    out = []
    for axis, tag, unit in (("rot", "Rot", "deg_per_m"),
                            ("trans", "Trans", "pct")):
        big = max(float(r["%s_abs_diff_%s" % (axis, unit)]) for r in rows)
        out += [("WindowBias%sAbsMax" % tag, at_most(big, "%.2g")),
                ("WindowBias%sOverDisplayMax" % tag, at_most(max(
                    float(r["%s_diff_over_display" % axis])
                    for r in rows), "%.2g"))]
        # 상한 표는 회전을 경로 100 m 당 각도로 적는다. 한 문장 안에서 둘을
        # 맞대려면 같은 단위여야 한다 — 백 배를 읽는 쪽에 맡기지 않는다.
        if axis == "rot":
            out.append(("WindowBiasRotAbsMaxPerHundred",
                        at_most(100.0 * big, "%.2g")))
    out.append(("WindowBiasOverDisplayCount", str(len(
        [r for r in rows
         if float(r["rot_diff_over_display"]) > 1.0
         or float(r["trans_diff_over_display"]) > 1.0]))))
    return out


def window_pooled_macros(rows):
    """모든 릴리스에서 창이 점수를 옮긴 양.

    줄 수와 릴리스 수는 자료에서 센다 — 설명에 적힌 수가 아니다. 공식
    프로그램이 있는 릴리스는 하나뿐이라 그쪽 측정만으로는 (릴리스, 규칙)
    조합 가운데 한 줄밖에 못 채운다. 여기 수는 공식과 같은 꼴을 배정밀도로
    직접 계산한 것이고, 그 계산이 공식과 같은 양을 내는지는 키티 줄에서
    따로 대조한다. 적는 자는 둘이다. 찍는 한 칸 대비는 표에 보이는지를
    가르지만 그 칸은 키티 순위표의 격자라, 순위표가 없는 릴리스에는 빌려
    쓴 자다. 점수가 키티와 다른 자릿수인 릴리스에서는 같은 움직임이 그 자로
    재면 크게도 작게도 보인다. 그래서 자기 점수로 나눈 비를 같이 적는다 —
    그쪽은 빌린 격자가 안 들어간다.
    """
    def read(r, col):
        """빈 칸은 0 으로 읽지 않는다 — "안 움직였다" 로 세어진다.

        여기 수는 전부 열아홉 줄의 최대이거나 문턱을 넘는 줄의 개수다. 결측을
        0 으로 채우면 최대는 덜 나오고 개수는 작게 나오는데, 둘 다 주장이
        느슨해지는 쪽이라 본문만 보고는 못 알아챈다.
        """
        raw = r[col]
        if not raw:
            sys.exit("%s(%s) 줄의 %s 칸이 비었다 — 창이 옮긴 양을 0 으로 "
                     "채우면 안 움직였다고 세어진다" % (r["release"],
                                                       r["rule"], col))
        return float(raw)

    def cell(r, axis, key):
        return read(r, "%s_%s_over_display" % (axis, key))

    def rel(r, axis, key):
        return read(r, "%s_%s_rel_pct" % (axis, key))

    CELLS = [(axis, key) for key in ("main", "sweep")
             for axis in ("rot", "trans")]
    AXIS_WORD = {"rot": "rotation", "trans": "translation"}
    WINDOW_WORD = {"main": "the release table's window",
                   "sweep": "the sweep's window"}

    def worst_cell(r):
        """한 칸 대비가 가장 큰 칸. (값, 축, 창) 이다.

        값만 꺼내면 그 값이 어느 축·어느 창의 것인지가 호출하는 쪽에 안 남고,
        본문이 회전 칸을 가리키면서 병진 비를 붙이는 일이 생긴다. 실제로
        생겼다 — 자기 점수 대비를 네 칸의 최대로 내는 바람에 튬 알지비디
        회전 줄에 병진 훑기 비가 붙었다.
        """
        return max((cell(r, a, k), a, k) for a, k in CELLS)

    def worst_rel_cell(r):
        return max((rel(r, a, k), a, k) for a, k in CELLS)

    def worst(r):
        return worst_cell(r)[0]

    def worst_rel(r):
        return worst_rel_cell(r)[0]

    out = [("PooledWindowRows", str(len(rows))),
           ("PooledWindowEps", rows[0]["eps_rad_per_frame"])]
    for key, tag in (("main", "Main"), ("sweep", "Sweep")):
        for axis, atag in (("rot", "Rot"), ("trans", "Trans")):
            out.append(("PooledWindow%s%sOverDisplayMax" % (tag, atag),
                        at_most(max(cell(r, axis, key) for r in rows))))

    over = [r for r in rows if worst(r) > 1.0]
    out.append(("PooledWindowOverDisplayCount", str(len(over))))
    under = [r for r in rows if worst(r) <= 1.0]
    if under:
        out.append(("PooledWindowUnderMax", at_most(max(
            worst(r) for r in under))))
    # 넘는 줄은 수 하나로 접지 않는다. 어느 릴리스의 어느 규칙인지가
    # 본문에 그대로 서야 읽는 쪽이 그 자리를 찾아갈 수 있다.
    if over:
        top = max(over, key=worst)
        _, top_axis, top_key = worst_cell(top)
        out += [("PooledWindowWorstRelease", LABELS[top["release"]]),
                ("PooledWindowWorstRule",
                 "100--800\\,m" if top["rule"] == "seg" else "1\\,m"),
                ("PooledWindowWorstMainRot",
                 "%.3g" % cell(top, "rot", "main")),
                ("PooledWindowWorstSweepRot",
                 "%.3g" % cell(top, "rot", "sweep")),
                ("PooledWindowWorstMainTrans",
                 "%.3g" % cell(top, "trans", "main")),
                ("PooledWindowWorstSweepTrans",
                 "%.3g" % cell(top, "trans", "sweep")),
                # 그 줄을 고르게 한 바로 그 칸의 비다. 네 칸의 최대가 아니다.
                ("PooledWindowWorstRelPct",
                 "%.3g" % rel(top, top_axis, top_key)),
                ("PooledWindowWorstAxis", AXIS_WORD[top_axis]),
                ("PooledWindowWorstWindow", WINDOW_WORD[top_key])]
    # 자기 점수 대비로 다시 세운 자리. 한 칸 대비로 고른 줄과 같을 이유가
    # 없고, 실제로 다르다면 "가장 크게 움직인 줄" 이 쓰는 자에 달렸다는
    # 뜻이라 그 사실 자체를 본문에 적어야 한다.
    hot = max(rows, key=worst_rel)
    hot_pct, hot_axis, hot_key = worst_rel_cell(hot)
    out += [("PooledWindowRelWorstRelease", LABELS[hot["release"]]),
            ("PooledWindowRelWorstRule",
             "100--800\\,m" if hot["rule"] == "seg" else "1\\,m"),
            # 본문이 "최대" 로 쓰는 자리라 바깥쪽으로 올린다.
            ("PooledWindowRelWorstPct", at_most(hot_pct)),
            ("PooledWindowRelWorstAxis", AXIS_WORD[hot_axis]),
            ("PooledWindowRelWorstWindow", WINDOW_WORD[hot_key]),
            # 같은 칸을 찍는 자로도 적는다. "한 칸 안" 이라고만 쓰면 0.857 이
            # 0.02 와 같은 말이 된다.
            ("PooledWindowRelWorstSteps",
             "%.3g" % cell(hot, hot_axis, hot_key))]
    return out


def tie_rule_macros(rows):
    """최근접 동률이 실제로 밟히는 릴리스 수.

    본문이 규칙을 적는 까닭은 그 자리가 실제로 밟히기 때문이다. 안 밟히면
    적을 것도 없는 세부라, 적는 문장 옆에 몇 군데서 밟히는지를 같이 둔다.
    릴리스 수로 센다 — 줄 수로 세면 같은 릴리스가 두 구간 규칙에서 두 번
    세어져서 일곱 중 몇이라는 말이 안 선다.
    """
    hit = {r["release"] for r in rows if int(r["n_index_flips"]) > 0}
    return [("TieRuleReleases", str(len(hit))),
            ("TieRuleTotalReleases", str(len({r["release"] for r in rows})))]


def angle_unit_macros(rows):
    """핏서티엠 각도 단위를 가린 결과.

    본문이 드는 수는 두 읽기의 가장 불리한 쪽이다 — 도로 읽었을 때 가장 큰
    중앙값과 라디안으로 읽었을 때 가장 작은 중앙값. 평균이나 한 기록의 값을
    들면 가장 나쁜 자리에서도 갈리는지가 안 보인다. 판정이 안 선 기록은
    세지 않는다(자세가 한 점뿐이라 이동 방향이 없는 기록이다).
    """
    judged = [r for r in rows if int(r["n_steps"]) > 0]
    if not judged:
        return []
    return [("AngleUnitLogs", str(len(judged))),
            ("AngleUnitTotalLogs", str(len(rows))),
            # 본문이 "최대 …" / "최소 …" 로 부르는 두 수라 바깥쪽으로 깎는다.
            ("AngleUnitDegWorst",
             at_most(max(float(r["median_deg_as_degree"]) for r in judged),
                     "%.2f")),
            ("AngleUnitRadBest",
             at_least(min(float(r["median_deg_as_radian"]) for r in judged),
                      "%.0f")),
            # 중앙값이 걸치는 걸음의 문턱. 이 속력을 엄격히 넘는 걸음만 든다
            # — 본문이 조건 없이 적으면 정지와 저속까지 든 중앙값으로 읽힌다.
            # 상수에서 뽑아 글자로 안 들게 한다.
            ("AngleUnitFast", "%g" % fast_threshold())]


def oxford_macros(rows):
    """상한 표 밖에 둔 릴리스 하나 — 공개 위치가 공개 속도와 어긋나는 몫.

    본문이 드는 것은 "녹화가 끊긴 자리에서 끊으면 되지 않나" 를 닫는 두
    수다. 시각이 끊긴 걸음 수와, 그 걸음들을 뺀 나머지에서 가장 큰 잔차.
    뒤쪽이 크면 끊는 자리를 시간축에서 못 찾는다는 뜻이고, 그러면 남는
    길은 우리가 문턱을 하나 고르는 것뿐이다.
    """
    if not rows:
        return []
    steps = sum(int(r["rows_published"]) - 1 for r in rows)
    return [("OxfordTraversals", str(len(rows))),
            ("OxfordSteps", group(steps)),
            ("OxfordGapSteps", str(sum(int(r["gap_steps"]) for r in rows))),
            # 본문이 "최대 … m 까지 어긋난다" 로 부르는 최댓값이라 위로
            # 깎는다. 아래로 깎으면 그 문장이 자기가 잰 가장 큰 어긋남을
            # 품지 못한다 — 2889.916 을 2889.9 로 적는 자리였다.
            ("OxfordWorstNoGap",
             at_most(max(float(r["resid_max_nogap_m"]) for r in rows),
                     "%.1f"))]


def time_axis_macros(rows):
    """키티-360 자세가 어느 시간축 위에 있는지, 틀린 축이 수를 얼마나 옮기나.

    공개 파일이 싣는 것은 프레임 번호라 축이 둘 선다 — 번호를 공개 주기로
    나눈 격자와, 따로 받는 oxts 시각이다. 가르는 수는 이동량비다(공식 간격이
    0.110 초인 자리와 이웃한 0.100 초 자리의 이동량 비). 달리는 중이면 1.10
    이어야 하고, 1.00 이면 자세가 그 시각을 안 따른다.

    옮겨가는 양은 비(시각축/격자축)로 적는다. 구간 규칙마다 방향이 달라서
    한 수로 못 묶는다 — 긴 구간에서는 조금 줄고 1 m 에서는 몇 배로 는다.
    """
    have = [r for r in rows if r["has_stamps"] == "yes"]
    none = [r for r in rows if r["has_stamps"] != "yes"]
    if not have:
        return []
    ratio = [float(r["step_ratio_median"]) for r in have]
    # 격자축 값도 시각이 있는 주행만 센다. 본문이 그 둘을 나란히 놓고
    # 비교하는데 시각축 값은 그 주행들에만 있다 — 없는 주행을 격자 쪽에만
    # 넣으면 두 수가 다른 집합 위의 값이 된다.
    jg = [float(r["speed_jitter_grid"]) for r in have]
    js = [float(r["speed_jitter_stamp"]) for r in have]
    seg = [float(r["seg_t_phase_half_ratio"]) for r in have]
    rpe = [float(r["rpe1m_t_phase_half_ratio"]) for r in have]
    return [("AxisDrivesStamped", str(len(have))),
            ("AxisDrivesTotal", str(len(rows))),
            ("AxisStampShort",
             shown(median([float(r["stamp_short_dt_s"]) for r in have]),
                   "%.3f")),
            ("AxisStampLong",
             shown(median([float(r["stamp_long_dt_s"]) for r in have]),
                   "%.3f")),
            ("AxisUnstampedDrives",
             "\\,and\\,".join(r["sequence"] for r in none)
             if len(none) == 2 else ", ".join(r["sequence"] for r in none)),
            # 1.10 이 기대값이라 양쪽으로 다 바깥쪽으로 깎는다.
            ("AxisStepRatioMin", at_least(min(ratio), "%.4f")),
            ("AxisStepRatioMax", at_most(max(ratio), "%.4f")),
            # 본문이 "at least … such pairs" 로 부르는 자리다. 정수 개수라
            # 깎을 자리가 없지만 방향을 적어 둔다 — 그래야 본문이 상한·하한으로
            # 부르는 매크로가 전부 깎는 함수를 거쳤다는 검사가 한 칸도 안 비운다.
            ("AxisStepRatioPairsMin",
             at_least(min(int(r["step_ratio_pairs"]) for r in have), "%.0f")),
            # 두 진단이 고르는 이웃쌍의 문턱. 앞 구간의 이동량이 이 값을
            # 엄격히 넘는 쌍만 센다 — 본문이 "프레임 사이" 로만 적으면 정지
            # 구간까지 든 통계로 읽힌다. 상수에서 뽑아 글자로 안 들게 한다.
            ("AxisMovingM", "%g" % moving_threshold()),
            ("AxisJitterGridMax", at_most(max(jg) * 100, "%.1f")),
            ("AxisJitterStampMin", at_least(min(js) * 100, "%.1f")),
            ("AxisSegTransDropMin", at_least((1 - max(seg)) * 100, "%.1f")),
            ("AxisSegTransDropMax", at_most((1 - min(seg)) * 100, "%.1f")),
            ("AxisRpeTransRatioMin", at_least(min(rpe), "%.2g")),
            ("AxisRpeTransRatioMax", at_most(max(rpe), "%.3g")),
            ("AxisStampGapMax",
             at_most(max(float(r["stamp_max_gap_s"]) for r in have), "%.0f"))]


def kitti_axis_macros(rows):
    """키티의 두 축이 상한을 얼마나 옮기나.

    이 릴리스도 자세를 프레임으로 색인하는데 축이 둘 열려 있다 — 시퀀스마다
    싣는 times.txt 와, 공식 프로그램이 꽂아 쓰는 0.1 초 격자다. 비는
    격자축을 시각축으로 나눈 것이라 1 보다 크면 격자 쪽이 더 큰 상한이다.

    1 m 규칙만 수로 적고 구간 규칙은 1 에서 떨어진 폭으로 적는다. 구간
    규칙은 양쪽으로 다 조금씩 갈려서 한쪽 끝만 들면 방향이 있는 것처럼
    읽힌다.
    """
    if not rows:
        return []
    rpe = [float(r["rpe1m_t_phase_half_ratio"]) for r in rows]
    seg = [abs(1 - float(r["seg_t_phase_half_ratio"])) for r in rows]
    return [("AxisKittiSeqs", str(len(rows))),
            ("AxisKittiRpeRatioMax", at_most(max(rpe), "%.2f")),
            ("AxisKittiSegShiftMax", at_most(max(seg) * 100, "%.1f"))]


def official_boundary_macros(rows):
    """공식 이진파일로 잰 산술 경계. 찍히는 값의 변화가 D 를 넘나."""
    if not rows:
        return []
    ratio = [float(r["change_over_D"]) for r in rows]
    out = [("OfficialBoundaryCases", str(len(rows))),
           ("OfficialBoundaryWorst", at_most(max(ratio), "%.3g"))]
    # 초과분은 절대 단위로 낸다. 전에는 이것을 "출력 격자 한 칸" 으로
    # 나눠 1 보다 작은 수를 적었는데, 그 분모가 0 바로 위 출력 수준이라
    # 가장 거친 칸이었다. 관측된 수준들은 등간격이 아니고(가장 좁은 칸은
    # 그 분모의 1/3 쯤이다), 순위표 표시 간격과도 다른 양이다. 셋을 한
    # 칸으로 묶으면 안 된다 — 프로그램 안의 단정밀도 셈에서 생기는 오차와
    # 순위표가 값을 적을 때 하는 반올림은 다른 것이다.
    excess = max(float(r["change_minus_D"]) for r in rows)
    out.append(("OfficialBoundaryExcessRad", at_most(excess, "%.3g")))
    out.append(("OfficialBoundaryExcessDeg",
                at_most(excess * 180.0 / math.pi, "%.3g")))
    # 이 사례가 불리한 까닭은 구간이 적어 반올림이 안 묻히는 것이다. 그
    # 수를 본문이 불러 쓸 수 있게 낸다 — 안 적으면 합성 구성이 실제
    # 시퀀스의 시간 이동 결과처럼 읽힌다.
    seg = {int(r["segments"]) for r in rows}
    if len(seg) == 1:
        out.append(("OfficialBoundarySegments", str(seg.pop())))
    return out


def time_gap_macros(rows):
    """같은 공백 규칙을 열 릴리스에 씌웠을 때 무엇이 걸리나.

    릴리스 수는 자료에서 세어 \\GapReleases 로 낸다 — 설명에 적힌 수가 아니다.
    """
    hit = {}
    for r in rows:
        g = int(r["gaps"])
        if g <= 0:
            continue
        v = float(r["max_gap_s"])
        if v > hit.get(r["release"], 0.0):
            hit[r["release"]] = v
    # 1 초 이상인 공백만 센다. 떨림은 공백이 아니다 — 문턱은 표본 간격의
    # 배수가 아니라 절대 1 초다(본문도 "at least a second" 로 적는다).
    big = {k: v for k, v in hit.items() if v >= 1.0}
    out = [("GapReleases", str(len({r["release"] for r in rows}))),
           ("GapWithReleases", str(len(big)))]
    if big:
        out.append(("GapWorstSec", at_most(max(big.values()), "%.4g")))
    # 나머지 릴리스의 가장 긴 공백. 100~800 m 규칙에서 거의 모든 구간이
    # 공백을 무는 릴리스가 여기 들어서, 그 일이 긴 공백 때문이 아니라
    # 구간이 길어서 생긴다는 것을 이 수가 가른다.
    small = [v for v in hit.values() if v < 1.0]
    if small:
        out.append(("GapSubSecWorstSec", at_most(max(small), "%.3g")))
    return out


def manifest_macros(rows):
    """원본 고정표. 폴더에 있는 파일 수와 상한이 실제로 여는 파일 수."""
    if not rows or "read_by_bound" not in rows[0]:
        return []
    return [("ManifestFiles", str(len(rows))),
            ("ManifestRead",
             str(sum(1 for r in rows if r["read_by_bound"] == "yes")))]


def gap_policy_macros(rows):
    """공백이 채점 구간에 들어가는 양. 적재 단계에서 버린 것은 따로 센다.

    두 규칙을 한 통에 넣지 않는다. 1 m 간격 규칙에서는 공백을 건너뛰는
    구간이 소수라 그것을 뺀 나머지와 견줄 수 있지만, 100~800 m 규칙에서는
    구간 하나가 수십 초를 덮어서 공백이 잦은 릴리스는 거의 모든 구간이
    공백을 문다 — 거기서는 "뺀 나머지" 가 비교 대상이 못 된다. 그래서
    견줄 수 있는 쪽의 수와 못 견디는 쪽의 몫을 따로 낸다.
    """
    if not rows or "rule" not in rows[0]:
        return []
    rels = sorted({r["release"] for r in rows})
    free = [rel for rel in rels
            if not any(int(r["segments_crossing"]) for r in rows
                       if r["release"] == rel)]
    out = [("GapCrossReleasesTotal", str(len(rels))),
           ("GapCrossFreeReleases", str(len(free)))]

    fine = [r for r in rows if r["rule"] == "rpe1m"]
    if fine:
        # 자리수를 둘에서 셋으로 올린다. 바깥으로 깎는 규칙이라 두 자리면
        # 29.2 가 30 으로 튀어서, 깎은 폭이 본문이 말하는 양에 견줄 만큼
        # 커진다. 셋이면 그 폭이 잰 값의 1 퍼센트 아래로 떨어진다.
        out.append(("GapRpeWeightMax",
                    at_most(max(float(r["weight_crossing"])
                                for r in fine) * 100.0, "%.3g")))
        # 공백을 건너뛰는 구간이 한 개도 없는 시퀀스는 뺄 것이 없어
        # 정의상 0 이다. 세면 최댓값만 묽어진다.
        moved = []
        for r in fine:
            if not int(r["segments_crossing"]):
                continue
            for a, b in (("D_rot_all", "D_rot_gapfree"),
                         ("D_trans_all", "D_trans_gapfree")):
                x, y = float(r[a]), float(r[b])
                if x > 0.0 and y == y:
                    moved.append(abs(y - x) / x * 100.0)
        if moved:
            out.append(("GapRpeDShiftMax", at_most(max(moved), "%.3g")))

    coarse = [r for r in rows if r["rule"] == "seg"]
    if coarse:
        worst = 0.0
        worst_rel = ""
        dominated = 0
        for rel in rels:
            got = [r for r in coarse if r["release"] == rel]
            n = sum(int(r["segments"]) for r in got)
            if n:
                c = sum(int(r["segments_crossing"]) for r in got)
                if c * 2 > n:
                    dominated += 1
                if c / n * 100.0 > worst:
                    worst, worst_rel = c / n * 100.0, rel
        # 공백이 지배하는 릴리스를 따로 센다. 그 줄의 수는 공백을 뺀
        # 나머지와 견줄 수 없으므로, 요약에서 같이 묶지 않는다.
        out.append(("GapSegDomReleases", str(dominated)))
        if worst_rel:
            out.append(("GapSegCrossWorstRelease", LABELS[worst_rel]))
        out.append(("GapSegCrossWorstPct", "%.3g" % worst))
        out += gap_seg_subset_macros(coarse, rels)
    out += gap_route_macros(rows)
    return out


def gap_route_macros(rows):
    """공백이 점수에 닿는 두 길의 크기. 분해된 몫이라 더하면 D 가 된다.

    끝점이 공백에 닿는 구간과, 공백이 구간 안에만 있어 경로 길이로만 닿는
    구간을 따로 더한다. 묶음 안에서 무게를 다시 맞춘 평균으로는 둘을 못
    견준다 — 분모가 묶음마다 달라서 구간 몇 개짜리 묶음이 전체보다 커
    보인다. 열이 없는 표(두 길을 가르기 전에 잰 것)에서는 아무것도 안 낸다.
    """
    if not rows or "share_rot_endpoint_touched" not in rows[0]:
        return []
    out = []
    for rule, tag in (("seg", "Seg"), ("rpe1m", "Rpe")):
        got = [r for r in rows if r["rule"] == rule]
        ends, inside = [], []
        for r in got:
            for ax in ("rot", "trans"):
                whole = float(r["D_%s_all" % ax])
                if whole <= 0.0:
                    continue
                ends.append(100.0 * float(
                    r["share_%s_endpoint_touched" % ax]) / whole)
                inside.append(100.0 * float(
                    r["share_%s_inside_only" % ax]) / whole)
        if not ends:
            continue
        # 둘 다 최댓값으로 적고 둘 다 바깥으로 깎는다. 하한을 적으면 공백이
        # 아예 없는 릴리스가 0 을 넣어 그 수가 아무것도 안 말한다 — 묻는 것은
        # "두 길이 각각 얼마까지 가나" 다.
        #
        # 몫은 구성상 100 을 넘지 못한다. 넘어 보이는 것은 분자와 분모를 따로
        # 더한 자리의 부동소수 찌꺼기뿐이라, 그 테두리 안이면 100 으로 눌러
        # 놓고 깎는다. 누르지 않고 올려 깎으면 100.00000001 이 101 이 되어
        # 없는 자리를 만든다. 테두리를 벗어나면 분해가 깨진 것이므로 멈춘다.
        hi = max(inside)
        if hi > 100.0 + 1e-6:
            raise SystemExit(
                "경로 길이 몫이 %r 로 100 을 넘는다. 분해가 깨졌다" % hi)
        out += [("GapRoute%sEndpointPctMax" % tag, at_most(max(ends), "%.3g")),
                ("GapRoute%sInsidePctMax" % tag,
                 at_most(min(hi, 100.0), "%.4g"))]
    if out:
        out.append(("GapRouteReleases",
                    str(len({r["release"] for r in rows}))))
    return out


def gap_seg_subset_macros(coarse, rels):
    """공백이 지배하는 릴리스에서 남은 구간만 다시 묶는다.

    "너무 적어서 못 견준다" 는 말로 끝내지 않는다. 남은 구간이 몇 개이고
    그것만 묶으면 D 가 몇 배가 되는지는 셀 수 있고, 그 수가 곧 왜 못
    견주는지의 근거다 — 남은 구간은 공백을 피한 자리라 고른 쪽으로 치우쳐
    있고, 그 치우침의 크기를 재는 것이 이 수다. 묶는 무게는 다른 자리와
    같은 구간 수이고, 공백을 뺀 쪽은 공백을 뺀 구간 수로 센다.
    """
    dom = []
    for rel in rels:
        got = [r for r in coarse if r["release"] == rel]
        n = sum(int(r["segments"]) for r in got)
        c = sum(int(r["segments_crossing"]) for r in got)
        if n and c * 2 > n:
            dom.append((n - c, rel, got))
    if not dom:
        return []
    # 남은 구간이 하나도 없는 릴리스는 뺄 것이 없어 묶을 수가 없다. 그런
    # 릴리스가 몇인지는 따로 세어 둔다 — 본문이 "적다" 와 "없다" 를 가른다.
    empty = sum(1 for free, _, _ in dom if not free)
    free, rel, got = max(dom)
    if not free:
        return [("GapSegDomEmptyReleases", str(empty))]
    have = [r for r in got if r["D_trans_gapfree"] not in ("", "nan")]
    n = sum(int(r["segments"]) for r in have)
    f = sum(int(r["segments"]) - int(r["segments_crossing"]) for r in have)
    out = [("GapSegDomEmptyReleases", str(empty)),
           ("GapSegSubRelease", LABELS[rel]),
           ("GapSegSubSeqs", str(len(have))),
           ("GapSegSubSeqsAll", str(len(got))),
           ("GapSegSubSegs", group(n)),
           ("GapSegSubFree", group(f)),
           ("GapSegSubFreePct", "%.3g" % (f / n * 100.0))]
    for tag, a, b in (("Trans", "D_trans_all", "D_trans_gapfree"),
                      ("Rot", "D_rot_all", "D_rot_gapfree")):
        x = sum(int(r["segments"]) * float(r[a]) for r in have) / n
        y = sum((int(r["segments"]) - int(r["segments_crossing"]))
                * float(r[b]) for r in have) / f
        out.append(("GapSegSub%sRatio" % tag, "%.3g" % (y / x)))
    return out


def gap_cause_macros(score, gaps):
    """공백이 지배하는 릴리스 가운데 공백 자체는 짧은 것을 짚는다.

    100~800 m 규칙에서 거의 모든 구간이 공백을 무는 까닭이 긴 공백이 아니라
    긴 구간이라는 말은, 공백이 지배하는 릴리스 자신에서 보여야 선다. 가장
    심한 릴리스는 공백도 길어서 그 자리를 못 맡는다 — 둘을 못 가린다.
    그래서 지배하는 쪽 가운데 공백이 1 초 아래인 릴리스를 따로 낸다.
    """
    if not score or "rule" not in score[0] or not gaps:
        return []
    worst_gap = {}
    for r in gaps:
        if int(r["gaps"]) > 0:
            v = float(r["max_gap_s"])
            if v > worst_gap.get(r["release"], 0.0):
                worst_gap[r["release"]] = v
    coarse = [r for r in score if r["rule"] == "seg"]
    best = None
    for rel in sorted({r["release"] for r in coarse}):
        got = [r for r in coarse if r["release"] == rel]
        n = sum(int(r["segments"]) for r in got)
        c = sum(int(r["segments_crossing"]) for r in got)
        if not n or c * 2 <= n or worst_gap.get(rel, 0.0) >= 1.0:
            continue
        if best is None or c / n > best[0]:
            best = (c / n, rel)
    if best is None:
        return []
    share, rel = best
    return [("GapSegDomSubSecRelease", LABELS[rel]),
            ("GapSegDomSubSecPct", "%.3g" % (share * 100.0)),
            ("GapSegDomSubSecGapSec", at_most(worst_gap[rel], "%.3g"))]


def gap_load_residual_macros(load_rows, audit_rows):
    """적재가 끊어 만든 토막인데도 공백이 다시 잡히는 자리.

    끊을 때는 시퀀스 전체의 중앙 간격을 쓰고 감사는 토막 자신의 중앙 간격을
    쓰기 때문에, 끊고 나면 공백이 없다고 말할 수 없다.

    세는 모집단은 적재가 남긴 토막 전부다. 채점 표에서 세면 두 가지로
    어긋난다 — 1 m 구간이 안 선 토막이 아예 안 보이고, 끊긴 파일의 첫
    토막은 이름에 번호가 안 붙어서(kitti_segment_D.py 의 "%s#%d" 는 k 가
    0 이면 안 붙는다) 이름으로 토막을 가리면 통째로 빠진다. 조건도 다르다 —
    본문이 말하는 것은 토막이 공백을 품는가이고 채점 표의 칸은 1 m 구간이
    공백을 건너뛰는가다.
    """
    cut = {r["release"] for r in load_rows}
    if not cut or not audit_rows:
        return []
    return [("GapLoadResidualPieces",
             str(sum(1 for r in audit_rows if r["release"] in cut
                     and int(r["gaps"]) > 0)))]


def gap_load_macros(rows):
    """적재가 끊고 버린 양. 끊는 릴리스가 없으면 아무 것도 안 낸다."""
    if not rows:
        return []
    files = sum(int(r["samples_file"]) for r in rows)
    kept = sum(int(r["samples_kept"]) for r in rows)
    if not files:
        return []
    return [("GapLoadRelease", sorted({r["release"] for r in rows})[0]),
            ("GapLoadDroppedPct",
             at_most((files - kept) / files * 100.0, "%.2g")),
            ("GapLoadDroppedPieces",
             str(sum(int(r["pieces_dropped"]) for r in rows))),
            # 버린 토막의 분모. 끊어서 난 토막 수다 — 표본이 하나도 없는
            # 파일은 끊을 것도 날 것도 없어서 0 으로 센다.
            ("GapLoadPiecesMade",
             str(sum(int(r["pieces_made"]) for r in rows))),
            ("GapLoadMinFrames", str(int(rows[0]["min_frames"])))]


def board_sequence_macros(rows):
    """순위표 환산이 공개 시퀀스 하나에 묶여 있는지.

    시퀀스마다의 기울기 비(가장 큰 쪽 나누기 가장 작은 쪽), 하나씩 뺀
    기울기의 범위 폭, 그리고 본문이 적는 되짚기 환산이 하나씩 뺄 때 묶음
    값에서 벗어나는 최대 폭을 낸다. 앞 둘은 기울기 통계량이고 마지막 하나만
    본문 수치와 같은 정의다 — 셋을 같은 이름으로 묶지 않는다.
    """
    out = []
    for metric, tag in (("trans", "Trans"), ("rot", "Rot")):
        seq = [float(r["slope_per_ms"]) for r in rows
               if r["metric"] == metric and r["kind"] == "sequence"]
        drop = [float(r["slope_per_ms"]) for r in rows
                if r["metric"] == metric and r["kind"] == "drop"]
        if not seq or not drop or min(seq) <= 0.0 or min(drop) <= 0.0:
            continue
        out += [("BoardSeq%sFold" % tag,
                 at_most(max(seq) / min(seq), "%.2g")),
                ("BoardSeqDrop%sPct" % tag,
                 at_most((max(drop) / min(drop) - 1.0) * 100.0, "%.2g"))]
        span = [float(r["span_ms"]) for r in rows
                if r["metric"] == metric and r["kind"] == "drop"
                and r.get("span_ms")]
        full = [float(r["span_ms"]) for r in rows
                if r["metric"] == metric and r["kind"] == "pooled"
                and r.get("span_ms")]
        if span and full and full[0] > 0.0:
            off = max(abs(v - full[0]) for v in span) / full[0] * 100.0
            out.append(("BoardSeqSpanDrop%sPct" % tag,
                        at_most(off, "%.2g")))
    return out


def stability_macros(rows):
    """공식 구간 선택이 기준을 바꿨을 때 자리를 옮기는 가중치의 몫.

    상한은 구간 집합이 고정됐을 때 선다. 공식 프로그램은 끝 프레임을 기준
    궤적에서 그때그때 다시 고르므로 그 전제가 자료 위에서 지켜지는지는 재야
    안다. 옮겨간 몫을 그대로 적으면 상한이 덮는 몫이 1 에서 그만큼 뺀 것이다.

    실내 손에 든 릴리스를 따로 가른다. 1 m 구간에 들어가는 표본 수가 차량
    릴리스보다 훨씬 많아서 끝점이 옮겨갈 자리도 그만큼 많다 — 한 범위로
    묶으면 그 폭이 통째로 차량 쪽 수치처럼 읽힌다.
    """
    if not rows:
        return []

    def frac(conv, group):
        sel = [float(r["weight_moved_frac"]) for r in rows
               if r["convention"] == conv
               and ((r["release"] in INDOOR) == (group == "indoor"))]
        return sel

    out = [("SegStabReleases", str(len({r["release"] for r in rows}))),
           ("SegStabNullMoved",
            str(max((int(r["moved"]) for r in rows
                     if r["convention"] == "null"), default=0)))]
    for conv, tag in (("tau_1ms", "TauOne"), ("tau_10ms", "TauTen"),
                      ("rate_0.5_nearest", "HalfRate")):
        for group, gtag in (("vehicle", "Vehicle"), ("indoor", "Indoor")):
            sel = frac(conv, group)
            if not sel:
                continue
            out.append(("SegStab%s%sMin" % (tag, gtag),
                        at_least(min(sel) * 100, "%.2g")))
            out.append(("SegStab%s%sMax" % (tag, gtag),
                        at_most(max(sel) * 100, "%.2g")))
    return out


def axiom_macros(rows):
    """거리 공리 점검의 (이름, 값) 들. 공개 행렬과 누른 행렬을 가른다.

    상한의 증명이 d 가 거리라는 데에만 기대므로, 그 전제가 공개 자료에서
    성립하는지를 수로 적는다. 두 묶음을 한 수로 합치지 않는다 — 합치면
    "누르기 전에는 거리가 아니었다" 라는 요지가 사라진다.
    """
    out = [("AxiomTrials", str(sum(int(r["triangle_trials"]) for r in rows
                                   if r["matrices"] == "released"))),
           ("AxiomNearTrials", str(sum(int(r["near_triangle_trials"])
                                       for r in rows
                                       if r["matrices"] == "released")))]
    for label, tag in (("released", "Released"), ("projected", "Projected")):
        sub = [r for r in rows if r["matrices"] == label]
        out.append(("AxiomSelf%s" % tag, at_most(max(
            float(r["self_distance_max_deg"]) for r in sub))))
        out.append(("AxiomTriangle%s" % tag, str(sum(
            int(r["triangle_violations"]) for r in sub))))
        # 항등 근처는 따로 센다. 무작위 삼조는 각이 큰 자리를 주로 뽑는데,
        # 역코사인이 못 견디는 곳은 그 반대쪽이다.
        out.append(("AxiomNearTriangle%s" % tag, str(sum(
            int(r["near_triangle_violations"]) for r in sub))))
        out.append(("AxiomNearWorst%s" % tag, at_most(max(
            float(r["near_triangle_worst_deg"]) for r in sub), "%.3g")))
        # 잘려서 0 으로 보인 자기거리. 0 으로 보이는 것과 0 인 것은 다르다.
        # 세는 수만으로는 못 가른다 — 정확한 회전도 끝자리 때문에 한 칸
        # 넘으므로, 넘은 폭을 같이 적어야 둘이 갈린다.
        out.append(("AxiomClipped%s" % tag, str(sum(
            int(r["self_distance_clipped"]) for r in sub))))
        out.append(("AxiomClipExcess%s" % tag, at_most(max(
            float(r["self_distance_clip_excess"]) for r in sub), "%.3g")))
    return out


def doc_macros(rows):
    """문서 대조의 (이름, 값) 들. 세는 수와 릴리스가 적어 둔 폭만 꺼낸다.

    인용문은 매크로로 안 만든다 — 본문에 싣는 것은 원문 그대로라 라텍에서
    탈출해야 하고, 자동 생성 파일을 거치면 어느 글자가 원문인지가 흐려진다.
    """
    named = [r for r in rows if r["method"] != "none"]
    nearest = [r for r in rows
               if "nearest" in (r["method"], r["eval_method"])]
    silent = [r for r in rows if r["published_hz"] == "미기재"]
    out = [("DocReleases", str(len(rows))),
           ("DocMethodNamed", str(len(named))),
           ("DocMethodSilent", str(len(rows) - len(named))),
           ("DocNearest", str(len(nearest))),
           ("DocRateSilent", str(len(silent)))]
    for r in rows:
        if r["release"] not in NAMES:
            continue
        if r["stated_error_ms"]:
            out.append(("DocStated%sMs" % NAMES[r["release"]],
                        r["stated_error_ms"]))
        if r["eval_tolerance_ms"]:
            out.append(("DocTolerance%sMs" % NAMES[r["release"]],
                        r["eval_tolerance_ms"]))
        # 릴리스가 적어 둔 주기와 실제 축이 어긋나는 자리를 본문이 부른다.
        if r["measured_hz"]:
            out.append(("DocMeasured%sHz" % NAMES[r["release"]],
                        r["measured_hz"]))
    return out


PROTOCOL_COLS = (("shift", "0.9"), ("decim2", "even"), ("decim2", "odd"),
                 ("rate", "0.5_slerp"), ("rate", "0.5_nearest"),
                 ("method", "0.5"), ("tau", "10ms"))


def table_protocol(rows, rule="seg"):
    """규약별 표 본문. 릴리스 한 줄에 변형 일곱을 병진 퍼센트로 둔다."""
    by = {(r["release"], r["axis"], r["param"]): r for r in rows
          if r["rule"] == rule}
    lines = []
    for rel in NAMES:
        cells = [LABELS[rel]]
        got = False
        for axis, param in PROTOCOL_COLS:
            r = by.get((rel, axis, param))
            if r is None:
                cells.append("--")
                continue
            got = True
            cells.append(tex_number("%.3g" % float(r["D_trans_pct"])))
        if got:
            lines.append(" & ".join(cells) + r" \\")
    return lines


def group(value):
    """구간 수처럼 큰 수는 세 자리마다 끊어 적는다."""
    try:
        return "{:,}".format(int(value))
    except (TypeError, ValueError):
        return value


def table_body(rows):
    """표 본문. 릴리스 한 줄에 두 규칙을 나란히 둔다."""
    by = {(r["release"], r["rule"]): r for r in rows}
    lines = []
    for rel in NAMES:
        s, p = by.get((rel, "seg")), by.get((rel, "rpe1m"))
        if s is None and p is None:
            continue
        cells = [LABELS[rel]]
        for r in (s, p):
            if r is None:
                cells += ["--", "--", "--"]
            else:
                cells += [group(r["n_segments"]),
                          "%.3g" % float(r["twoD_trans_pct"]),
                          "%.3g" % float(r["twoD_rot_deg_per_100m"])]
        lines.append(" & ".join(cells) + r" \\")
    return lines


def table_spread(rows):
    """시퀀스별 폭 표의 본문. 묶음 값 하나가 가리는 범위를 본문에 싣는다.

    경로로 가리키지 않는다 — 코드가 아직 어디에 공개될지 정해지지 않은
    동안에는 "파일을 보라" 가 읽는 쪽에서 닿을 수 없는 안내다.
    """
    by = {(r["release"], r["rule"]): r for r in rows}
    lines = []
    for rel in NAMES:
        s, p = by.get((rel, "seg")), by.get((rel, "rpe1m"))
        if s is None and p is None:
            continue
        cells = [LABELS[rel]]
        for r in (s, p):
            if r is None or not r.get("n_sequences"):
                cells += ["--", "--", "--"]
                continue
            cells.append(r["n_sequences"])
            for lo, hi in (("twoD_trans_seq_min_pct", "twoD_trans_seq_max_pct"),
                           ("twoD_rot_seq_min_deg_per_100m",
                            "twoD_rot_seq_max_deg_per_100m")):
                a, b = r.get(lo, ""), r.get(hi, "")
                cells.append("--" if a in ("", "nan") or b in ("", "nan")
                             else "%.3g--%.3g" % (float(a), float(b)))
        lines.append(" & ".join(cells) + r" \\")
    return lines


ENV_STORE = "results/next/environment.json"


def live_environment():
    """지금 이 해석기·컴파일러·기계의 판본."""
    import numpy
    import platform
    import scipy
    import subprocess

    # 공식 이진 대조는 개발도구를 g++ -O2 로 직접 빌드한다. 어느 판본으로
    # 어느 기계에서 빌드했는지가 그 비교의 조건이라 같이 적는다 — 부동소수
    # 결과는 컴파일러와 아키텍처에 묶인다(27 회차 권고).
    try:
        ver = subprocess.run(["g++", "-dumpfullversion"], check=True,
                             capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        ver = "미확인"
    return {"python": ".".join(str(v) for v in sys.version_info[:3]),
            "numpy": numpy.__version__, "scipy": scipy.__version__,
            "gpp": ver, "arch": platform.machine()}


def environment_macros(store=ENV_STORE):
    """재현에 필요한 판본 (이름, 값) 들. 본문에 손으로 적지 않기 위해서다.

    본문이 적는 것은 그 수를 잰 환경이다. 그런데 이 파일은 조판할 때마다
    돌 수 있어서, 기존 결과표를 다른 기계에서 다시 조판하면 수는 그대로인데
    판본 줄만 조용히 바뀐다. 그래서 처음 낼 때 그 환경을 결과 폴더에 적어
    두고 그 뒤로는 적어 둔 것을 쓴다 — 지금 환경이 다르면 경고만 하고
    기록된 쪽을 쓴다(31 회차 권고 2).
    """
    live = live_environment()
    if store and os.path.exists(store):
        with open(store) as f:
            use = json.load(f)
        if use != live:
            print("경고: 측정 당시 환경 기록을 쓴다 — 지금 환경과 다르다"
                  " (%s)" % store)
    else:
        use = live
        if store:
            os.makedirs(os.path.dirname(store), exist_ok=True)
            with open(store, "w") as f:
                json.dump(live, f, ensure_ascii=False, indent=1,
                          sort_keys=True)
    return [("EnvPython", use["python"]),
            ("EnvNumpy", use["numpy"]),
            ("EnvScipy", use["scipy"]),
            ("EnvGpp", use["gpp"]),
            # x86_64 의 밑줄은 본문에서 첨자로 읽혀 빌드가 멈춘다. 수가 아닌
            # 칸이라 tex_number 가 손대지 않으므로 여기서 걷어낸다.
            ("EnvArch", use["arch"].replace("_", "\\_"))]


def retrieval_macros(rows):
    """순위표를 언제 읽었나. 출처 문장 안에 적힌 날짜 하나를 꺼낸다."""
    dates = set()
    for r in rows:
        m = re.search(r"(\d{4}-\d{2}-\d{2})", r.get("source", ""))
        if m:
            dates.add(m.group(1))
    if not dates:
        return []
    # 날짜가 여럿이면 가장 늦은 것을 적는다. 하나로 뭉뚱그리지 않으려면
    # 본문이 둘을 따로 불러야 하는데, 지금은 두 순위표를 같은 날 읽었다.
    return [("BoardRetrievalDate", max(dates))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="results/next/release_bound_table.csv")
    ap.add_argument("--sweep", default="results/next/protocol_sweep_table.csv")
    ap.add_argument("--boards",
                    default="results/next/offset_tolerance_summary.csv")
    ap.add_argument("--decomp",
                    default="results/next/start_frame_decomposition.csv")
    ap.add_argument("--floor",
                    default="results/next/orthonormality_floor.csv")
    ap.add_argument("--doc",
                    default="results/next/reconstruction_doc_audit.csv")
    ap.add_argument("--cancel", default="results/next/pair_cancellation.csv")
    ap.add_argument("--boundary",
                    default="results/next/official_boundary_case.csv")
    ap.add_argument("--gaps",
                    default="results/next/time_gap_audit.csv")
    ap.add_argument("--manifest",
                    default="results/next/source_manifest.csv")
    ap.add_argument("--gap-score",
                    default="results/next/gap_policy_score.csv")
    ap.add_argument("--gap-load",
                    default="results/next/gap_policy_load.csv")
    ap.add_argument("--board-seq",
                    default="results/next/board_sequence_sensitivity.csv")
    ap.add_argument("--official",
                    default="results/next/kitti_official_binary.csv")
    ap.add_argument("--method-scale",
                    default="results/next/method_scale_probe.csv")
    ap.add_argument("--official-floor",
                    default="results/next/kitti_official_floor.csv")
    ap.add_argument("--axioms",
                    default="results/next/metric_axioms_probe.csv")
    ap.add_argument("--known",
                    default="results/next/known_answer_probe.csv")
    ap.add_argument("--rank", default="results/next/kitti_rank_check.csv")
    ap.add_argument("--bound-check",
                    default="results/next/resample_bound_check.csv")
    ap.add_argument("--window-counts",
                    default="results/next/window_exclusion_counts.csv")
    ap.add_argument("--window-bias",
                    default="results/next/window_exclusion_bias.csv")
    ap.add_argument("--window-pooled",
                    default="results/next/window_exclusion_pooled.csv")
    ap.add_argument("--tie-rule", default="results/next/tie_rule_probe.csv")
    ap.add_argument("--angle-unit",
                    default="results/next/pit30m_angle_unit.csv")
    ap.add_argument("--time-axis",
                    default="results/next/kitti360_time_axis.csv")
    ap.add_argument("--kitti-axis",
                    default="results/next/kitti_time_axis.csv")
    ap.add_argument("--stability",
                    default="results/next/segment_set_stability.csv")
    ap.add_argument("--oxford",
                    default="results/next/oxford_rtk_quality.csv")
    ap.add_argument("--register",
                    default="results/next/dataset_register.csv")
    ap.add_argument("--rule-ratio",
                    default="results/next/rule_ratio_common.csv")
    ap.add_argument("--crosscheck",
                    default="results/next/kitti_official_crosscheck.csv")
    ap.add_argument("--out-dir", default="paper_tim")
    args = ap.parse_args()

    rows = read_table(args.table)
    os.makedirs(args.out_dir, exist_ok=True)

    all_macros = (list(macros(rows)) + list(ratio_macros(rows))
                  + list(per_sequence_macros(rows))
                  + list(sweep_population_macros(rows))
                  + list(segment_rule_population_macros(
                      rows, read_table(args.register)))
                  + list(release_count_macros(rows))
                  + list(table_min_macros(rows)))
    for path, fn in ((args.sweep, sweep_macros), (args.boards, board_macros),
                     (args.decomp, decomp_macros), (args.floor, floor_macros),
                     (args.doc, doc_macros),
                     (args.cancel, cancellation_macros),
                     (args.official, official_macros),
                     (args.official_floor, official_floor_macros),
                     (args.method_scale, method_scale_macros),
                     (args.axioms, axiom_macros),
                     (args.known, known_macros),
                     (args.rank, retrieval_macros),
                     (args.bound_check, bound_check_macros),
                     (args.window_counts, window_counts_macros),
                     (args.window_bias, window_bias_macros),
                     (args.window_pooled, window_pooled_macros),
                     (args.tie_rule, tie_rule_macros),
                     (args.angle_unit, angle_unit_macros),
                     (args.time_axis, time_axis_macros),
                     (args.kitti_axis, kitti_axis_macros),
                     (args.stability, stability_macros),
                     (args.oxford, oxford_macros),
                     (args.board_seq, board_sequence_macros),
                     (args.gaps, time_gap_macros),
                     (args.manifest, manifest_macros),
                     (args.gap_score, gap_policy_macros),
                     (args.gap_load, gap_load_macros),
                     (args.rule_ratio, rule_ratio_common_macros),
                     (args.gap_load, wheeled_indoor_macros),
                     (args.crosscheck, crosscheck_macros),
                     (args.boundary, official_boundary_macros)):
        if os.path.exists(path):
            all_macros += fn(read_table(path))
    # 두 표를 함께 봐야 나오는 값이라 고리 밖에서 따로 부른다.
    if os.path.exists(args.gap_score) and os.path.exists(args.gaps):
        all_macros += gap_cause_macros(read_table(args.gap_score),
                                       read_table(args.gaps))
    if os.path.exists(args.gap_load) and os.path.exists(args.gaps):
        all_macros += gap_load_residual_macros(read_table(args.gap_load),
                                               read_table(args.gaps))
    if os.path.exists(args.register):
        all_macros += register_count_macros(read_table(args.register))
        all_macros += rate_group_macros(
            rows, read_table(args.register),
            read_table(args.sweep) if os.path.exists(args.sweep) else None)
    all_macros += sweep_ratio_macros()
    all_macros += public_commit_macros(args.out_dir)
    # 환경 기록은 그것이 설명하는 표들 옆에 둔다 — 다른 폴더에서 돌렸을 때
    # 기록만 딴 데 생기면 그 회차의 판본이 어느 표의 것인지가 끊긴다.
    all_macros += environment_macros(
        os.path.join(os.path.dirname(args.sweep) or ".", "environment.json"))

    nums = os.path.join(args.out_dir, "numbers.tex")
    with open(nums, "w") as f:
        f.write("%% scripts/make_paper_macros.py 가 만든 파일이다."
                " 손으로 고치지 마라.\n")
        for name, value in all_macros:
            f.write("\\newcommand{\\%s}{%s}\n" % (name, tex_number(value)))

    tbl = os.path.join(args.out_dir, "table_bounds.tex")
    with open(tbl, "w") as f:
        f.write("%% scripts/make_paper_macros.py 가 만든 파일이다."
                " 손으로 고치지 마라.\n")
        for line in table_body(rows):
            f.write(line + "\n")

    spread = os.path.join(args.out_dir, "table_spread.tex")
    with open(spread, "w") as f:
        f.write("%% scripts/make_paper_macros.py 가 만든 파일이다."
                " 손으로 고치지 마라.\n")
        for line in table_spread(rows):
            f.write(line + "\n")
    print("시퀀스별 폭 표 %d 줄 -> %s" % (len(table_spread(rows)), spread))

    if os.path.exists(args.sweep):
        prot = os.path.join(args.out_dir, "table_protocol.tex")
        lines = table_protocol(read_table(args.sweep))
        with open(prot, "w") as f:
            f.write("%% scripts/make_paper_macros.py 가 만든 파일이다."
                    " 손으로 고치지 마라.\n")
            for line in lines:
                f.write(line + "\n")
        print("규약 표 %d 줄 -> %s" % (len(lines), prot))

    print("매크로 %d 개 -> %s" % (len(all_macros), nums))
    print("표 %d 줄 -> %s" % (len(table_body(rows)), tbl))


if __name__ == "__main__":
    main()
