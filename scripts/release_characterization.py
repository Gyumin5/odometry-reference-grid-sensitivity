#!/usr/bin/env python3
"""릴리스마다 표현 성질 여섯을 한 줄로 묶는다.

어느 갈래로 쓰든 필요한 표다. 품질(얼마나 정확한가)이 아니라 표현(무엇을 어떻게
실었는가)을 적는다 — 공개된 파일 하나를 열면 바로 읽히는 것만 담는다.

  게시 주기      그 파일이 몇 Hz 로 실렸나
  시각 흔들림    표본 간격이 얼마나 고르나. 시각을 지어낸 릴리스는 빈칸
  자세 격자      자세 열이 유한 격자에 놓였나, 놓였다면 그 간격
  대역 기울기    2~4 Hz 에서 스펙트럼이 어떻게 기우나. 0 이면 백색, 큰 음수면 평활됨
  유효 대역      바닥의 열 배를 넘는 성분이 남아 있는 마지막 주파수
  열 사이 일치   같은 파일의 자세 열과 각속도 열이 서로를 설명하나

입력은 results/next 의 CSV 들이고 새로 재지 않는다.
"""

import argparse
import csv
import os
import statistics as st


def num(rows, key):
    out = []
    for r in rows:
        v = r.get(key, "")
        if v in ("", "nan", None):
            continue
        try:
            out.append(float(v))
        except ValueError:
            continue
    return out


def span(vals, fmt="%.2f"):
    if not vals:
        return ""
    if len(vals) == 1:
        return fmt % vals[0]
    return (fmt + " (" + fmt + "~" + fmt + ")") % (st.median(vals), min(vals), max(vals))


def load(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/next")
    ap.add_argument("--out", default="results/next/release_characterization.csv")
    args = ap.parse_args()

    res = load(os.path.join(args.results, "reference_resolution.csv"))
    cons = load(os.path.join(args.results, "rate_channel_consistency.csv"))
    pair = load(os.path.join(args.results, "two_rate_pair.csv"))
    if not res:
        print("reference_resolution.csv 가 없다")
        return

    by = {}
    for r in res:
        by.setdefault(r["release"], []).append(r)
    cons_by, pair_by = {}, {}
    for r in cons:
        cons_by.setdefault(r["release"], []).append(r)
    for r in pair:
        pair_by.setdefault(r["release"], []).append(r)

    rows = []
    for rel, rs in sorted(by.items()):
        slopes = num(rs, "band_slope")
        white = sum(r.get("model") == "백색" for r in rs)
        # 저장 분해능은 "<=1.4e-07" 꼴로 들어온다. 문자열을 정렬해 앞의 셋만 보이면
        # 가장 거친 릴리스가 목록에서 빠진다 — 숫자로 되읽어 중앙과 범위를 적는다.
        quanta = []
        for r in rs:
            v = r.get("quantum_attitude", "")
            if v.startswith("<="):
                try:
                    quanta.append(float(v[2:]))
                except ValueError:
                    pass
        cr = cons_by.get(rel, [])
        pr = pair_by.get(rel, [])
        rows.append({
            "release": rel,
            "n_sequences": len(rs),
            "duration_s_median": round(st.median(num(rs, "duration_s")), 1)
            if num(rs, "duration_s") else "",
            "fs_hz": span(num(rs, "fs_hz")),
            "jitter_rms_ms": span(num(rs, "jitter_rms_ms"), "%.3f"),
            "quantum_attitude": ("<=" + span(quanta, "%.3g")) if quanta else "못 잼",
            "band_slope": span(slopes, "%.2f"),
            "white_pass": "%d/%d" % (white, len(rs)),
            "f_eff_min_hz": span(num(rs, "f_eff_min_hz")),
            "kept_frac": span(num(rs, "kept_frac"), "%.3f"),
            "rate_channel_residual": span(num(cr, "residual_frac"), "%.3f"),
            "two_rate_excess": span(num(pr, "ratio_excess"), "%.3f"),
        })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    head = ("%-15s %4s %-18s %-17s %-19s %-8s %-16s %-16s"
            % ("릴리스", "n", "주기(Hz)", "흔들림(ms)", "대역기울기",
               "백색", "유효대역(Hz)", "두주기초과"))
    print(head)
    print("-" * len(head))
    for r in rows:
        print("%-15s %4d %-18s %-17s %-19s %-8s %-16s %-16s"
              % (r["release"], r["n_sequences"], r["fs_hz"], r["jitter_rms_ms"],
                 r["band_slope"], r["white_pass"], r["f_eff_min_hz"],
                 r["two_rate_excess"] or "-"))
    print()
    for r in rows:
        print("%-15s 자세격자 %-22s 열사이잔차 %-16s 보존율 %s"
              % (r["release"], r["quantum_attitude"] or "-",
                 r["rate_channel_residual"] or "-", r["kept_frac"]))
    print("\n%d 릴리스 -> %s" % (len(rows), args.out))


if __name__ == "__main__":
    main()
