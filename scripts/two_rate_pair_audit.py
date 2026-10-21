#!/usr/bin/env python3
"""한 릴리스가 같은 항법 해를 두 주기로 싣고 있을 때 그 둘을 맞댄다.

통제 교정 실험은 평활이 대역 밀도를 얼마나 누르는지 합성 자료로 보였다. 같은 해를
두 주기로 실은 공개본이 있으면 그 기제를 실제 자료에서 그대로 볼 수 있다 — 두 벌은
같은 추정의 산출물이라 '어느 쪽이 더 정확한가' 가 아니라 '어떻게 실었나' 만 다르다.

맞대는 것 둘.
  ⑴ 두 벌이 같은 해인가. 느린 쪽 시각으로 빠른 쪽을 보간해 자세 증분을 맞추고
    남는 잔차를 신호 대비로 적는다. 작으면 같은 해다.
  ⑵ 같은 해인데도 공통 대역에서 밀도가 갈리는가. 갈리면 그 차이는 기준의 품질이
    아니라 그 파일을 어떻게 실었느냐다.

대상 쌍:
  보레아스  applanix/lidar_poses.csv (라이다 시각) 대 applanix/gps_post_process.csv (200 Hz)
  헬리퍼    LiDAR_GT/<센서>_gt.txt (10 Hz) 대 Inertial_data/inspva.csv (50 Hz)
"""

import argparse
import csv
import glob
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helipr_inertial_cadence import load_inspva  # noqa: E402
from reference_resolution_audit import (  # noqa: E402
    audit,
    load_boreas_gps_post,
    load_boreas_lidar_poses,
    load_tum,
)
from reference_timing_audit import best_tau, fit_at_tau, rate_from_attitude  # noqa: E402


def pairs(args):
    out = []
    if os.path.isdir(args.boreas_root):
        for seq in sorted(os.listdir(args.boreas_root)):
            slow = os.path.join(args.boreas_root, seq, "applanix", "lidar_poses.csv")
            fast = os.path.join(args.boreas_root, seq, "applanix", "gps_post_process.csv")
            if os.path.exists(slow) and os.path.exists(fast):
                out.append(("boreas", seq, slow, load_boreas_lidar_poses,
                            fast, load_boreas_gps_post))
    if os.path.isdir(args.helipr_root):
        for seq in sorted(os.listdir(args.helipr_root)):
            fast = os.path.join(args.helipr_root, seq, "Inertial_data", "inspva.csv")
            if not os.path.exists(fast):
                continue
            gt = sorted(glob.glob(os.path.join(args.helipr_root, seq,
                                               "LiDAR_GT", "*_gt.txt")))
            if gt:
                out.append(("helipr", seq, gt[0], load_tum, fast, load_inspva))
    return out


def differencing_ratio(fs_slow, fs_fast, lo, hi, n=64):
    """차분 창만으로 설명되는 밀도비.

    증분을 간격으로 나눈 각속도는 그 간격 동안의 이동평균이라 |sinc(pi f dt)| 만큼
    눌린다. 느린 쪽은 간격이 길어 더 눌리므로, 같은 해를 두 주기로 실으면 그것만으로도
    빠른 쪽 밀도가 높게 나온다. 관측된 밀도비를 이 값으로 나눈 것이 차분 창 밖의 평활
    — 공개 전에 한 번 더 깎였다는 몫이다.

    되접힘은 여기 안 넣는다. 느린 쪽 나이퀴스트 위의 참 스펙트럼이 접혀 들어와 느린
    쪽을 오히려 올리므로, 이 예측은 설명 몫의 상한이고 따라서 초과분은 하한이다.
    """
    def att(f, fs):
        x = math.pi * f / fs
        return 1.0 if x == 0 else (math.sin(x) / x) ** 2
    rs = []
    for i in range(n):
        f = lo + (hi - lo) * i / (n - 1)
        d = att(f, fs_slow)
        if d > 0:
            rs.append(att(f, fs_fast) / d)
    rs.sort()
    return rs[len(rs) // 2] if rs else float("nan")


def agreement(t_slow, rot_slow, t_fast, rot_fast, tau_max, tau_step):
    """느린 쪽 자세 증분을 빠른 쪽으로 설명하고 남는 잔차 비를 돌려준다.

    느린 쪽 증분은 그 간격 동안의 평균 각속도고 빠른 쪽은 중점 값이라, 간격 안에서
    움직임이 휘는 만큼은 잔차에 남는다. 그래서 이 값은 '두 벌이 다른 해다' 의 상한이지
    하한이 아니다 — 작게 나오면 같은 해라고 말할 수 있고, 조금 크게 나온 것만으로
    다른 해라고 말할 수는 없다.
    """
    tc_s, w_s = rate_from_attitude(t_slow, rot_slow)
    tc_f, w_f = rate_from_attitude(t_fast, rot_fast)
    lo = max(tc_s[0], tc_f[0] + tau_max)
    hi = min(tc_s[-1], tc_f[-1] - tau_max)
    m = (tc_s >= lo) & (tc_s <= hi)
    if m.sum() < 128:
        return None
    tc_s, w_s = tc_s[m], w_s[m]
    tau, _, _ = best_tau(tc_s, w_s, tc_f, w_f, -tau_max, tau_max, tau_step)
    res, _ = fit_at_tau(tc_s, w_s, tc_f, w_f, tau)
    # 평균을 뺀 신호 대비로 잰다 — fit_at_tau 가 양쪽 평균을 빼고 맞추기 때문이다.
    amp = float(((w_s - w_s.mean(axis=0)) ** 2).mean())
    return {"n_pair": int(m.sum()),
            "residual_frac": "%.4g" % (float((res ** 2).mean() / amp) ** 0.5
                                       if amp else float("nan")),
            "tau_ms": round(tau * 1000, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boreas-root", default="data/boreas")
    ap.add_argument("--helipr-root", default="/mnt/Data/velref/helipr")
    ap.add_argument("--band", default="2,4")
    ap.add_argument("--win-s", type=float, default=16.0)
    ap.add_argument("--tau-max", type=float, default=0.2)
    ap.add_argument("--tau-step", type=float, default=0.005)
    ap.add_argument("--out", default="results/next/two_rate_pair.csv")
    args = ap.parse_args()
    band_lo, band_hi = (float(v) for v in args.band.split(","))

    rows = []
    for release, seq, p_slow, ld_slow, p_fast, ld_fast in pairs(args):
        try:
            t_s, rot_s, _ = ld_slow(p_slow)
            t_f, rot_f, _ = ld_fast(p_fast)
        except Exception as e:
            print("[%s] %s 읽기 실패: %s" % (release, seq, e))
            continue
        a_s = audit(t_s, rot_s, seq, 1.0, 10.0, band_lo, band_hi, args.win_s, "x") or {}
        a_f = audit(t_f, rot_f, seq, 1.0, 10.0, band_lo, band_hi, args.win_s, "x") or {}
        if "S0_total_rad2ps2perHz" not in a_s or "S0_total_rad2ps2perHz" not in a_f:
            print("[%s] %s 한쪽이 대역을 못 채운다" % (release, seq))
            continue
        s_slow = float(a_s["S0_total_rad2ps2perHz"])
        s_fast = float(a_f["S0_total_rad2ps2perHz"])
        rec = {
            "release": release, "sequence": seq,
            "fs_slow_hz": a_s["fs_hz"], "fs_fast_hz": a_f["fs_hz"],
            "slope_slow": a_s["band_slope"], "slope_fast": a_f["band_slope"],
            "model_slow": a_s.get("model", ""), "model_fast": a_f.get("model", ""),
            "S0_slow": "%.4g" % s_slow, "S0_fast": "%.4g" % s_fast,
            "S0_ratio_fast_over_slow": round(s_fast / s_slow, 4) if s_slow else "",
        }
        pred = differencing_ratio(a_s["fs_hz"], a_f["fs_hz"], band_lo, band_hi)
        rec["ratio_from_differencing"] = round(pred, 4)
        if s_slow and pred:
            rec["ratio_excess"] = round((s_fast / s_slow) / pred, 4)
        ag = agreement(t_s, rot_s, t_f, rot_f, args.tau_max, args.tau_step)
        if ag:
            rec.update(ag)
        rows.append(rec)
        print("%-8s %-14s %6.1f/%6.1f Hz  기울기 %6s/%6s  밀도비 %7s "
              "차분설명 %6s  초과 %7s  잔차 %8s  tau %7s ms"
              % (release, seq[:14], rec["fs_slow_hz"], rec["fs_fast_hz"],
                 rec["slope_slow"], rec["slope_fast"],
                 rec["S0_ratio_fast_over_slow"], rec["ratio_from_differencing"],
                 rec.get("ratio_excess", "-"),
                 rec.get("residual_frac", "-"), rec.get("tau_ms", "-")))

    if not rows:
        print("맞댈 쌍이 없다")
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
    print("%d 쌍 -> %s" % (len(rows), args.out))

    by = {}
    for r in rows:
        by.setdefault(r["release"], []).append(r)
    for rel, rs in sorted(by.items()):
        ratio = sorted(r["S0_ratio_fast_over_slow"] for r in rs
                       if r["S0_ratio_fast_over_slow"] != "")
        res = sorted(float(r["residual_frac"]) for r in rs if r.get("residual_frac"))
        exc = sorted(r["ratio_excess"] for r in rs if r.get("ratio_excess"))
        print("%-8s n=%-3d 밀도비 중앙 %.3f (%.3f~%.3f)  차분설명 %.3f  "
              "초과 중앙 %s  잔차 중앙 %s"
              % (rel, len(rs), ratio[len(ratio) // 2], ratio[0], ratio[-1],
                 rs[0]["ratio_from_differencing"],
                 "%.3f" % exc[len(exc) // 2] if exc else "-",
                 "%.3f" % res[len(res) // 2] if res else "-"))


if __name__ == "__main__":
    main()
