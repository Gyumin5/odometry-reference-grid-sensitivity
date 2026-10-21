"""규약 훑기 묶음표를 묶는다.

틀리면 조용한 쪽으로 틀린다. 변형 이름을 축과 값으로 가를 때 밑줄이 하나 더
들어간 이름(rate_0.5_nearest)을 잘못 끊으면 서로 다른 변형이 한 줄에 겹치고,
회전 칸과 병진 칸이 바뀌어도 표는 멀쩡해 보인다.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

from protocol_sweep_table import (collect, flatten, pooled_row,  # noqa: E402
                                  split_variant)

COLS = ["sequence", "n_frames", "n_segments",
        "D_shift_0.5_deg_per_100m", "D_shift_0.5_pct",
        "D_rate_0.5_nearest_deg_per_100m", "D_rate_0.5_nearest_pct"]


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)


def row(sequence, rot="0.3", trans="0.9"):
    return {"sequence": sequence, "n_frames": "100", "n_segments": "50",
            "D_shift_0.5_deg_per_100m": rot, "D_shift_0.5_pct": trans,
            "D_rate_0.5_nearest_deg_per_100m": "1.1",
            "D_rate_0.5_nearest_pct": "2.2"}


def test_the_axis_is_cut_off_before_the_first_value_not_after():
    assert split_variant("D_shift_0.5_pct") == ("shift", "0.5")
    assert split_variant("D_decim2_even_pct") == ("decim2", "even")


def test_a_variant_whose_value_itself_holds_an_underscore_stays_whole():
    # rate_0.5_nearest 와 rate_0.5_slerp 가 한 줄로 겹치면 보간 방식 비교가
    # 통째로 사라지는데, 표는 줄 수만 줄어들 뿐 멀쩡해 보인다.
    assert split_variant("D_rate_0.5_nearest_pct") == ("rate", "0.5_nearest")
    assert split_variant("D_rate_0.5_slerp_pct") == ("rate", "0.5_slerp")


def test_the_two_units_land_in_their_own_columns():
    got = flatten("kitti", "seg", row("ALL", rot="0.34", trans="0.89"))
    shift = [r for r in got if r["axis"] == "shift"][0]
    assert shift["D_rot_deg_per_100m"] == "0.34"
    assert shift["D_trans_pct"] == "0.89"


def test_only_the_pooled_row_is_read(tmp_path):
    p = tmp_path / "kitti_segment_sweep.csv"
    write(str(p), [row("00", rot="9.9"), row("ALL", rot="0.34")])
    assert pooled_row(str(p))["D_shift_0.5_deg_per_100m"] == "0.34"


def test_a_sweep_without_a_pooled_row_is_not_guessed_from_one_sequence(tmp_path):
    p = tmp_path / "kitti_segment_sweep.csv"
    write(str(p), [row("00"), row("01")])
    assert pooled_row(str(p)) is None


def test_a_missing_sweep_drops_out_instead_of_stopping_the_table(tmp_path):
    p = tmp_path / "kitti_segment_sweep.csv"
    write(str(p), [row("ALL")])
    got = collect(str(tmp_path), releases=("kitti", "boreas"),
                  rules=(("seg", "segment"),))
    assert {r["release"] for r in got} == {"kitti"}


def test_the_two_rules_stay_apart_in_the_long_table(tmp_path):
    write(str(tmp_path / "kitti_segment_sweep.csv"), [row("ALL", rot="0.34")])
    write(str(tmp_path / "kitti_rpe1m_sweep.csv"), [row("ALL", rot="26.8")])
    got = collect(str(tmp_path))
    by = {(r["rule"], r["axis"]): r for r in got}
    assert by[("seg", "shift")]["D_rot_deg_per_100m"] == "0.34"
    assert by[("rpe1m", "shift")]["D_rot_deg_per_100m"] == "26.8"
