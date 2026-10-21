"""릴리스별 상한을 한 장으로 모을 때 조용히 틀리는 자리를 묶는다.

셋이다. 안 잰 릴리스를 0 으로 채워 표에 올리는 것(없는 측정이 측정처럼
보인다), 묶음 행 대신 첫 시퀀스를 집는 것(한 시퀀스의 성질이 릴리스
전체의 값으로 둔갑한다), 시퀀스별 폭을 잴 때 묶음 행을 같이 세는 것
(폭이 묶음 값 쪽으로 좁아진다).
"""

import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

from release_bound_table import POOLED, collect, read_rows  # noqa: E402

COLS = ["sequence", "n_frames", "n_segments",
        "twoD_phase_half_deg_per_100m", "twoD_t_phase_half_pct",
        "twoD_phase_half_rms_deg_per_100m", "twoD_t_phase_half_rms_pct"]


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)


def row(seq, rot="1.0", trans="2.0"):
    return {"sequence": seq, "n_frames": "100", "n_segments": "10",
            "twoD_phase_half_deg_per_100m": rot,
            "twoD_t_phase_half_pct": trans,
            "twoD_phase_half_rms_deg_per_100m": rot,
            "twoD_t_phase_half_rms_pct": trans}


def test_the_pooled_row_is_the_one_that_is_read(tmp_path):
    p = tmp_path / "kitti_segment_D.csv"
    write_csv(str(p), [row("00", "9.0", "9.0"), row(POOLED, "1.0", "2.0")])
    got, per_seq = read_rows(str(p))
    assert got["sequence"] == POOLED
    assert got["twoD_phase_half_deg_per_100m"] == "1.0"
    # 묶음 행은 시퀀스 쪽에 끼지 않는다. 끼면 폭이 그쪽으로 좁아진다.
    assert [r["sequence"] for r in per_seq] == ["00"]


def test_a_file_without_a_pooled_row_gives_nothing(tmp_path):
    p = tmp_path / "kitti_segment_D.csv"
    write_csv(str(p), [row("00")])
    assert read_rows(str(p))[0] is None


def test_a_missing_file_gives_nothing_rather_than_an_empty_row(tmp_path):
    assert read_rows(str(tmp_path / "nope.csv")) == (None, [])


def test_the_per_sequence_range_comes_from_the_sequences(tmp_path):
    write_csv(str(tmp_path / "kitti_segment_D.csv"),
              [row("00", "0.2", "0.5"), row("01", "0.8", "0.9"),
               row(POOLED, "0.5", "0.7")])
    got = collect(str(tmp_path))[0]
    assert got["twoD_rot_seq_min_deg_per_100m"] == "0.2"
    assert got["twoD_rot_seq_max_deg_per_100m"] == "0.8"
    assert got["n_sequences"] == "2"


def test_only_the_releases_that_were_measured_reach_the_table(tmp_path):
    # 안 잰 릴리스가 빈 값으로 표에 오르면, 읽는 사람은 그 릴리스에서 상한이
    # 작다고 읽는다. 아예 줄이 없어야 한다.
    write_csv(str(tmp_path / "kitti_segment_D.csv"), [row(POOLED)])
    write_csv(str(tmp_path / "kitti_rpe1m_D.csv"), [row(POOLED)])
    write_csv(str(tmp_path / "helipr_segment_D.csv"), [row(POOLED)])
    rows = collect(str(tmp_path))
    assert {(r["release"], r["rule"]) for r in rows} == {
        ("kitti", "seg"), ("kitti", "rpe1m"), ("helipr", "seg")}


def test_the_two_segment_rules_are_kept_apart(tmp_path):
    # 같은 릴리스의 두 규칙이 한 줄로 합쳐지면 100~800 m 값과 1 m 값이 섞인다.
    write_csv(str(tmp_path / "kitti_segment_D.csv"), [row(POOLED, "0.3")])
    write_csv(str(tmp_path / "kitti_rpe1m_D.csv"), [row(POOLED, "27.9")])
    rows = collect(str(tmp_path))
    by = {r["rule"]: r for r in rows}
    assert by["seg"]["twoD_rot_deg_per_100m"] == "0.3"
    assert by["rpe1m"]["twoD_rot_deg_per_100m"] == "27.9"
    assert by["seg"]["segment_length"] != by["rpe1m"]["segment_length"]
