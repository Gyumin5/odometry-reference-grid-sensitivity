"""그림이 표와 다른 수를 그리지 않게 묶는다.

그림은 빌드가 깨지지 않으므로 틀려도 안 보인다. 본문이 2D 로 말하는데 그림이
D 를 그리면 읽는 사람은 둘이 같은 양이라고 읽는다. 구간이 하나도 안 서는
릴리스가 0 으로 끼어드는 것도 같은 종류다.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

from make_offset_figure import (display_step, series,  # noqa: E402
                                worst_floor)


def sweep_row(release, rule, axis, param, rot, trans):
    return {"release": release, "rule": rule, "axis": axis, "param": param,
            "n_segments": "10", "D_rot_deg_per_100m": rot,
            "D_trans_pct": trans}


def test_the_figure_plots_twice_D_because_the_text_does():
    """표는 D 를 적고 본문은 2D 로 말한다. 그림은 본문 쪽이다."""
    rows = [sweep_row("kitti", "seg", "tau", "1ms", "0.003", "0.008"),
            sweep_row("kitti", "seg", "tau", "10ms", "0.03", "0.08")]
    got = series(rows, "kitti", "seg", "D_trans_pct")
    assert got == [(1.0, 0.016), (10.0, 0.16)]


def test_a_release_without_that_rule_draws_no_line():
    """튬 알지비디는 100~800 m 규칙에서 구간이 하나도 안 선다."""
    rows = [sweep_row("tumrgbd", "rpe1m", "tau", "1ms", "0.1", "0.2")]
    assert series(rows, "tumrgbd", "seg", "D_trans_pct") == []


def test_only_the_offset_axis_is_plotted():
    """위상·솎기·주기 변형은 같은 표에 있지만 가로축이 밀리초가 아니다."""
    rows = [sweep_row("kitti", "seg", "decim2", "even", "0.04", "0.07"),
            sweep_row("kitti", "seg", "tau", "2ms", "0.006", "0.017")]
    assert [p[0] for p in series(rows, "kitti", "seg", "D_trans_pct")] == [2.0]


def test_the_residue_line_is_the_worst_of_that_rule():
    """바닥은 릴리스마다 다르다. 한 줄만 그리면 가장 높은 것이어야 한다."""
    rows = [{"release": "kitti", "rule": "seg", "floor_deg_per_100m": "0.004"},
            {"release": "boreas", "rule": "seg",
             "floor_deg_per_100m": "1.7e-07"},
            {"release": "kitti", "rule": "rpe1m",
             "floor_deg_per_100m": "1.122"}]
    assert worst_floor(rows, "seg", ["kitti", "boreas"]) == 0.004


def test_the_residue_line_ignores_a_release_the_figure_does_not_draw():
    # 설명문은 이 선을 "그려진 선들의 잔여 중 가장 큰 것" 으로 읽는다. 선이
    # 없는 릴리스가 최대를 쥐면 그림에 근거가 없는 높이에 선이 그어진다.
    rows = [{"release": "kitti", "rule": "seg", "floor_deg_per_100m": "0.004"},
            {"release": "tumrgbd", "rule": "seg",
             "floor_deg_per_100m": "9.9"}]
    assert worst_floor(rows, "seg", ["kitti"]) == 0.004
    with pytest.raises(SystemExit):
        worst_floor(rows, "seg", ["nuscenes"])


def test_the_printed_step_comes_from_the_board_that_prints_it():
    """칸 폭은 순위표와 칸마다 다르다. 병진 폭을 회전 선에 쓰면 백 배 틀린다."""
    rows = [{"board": "kitti", "metric": "translation", "display_term": "0.01"},
            {"board": "kitti", "metric": "rotation", "display_term": "0.0001"}]
    assert display_step(rows, "kitti", "rotation") == 0.0001
    assert display_step(rows, "kitti360", "rotation") is None


def test_the_figure_knows_every_release_the_table_measures():
    # 그림은 자기 이름표에 있는 릴리스만 돈다. 릴리스를 더하면서 이 표를 안
    # 고치면 그 선이 조용히 빠지는데, 설명문의 분류는 훑기표에서 나오므로
    # 그려지지 않은 이름이 "그려진 선" 으로 적힌다. 실제로 그랬다 — 열째
    # 릴리스 유로크가 그림에 없는 채로 설명문에 들어갔다.
    from kitti_segment_D import RELEASES
    from make_offset_figure import LABELS, MARKERS
    assert set(RELEASES) <= set(LABELS), sorted(set(RELEASES) - set(LABELS))
    assert set(LABELS) == set(MARKERS)
    assert len(set(MARKERS.values())) == len(MARKERS)
