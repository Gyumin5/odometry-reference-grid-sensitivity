"""순위표 간격을 밀리초로 옮기는 환산을 묶는다.

이 환산의 전부는 "D 가 어긋남에 비례한다" 는 전제다. 전제가 깨져도 기울기는
계산되므로 표는 멀쩡해 보이면서 수가 틀린다. 그래서 벗어난 폭을 같이 내고,
그 값이 실제로 휘어짐을 잡는지 시험한다. 단위 환산도 같은 종류다 — 도/100 m
를 도/미터로 안 옮기면 백 배 틀린 수가 조용히 들어간다.
"""

import csv
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

from offset_tolerance import (exceeds, pairs, slope,  # noqa: E402
                              sweep_taus, tolerance_ms)

COLS = ["release", "rule", "axis", "param", "n_segments",
        "D_rot_deg_per_100m", "D_trans_pct"]


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)


def tau_rows(release="kitti", rule="seg", per_ms=0.01):
    out = []
    for ms in (1.0, 2.0, 5.0, 10.0, 25.0, 50.0):
        out.append({"release": release, "rule": rule, "axis": "tau",
                    "param": "%gms" % ms, "n_segments": "100",
                    "D_rot_deg_per_100m": "%g" % (per_ms * ms * 10.0),
                    "D_trans_pct": "%g" % (per_ms * ms)})
    return out


def test_a_straight_line_through_the_origin_gives_back_its_slope():
    k, dev = slope([(1.0, 0.01), (10.0, 0.1), (50.0, 0.5)])
    assert abs(k - 0.01) < 1e-12
    assert dev < 1e-12


def test_a_bent_curve_is_reported_as_bent_instead_of_passing_quietly():
    # 어긋남이 커질수록 꺾이는 모양. 기울기는 그래도 나오므로 벗어난 폭이
    # 유일한 신호다.
    _, dev = slope([(1.0, 0.01), (10.0, 0.1), (50.0, 0.2)])
    assert dev > 0.3


def test_the_tolerance_is_read_back_off_the_measured_points():
    pts = [(1.0, 0.01), (10.0, 0.1), (50.0, 0.5)]
    ms, where = tolerance_ms(0.05, pts)
    assert where == "inside"
    assert abs(ms - 5.0) < 1e-12


def test_a_bent_column_is_read_on_its_own_bend_not_on_a_straight_line():
    # 지어낸 휜 칸이다. 실제 자료는 두 칸 다 비례하지만, 비례가 깨진 칸이
    # 들어와도 환산이 직선으로 눌러 담지 않는다는 것을 못박는다. 직선으로
    # 나눴다면 0.0056 을 기울기로 써서 2.7 ms 가 나온다.
    pts = [(1.0, 0.0056), (5.0, 0.0173), (10.0, 0.0333), (50.0, 0.1637)]
    ms, where = tolerance_ms(0.0173, pts)
    assert where == "inside"
    assert abs(ms - 5.0) < 1e-9


def test_a_gap_outside_the_measured_range_is_labelled_not_extrapolated():
    pts = [(1.0, 0.01), (50.0, 0.5)]
    assert tolerance_ms(0.001, pts) == (1.0, "below")
    assert tolerance_ms(9.0, pts) == (50.0, "above")


def test_the_two_ends_of_the_measured_range_are_judged_by_the_same_rule():
    """아래 끝과 위 끝이 다른 부등호를 쓰면 같은 차이가 다르게 갈린다.

    잰 가장 큰 어긋남의 상한과 마지막 비트만 다른 차이는 위로 벗어난 것이다.
    날 부등호로 재면 그 자리가 내부로 떨어져서, 되짚은 값이 50 밀리초 바로
    아래로 적히고 꼬리표도 경계가 아니라 내부가 된다. 아래 끝은 이미
    exceeds 로 재고 있었으므로 같은 차이가 양 끝에서 다르게 갈렸다.
    """
    pts = [(1.0, 0.01), (50.0, 0.5)]
    assert tolerance_ms(0.5 * (1.0 - 1e-12), pts) == (50.0, "above")
    assert tolerance_ms(0.01 * (1.0 + 1e-12), pts) == (1.0, "below")


def test_the_loosened_bracket_still_reads_inside_the_two_points():
    # 느슨한 부등호는 잰 점 바깥의 차이도 구간에 받아들인다. 그대로 비례식에
    # 넣으면 되짚은 밀리초가 그 구간 밖으로 나간다.
    pts = [(1.0, 0.01), (10.0, 0.1), (50.0, 0.5)]
    for gap in (0.1 * (1.0 - 1e-12), 0.1 * (1.0 + 1e-12)):
        ms, where = tolerance_ms(gap, pts)
        assert where == "inside"
        assert 1.0 <= ms <= 50.0


def test_a_pair_that_is_already_tied_is_not_counted_as_offset_sensitive():
    # 찍는 자리수만으로 안 갈리는 쌍은 어긋남과 아무 상관이 없다. 가장 작은
    # 어긋남에서 덮였다고 적으면 그 수가 시간 민감도처럼 읽힌다.
    assert tolerance_ms(0.0, [(1.0, 0.01), (50.0, 0.5)]) == (0.0, "dead")


def test_the_two_reasons_a_pair_fails_get_different_labels():
    pts = [(1.0, 0.01), (50.0, 0.5)]
    assert tolerance_ms(0.0, pts)[1] == "dead"
    assert tolerance_ms(0.001, pts)[1] == "below"
    assert tolerance_ms(0.02, pts)[1] == "inside"


def test_a_gap_equal_to_the_bound_is_not_certified():
    # 0 이 아닌 문턱에서도 마지막 비트가 판정을 가르면 안 된다. 적힌 값
    # 0.0009 와 0.0013 의 쓸 수 있는 간격은 세 칸 빼기 하나라 0.0003 이고,
    # 문턱도 0.0003 이다. 수학적으로 등호라 떨어져야 하는데 이진 곱셈이
    # 0.00030000000000000003 을 남겨 엄격 부등호를 통과했다.
    b = {"entries": [("a", {"rotation": 0.0009}),
                     ("b", {"rotation": 0.0013})]}
    g = pairs(b, "rotation", 4)[0][4]
    assert not exceeds(g, 0.0003), g
    assert exceeds(g, 0.0002)


def test_rotation_is_carried_over_to_the_unit_the_board_prints(tmp_path):
    # 훑기 표는 도/100 m 로 적고 키티 순위표는 도/미터로 적는다. 2D 라
    # 곱은 2/100 이다.
    p = tmp_path / "s.csv"
    write(str(p), tau_rows())
    got = dict(sweep_taus(str(p), "kitti", "seg", "rotation"))
    assert abs(got[10.0] - 0.02 * (0.01 * 10.0 * 10.0)) < 1e-12


def test_translation_is_doubled_and_left_in_percent(tmp_path):
    p = tmp_path / "s.csv"
    write(str(p), tau_rows())
    got = dict(sweep_taus(str(p), "kitti", "seg", "translation"))
    assert abs(got[10.0] - 2.0 * 0.1) < 1e-12


def test_only_the_asked_release_and_rule_are_read(tmp_path):
    p = tmp_path / "s.csv"
    write(str(p), tau_rows(per_ms=0.01)
          + tau_rows(release="boreas", per_ms=0.5)
          + tau_rows(rule="rpe1m", per_ms=0.9))
    got = dict(sweep_taus(str(p), "kitti", "seg", "translation"))
    assert abs(got[1.0] - 2.0 * 0.01) < 1e-12


def test_the_display_width_is_taken_off_the_gap_before_converting():
    # 적힌 차이 그대로 되짚으면 두 수가 반대쪽으로 잘린 경우를 못 버틴다.
    b = {"entries": [("a", {"translation": 0.53}),
                     ("b", {"translation": 0.60})]}
    assert abs(pairs(b, "translation", 2)[0][4] - 0.06) < 1e-12


def test_a_pair_exactly_one_printed_step_apart_has_nothing_to_spend():
    """정확히 한 칸 떨어진 쌍은 어긋남이 0 이어도 조건을 통과하면 안 된다.

    두 수가 서로 반대쪽으로 잘렸으면 참값이 겹치는 쌍이다. 이진 부동소수로
    빼면 0 대신 1e-18 쯤이 남는데 그 부호가 값에 따라 갈려서, 양수로 남은
    쌍만 조용히 통과했다. 실제로 키티 병진에서 한 칸 쌍 45 가운데 41 이
    그렇게 통과했다 — 같은 자리에서 음수로 남은 4 는 탈락했다.
    """
    for lo, hi in ((0.53, 0.54), (0.52, 0.53), (0.81, 0.82)):
        b = {"entries": [("a", {"translation": lo}),
                         ("b", {"translation": hi})]}
        g = pairs(b, "translation", 2)[0][4]
        assert g == 0.0, (lo, hi)
        assert not g > 0.0, (lo, hi)


def test_a_gap_narrower_than_the_display_width_does_not_go_negative():
    # 음수 차이는 뜻이 없다. 0 으로 두면 가장 작은 어긋남에서 이미 덮인다는
    # 판정이 그대로 나온다.
    b = {"entries": [("a", {"translation": 0.53}),
                     ("b", {"translation": 0.54})]}
    g = pairs(b, "translation", 1)[0][4]
    assert g == 0.0
    assert tolerance_ms(g, [(1.0, 0.01), (50.0, 0.5)]) == (0.0, "dead")


def test_a_release_without_offset_rows_comes_back_empty_not_wrong(tmp_path):
    p = tmp_path / "s.csv"
    write(str(p), tau_rows())
    assert sweep_taus(str(p), "pit30m", "seg", "translation") == []
    with pytest.raises(ValueError):
        slope([])
