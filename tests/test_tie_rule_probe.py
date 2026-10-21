"""동률 탐침이 세는 것이 실제로 동률인지 본다.

이 탐침의 값어치는 "밟히나 안 밟히나" 한 줄에 있다. 세는 조건이 헐거우면
안 밟히는 자리를 밟힌다고 적고, 빡빡하면 밟히는 자리를 놓친다. 둘 다
규칙을 적을지 말지를 거꾸로 돌린다.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

from tie_rule_probe import RULES, flips, method_gap  # noqa: E402
from tie_rule_probe import reason, worst  # noqa: E402


def test_a_binary_exact_grid_lands_queries_exactly_between_samples():
    """하나 걸러 고른 격자는 남은 질의를 두 표본 한가운데에 놓는다.

    이것이 동률이 우연이 아닌 이유다. 다만 한가운데가 정확한 동률이 되려면
    두 간격이 이진으로 같아야 한다 — 0.25 초 격자는 그렇다.
    """
    t = np.arange(200, dtype=float) * 0.25
    n_q, n_diff, closest = flips(t)
    assert n_q > 0
    assert n_diff > 0
    assert closest == 0.0


def test_a_ten_hertz_decimal_grid_does_not_make_an_exact_tie():
    """십진 0.1 초는 이진으로 안 떨어져서 한가운데가 동률이 아니다.

    실제 자료가 그렇다 — 키티는 질의 69,525 가운데 동률이 0 이고 가장
    한가운데에 가까운 것도 1.38e-11 초 치우쳐 있다. 동률을 "고른 간격이면
    생긴다" 로 적으면 그 릴리스를 틀리게 센다.
    """
    _, n_diff, closest = flips(np.arange(200, dtype=float) / 10.0)
    assert n_diff == 0
    assert 0.0 < closest < 1e-9


def test_an_unevenly_sampled_release_never_hits_a_tie():
    # 간격이 제각각이면 한가운데에 정확히 떨어질 이유가 없다. 그런 자료에서
    # 동률을 세면 그건 세는 쪽이 느슨한 것이다.
    rng = np.random.default_rng(0)
    t = np.cumsum(0.1 + 0.03 * rng.random(300))
    _, n_diff, closest = flips(t)
    assert n_diff == 0
    assert closest > 0.0


def test_too_short_a_sequence_is_counted_as_nothing_rather_than_guessed():
    assert flips(np.arange(5, dtype=float)) == (0, 0, None)


def test_the_worst_split_ignores_the_variants_the_rule_cannot_touch():
    """최근접이 안 들어간 변형은 두 규칙에서 정확히 같다.

    그 변형까지 후보에 넣으면 0 이 최대로 뽑히는 일은 없지만, 뒤집기와
    무관한 자리가 "가장 크게 갈린 변형" 으로 적힐 수 있다.
    """
    low = {"shift_0.5": 1.0, "rate_0.5_nearest": 1.0, "t_method_0.7": 2.0}
    high = {"shift_0.5": 9.0, "rate_0.5_nearest": 1.02, "t_method_0.7": 2.0}
    name, a, b, pct = worst(low, high)
    assert name == "rate_0.5_nearest"
    assert (a, b) == (1.0, 1.02)
    assert abs(pct - 2.0) < 1e-9


def test_a_release_with_nothing_measured_names_no_variant():
    assert worst({}, {}) is None


def test_the_method_share_is_read_as_a_ratio_of_the_two_rules():
    """방식의 몫이 규칙이 만든 것인지 보려면 뒤집기 전후의 비를 견줘야 한다.

    차를 보면 D 가 큰 릴리스가 늘 더 크게 움직인 것처럼 보인다. 비는 그
    크기를 나눠 없앤다. 주기 가운데 비가 가장 크게 달라진 자리를 고른다.
    """
    low = {"rate_0.5_slerp": 1.0, "rate_0.5_nearest": 10.0,
           "rate_0.7_slerp": 1.0, "rate_0.7_nearest": 4.0}
    high = {"rate_0.5_slerp": 1.0, "rate_0.5_nearest": 10.1,
            "rate_0.7_slerp": 1.0, "rate_0.7_nearest": 2.0}
    (rate, a, b), swaps = method_gap(low, high)
    assert rate == 0.7
    assert (a, b) == (4.0, 2.0)
    # 0.7 최근접이 0.5 최근접 아래로 안 내려가 순서는 그대로다.
    assert swaps == 0


def test_a_variant_that_changes_places_is_counted():
    """순서가 바뀌면 그 뒤집기가 변형들의 줄 세우기를 바꾼 것이다."""
    low = {"a": 1.0, "b": 2.0}
    high = {"a": 3.0, "b": 2.0}
    _, swaps = method_gap(low, high)
    assert swaps == 1


def test_a_release_with_no_rate_pair_gives_no_ratio():
    (best), swaps = method_gap({"null": 1.0}, {"null": 1.0})
    assert best is None
    assert swaps == 0


def test_both_segment_rules_are_measured_not_just_the_long_one():
    """한 규칙만 재면 그 규칙에서 구간이 안 서는 릴리스가 통째로 빈다.

    튬 알지비디는 실내 궤적이 짧아 100~800 m 가 한 구간도 안 선다. 상한 표는
    그 릴리스를 1 m 규칙으로 싣는데, 동률 탐침이 긴 규칙만 보면 실린 줄에
    대해서는 아무것도 안 잰 셈이 된다.
    """
    names = [name for name, _, _ in RULES]
    assert names == ["seg", "rpe1m"]
    assert dict((n, (lengths, step)) for n, lengths, step in RULES)["rpe1m"] \
        == ((1.0,), 1)


def test_an_empty_cell_says_which_kind_of_empty_it_is():
    """빈 칸이 "잴 것이 없었다" 인지 "재다 못 냈다" 인지 글로 남는다.

    둘을 안 가르면 읽는 쪽이 빈 칸을 측정 누락으로 읽는다. 앞쪽은 누락이
    아니라 결과다.
    """
    assert "한 구간도" in reason(0, None)
    assert "최근접" in reason(5, None)
    assert reason(0, ("rate_0.5_nearest", 1.0, 1.0, 0.0)) == ""
