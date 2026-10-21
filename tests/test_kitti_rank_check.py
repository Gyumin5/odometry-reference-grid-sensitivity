"""공식 순위표와 2D 를 맞대는 판정이 단위와 기준 행을 안 섞는지 본다.

여기서 틀리면 조용히 틀린다. 순위표는 회전을 도/미터로 적는데 우리 보고서는
도/100 m 를 쓰므로, 환산을 한 번 더 걸면 문턱이 100 배 커져 아무것도 인증되지
않는다. 반대로 묶은 행 대신 시퀀스 한 줄을 집으면 문턱이 그 시퀀스의 성질이
되어 릴리스 전체의 판정처럼 보고된다. 순위표가 둘이 되면서 자리가 하나 더
생겼다 — 한 순위표의 상한을 다른 순위표에 들이대는 것이다.
"""

import csv
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

from kitti_rank_check import (  # noqa: E402
    LEADERBOARDS, METRICS, POOLED, PROTOCOLS, display_step, display_term,
    first_certified_distance, pooled_two_d, rounds_to_display, scores,
    verdicts,
)


def row(seq, d_deg_per_m, d_pct=1.0, key=PROTOCOLS[0]):
    return {
        "sequence": seq,
        "D_%s_deg_per_m" % key: "%r" % d_deg_per_m,
        "D_t_%s_pct" % key: "%r" % d_pct,
    }


def board(pairs, metric="translation"):
    return {"entries": [(n, {metric: v}) for n, v in pairs]}


def test_the_bound_comes_from_the_pooled_row_not_a_sequence():
    # 공식 점수는 구간을 통째로 모아 평균한다. 시퀀스 한 줄을 집으면 그
    # 시퀀스의 성질이 릴리스 전체의 판정으로 둔갑한다.
    rows = [row("00", 0.009, 9.0), row(POOLED, 0.001, 0.5)]
    assert abs(pooled_two_d(rows, "rotation")[0] - 0.002) < 1e-12
    assert abs(pooled_two_d(rows, "translation")[0] - 1.0) < 1e-12


def test_the_threshold_takes_the_worst_protocol_not_a_chosen_one():
    # 본문은 이 판정을 "기준 보간 규약에 흔들리지 않는다" 로 읽는다. 그건
    # 규약 하나가 아니라 잰 규약 전부에 대한 말이라, 한 규약으로 문턱을
    # 세우면 다른 규약이 더 큰 값을 낼 때 그 문장이 거짓이 된다.
    r = dict(row(POOLED, 0.001, 0.5))
    r.update(row(POOLED, 0.004, 0.2, key="decim2"))
    r.update(row(POOLED, 0.002, 0.3, key="interp"))
    assert pooled_two_d([r], "rotation") == (pytest.approx(0.008), "decim2")
    assert pooled_two_d([r], "translation") == (pytest.approx(1.0),
                                                "phase_half")


def test_a_pooled_row_without_any_protocol_column_is_an_error():
    with pytest.raises(KeyError):
        pooled_two_d([{"sequence": POOLED}], "rotation")


def test_a_missing_pooled_row_is_an_error_not_a_silent_fallback():
    # 묶은 행이 없으면 옛 CSV 다. 조용히 첫 줄을 쓰면 판정이 바뀐 걸 아무도
    # 모른다.
    with pytest.raises(KeyError):
        pooled_two_d([row("00", 0.001)], "rotation")


def test_the_rotation_bound_stays_in_degrees_per_metre():
    # 순위표가 도/미터로 적으므로 환산은 2 배뿐이다. 보고서용 도/100 m 환산을
    # 여기에 또 걸면 200 배가 되어 모든 쌍이 인증되는 것처럼 보인다.
    assert METRICS["rotation"]["scale"] == 2.0
    assert METRICS["rotation"]["unit"] == "도/미터"


def test_every_unordered_pair_is_counted_once_in_rank_order():
    recs = verdicts(board([("a", 1.0), ("b", 2.0), ("c", 3.0), ("d", 4.0)]),
                    "translation", 10.0)
    assert len(recs) == 6
    assert all(ia < ib for ia, ib, _, _, _, _ in recs)
    assert len({(ia, ib) for ia, ib, _, _, _, _ in recs}) == 6
    assert sorted(r[1] - r[0] for r in recs) == [1, 1, 1, 2, 2, 3]


def test_a_gap_wider_than_the_bound_is_certified():
    recs = verdicts(board([("a", 1.0), ("b", 4.0)]), "translation", 2.0)
    assert recs[0][4] == 3.0 and recs[0][5] is True


def test_a_gap_inside_the_bound_is_not_certified():
    recs = verdicts(board([("a", 1.0), ("b", 2.0)]), "translation", 2.0)
    assert recs[0][5] is False


def test_a_gap_exactly_at_the_bound_counts_as_not_certified():
    recs = verdicts(board([("a", 1.0), ("b", 3.0)]), "translation", 2.0)
    assert recs[0][4] == 2.0 and recs[0][5] is False


def test_each_column_is_ranked_within_itself():
    # 병진으로 줄 세운 순서가 회전 순서와 같지 않다. 병진 등수를 회전 칸에
    # 돌려 쓰면 "바로 옆 등수" 가 그 칸에서는 옆이 아닌 자리를 가리킨다.
    b = {"entries": [("a", {"translation": 1.0, "rotation": 0.9}),
                     ("b", {"translation": 2.0, "rotation": 0.1})]}
    assert scores(b, "translation") == [("a", 1.0), ("b", 2.0)]
    assert scores(b, "rotation") == [("b", 0.1), ("a", 0.9)]


def test_a_board_without_that_column_reports_nothing_for_it():
    # 키티-360 은 회전 칸이 없다. 빈 칸을 0 으로 채우면 모든 쌍이 차이 0 인
    # 비교로 세어져 "하나도 인증 안 됨" 이 공짜로 나온다.
    b = board([("a", 1.0), ("b", 2.0)])
    assert scores(b, "rotation") == []
    assert verdicts(b, "rotation", 1.0) == []


def test_the_nearest_certified_rank_distance_is_the_smallest_one():
    # 1~2 등은 못 가리고 1~4 등은 가린다면 답은 3 이다. 큰 쪽을 돌려주면
    # "몇 등 떨어져야 가려지나" 를 실제보다 멀게 보고한다.
    recs = verdicts(board([("a", 1.0), ("b", 1.4), ("c", 1.8), ("d", 2.2)]),
                    "translation", 1.0)
    assert [r[5] for r in recs] == [False, False, True, False, False, False]
    assert first_certified_distance(recs) == 3
    assert first_certified_distance([(1, 2, "a", "b", 0.1, False)]) is None


def test_the_display_term_is_one_whole_step_inside_a_single_column():
    # 두 항목이 반대쪽으로 잘렸을 때까지 버텨야 충분조건이다. 한 칸 안에서는
    # (h_A+h_B)/2 가 h 와 같다. 절반만 더하면 문턱이 모자란다.
    assert display_step(2) == 0.01
    assert abs(display_term(2) - 0.01) < 1e-15
    assert abs(display_term(2, 4) - 0.00505) < 1e-15


def test_a_column_whose_scores_carry_more_digits_than_claimed_is_caught():
    # 자리수를 크게 잡으면 반올림 항이 작아져 인증이 공짜로 늘어난다.
    b = board([("a", 0.53), ("b", 0.545)])
    assert rounds_to_display(b, "translation", 3) is True
    assert rounds_to_display(b, "translation", 2) is False


def test_the_display_term_moves_a_pair_out_of_certification():
    # 적힌 차이 0.02, 상한 0.015 면 2D 만으로는 인증이지만 폭 0.01 을 더하면
    # 문턱이 0.025 라 아니다. 이 자리가 빠져 있던 것이 이번 수정이다.
    b = board([("a", 0.53), ("b", 0.55)])
    assert verdicts(b, "translation", 0.015)[0][5] is True
    assert verdicts(b, "translation", 0.015 + display_term(2))[0][5] is False


def test_every_stored_leaderboard_declares_the_digits_it_prints():
    for key, b in LEADERBOARDS.items():
        for metric, _, decimals in b["columns"]:
            assert decimals > 0, (key, metric)
            # 적어 둔 값이 그 자리수로 찍힌 수여야 하고, 한 자리 줄이면 안
            # 맞아야 한다 — 자리수를 크게 잡으면 반올림 항이 작아진다.
            assert rounds_to_display(b, metric, decimals), (key, metric)
            assert not rounds_to_display(b, metric, decimals - 1), \
                (key, metric)


def test_every_stored_leaderboard_says_how_its_column_is_aggregated():
    """묶는 꼴을 적어 두지 않은 순위표가 몰래 인증을 받지 못하게 한다.

    상한은 가중 평균 꼴 점수에만 선다. 전에는 안 적힌 자리에서 평균 꼴을
    가정했는데, 그건 인증을 더 쉽게 내주는 쪽이라 틀렸을 때 거짓이 된다.
    """
    for key, b in LEADERBOARDS.items():
        assert b["aggregation"] in ("mean", "unconfirmed"), key
    # 키티는 공식 개발도구 코드에서 평균 꼴을 확인했다. 키티-360 의 상대오차
    # 칸은 어디에도 안 적혀 있다 — 확인되면 그때 이 줄을 바꾼다.
    assert LEADERBOARDS["kitti"]["aggregation"] == "mean"
    assert LEADERBOARDS["kitti360"]["aggregation"] == "unconfirmed"


def test_every_stored_leaderboard_has_no_repeats_and_real_scores():
    for key, b in LEADERBOARDS.items():
        names = [e[0] for e in b["entries"]]
        assert len(set(names)) == len(names), key
        for metric, _, _ in b["columns"]:
            vals = [v for _, v in scores(b, metric)]
            assert len(vals) == len(names), (key, metric)
            assert all(0.0 < v < 10.0 for v in vals), (key, metric)


def test_only_the_datasets_own_leaderboard_is_used():
    # 다른 논문의 결과표를 넣으면 어느 방법을 싣느냐로 결론이 달라진다.
    # 출처가 그 데이터셋 자신의 공식 평가 페이지인지 자료에 묶어 둔다.
    for key, b in LEADERBOARDS.items():
        assert "공식 순위표" in b["source"], key
        assert "cvlibs.net" in b["source"], key
        assert b["measured"], key


def test_the_verdicts_do_not_turn_on_a_binary_subtraction_residue():
    # 적힌 두 수를 이진 부동소수로 빼면 1e-16 쯤의 잔차가 남고 그 부호는 두
    # 값이 무엇이냐에 따라 갈린다. offset_tolerance 는 문턱이 0 이라 그 잔차가
    # 그대로 판정을 갈랐다. 여기 문턱은 2D + h 로 양수라 안 걸린다고 읽었는데,
    # 읽은 말은 잰 값이 아니다. 뺄셈을 표시 격자의 정수 칸으로 바꿔도 판정이
    # 하나도 안 바뀌는지 자료에 묶어 둔다.
    for key, b in LEADERBOARDS.items():
        with open(os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), b["csv"])) as f:
            rows = list(csv.DictReader(f))
        for metric, _, decimals in b["columns"]:
            lim = pooled_two_d(rows, metric)[0] + display_term(decimals)
            thr = lim * (1.0 + 1e-9)
            table = dict(scores(b, metric))
            grid = 10 ** decimals
            for _, _, na, nb, _, ok in verdicts(b, metric, lim):
                on_grid = abs(round(table[na] * grid)
                              - round(table[nb] * grid)) / float(grid)
                assert (on_grid > thr) == ok, (key, metric, na, nb)


def test_each_board_reads_its_own_measurement_file():
    # 상한은 그 릴리스에서 그 구간 규칙으로 잰 것이어야 한다. 키티의 묶음
    # 행을 키티-360 순위표에 들이대면 구간 길이가 100~800 m 대 1 m 로 달라
    # 문턱이 자릿수째 틀린다.
    paths = {key: b["csv"] for key, b in LEADERBOARDS.items()}
    assert len(set(paths.values())) == len(paths)
    assert "kitti360" in paths["kitti360"] and "rpe1m" in paths["kitti360"]
