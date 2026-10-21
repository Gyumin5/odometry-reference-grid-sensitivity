"""공식 구간 선택 규칙을 그대로 옮겼는지, 흔들림을 세는 쪽이 민감한지 본다.

상한의 전제 중 하나가 구간 집합과 가중치의 고정이다. 공식 평가 프로그램은
기준 궤적에서 누적 이동거리를 다시 재서 끝 프레임을 고르므로, 기준이 달랐으면
그 선택도 달라질 수 있다. 그 달라짐을 세는 것이 이 표인데, 세는 쪽이 둔하면
"안 움직였다" 가 공짜로 나온다 — 규칙을 한 칸 느슨하게 옮겨 적기만 해도 그렇다.

그래서 묶는 것 셋이다. 부등호가 strict 인가(딱 닿는 자리는 안 받는다), 끝까지
못 가면 버리는가, 한 프레임짜리 움직임이 실제로 세어지는가.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

from segment_set_stability import arc, endpoints, tally  # noqa: E402


def line(n=20, step=1.0):
    p = np.zeros((n, 3))
    p[:, 0] = step * np.arange(n)
    return p


def test_the_rule_does_not_accept_the_frame_that_lands_exactly_on_it():
    # evaluate_odometry.cpp 는 dist[i] > dist[first] + len 이다. 1 m 간격
    # 직선에서 3 m 를 물으면 거리가 정확히 3 인 프레임 3 이 아니라 4 다.
    j, d = endpoints(line(), [(0, 3.0)])
    assert np.isclose(d[3], 3.0)
    assert j[0] == 4


def test_a_segment_that_runs_off_the_end_is_dropped():
    j, _ = endpoints(line(n=5), [(0, 3.0), (0, 99.0), (3, 1.0)])
    assert list(j) == [4, -1, -1]


def test_a_rigid_shift_of_the_whole_path_moves_nothing():
    # 누적 거리는 차이로만 쓰이므로 통째로 옮기면 같은 프레임이 나와야 한다.
    p = line()
    j0, _ = endpoints(p, [(0, 3.0), (5, 7.0)])
    j1, _ = endpoints(p + np.array([100.0, -3.0, 7.0]), [(0, 3.0), (5, 7.0)])
    assert list(j0) == list(j1)


def test_a_one_frame_move_is_actually_counted():
    # 경계 바로 앞 프레임을 조금 늘려 문턱을 넘기면 끝이 한 칸 당겨진다.
    p = line()
    q = p.copy()
    q[3:, 0] += 0.01
    j0, _ = endpoints(p, [(0, 3.0)])
    j1, _ = endpoints(q, [(0, 3.0)])
    assert j0[0] == 4 and j1[0] == 3


def drive(n, rate=10.0, step=1.0):
    """시각·자세·위치 세 묶음. 살짝 휘어야 재표집이 뭔가를 바꾼다."""
    t = np.arange(n) / rate
    yaw = 0.002 * np.arange(n)
    c, s = np.cos(yaw), np.sin(yaw)
    z, o = np.zeros(n), np.ones(n)
    rot = np.stack([c, -s, z, s, c, z, z, z, o], -1).reshape(-1, 3, 3)
    pos = np.cumsum(step * np.stack([c, s, z], -1), axis=0)
    return t, rot, pos


def test_the_weight_pools_sequences_the_way_the_bound_table_does():
    # 상한 표는 릴리스의 모든 구간을 한 자루에 담아 센다. 여기도 같은 자루여야
    # 두 수가 같은 것을 잰다 — 시퀀스마다 그 시퀀스의 구간 수로 나누면 짧은
    # 시퀀스의 구간 하나가 긴 시퀀스의 구간 하나보다 무거워진다. 길이가 다른
    # 둘로 건다. 1 m 규칙은 L 이 다 1 이라 분모가 구간 수와 같아야 한다.
    src = [("a", drive(400)), ("b", drive(900))]
    nseq, acc = tally("_synthetic", (1.0,), 1, source=src)
    assert nseq == 2
    for name, (mv, tot, wm, _sh, _dr, _lc, wtot) in acc.items():
        assert abs(wtot - tot) < 1e-9, name
        assert abs(wm - mv) < 1e-9, name
        if name == "null":
            assert mv == 0


def test_a_segment_that_disappears_is_in_the_denominator_only():
    """사라진 구간은 분모에만 남는다 — 본문이 그렇게 읽히게 적어야 한다.

    심사 26 회차가 짚은 자리다. 누적하는 쪽은 분자를 live & (j0 != j1) 로,
    분모를 w 전체로 쌓는다. 그래서 이 몫은 "구간 집합이 얼마나 바뀌었나" 가
    아니라 "살아남은 구간 중 끝이 옮겨간 것의 가중치 몫" 이다. 구간 하나가
    통째로 사라져도 끝이 옮겨간 것이 없으면 몫은 0 이다.
    """
    base = line(n=12)
    short = line(n=8)          # 뒤가 잘려 마지막 구간이 못 선다
    pairs = [(0, 3.0), (2, 3.0), (6, 4.0)]
    j0, _ = endpoints(base, pairs)
    j1, _ = endpoints(short, pairs)
    w = np.asarray([1.0 / L for _i, L in pairs])

    live = (j0 >= 0) & (j1 >= 0)
    moved = live & (j0 != j1)

    assert not live.all(), (list(j0), list(j1))   # 하나는 죽었다
    assert not moved.any()                        # 산 것은 안 옮겨졌다
    assert w[moved].sum() == 0.0                  # 분자
    assert w.sum() > w[live].sum()                # 분모엔 죽은 것이 남아 있다


def test_the_baseline_is_reselected_on_the_window_and_can_differ_at_ties():
    """기준 끝점은 공통 창에서 다시 고른 것이다 — 30 회차.

    segments 는 궤적 전체의 누적거리에서 끝점을 고르고 endpoints 는 창만 떼어
    다시 고른다. 공식 규칙이 엄격한 초과 판정이라, 경계에 정확히 닿는 구간은
    두 누적 방식의 부동소수 차이만으로 갈린다. 간격마다 정확히 0.1 m 를 가는
    균일 궤적에서 1 m 규칙을 걸면 경계에 닿는 구간이 수백 개라 그 자리가
    실제로 밟힌다.

    기준을 창에서 다시 고르는 쪽을 유지한다. 원본 목록을 기준으로 바꾸면 아무
    것도 안 바꾼 규약(null)에서도 그 동률 차이가 "옮겨간 줄" 로 세어져서, 보고
    하는 모든 수가 규약 몫이 아니라 동률 잡음을 얹고 올라간다 — 26 회차가 묶은
    "null 은 0" 이 그 불변량이다. 대신 이 검사가 그 차이의 존재를 못박고 본문이
    기준을 그렇게 적는지 본다.
    """
    from protocol_sweep import window, window_segments
    from segment_set_stability import RULES

    n = 600
    t = 0.1 * np.arange(n)
    pos = np.zeros((n, 3))
    pos[:, 0] = 0.1 * np.arange(n)

    lengths, step = RULES[1][1], RULES[1][2]      # 1 m 규칙
    dt = float(np.median(np.diff(t)))
    lo, hi = window(t, dt)
    m = (t >= lo) & (t <= hi)
    idx = window_segments(pos, m, lengths, step)
    assert idx

    pairs = [(i, L) for (i, _j, L) in idx]
    orig_j = np.asarray([j for (_i, j, _L) in idx])
    reselected, _ = endpoints(np.asarray(pos)[m], pairs)

    assert int((reselected != orig_j).sum()) > 0, "이 입력에서는 갈려야 한다"


def test_the_cumulative_distance_starts_at_zero_and_is_monotone():
    d = arc(line(step=2.0))
    assert d[0] == 0.0
    assert np.all(np.diff(d) > 0)
    assert np.isclose(d[-1], 38.0)
