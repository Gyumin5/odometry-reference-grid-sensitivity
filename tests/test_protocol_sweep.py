"""위상·주기 훑기가 가른다고 말한 것을 실제로 가르는지 묶는다.

이 측정의 값은 "한 점에 기대지 않는다" 와 "주기와 보간 방식을 섞지 않는다"
둘이다. 그 둘이 코드에서 지켜지지 않으면 표의 숫자가 늘어나기만 하고
가리키는 것이 없다.
"""

import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

from kitti_segment_D import (euclidean, geodesic, proper,  # noqa: E402
                             relative, relative_translation, segment_weights,
                             segments)
from kitti_segment_D import LENGTHS, STEP  # noqa: E402
from protocol_sweep import (RATIOS, SHIFTS, TAUS_MS, as_row,  # noqa: E402
                            coarse_grid, measure, window, window_segments,
                            window_segments_original)

LEN1 = ((1.0,), 1)


def straight(n=400, dt=0.1, v=10.0):
    """등속 직진, 자세 고정. 어떤 보간도 정확히 맞는 궤적이다."""
    t = np.arange(n) * dt
    rot = np.repeat(np.eye(3)[None], n, axis=0)
    pos = np.stack([v * t, np.zeros(n), np.zeros(n)], axis=1)
    return t, rot, pos


def curved(n=400, dt=0.1, v=10.0, omega=0.3):
    """일정 곡률 선회. 보간 방식이 실제로 갈리는 궤적이다.

    각속도가 일정하면 구면 선형보간이 사실상 정확해서, 연속 보간 복원이
    기준과 거의 같아진다. 방식 몫과 주기 몫을 가르는 시험에는 아래
    wiggly 를 쓴다.
    """
    t = np.arange(n) * dt
    yaw = omega * t
    rot = Rotation.from_euler("z", yaw).as_matrix()
    r = v / omega
    pos = np.stack([r * np.sin(yaw), r * (1 - np.cos(yaw)), np.zeros(n)],
                   axis=1)
    return t, rot, pos


def wiggly(n=400, dt=0.1, v=10.0):
    """각속도와 속도가 함께 변하는 궤적. 어떤 복원도 정확하지 않다."""
    t = np.arange(n) * dt
    yaw = 0.6 * np.sin(0.8 * t) + 0.2 * t
    speed = v * (1.0 + 0.4 * np.sin(1.3 * t))
    rot = Rotation.from_euler("z", yaw).as_matrix()
    dpos = np.stack([speed * np.cos(yaw), speed * np.sin(yaw),
                     np.zeros(n)], axis=1) * dt
    pos = np.cumsum(dpos, axis=0) - dpos[0]
    return t, rot, pos


def test_every_query_time_stays_inside_its_own_source_range():
    # 창을 잘못 잡으면 끝에서 자르게 되고, 자른 자리는 중복 끝점이 된다.
    t, _, _ = curved()
    dt = float(np.median(np.diff(t)))
    lo, hi = window(t, dt)
    assert lo >= t[0] and hi <= t[-1]
    assert hi + max(SHIFTS) * dt <= t[-1] + 1e-12
    assert hi + max(TAUS_MS) * 1e-3 <= t[-1] + 1e-12
    assert lo >= t[1] - 1e-12          # 홀수 솎기의 원본 시작
    assert hi <= t[1::2][-1] + 1e-12   # 홀수 솎기의 원본 끝
    for r in RATIOS:
        assert hi <= coarse_grid(t, dt, r)[-1] + 1e-12


def test_a_perfectly_interpolable_trajectory_moves_by_nothing():
    # 연속 보간으로 복원하는 변형은 등속 직진에서 정확히 0 이어야 한다.
    # 0 이 아니면 경계 처리나 구간 대응이 틀어졌다는 뜻이다.
    d, nseg = measure(*straight(), lengths=LEN1[0], step=LEN1[1])
    assert nseg > 0
    for k, v in d.items():
        if "nearest" in k or k.startswith(("method_", "t_method_")):
            continue
        assert abs(v) < 1e-9, (k, v)


def test_nearest_reconstruction_is_not_exact_even_on_a_straight_line():
    # 최근접 복원의 D 는 궤적이 굽어서가 아니라 격자에 끊겨서 생긴다. 이것을
    # 주기 변경의 몫과 한 수로 묶으면 무엇 때문에 큰지 못 읽는다.
    d, _ = measure(*straight(), lengths=LEN1[0], step=LEN1[1])
    assert d["rate_0.5_slerp"] == 0.0 and d["t_rate_0.5_slerp"] < 1e-12
    assert d["t_rate_0.5_nearest"] > 0.01


def test_both_decimation_phases_are_measured_and_can_differ():
    d, _ = measure(*curved(), lengths=LEN1[0], step=LEN1[1])
    assert "decim2_even" in d and "decim2_odd" in d
    assert d["decim2_even"] > 0 and d["decim2_odd"] > 0


def swerving(n=400, dt=0.1, v=10.0):
    """속도는 일정하고 요각속도만 변하는 궤적.

    한 걸음이 정확히 v*dt 미터라 거리로 고른 구간의 끝이 시작과 같은 위상에
    떨어진다. 솎기 위상과 평가 격자의 맞물림만 남겨 보려는 궤적이다.
    """
    t = np.arange(n) * dt
    yaw = 0.6 * np.sin(0.8 * t)
    rot = Rotation.from_euler("z", yaw).as_matrix()
    dpos = np.stack([v * np.cos(yaw), v * np.sin(yaw), np.zeros(n)],
                    axis=1) * dt
    pos = np.cumsum(dpos, axis=0) - dpos[0]
    return t, rot, pos


def trans_D(rot, pos, idx, alt_rot, alt_pos):
    """구간 길이로 나눈 병진 이동량. pool() 이 묶는 것과 같은 양이다."""
    a = relative_translation(rot, pos, idx)
    b = relative_translation(alt_rot, alt_pos, idx)
    L = np.asarray([p[2] for p in idx], dtype=float)
    return float(np.mean(euclidean(a, b) / L))


def test_an_orientation_error_at_the_segment_start_moves_translation_by_that_angle():
    # 긴 구간의 병진 이동량을 정하는 것은 끝점 위치가 아니라 시작 자세다.
    # 상대 병진이 시작 자세의 역으로 돌려진 벡터라, 시작에서 각도가 d 만큼
    # 틀어지면 시작과 끝을 잇는 직선 벡터가 그 각도만큼 돌아간다. 그 길이를
    # 구간 길이로 나누므로 남는 것은 d 에 직선/경로 비를 곱한 값이고, 구간을
    # 길게 잡아도 길이로 줄지 않는다 — 릴리스 측정에서 솎기 두 위상이 병진만
    # 쉰 배 넘게 갈린 까닭이 이것이다(한쪽만 시작 프레임을 품는다).
    _, rot, pos = swerving(n=1200)
    delta = 1e-3
    turn = Rotation.from_euler("z", delta).as_matrix()
    got = []
    for length in (50.0, 400.0):
        idx = segments(pos, (length,), 10)
        assert idx
        alt = rot.copy()
        for i, _, _ in idx:
            alt[i] = rot[i] @ turn
        chord = np.asarray([np.linalg.norm(pos[j] - pos[i])
                            for i, j, _ in idx])
        want = delta * float(np.mean(chord / length))
        d = trans_D(rot, pos, idx, alt, pos)
        assert abs(d - want) < 0.01 * want
        got.append(d)
    # 길이를 여덟 배로 늘려도 같은 자릿수에 남는다.
    assert 0.5 < got[1] / got[0] < 2.0


def test_a_position_error_at_the_segment_end_fades_as_the_segment_grows():
    # 끝점 위치 오차는 길이로 나뉘어 들어간다. 그래서 1 m 규칙에서는 이쪽이
    # 지배하고 100~800 m 규칙에서는 사라진다 — 같은 복원인데 두 규칙의
    # 상한이 열 배 넘게 갈리는 까닭이다.
    _, rot, pos = swerving(n=1200)
    bump = 0.01
    got = []
    for length in (50.0, 400.0):
        idx = segments(pos, (length,), 10)
        alt = pos.copy()
        for _, j, _ in idx:
            alt[j] = pos[j] + np.array([0.0, bump, 0.0])
        d = trans_D(rot, pos, idx, rot, alt)
        assert abs(d - bump / length) < 0.02 * bump / length
        got.append(d)
    assert got[0] > 5.0 * got[1]


def test_the_untouched_convention_is_measured_so_the_floor_is_visible():
    # 직교하는 자세로 만든 궤적에서는 바닥이 0 이어야 한다. 0 이 아닌 값이
    # 나오면 그 수는 규약 몫이 아니라 자료의 수치 바닥이다.
    d, _ = measure(*wiggly(), lengths=LEN1[0], step=LEN1[1])
    assert "null" in d
    # 배정밀도 왕복분만 남는다. 가장 작은 실제 변형보다 세 자릿수 아래다.
    assert d["null"] < 1e-8 and d["t_null"] < 1e-8
    assert d["null"] < d["tau_1ms"] / 1000.0


def test_a_release_whose_matrices_are_not_quite_orthonormal_shows_a_floor():
    # 공개 자료를 흉내 낸다 — 행렬에 1e-7 짜리 흠집을 내면 회전으로 바꿀 때
    # 가장 가까운 회전으로 눌리고, 그 눌림이 바닥으로 나타난다.
    t, rot, pos = wiggly()
    rng = np.random.default_rng(0)
    rough = rot + rng.normal(scale=1e-7, size=rot.shape)
    d, _ = measure(t, rough, pos, lengths=LEN1[0], step=LEN1[1])
    assert d["null"] > 1e-9
    # 읽을 때 한 번 눌러 두면 그 바닥이 사라진다 — gather 가 그렇게 읽는다.
    fixed, _ = measure(t, proper(rough), pos, lengths=LEN1[0], step=LEN1[1])
    assert fixed["null"] < d["null"] / 100.0


def test_the_shift_sweep_covers_more_than_the_half_sample_point():
    d, _ = measure(*curved(), lengths=LEN1[0], step=LEN1[1])
    got = {k for k in d if k.startswith("shift_")}
    assert got == {"shift_%g" % s for s in SHIFTS}
    # 반 표본이 늘 최댓값이라고 부르지 않으려면 다른 위상도 실제로 달라야 한다.
    assert len({round(d["shift_%g" % s], 12) for s in SHIFTS}) > 1


def test_the_offset_is_also_measured_in_milliseconds():
    # 표본 비율만으로는 릴리스를 가로질러 못 읽는다. 같은 밀리초 어긋남이
    # 주기와 무관하게 비교되려면 절대 단위 칸이 따로 있어야 한다.
    d, _ = measure(*wiggly(), lengths=LEN1[0], step=LEN1[1])
    got = {k for k in d if k.startswith("tau_")}
    assert got == {"tau_%gms" % ms for ms in TAUS_MS}
    v = [d["tau_%gms" % ms] for ms in sorted(TAUS_MS)]
    assert all(b >= a for a, b in zip(v, v[1:]))  # 커질수록 이동량도 커진다
    # 같은 양(반 표본 = 50 밀리초, dt=0.1 초)은 같은 수로 나와야 한다.
    assert abs(d["tau_50ms"] - d["shift_0.5"]) < 1e-12


def test_the_interpolation_method_is_separated_from_the_rate_change():
    # 복원이 둘 다 부정확한 궤적에서는 셋이 서로 다른 수여야 한다. 같아지면
    # 방식 몫이 주기 몫을 그대로 베낀 것이다.
    d, _ = measure(*wiggly(), lengths=LEN1[0], step=LEN1[1])
    for r in RATIOS:
        a, b, m = ("rate_%g_slerp" % r, "rate_%g_nearest" % r, "method_%g" % r)
        assert a in d and b in d and m in d
        assert d[a] > 0 and d[b] > 0 and d[m] > 0
        # 방식 몫은 두 복원 사이의 거리라 삼각부등식 안에 있어야 한다.
        assert d[m] <= d[a] + d[b] + 1e-12
        assert abs(d[m] - d[b]) > 1e-9
        assert abs(d[m] - d[a]) > 1e-9


def sweep_mask(t):
    dt = float(np.median(np.diff(t)))
    lo, hi = window(t, dt)
    return (t >= lo) & (t <= hi)


def original_ids(idx, off):
    """구간을 원본 색인의 (시작, 끝, 요구 길이) 로 적는다."""
    return {(i + off, j + off, L) for (i, j, L) in idx}


def test_the_sweep_set_is_a_subset_of_the_official_set():
    # 훑기 창은 홀수 솎기 때문에 원본 색인 1 에서 시작한다. 그 창의 자세만
    # 떼어 구간을 다시 고르면 시작점이 밀려 공식 집합과 한 점도 안 겹친다.
    # 두 규칙 모두에서 실제 부분집합인지를 원본 색인으로 확인한다.
    t, _, pos = straight(n=600)
    m = sweep_mask(t)
    off = int(np.flatnonzero(m)[0])
    assert off == 1
    for lengths, step in ((LENGTHS, STEP), LEN1):
        official = set(segments(pos, lengths=lengths, step=step))
        got = original_ids(window_segments(pos, m, lengths, step), off)
        assert got, (lengths, step)
        assert got <= official, (lengths, step)
        # 빠진 것은 창 밖에서 시작하거나 창 밖에서 끝나는 것뿐이다.
        last = int(np.flatnonzero(m)[-1])
        for (i, j, _) in official - got:
            assert i < off or j > last


def test_reselecting_inside_the_window_would_have_shifted_the_start_phase():
    # 고친 자리를 못으로 박는다. 창 안에서 다시 고르면 개수는 비슷한데
    # 집합이 통째로 어긋난다 — 그 수를 제외 수라고 부르면 안 된다.
    t, _, pos = straight(n=600)
    m = sweep_mask(t)
    off = int(np.flatnonzero(m)[0])
    official = set(segments(pos, lengths=LENGTHS, step=STEP))
    reselected = original_ids(
        segments(pos[m], lengths=LENGTHS, step=STEP), off)
    anchored = original_ids(window_segments(pos, m, LENGTHS, STEP), off)
    assert not (reselected & official)       # 교집합이 비어 있다
    assert abs(len(reselected) - len(anchored)) <= len(LENGTHS)
    assert anchored <= official

    # 1 m 규칙은 시작점 간격이 1 이라 위상이 밀리지 않는다. 그래서 이
    # 결함은 100~800 m 규칙에서만 났고, 규칙마다 따로 확인해야 한다.
    one = original_ids(segments(pos[m], lengths=LEN1[0], step=LEN1[1]), off)
    assert one <= set(segments(pos, lengths=LEN1[0], step=LEN1[1]))


def test_the_two_index_spaces_describe_the_same_window_set():
    # 같은 집합을 원본 색인과 창 기준 색인으로 각각 돌려준다. 둘이 갈리면
    # 점수를 엉뚱한 배열에 매기거나 구간을 두 번 세게 된다.
    t, _, pos = straight(n=600)
    m = sweep_mask(t)
    off = int(np.flatnonzero(m)[0])
    for lengths, step in ((LENGTHS, STEP), LEN1):
        orig = set(window_segments_original(pos, m, lengths, step))
        local = original_ids(window_segments(pos, m, lengths, step), off)
        assert orig, (lengths, step)
        assert orig == local, (lengths, step)


def test_an_empty_window_yields_no_segments_in_either_space():
    t, _, pos = straight(n=600)
    m = np.zeros(len(t), dtype=bool)
    assert window_segments_original(pos, m, LENGTHS, STEP) == []
    assert window_segments(pos, m, LENGTHS, STEP) == []


def test_the_bound_still_holds_for_every_variant():
    # 고정된 추정 하나를 놓고, 기준을 바꿨을 때 점수 변화가 D 를 넘지 않는가.
    rng = np.random.default_rng(7)
    t, rot, pos = curved()
    dt = float(np.median(np.diff(t)))
    lo, hi = window(t, dt)
    m = (t >= lo) & (t <= hi)
    idx = window_segments(pos, m, *LEN1)
    w = segment_weights(idx)
    L = np.asarray([p[2] for p in idx], dtype=float)

    # 추정은 기준에 작은 잡음을 얹은 것으로 둔다. 어떤 추정이든 상한이 서야 한다.
    noise = Rotation.from_rotvec(0.02 * rng.standard_normal((m.sum(), 3)))
    est_rot = np.einsum("nab,nbc->nac", rot[m], noise.as_matrix())
    est_pos = pos[m] + 0.2 * rng.standard_normal((m.sum(), 3))

    from protocol_sweep import variants
    d, _ = measure(t, rot, pos, lengths=LEN1[0], step=LEN1[1])
    qe = relative(est_rot, idx)
    qte = relative_translation(est_rot, est_pos, idx)
    base_r = float(np.dot(w, geodesic(qe, relative(rot[m], idx))))
    base_t = float(np.dot(w, euclidean(
        qte, relative_translation(rot[m], pos[m], idx))))

    for v in variants(t, rot, pos, dt, lo, hi):
        name = v[0]
        ar, ap = (v[1], v[2]) if len(v) == 3 else (v[1], v[2])
        alt_r = float(np.dot(w, geodesic(qe, relative(ar, idx))))
        alt_t = float(np.dot(w, euclidean(
            qte, relative_translation(ar, ap, idx))))
        if len(v) == 3:
            assert abs(alt_r - base_r) <= d[name] + 1e-12, name
            assert abs(alt_t - base_t) <= d["t_" + name] + 1e-12, name
    assert L.size == len(idx)


def test_the_same_instants_measure_the_same_from_any_clock_origin():
    """시각을 유닉스 기원으로 옮겨 적어도 D 가 그대로여야 한다.

    여기 쓰는 간격 0.25 초는 이진으로 떨어져서, 기원 1.7e9 에 얹어도 모든
    표본이 정확히 같은 순간을 가리킨다. 그러니 두 판은 같은 질문이고 답도
    같아야 한다. 원점을 안 빼면 안 같다 — 1~50 밀리초 어긋남을 더하는 순간
    그 덧셈이 기원 크기의 칸(2.4e-07 초)에 맞춰 떨어져서, 1 밀리초를
    주문했는데 다른 양이 들어간다.
    """
    t, rot, pos = wiggly(n=600, dt=0.25)
    assert np.array_equal(1.7e9 + t - 1.7e9, t)      # 같은 순간임을 못박는다
    near, _ = measure(t, rot, pos, lengths=LEN1[0], step=LEN1[1])
    far, _ = measure(1.7e9 + t, rot, pos, lengths=LEN1[0], step=LEN1[1])
    assert near is not None and far is not None
    assert near == far


def test_the_written_row_reads_back_as_the_number_it_measured():
    """파일로 나간 D 를 되읽으면 잰 값 그대로여야 한다.

    이 줄을 되읽는 쪽은 수를 견준다 — 어긋남 환산은 2D 가 순위표 간격과
    같아지는 자리를 이 점들 사이에서 되짚고, 묶음 표는 변형들을 D 로 줄
    세운다. 다섯 자리로 깎으면 되짚는 점이 상대 5e-06 만큼 움직이고, 간발의
    차로 갈리던 두 변형이 같은 수로 적혀 순서가 사라진다.
    """
    d = {"shift_0.5": 0.0012345678901234567,
         "t_shift_0.5": 0.0023456789012345678}
    row = as_row("00", d, 100, 10)
    assert float(row["D_shift_0.5_pct"]) == 100.0 * d["t_shift_0.5"]
    assert float(row["D_shift_0.5_deg_per_100m"]) == 100.0 * np.degrees(
        d["shift_0.5"])


def test_no_variant_asks_for_a_time_outside_the_released_samples():
    """공통 창이 지키기로 한 성질을 그대로 건다.

    창의 쓸모는 어느 변형도 공개 표본 밖을 묻지 않게 하는 것 하나다.
    밖을 묻는 순간 구면보간은 멈추고 최근접은 조용히 끝 표본을 쓴다 —
    뒤쪽이 더 나쁘다. 한 프레임만 다른 규약으로 섞여 들어가고 아무 신호도
    안 난다. 끝자리 한 눈금 차이로 새는 자리라 주기를 여럿 두고 본다.
    """
    for dt in (0.01, 0.05, 0.1, 0.25, 1.0 / 3.0):
        t = np.arange(600, dtype=float) * dt
        lo, hi = window(t, dt)
        tw = t[(t >= lo) & (t <= hi)]
        assert len(tw) > 0
        for s in SHIFTS:
            assert (tw + s * dt)[-1] <= t[-1]
        for ms in TAUS_MS:
            assert (tw + ms * 1e-3)[-1] <= t[-1]
        for r in RATIOS:
            tc = coarse_grid(t, dt, r)
            assert tc[-1] <= t[-1]
            assert tw[-1] <= tc[-1]
        for sl in (slice(None, None, 2), slice(1, None, 2)):
            assert tw[-1] <= t[sl][-1]
            assert tw[0] >= t[sl][0]


def test_every_variant_is_scored_on_the_one_segment_list():
    """구간·가중치·정규화 길이가 변형마다 다시 골라지지 않는지 본다.

    상한이 서는 조건이 "구간·대응·가중치가 고정" 이라서, 비교 한쪽이
    자기 자세로 구간을 다시 고르면 그 비교는 정리가 말하는 양이 아니다.
    훑기는 원본 자세에서 창 안 구간을 한 번만 고르고 모든 변형에 그것을
    쓴다 — 그 사실을 변형별 정규화 길이로 확인한다. 길이가 한 변형에서만
    달라지면 그 변형이 자기 집합을 들고 있다는 뜻이다.
    """
    import protocol_sweep as ps

    t, rot, pos = wiggly()
    seen = []
    real = ps.segments

    def spy(p, **kw):
        seen.append(np.array(p, dtype=float, copy=True))
        return real(p, **kw)

    ps.segments = spy
    try:
        raw = {}
        d, nseg = measure(t, rot, pos, raw=raw,
                          lengths=LENGTHS, step=STEP)
    finally:
        ps.segments = real
    assert d and nseg > 0
    assert len(raw) > 1

    # 구간을 고르는 호출은 한 번뿐이고, 그 입력은 원본 위치다. 변형 자세가
    # 여기 들어오면 그 변형은 자기 구간 집합을 들고 비교하는 것이 된다.
    assert len(seen) == 1
    assert np.array_equal(seen[0], np.asarray(pos, dtype=float))

    # 모든 변형이 그 하나에 점수를 매긴다. 가중치는 합이 1 이다.
    m = sweep_mask(t)
    idx = window_segments(pos, m, LENGTHS, STEP)
    want = np.asarray([p[2] for p in idx], dtype=float)
    assert len(want) == nseg
    for name, (g, gt, L) in raw.items():
        assert len(g) == nseg, name
        assert len(gt) == nseg, name
        assert np.array_equal(np.asarray(L, dtype=float), want), name
    # 가중치와 정규화 길이도 그 하나에서 나온다: w_k = 1/(N L_k).
    w = segment_weights(idx)
    assert len(w) == nseg
    assert np.allclose(w, 1.0 / (nseg * want), rtol=0, atol=1e-15)


def test_two_variants_that_differ_late_do_not_collapse_into_one_number():
    # 깎는 자리수 아래에서만 갈리는 두 변형이 같은 글자로 적히면, 되읽는
    # 쪽에서 순서가 동률이 된다. 줄 세우기가 거기서 뒤집힌다.
    a = as_row("00", {"t_x": 0.01000000001}, 1, 1)["D_x_pct"]
    b = as_row("00", {"t_x": 0.01000000002}, 1, 1)["D_x_pct"]
    assert a != b
    assert float(a) < float(b)
