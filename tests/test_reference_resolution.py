"""기준 궤적 분해능 측정기가 심어 둔 잡음을 되찾는지 본다.

각속도에 분산이 알려진 백색 잡음을 넣고 자세를 적분해 만든 뒤, 측정기가 그 잡음의
각도 임의보행 계수를 되찾는지 확인한다. 참값은 ARW = sigma_w / sqrt(fs) 다 —
표집률 fs 의 이산 백색 잡음은 단측 스펙트럼 밀도 S0 = 2*sigma_w^2/fs 를 갖고
ARW = sqrt(S0/2) 이기 때문이다.

창 길이는 초로 준다. fs 50 Hz 에서 10.24 s 면 한 창이 512 표본이다.
"""

import math
import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from reference_resolution_audit import (  # noqa: E402
    audit,
    load_kitti360_poses,
    quantum,
    quantum_report,
    so3_log,
    ypr_to_rot,
)

WIN = 10.24


def make_track(fs, duration, sigma_w, seed, flat=False, scale=(1.0, 1.0, 1.0)):
    """심어 둔 잡음을 가진 자세 시계열. 회전은 Z-Y-X 로 조립한다."""
    n = int(fs * duration)
    t = np.arange(n) / fs
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, sigma_w, size=(n, 3)) * np.array(scale)
    w[:, 2] += 0.3 * np.sin(2 * math.pi * 0.2 * t)
    if flat:
        w[:, 0] = 0.0
        w[:, 1] = 0.0
    ang = np.cumsum(w, axis=0) / fs
    rot = ypr_to_rot(ang[:, 2], ang[:, 1], ang[:, 0])
    return t, rot


def run(t, rot, **kw):
    kw.setdefault("horizon", 1.0)
    kw.setdefault("speed", 10.0)
    kw.setdefault("win_s", WIN)
    row = audit(t, rot, "synthetic", **kw)
    assert row is not None
    return row


def test_recovers_injected_random_walk():
    fs, sigma_w = 50.0, 0.004
    t, rot = make_track(fs, 400.0, sigma_w, seed=7)
    row = run(t, rot)
    got = float(row["ARW_rad_per_sqrt_s"])
    want = math.sqrt(3) * sigma_w / math.sqrt(fs)   # 세 축 다 같은 밀도로 넣었다
    assert abs(got - want) / want < 0.2, (got, want)
    # 위치 오차용 계수는 수직 두 축만 센다.
    got_perp = float(row["ARW_perp_rad_per_sqrt_s"])
    want_perp = math.sqrt(2) * sigma_w / math.sqrt(fs)
    assert abs(got_perp - want_perp) / want_perp < 0.2, (got_perp, want_perp)


def test_random_walk_is_independent_of_sampling_rate():
    """같은 잡음 밀도를 다른 표집률로 내보내도 계수는 같아야 한다."""
    vals = []
    for fs in (25.0, 100.0):
        sigma_w = 0.004 * math.sqrt(fs / 50.0)   # S0 를 고정한다
        t, rot = make_track(fs, 400.0, sigma_w, seed=11)
        vals.append(float(run(t, rot)["ARW_rad_per_sqrt_s"]))
    assert abs(vals[0] - vals[1]) / vals[0] < 0.2, vals


def test_total_is_invariant_under_a_constant_body_rotation():
    """몸체 좌표계를 상수 회전만 시켜도 총계가 바뀌면 안 된다.

    같은 운동을 다른 축 규약으로 실은 두 공개본이 다른 값을 받으면 릴리스 비교가
    규약 차이를 재는 것이 된다. 축마다 중앙값을 내고 더하면 그렇게 된다 — 중앙값은
    선형이 아니라 축을 섞으면 합이 안 보존된다. 세 축 스펙트럼을 먼저 더한 뒤
    중앙값을 내면(스펙트럼 행렬의 대각합) 보존된다.
    """
    fs, sigma_w = 50.0, 0.004
    t, rot = make_track(fs, 400.0, sigma_w, seed=19, scale=(10.0, 1.0, 3.0))
    plain = float(run(t, rot)["ARW_rad_per_sqrt_s"])

    a = 0.7
    c = np.array([[math.cos(a), 0.0, math.sin(a)],
                  [0.0, 1.0, 0.0],
                  [-math.sin(a), 0.0, math.cos(a)]])
    turned = float(run(t, rot @ c)["ARW_rad_per_sqrt_s"])
    assert abs(turned - plain) / plain < 0.05, (plain, turned)


def test_forward_axis_selects_which_axes_feed_position():
    """전진축은 릴리스마다 다르다. 박아 두면 넣을 축과 뺄 축이 뒤바뀐다."""
    fs, sigma_w = 50.0, 0.004
    t, rot = make_track(fs, 400.0, sigma_w, seed=31, scale=(10.0, 1.0, 1.0))
    fwd_x = float(run(t, rot, forward="x")["ARW_perp_rad_per_sqrt_s"])
    fwd_z = float(run(t, rot, forward="z")["ARW_perp_rad_per_sqrt_s"])
    # x 축에만 열 배 잡음을 넣었다. x 를 앞으로 보면 빠지고, z 를 앞으로 보면 들어온다.
    assert fwd_z > 3 * fwd_x, (fwd_x, fwd_z)


def test_position_error_grows_with_horizon_to_the_three_halves():
    """각도 오차에 속도를 곱하면 미터가 아니라 초당 미터다. 백색 각속도 오차를 두 번
    적분해야 위치가 되고, 그래서 구간 h 에 h^(3/2) 로 는다."""
    fs, sigma_w = 50.0, 0.004
    t, rot = make_track(fs, 400.0, sigma_w, seed=41)
    one = float(run(t, rot, horizon=1.0)["e_pos_m"])
    four = float(run(t, rot, horizon=4.0)["e_pos_m"])
    assert abs(four / one - 8.0) / 8.0 < 1e-3, (one, four)

    row = run(t, rot, horizon=2.0, speed=7.0)
    want = 7.0 * float(row["ARW_perp_rad_per_sqrt_s"]) * 2.0 ** 1.5 / math.sqrt(3)
    assert abs(float(row["e_pos_m"]) - want) / want < 1e-3


def test_flat_axes_are_reported_not_counted_as_noise():
    fs, sigma_w = 50.0, 0.004
    t, rot = make_track(fs, 400.0, sigma_w, seed=3, flat=True)
    row = run(t, rot)
    assert row["flat_axes"] == "xy"
    assert row["S0_x"] == "0" and row["S0_y"] == "0"
    # 고정축이 잡음으로 세어지면 계수가 sqrt(3) 배로 부풀어 오른다.
    assert float(row["ARW_perp_rad_per_sqrt_s"]) == float(row["ARW_rad_per_sqrt_s"])
    got = float(row["ARW_rad_per_sqrt_s"])
    want = sigma_w / math.sqrt(fs)
    assert abs(got - want) / want < 0.2, (got, want)


def test_effective_bandwidth_brackets_the_signal():
    fs, sigma_w = 50.0, 0.004
    t, rot = make_track(fs, 400.0, sigma_w, seed=5)
    f_eff = run(t, rot)["f_eff_z_hz"]
    assert 0.1 < f_eff < fs / 2, f_eff


def test_resampling_never_makes_more_samples_than_the_file_has():
    """공백을 건너뛰는 증분을 앞 토막에 남겨 두면 보간이 공백 안으로 들어가 없는
    표본을 만든다. 남긴 비율이 1 을 넘으면 그 일이 일어난 것이다."""
    fs, sigma_w = 50.0, 0.004
    t, rot = make_track(fs, 200.0, sigma_w, seed=13)
    t = t.copy()
    t[len(t) // 2:] += 20.0                       # 20 초짜리 공백 하나
    row = run(t, rot)
    assert row["kept_frac"] <= 1.0, row["kept_frac"]
    assert row["n_segments"] == 2, row["n_segments"]


def test_so3_log_round_trips_small_rotations():
    ang = np.array([[0.01, -0.02, 0.03], [0.0, 0.0, 0.0], [0.2, 0.1, -0.15]])
    rot = ypr_to_rot(ang[:, 2], ang[:, 1], ang[:, 0])
    back = so3_log(rot)
    assert np.linalg.norm(back[1]) < 1e-12
    # 작은 각에서는 Z-Y-X 각과 회전벡터가 1차까지 같다.
    assert np.allclose(back[0], ang[0], atol=2e-3)


def allan_dev(x, fs, m):
    """겹치는 앨런 편차. 평균 구간 m 표본에서의 값."""
    theta = np.cumsum(x) / fs                      # 각속도를 적분한 각도
    n = len(theta)
    k = n - 2 * m
    d = theta[2 * m:] - 2 * theta[m:m + k] + theta[:k]
    tau = m / fs
    return math.sqrt(float((d ** 2).mean()) / (2 * tau ** 2))


def test_spectral_coefficient_matches_allan_deviation():
    """같은 잡음을 두 방법으로 읽어 맞는지 본다. 백색 잡음은 tau=1 s 에서 앨런 편차가
    곧 각도 임의보행 계수다."""
    fs, sigma_w = 50.0, 0.004
    t, rot = make_track(fs, 400.0, sigma_w, seed=23, flat=True)
    spectral = float(run(t, rot)["ARW_rad_per_sqrt_s"])

    # z 축만 움직이므로 그 축의 잡음만 남기고 본다. 0.2 Hz 운동은 고역통과로 뺀다.
    rng = np.random.default_rng(23)
    w = rng.normal(0.0, sigma_w, size=int(fs * 400.0))
    allan = allan_dev(w, fs, int(fs))              # tau = 1 s
    assert abs(spectral - allan) / allan < 0.2, (spectral, allan)


def test_random_walk_is_insensitive_to_where_the_band_sits():
    """운동 대역보다 위라면 어느 대역을 보든 같은 계수가 나와야 한다.

    비교 대역은 손으로 고르는 값이라 결과가 그 선택에 끌려가면 안 된다. 운동은
    0.2 Hz 사인 하나뿐이므로 1 Hz 위는 어디나 같은 백색 잡음 바닥이다.
    """
    fs, sigma_w = 50.0, 0.004
    t, rot = make_track(fs, 400.0, sigma_w, seed=23)
    vals = [float(run(t, rot, band_lo=lo, band_hi=hi)["ARW_rad_per_sqrt_s"])
            for lo, hi in ((1.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 16.0))]
    assert max(vals) / min(vals) < 1.1, vals


def test_independent_attitude_error_is_refused_not_converted():
    """자세에 표본마다 독립인 오차가 실려 있으면 그것을 미분한 스펙트럼은 f^2 로
    올라간다. 그 높이를 h 초 구간으로 끌어내리는 적분 환산은 근거가 없다 —
    숫자를 내는 대신 식별 불가라고 말해야 한다."""
    fs = 50.0
    t, rot = make_track(fs, 400.0, 0.0, seed=37)
    rng = np.random.default_rng(37)
    e = rng.normal(0.0, 2e-4, size=(len(t), 3))
    row = run(t, rot @ ypr_to_rot(e[:, 2], e[:, 1], e[:, 0]))
    assert row["model"] == "f^2 경사", (row["model"], row["band_slope"])
    assert row["band_slope"] > 1.5
    assert "ARW_rad_per_sqrt_s" not in row
    # 밀도 자체는 관측이므로 남긴다. 막는 것은 환산이다.
    assert float(row["S0_total_rad2ps2perHz"]) > 0


def test_effective_bandwidth_is_reported_even_when_conversion_is_refused():
    """유효 대역은 그 파일이 무엇을 싣고 있는지의 성질이라 환산과 운명을 같이하지
    않는다. 거부된 릴리스야말로 '이 대역 위로는 아무것도 없다' 를 보여야 하는 쪽이다."""
    fs = 50.0
    t, rot = make_track(fs, 400.0, 0.0, seed=51)
    rng = np.random.default_rng(51)
    e = rng.normal(0.0, 2e-4, size=(len(t), 3))
    row = run(t, rot @ ypr_to_rot(e[:, 2], e[:, 1], e[:, 0]))
    assert "ARW_rad_per_sqrt_s" not in row
    assert "f_eff_min_hz" in row
    assert row["f_eff_min_hz"] >= 0.0


def test_white_rate_noise_passes_the_gate():
    fs, sigma_w = 50.0, 0.004
    t, rot = make_track(fs, 400.0, sigma_w, seed=43)
    row = run(t, rot)
    assert row["model"] == "백색", (row["model"], row["band_slope"])
    assert abs(row["band_slope"]) < 0.5


def test_band_above_nyquist_is_refused_not_guessed():
    fs, sigma_w = 2.0, 0.004
    t, rot = make_track(fs, 400.0, sigma_w, seed=29)
    row = audit(t, rot, "synthetic", horizon=1.0, speed=10.0, win_s=64.0)
    assert row is not None
    assert "ARW_rad_per_sqrt_s" not in row
    assert "나이퀴스트" in row["note"]


def test_quantum_is_an_upper_bound_measured_per_column():
    """열마다 단위가 다르다. 섞어서 하나로 재면 단위가 다른 열이 한 숫자로 뭉개진다.
    릴리스 하나로는 가장 거친 열의 상한을 적는다 — 그 열이 결과를 지배한다."""
    rng = np.random.default_rng(3)
    rows = [[round(float(v), 3), round(float(w), 6)]
            for v, w in zip(rng.normal(size=400), rng.normal(size=400))]
    assert quantum_report(rows) == "<=0.001"


def test_the_bound_never_claims_a_finer_grid_than_the_data_shows():
    """이웃한 두 값이 격자 한 칸 차이로 떨어지지 않으면 최소 간격은 격자의 정수배다.
    그걸 격자라고 단정하지 않고 상한으로 적는다 — 키티-360 에서 같은 1e-10 격자가
    열마다 1e-10, 9e-10, 2.5e-9 로 다르게 잡히는 것이 그 경우다."""
    # 간격이 늘 세 칸인 열. 참 격자는 1e-10 이지만 자료에 남은 증거는 3e-10 뿐이다.
    col = [round(-0.92 + 3e-10 * i, 10) for i in range(400)]
    got = quantum(col)
    assert got is not None
    assert got > 1e-10, "자료에 없는 더 고운 격자를 지어냈다"
    assert abs(got - 3e-10) < 1e-5 * 3e-10


def test_full_precision_values_have_no_measurable_bound():
    """배정밀도 그대로 실린 열은 저장 분해능을 못 잰다. 최소 간격을 적으면 그건
    저장 분해능이 아니라 부동소수점 해상도를 보고하는 것이다."""
    ph = np.linspace(0.0, 2 * np.pi, 600)
    assert quantum([float(v) for v in -0.92 + 1e-14 * np.sin(ph)]) is None
    assert quantum_report([[float(v)] for v in -0.92 + 1e-14 * np.sin(ph)]) == "못 잼"


def test_the_bound_on_continuous_values_shrinks_with_more_samples():
    """격자가 없는 열에서는 이 값이 표본 수를 따라 작아진다. 그래서 격자로 읽으면
    안 되고, 작게 나왔다는 것은 '거친 양자화는 없다' 까지만 말한다."""
    rng = np.random.default_rng(13)
    few = quantum([float(v) * 0.01 - 0.92 for v in rng.normal(size=300)])
    many = quantum([float(v) * 0.01 - 0.92 for v in rng.normal(size=6000)])
    assert few is not None and many is not None
    assert many < few


def test_kitti360_loader_reads_frame_index_as_time():
    """키티-360 은 시각 파일이 없어 프레임 번호에 0.1 s 를 곱한다."""
    import tempfile

    lines = []
    for i in (1, 2, 3, 20):
        m = ypr_to_rot(np.array([0.01 * i]), np.array([0.0]), np.array([0.0]))[0]
        lines.append(" ".join([str(i)] + ["%.10f" % v for v in
                                          np.hstack([m, np.zeros((3, 1))]).ravel()]))
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("\n".join(lines) + "\n")
        path = f.name
    try:
        t, rot, raw = load_kitti360_poses(path)
    finally:
        os.unlink(path)
    assert np.allclose(t, [0.1, 0.2, 0.3, 2.0])
    assert rot.shape == (4, 3, 3)
    assert len(raw[0]) == 9
    # 첫 두 자세 사이의 각은 심어 둔 0.01 rad 이어야 한다.
    inc = rot[0].T @ rot[1]
    assert abs(np.linalg.norm(so3_log(inc[None])[0]) - 0.01) < 1e-9
