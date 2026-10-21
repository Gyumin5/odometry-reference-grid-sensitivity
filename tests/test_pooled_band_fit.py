"""시퀀스를 묶어 대역 모양을 보는 쪽이 한 건씩 보는 쪽보다 무엇을 더 아는지 본다.

한 건씩 보면 짧은 기록에서 참으로 백색인 것도 절반만 백색으로 판정된다. 묶으면
그 흔들림이 시퀀스 수만큼 줄어야 한다. 그게 안 되면 묶는 이유가 없다.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

from pooled_band_fit import fit, sequence_spectrum  # noqa: E402
from reference_resolution_audit import rate_segments, ypr_to_rot  # noqa: E402

FREQS = np.linspace(0.1, 25.0, 500)
BAND = (FREQS >= 2.0) & (FREQS <= 4.0)


def test_a_flat_spectrum_reads_as_zero_slope_and_zero_curvature():
    s = np.full_like(FREQS, 3.7)
    slope, curv = fit(FREQS, s, BAND)
    assert abs(slope) < 1e-9
    assert abs(curv) < 1e-9


def test_a_power_law_reads_back_its_exponent():
    for k in (-4.0, -2.0, 2.0):
        slope, curv = fit(FREQS, FREQS ** k, BAND)
        assert abs(slope - k) < 1e-9
        assert abs(curv) < 1e-9


def test_a_bent_band_shows_up_as_curvature_not_as_slope():
    # 내려오는 성분과 올라가는 성분이 겹치면 합이 U 자가 되고 직선 기울기는 0 을
    # 지난다. 굽힘을 같이 재는 이유가 이것이다.
    s = FREQS ** -4.0 + 4e-3 * FREQS ** 2.0
    slope, curv = fit(FREQS, s, BAND)
    assert abs(slope) < 0.5     # 직선으로 보면 백색과 구별이 안 된다
    assert curv > 1.0           # 굽힘이 그 겹침을 드러낸다


def test_pooling_shrinks_the_scatter_of_the_slope():
    rng = np.random.default_rng(5)
    fs, n = 50.0, 975
    t = np.arange(n) / fs
    singles = []
    for _ in range(40):
        w = rng.normal(0, 0.004 * np.sqrt(fs / 50.0), (n, 3))
        ang = np.cumsum(w / fs, axis=0)
        rot = ypr_to_rot(ang[:, 2], ang[:, 1], ang[:, 0])
        f, s = sequence_spectrum(t, rot, max(16, int(round(16.0 * fs))))
        assert f is not None
        singles.append(s)
    band = (f >= 2.0) & (f <= 4.0)
    per = np.array([fit(f, s, band)[0] for s in singles])
    pooled = fit(f, np.mean(singles, axis=0), band)[0]
    # 한 건씩 보면 ±1 넘게 흩어지는데 묶으면 0 근처로 모여야 한다.
    assert per.std() > 0.3
    assert abs(pooled) < per.std()


def test_a_fixed_axis_is_left_out_rather_than_counted_as_quiet():
    # 평면으로 배포된 기준에서 비어 있는 축을 0 으로 더하면 총계가 내려가
    # 조용한 기준으로 읽힌다. 그 축은 빼야 한다.
    rng = np.random.default_rng(6)
    fs, n = 50.0, 2000
    t = np.arange(n) / fs
    yaw = np.cumsum(rng.normal(0, 0.01, n)) / fs
    rot = ypr_to_rot(yaw, np.zeros(n), np.zeros(n))
    f, s = sequence_spectrum(t, rot, 800)
    assert f is not None and np.all(s > 0)


def test_too_short_a_sequence_gives_nothing():
    t = np.arange(40) / 50.0
    rot = ypr_to_rot(np.zeros(40), np.zeros(40), np.zeros(40))
    assert sequence_spectrum(t, rot, 800) == (None, None)


def test_the_shared_segment_split_is_what_the_audit_uses():
    # 묶음 분석과 한 건 분석이 같은 토막을 봐야 한다. 공백을 넣어 두고 둘이
    # 같은 토막 수를 세는지 확인한다.
    fs, n = 50.0, 1200
    t = np.arange(n) / fs
    t[600:] += 5.0  # 5 초 공백
    rng = np.random.default_rng(7)
    ang = np.cumsum(rng.normal(0, 0.01, (n, 3)), axis=0) / fs
    rot = ypr_to_rot(ang[:, 2], ang[:, 1], ang[:, 0])
    segs, dt_med, got_fs, kept, n_inc = rate_segments(t, rot)
    assert len(segs) == 2
    assert abs(got_fs - fs) < 1e-6
    assert kept < n_inc
