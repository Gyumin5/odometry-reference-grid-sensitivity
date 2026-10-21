"""차분 창 예측이 맞는지 본다.

같은 해를 두 주기로 실으면 느린 쪽이 더 눌린다. 그 눌림 중 차분 창으로 설명되는
몫을 떼어내야 '공개 전에 한 번 더 깎였다' 를 말할 수 있다.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

from two_rate_pair_audit import differencing_ratio  # noqa: E402


def test_same_rate_predicts_no_gap():
    assert abs(differencing_ratio(50.0, 50.0, 2.0, 4.0) - 1.0) < 1e-12


def test_faster_publication_is_predicted_louder():
    assert differencing_ratio(10.0, 50.0, 2.0, 4.0) > 1.0


def test_prediction_matches_the_sinc_ratio_at_the_band_middle():
    def att(f, fs):
        x = math.pi * f / fs
        return (math.sin(x) / x) ** 2
    hand = att(3.0, 50.0) / att(3.0, 10.0)
    assert abs(differencing_ratio(10.0, 50.0, 2.0, 4.0) - hand) < 0.02


def test_a_wider_rate_gap_predicts_a_bigger_ratio():
    near = differencing_ratio(10.0, 20.0, 2.0, 4.0)
    far = differencing_ratio(10.0, 200.0, 2.0, 4.0)
    assert far > near > 1.0


def test_band_position_changes_the_prediction():
    low = differencing_ratio(10.0, 100.0, 0.5, 1.0)
    high = differencing_ratio(10.0, 100.0, 2.0, 4.0)
    assert high > low > 1.0
