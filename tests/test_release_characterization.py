"""릴리스 표를 묶는 함수들이 빈칸과 한 건짜리를 어떻게 다루는지 고정한다.

표에 들어갈 숫자라 조용히 틀리면 안 된다. 특히 빈칸을 0 으로 읽으면 '안 쟀다' 가
'0 이다' 로 바뀐다.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

from release_characterization import num, span  # noqa: E402


def test_blank_and_nan_are_dropped_not_read_as_zero():
    rows = [{"v": "1.0"}, {"v": ""}, {"v": "nan"}, {"v": "3.0"}, {}]
    assert num(rows, "v") == [1.0, 3.0]


def test_a_non_number_is_dropped_instead_of_raising():
    assert num([{"v": "격자 아님"}, {"v": "2"}], "v") == [2.0]


def test_span_reports_median_and_range():
    assert span([1.0, 2.0, 3.0]) == "2.00 (1.00~3.00)"


def test_a_single_value_has_no_range():
    assert span([2.5]) == "2.50"


def test_nothing_measured_stays_blank():
    assert span([]) == ""
