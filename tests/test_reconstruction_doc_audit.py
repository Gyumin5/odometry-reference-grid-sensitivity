"""문서 대조표가 인용으로만 서 있는지 본다.

이 표는 재면 나오는 값이 아니라 남의 문서에서 옮겨 적은 것이다. 그래서 틀리는
방식이 다르다 — 측정 쪽은 다시 돌리면 드러나지만 이쪽은 한 줄을 잘못 적어도
영원히 그대로 있고, 본문은 그걸 "릴리스 몇 중 몇만 적어 뒀다" 같은 수로
바꿔 싣는다. 묶는 것 셋이다. 세는 데 쓰는 이름이 닫힌 목록 안에 있는가, 방식을
적었다고 센 줄마다 원문과 출처가 붙어 있는가, 잰 값과 적힌 값이 같은 칸에
섞이지 않았는가.
"""

import csv
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

from reconstruction_doc_audit import (AUDIT, FIELDS, METHODS,  # noqa: E402
                                      check, main, tally)

BOUND_TABLE = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "results", "next", "release_bound_table.csv")


def test_every_release_in_the_bound_table_has_a_documentation_row():
    # 표에 있는 릴리스가 여기 없으면 본문의 "몇 중 몇" 이 조용히 틀린다.
    with open(BOUND_TABLE) as f:
        want = {r["release"] for r in csv.DictReader(f)}
    assert {r["release"] for r in AUDIT} == want


def test_the_method_names_come_from_a_closed_list():
    assert check() == len(AUDIT)
    for r in AUDIT:
        assert r["method"] in METHODS
        assert r["eval_method"] in METHODS


def test_a_method_claimed_without_a_quote_is_rejected():
    # 인용 없이 방식만 적으면 그 줄이 센 값에 들어가는데 되짚을 근거가 없다.
    bad = dict(AUDIT[0], quote="")
    with pytest.raises(ValueError):
        check([bad])


def test_an_unknown_method_name_is_rejected():
    bad = dict(AUDIT[0], method="linear")
    with pytest.raises(ValueError):
        check([bad])


def test_the_same_release_cannot_be_listed_twice():
    with pytest.raises(ValueError):
        check([AUDIT[0], dict(AUDIT[0])])


def test_what_we_measured_is_never_written_into_what_they_stated():
    # 문서가 주기를 안 적은 릴리스는 published_hz 가 "미기재" 여야 한다.
    # 우리가 잰 값을 그 칸에 넣으면 그 릴리스가 적어 둔 것처럼 읽힌다.
    for r in AUDIT:
        if r["published_hz"] == "미기재":
            assert r["measured_hz"] not in ("", "미기재"), r["release"]
            assert r["published_hz"] != r["measured_hz"]


def test_the_tally_matches_the_rows_it_counts():
    t = tally()
    assert t["n_releases"] == len(AUDIT)
    assert t["n_method_named"] + t["n_method_silent"] == t["n_releases"]
    assert t["n_rate_stated"] + t["n_rate_silent"] == t["n_releases"]
    assert t["n_method_named"] == len(
        [r for r in AUDIT if r["method"] != "none"])
    assert t["n_nearest"] == len(
        [r for r in AUDIT if "nearest" in (r["method"], r["eval_method"])])


def test_the_releases_that_name_a_method_are_the_ones_quoted():
    named = {r["release"]: r["method"] for r in AUDIT
             if r["method"] != "none"}
    assert named == {"kitti": "nearest", "helipr": "bspline",
                     "nclt": "odom"}
    quote = dict((r["release"], r["quote"]) for r in AUDIT)
    assert "closest timestamp" in quote["kitti"]
    assert "B-spline" in quote["helipr"]
    assert "interpolation using odometry" in quote["nclt"]


def test_the_written_file_carries_every_column(tmp_path):
    out = tmp_path / "a.csv"
    sys.argv = ["x", "--out", str(out)]
    main()
    with open(out) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(AUDIT)
    assert list(rows[0].keys()) == list(FIELDS)
    assert all(r["url"].startswith("https://") for r in rows)
