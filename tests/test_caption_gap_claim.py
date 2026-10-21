"""표 캡션이 "TUM VI 의 100~800 m 구간은 전부 공백을 가로지른다" 고 적는다.

그 문장에는 수가 없어서 느슨한 수 검사에 안 걸리고, 매크로도 아니라서 자료가
바뀌어도 저절로 안 따라간다. 자료가 그 말을 더는 받치지 않게 되면 원고만 틀린
채 남으므로 여기서 묶는다. 걸리면 자료를 보고 글을 고친다 — 반대로 하지 않는다.
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SCORE = os.path.join(HERE, os.pardir, "results", "next",
                     "gap_policy_score.csv")


def seg_rows(release):
    with open(SCORE) as f:
        return [r for r in csv.DictReader(f)
                if r["release"] == release and r["rule"] == "seg"]


def test_every_tumvi_segment_crosses_a_gap():
    rows = seg_rows("tumvi")
    assert rows, "TUM VI 의 구간 규칙 줄이 하나도 없다"
    for r in rows:
        assert r["segments_crossing"] == r["segments"], (
            "%s 는 %s 구간 중 %s 개만 공백을 가로지른다 — 캡션의 '전부' 가 "
            "더는 맞지 않는다" % (r["sequence"], r["segments"],
                                  r["segments_crossing"]))


def test_the_claim_is_not_vacuously_true_everywhere():
    """다른 릴리스에서는 이 말이 안 선다는 것까지 본다.

    모든 릴리스가 전부 가로지르면 위 검사는 자료의 어떤 성질도 안 가린다.
    """
    partial = [r for r in seg_rows("kitti")
               if r["segments_crossing"] != r["segments"]]
    assert partial, "KITTI 도 전부 가로지르면 TUM VI 를 따로 짚을 이유가 없다"
