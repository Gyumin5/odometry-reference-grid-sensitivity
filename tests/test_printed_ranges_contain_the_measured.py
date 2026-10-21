"""찍힌 상한·하한이 잰 값을 수로 품는지 본다.

구문나무 검사(tests/test_make_paper_macros.py 의
test_no_range_end_is_rounded_the_plain_way)와 축이 다르다. 그쪽은 깎는 함수를
거쳤는지를 보고, 이쪽은 깎기 전 값과 찍힌 값을 수로 맞댄다 — 방향을 거꾸로
깎아도 구문나무 검사는 그대로 통과하기 때문이다. 실제로 최댓값에 아래쪽
깎기를 붙여 37.628 을 37.6 으로 찍은 적이 있다.

깎는 두 함수를 방향 표식을 붙여 돌려주는 함수로 바꿔 생성기를 한 번 더 돌리고,
그 결과를 본문이 싣는 numbers.tex 와 이름별로 맞댄다. 이름에 Min·Max 가 있는지
보지 않는다 — 어느 방향으로 깎기로 한 자리인지는 생성기가 어느 함수를 불렀는지가
정하고, 그 표식이 그걸 그대로 들고 온다. 이름을 손으로 베껴 적지 않으므로
생성기에 매크로가 늘어도 이 검사가 같이 늘어난다.

반대쪽에서 들어가는 검사는 tests/test_body_bound_phrases.py 에 따로 있다. 거기는
원고에서 시작해 본문이 상한으로 부르는 자리를 뽑는다. 둘을 한 파일에 두지 않는
까닭은 이 파일이 공개 나무에 담기기 때문이다 — 원고를 여는 검사가 같이 있으면
파일 없음으로 깨진다.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import make_paper_macros as M          # noqa: E402

MARK = re.compile(r"<(LO|HI) ([^>]+)>")


def generate(out_dir):
    """생성기를 그 폴더에 한 번 돌리고 매크로를 이름->글자 로 읽어 온다."""
    argv = sys.argv
    sys.argv = ["make_paper_macros.py", "--out-dir", out_dir]
    try:
        M.main()
    finally:
        sys.argv = argv
    got = {}
    with open(os.path.join(out_dir, "numbers.tex")) as f:
        for line in f:
            if not line.startswith("\\newcommand{\\"):
                continue
            name, _, rest = line[len("\\newcommand{\\"):].partition("}{")
            got[name] = rest.rstrip()[:-1]
    return got


def number(text):
    """조판 글자에서 수를 읽는다. 수가 아니면 None — 이름·날짜가 그렇다.

    지수 꼴은 파일에 조판 꼴로 적히므로 먼저 되돌린다. 안 되돌리면 그 50 개가
    수가 아닌 것으로 읽혀 방향 검사에서 조용히 빠진다.
    """
    try:
        return float(M.plain_number(text).replace("\\,", "").replace(",", ""))
    except ValueError:
        return None


def both_runs(tmpdir):
    """찍힌 판과 방향 표식만 붙인 판을 같이 만든다."""
    here = os.getcwd()
    os.chdir(ROOT)                     # 기본 경로가 저장소 뿌리 기준이다
    try:
        shown = generate(os.path.join(str(tmpdir), "shown"))
        keep = (M.at_least, M.at_most)
        M.at_least = lambda v, fmt="%.3g": "<LO %r>" % float(v)
        M.at_most = lambda v, fmt="%.3g": "<HI %r>" % float(v)
        try:
            marked = generate(os.path.join(str(tmpdir), "raw"))
        finally:
            M.at_least, M.at_most = keep
    finally:
        os.chdir(here)
    return shown, marked


def test_every_printed_bound_contains_the_measured_value(tmpdir):
    shown, marked = both_runs(tmpdir)
    bad, checked = [], 0
    for name, raw in marked.items():
        hits = MARK.findall(raw)
        if len(hits) != 1:
            continue               # 한 칸에 두 끝을 엮어 적는 자리는 건너뛴다
        side, text = hits[0]
        want, got = number(text), number(shown.get(name, ""))
        if want is None or got is None:
            continue
        checked += 1
        # 이진 표기 찌꺼기로 넘어지지 않게 상대 여유를 둔다. 자리수를 하나
        # 깎은 폭보다 한참 작아서 방향이 뒤집힌 것은 그대로 걸린다.
        slack = abs(want) * 1e-12
        if side == "LO" and got > want + slack:
            bad.append((name, shown[name], text))
        if side == "HI" and got < want - slack:
            bad.append((name, shown[name], text))
    assert not bad, bad
    # 생성기가 통째로 비어도 통과하는 검사가 되지 않게 아래를 고정한다.
    assert checked > 60, checked


