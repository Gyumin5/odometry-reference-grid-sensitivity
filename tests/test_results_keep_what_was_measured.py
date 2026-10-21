"""결과표가 측정한 수를 그대로 담는지 본다.

본문에 실리는 상한은 두 단을 거친다. 측정이 결과표에 적고, 생성기가 그 열을
읽어 바깥으로 깎아 매크로로 낸다. 바깥으로 깎는 것은 뒷단에만 있으므로,
앞단이 먼저 깎아 두면 보장이 통째로 사라진다 — 생성기가 받는 수가 이미 잰
값보다 작으면, 거기서 다시 올려 깎아도 본문의 "최대 이만큼" 은 잰 값을 안
품는다. 깎는 함수를 부른 자국이 남아서 겉보기에는 지켜진 것처럼 읽힌다.

실제로 그랬다. 직교 이탈 최대와 공식 프로그램 탐침들이 결과표에 세 자리로
적히고 있었고, 생성기가 그 세 자리를 다시 세 자리로 깎아 아무것도 안 바꿨다.

그래서 규칙은 하나다 — 생성기가 읽는 열은 결과표에 잰 값 그대로 적는다.
깎는 것은 본문에 실을 때 방향을 정해 한 번만 한다.
"""

import ast
import glob
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = os.path.join(ROOT, "scripts", "make_paper_macros.py")

# 잰 수가 아니라 이름표인 열. 훑기 한 회차를 가리키는 값이라 측정에서 나오지
# 않고, 다른 표와 맞댈 때 글자 그대로 쓰인다. 여기에 더할 때는 그 열이
# 측정이 아니라는 근거를 같이 적는다.
LABEL_COLUMNS = {
    "offset_ms",   # 훑은 어긋남 크기. 우리가 고른 1·2·5·10 이다.
    "tau_ms",      # 같은 수를 순위표 쪽에서 부르는 이름.
}

FORMAT = re.compile(r"%\.?[0-9]*[gfe]$")


def key_name(node):
    """열 이름 틀. 글자 그대로면 그 글자, 조립한 것이면 왼쪽 틀이다.

    측정 쪽은 열 이름을 "rot_%s_rel_pct" % key 처럼 조립해 쓴다. 글자만
    받으면 그 자리가 통째로 안 보인다.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod)
            and isinstance(node.left, ast.Constant)
            and isinstance(node.left.value, str)):
        return node.left.value
    return None


def rounds(node):
    """그 값 안 어딘가에서 서식으로 깎는지. 조건식 안쪽까지 본다.

    "%.4g" % x 를 괄호나 조건식으로 감싸면 글자로 맞대는 검사는 못 본다 —
    실제로 (exact(x) if y else "") 꼴로 적힌 열 넷이 그렇게 숨어 있었다.
    값 아래를 통째로 걸어서 서식 BinOp 하나라도 있으면 깎는 것으로 본다.
    """
    for n in ast.walk(node):
        if (isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mod)
                and isinstance(n.left, ast.Constant)
                and isinstance(n.left.value, str)
                and FORMAT.search(n.left.value)):
            return True
    return False


def rounded_columns(path):
    """그 대본이 깎아서 적는 (열 이름 틀) 들."""
    tree = ast.parse(open(path).read())
    out = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Dict):
            continue
        for k, v in zip(n.keys, n.values):
            name = key_name(k) if k is not None else None
            if name is not None and rounds(v):
                out.append(name)
    return out


def writers():
    return [p for p in sorted(glob.glob(os.path.join(ROOT, "scripts", "*.py")))
            if os.path.basename(p) != "make_paper_macros.py"]


def columns_the_generator_reads():
    """생성기 안에서 글자 그대로 쓰이는 열 이름들."""
    tree = ast.parse(open(GEN).read())
    return {n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)}


def as_pattern(name):
    """열 이름 하나를 정규식으로 바꾼다. %s 자리는 아무 조각이나 받는다.

    양쪽이 다 이름을 조립해 쓴다 — 측정은 "rot_%s_abs_diff" % key 로 적고
    생성기는 "%s_abs_diff_%s" % (axis, unit) 로 읽는다. 글자 그대로 맞대면
    두 이름이 영원히 안 겹쳐서, 앞단이 먼저 깎아도 검사가 조용하다.
    """
    return re.compile("^" + "[A-Za-z0-9_]+".join(
        re.escape(p) for p in name.split("%s")) + "$")


def test_no_column_the_generator_reads_is_rounded_before_it_is_stored():
    read = columns_the_generator_reads()
    # 생성기 쪽에서 조립되는 이름만 추린다. 열 이름이 아닌 글자도 섞여 있으니
    # 밑줄과 %s 를 다 가진 것으로 좁힌다.
    templates = [(g, as_pattern(g)) for g in read
                 if "%s" in g and "_" in g.replace("%s", "")]
    bad = []
    for path in writers():
        for col in rounded_columns(path):
            if col in LABEL_COLUMNS:
                continue
            if col in read:
                bad.append("%s 의 %s" % (os.path.basename(path), col))
                continue
            # 측정 쪽도 조립해 적으면 두 이름에 다 구멍이 있어 맞댈 수가
            # 없다. 측정 쪽 구멍을 아무 조각으로 메운 뒤 틀에 넣어 본다.
            probe = col.replace("%s", "zzz")
            for g, pat in templates:
                if pat.match(probe):
                    bad.append("%s 의 %s (생성기가 %s 로 읽는다)"
                               % (os.path.basename(path), col, g))
                    break
    assert not bad, (
        "생성기가 읽는 열을 결과표가 먼저 깎아 적는다. 잰 값을 그대로 적어라"
        " — repr(float(x)): " + ", ".join(sorted(set(bad))))


def test_the_label_exception_stays_small_and_is_really_a_label():
    """이름표 예외가 측정값을 숨기는 구멍으로 자라지 않게 한다."""
    assert len(LABEL_COLUMNS) <= 4
    read = columns_the_generator_reads()
    for col in LABEL_COLUMNS:
        assert col in read, "%s 는 생성기가 읽지도 않는다 — 예외를 지워라" % col
        assert col.endswith("_ms"), col
