"""공개 나무의 고르는 규칙을 묶는다.

손 목록이 아니라 규칙으로 고르기 때문에 묶을 것은 그 규칙이다. 검사가 부르는
대본이 들어오는가, 그 대본이 다시 부르는 것까지 닫히는가, 아무도 안 부르는
대본은 빠지는가, 표는 이름이 적힌 것과 릴리스별로 만들어지는 것만 들어오는가.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

from make_public_release import (manifest, result_names,  # noqa: E402
                                 support_names)


def tree(tmp_path, scripts, tests):
    """대본과 검사를 적어 두고 그 두 디렉토리 경로를 준다."""
    s, t = tmp_path / "scripts", tmp_path / "tests"
    s.mkdir()
    t.mkdir()
    for name, body in scripts.items():
        (s / (name + ".py")).write_text(body)
    for name, body in tests.items():
        (t / (name + ".py")).write_text(body)
    return str(s), str(t)


def test_a_script_a_test_imports_comes_along(tmp_path):
    s, t = tree(tmp_path, {"core": "x = 1", "unused": "y = 2"},
                {"test_core": "from core import x"})
    assert manifest(s, t) == ["core"]


def test_what_that_script_imports_comes_along_too(tmp_path):
    """한 겹만 보면 적재기가 빠진 채 상한 대본만 담긴다."""
    s, t = tree(tmp_path, {"core": "import loader", "loader": "z = 3",
                           "unused": "y = 2"},
                {"test_core": "import core"})
    assert manifest(s, t) == ["core", "loader"]


def test_a_cycle_between_scripts_does_not_hang(tmp_path):
    s, t = tree(tmp_path, {"a": "import b", "b": "import a"},
                {"test_a": "import a"})
    assert manifest(s, t) == ["a", "b"]


def test_only_the_tables_the_kept_scripts_name_are_copied(tmp_path):
    s, _ = tree(tmp_path, {"core": 'OUT = "results/next/kept.csv"'}, {})
    r = tmp_path / "results"
    r.mkdir()
    for f in ("kept.csv", "stray.csv", "kitti_rpe1m_D.csv", "notes.txt"):
        (r / f).write_text("")
    got = result_names(["core"], s, str(r))
    # 릴리스별 이름은 대본 안에 글자로 안 적힌다 — 꼴로 고른다.
    assert got == ["kept.csv", "kitti_rpe1m_D.csv"]


def test_a_sidecar_file_a_kept_script_names_comes_along(tmp_path):
    """import 로 안 걸리는 곁파일 — 빠지면 받는 쪽에서 그 대조가 안 돈다."""
    s, _ = tree(tmp_path, {"core": 'SRC = "official_mean.cpp"'}, {})
    for f in ("official_mean.cpp", "stray.cpp"):
        with open(os.path.join(s, f), "w") as fh:
            fh.write("int main(){}")
    assert support_names(["core"], s) == ["official_mean.cpp"]


def test_the_official_comparison_keeps_its_cpp_in_the_real_tree():
    """실제 나무에서도 빠지지 않는다 — 규칙이 이 자리를 덮는지 본다."""
    assert "official_mean.cpp" in support_names(manifest())


def test_a_directory_whose_name_a_script_mentions_is_not_a_sidecar(tmp_path):
    """곁파일 고르는 규칙은 이름만 본다 — 디렉토리가 걸리면 복사가 터진다."""
    s, _ = tree(tmp_path, {"core": 'IGNORE = "__pycache__/"'}, {})
    os.mkdir(os.path.join(s, "__pycache__"))
    assert support_names(["core"], s) == []


def test_the_released_tree_ignores_what_running_the_tests_leaves(tmp_path):
    """받는 쪽이 검사를 돌리면 바이트 묶음이 생긴다.

    그게 저장소로 들어가면 공개본에 이 기계의 경로가 박힌 채 실린다. 실제로
    한 번 실려서 85 개를 걷어냈다.
    """
    import subprocess
    out = tmp_path / "tree"
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=os.path.join(root, "src"))
    subprocess.check_call(
        [sys.executable, os.path.join(root, "scripts",
                                      "make_public_release.py"),
         "--out", str(out)], env=env, stdout=subprocess.DEVNULL)
    ignored = (out / ".gitignore").read_text().split()
    assert "__pycache__/" in ignored and "*.pyc" in ignored

    # 32 회차 권고 2. 표를 낸 기계의 판본 기록이 표와 같이 가야 한다.
    # 없으면 받는 쪽에서 매크로를 다시 낼 때 그 기계의 환경이 적히고,
    # 같은 표가 다른 판본 줄을 달고 나온다.
    src = os.path.join(root, "results", "next", "environment.json")
    if os.path.exists(src):
        got = out / "results" / "next" / "environment.json"
        assert got.exists()
        assert got.read_text() == open(src).read()
