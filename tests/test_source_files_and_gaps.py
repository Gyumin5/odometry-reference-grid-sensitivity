"""측정이 여는 파일 목록과 공백 처리를 묶는다.

묶는 것 넷이다. 고름이 자세 파일만이 아니라 읽개가 같이 여는 보조 입력까지
내는가, 원본 고정표가 그 목록을 그대로 받아 적는가, 공백 판정이 간격의
배수로만 서는가, 공백 매크로가 두 규칙을 한 통에 섞지 않는가.

앞 둘이 이 묶음의 까닭이다. 폴더를 따로 훑어 "읽은 파일" 을 적던 때는 그
목록이 측정과 조용히 갈렸다 — 헬리퍼의 전역 좌표판은 측정이 건너뛰는데
목록에는 남았고, 키티의 시각 파일은 측정이 여는데 목록에서 빠졌다.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

import numpy as np  # noqa: E402

from gap_policy_audit import (class_score, class_share, crossing,  # noqa: E402
                              endpoint_touched, gap_index, refuse_to_shrink,
                              support_start)
from kitti_segment_D import RELEASES, source_files  # noqa: E402
from make_paper_macros import gap_policy_macros  # noqa: E402
from source_manifest import check as manifest_check  # noqa: E402


def test_every_release_yields_a_tuple_of_paths(tmp_path):
    # 뿌리가 비어 있어도 계약은 꼴로 선다 — 빈 뿌리에서는 아무 것도 안 난다.
    for release in RELEASES:
        got = list(source_files(release, str(tmp_path)))
        assert got == []


def test_kitti_carries_the_time_file_the_loader_also_opens(tmp_path):
    root = tmp_path / "kitti"
    (root / "poses").mkdir(parents=True)
    (root / "sequences" / "00").mkdir(parents=True)
    (root / "poses" / "00.txt").write_text("")
    (root / "sequences" / "00" / "times.txt").write_text("")
    # 시각 파일이 없는 시퀀스도 있다. 그때는 자세 파일 하나만 난다.
    (root / "poses" / "01.txt").write_text("")

    got = dict(source_files("kitti", str(root)))
    assert got["00"] == (str(root / "poses" / "00.txt"),
                         str(root / "sequences" / "00" / "times.txt"))
    assert got["01"] == (str(root / "poses" / "01.txt"),)


def test_the_manifest_flags_every_path_the_loader_yields(tmp_path):
    root = tmp_path / "kitti"
    (root / "poses").mkdir(parents=True)
    (root / "sequences" / "00").mkdir(parents=True)
    (root / "poses" / "00.txt").write_text("")
    (root / "sequences" / "00" / "times.txt").write_text("")
    # 측정이 안 여는 파일 하나. 유리글에는 걸리고 칸은 no 여야 한다.
    (root / "sequences" / "01").mkdir(parents=True)
    (root / "sequences" / "01" / "times.txt").write_text("")

    read = {os.path.realpath(p)
            for _n, ps in source_files("kitti", str(root)) for p in ps}
    assert os.path.realpath(
        str(root / "sequences" / "00" / "times.txt")) in read
    assert os.path.realpath(
        str(root / "sequences" / "01" / "times.txt")) not in read


def test_the_manifest_check_catches_a_changed_file(tmp_path):
    root = tmp_path / "rel"
    root.mkdir()
    (root / "a.txt").write_text("hello")
    table = tmp_path / "m.csv"
    import csv
    import hashlib
    with open(table, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["release", "path", "bytes",
                                          "sha256", "read_by_bound"])
        w.writeheader()
        w.writerow({"release": "rel", "path": "a.txt", "bytes": 5,
                    "sha256": hashlib.sha256(b"hello").hexdigest(),
                    "read_by_bound": "yes"})
    found = {"rel": str(root)}
    assert manifest_check(str(table), found) == 0

    # 같은 크기로 내용만 바꾼다 — 크기만 보면 못 잡는 자리다.
    (root / "a.txt").write_text("world")
    assert manifest_check(str(table), found) == 1

    (root / "a.txt").unlink()
    assert manifest_check(str(table), found) == 1


def test_the_manifest_check_catches_one_trajectory_read_twice(tmp_path):
    # 튬 알지비디의 fr1_xyz 가 이 꼴이었다 — 같은 바이트가 두 이름으로 깔려
    # 있어서 그 주행이 집계에 두 번 들어갔다. 수는 멀쩡해 보인다.
    import csv
    import hashlib
    root = tmp_path / "rel"
    root.mkdir()
    for name in ("short.txt", "canonical.txt"):
        (root / name).write_text("same bytes")
    table = tmp_path / "m.csv"
    h = hashlib.sha256(b"same bytes").hexdigest()
    with open(table, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["release", "path", "bytes",
                                          "sha256", "read_by_bound"])
        w.writeheader()
        for name in ("short.txt", "canonical.txt"):
            w.writerow({"release": "rel", "path": name, "bytes": 10,
                        "sha256": h, "read_by_bound": "yes"})
    assert manifest_check(str(table), {"rel": str(root)}) == 1

    # 상한이 한쪽만 열면 겹침이 아니다 — 곁파일은 같아도 된다.
    with open(table, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["release", "path", "bytes",
                                          "sha256", "read_by_bound"])
        w.writeheader()
        w.writerow({"release": "rel", "path": "short.txt", "bytes": 10,
                    "sha256": h, "read_by_bound": "no"})
        w.writerow({"release": "rel", "path": "canonical.txt", "bytes": 10,
                    "sha256": h, "read_by_bound": "yes"})
    assert manifest_check(str(table), {"rel": str(root)}) == 0


def test_a_gap_is_five_median_intervals_not_a_fixed_second():
    # 간격 0.01 초짜리 축에서 0.06 초는 공백이고 0.04 초는 아니다. 같은
    # 모양을 1000 배로 늘려도 판정이 안 바뀐다 — 초가 아니라 배수다.
    for scale in (1.0, 1000.0):
        small = np.cumsum([0.0] + [0.01] * 5 + [0.04] + [0.01] * 5) * scale
        big = np.cumsum([0.0] + [0.01] * 5 + [0.06] + [0.01] * 5) * scale
        assert len(gap_index(small)) == 0
        assert list(gap_index(big)) == [6]


def test_the_manifest_check_tells_three_kinds_of_change_apart(tmp_path):
    """읽개가 여는 집합과 적힌 집합을 맞추되 셋을 가른다.

    쓰지 않는 파일이 폴더에 하나 늘어난 것은 어긋남이 아니다. 읽개가 새로
    여는 파일이 생긴 것과, 적혀 있는데 더는 안 여는 것은 어긋남이다.
    """
    import csv
    import hashlib

    root = tmp_path / "kitti"
    (root / "poses").mkdir(parents=True)
    (root / "sequences" / "00").mkdir(parents=True)
    (root / "poses" / "00.txt").write_text("a")
    (root / "sequences" / "00" / "times.txt").write_text("b")

    def write(table, entries):
        with open(table, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["release", "path", "bytes",
                                              "sha256", "read_by_bound"])
            w.writeheader()
            for rel, body, flag in entries:
                w.writerow({"release": "kitti", "path": rel,
                            "bytes": len(body),
                            "sha256": hashlib.sha256(body).hexdigest(),
                            "read_by_bound": flag})

    table = tmp_path / "m.csv"
    full = [("poses/00.txt", b"a", "yes"),
            ("sequences/00/times.txt", b"b", "yes")]
    write(table, full)
    found = {"kitti": str(root)}
    assert manifest_check(str(table), found) == 0

    # 안 읽는 파일이 폴더에 하나 늘었다. 표에 없어도 통과해야 한다.
    (root / "sequences" / "00" / "notes.txt").write_text("c")
    assert manifest_check(str(table), found) == 0

    # 읽개가 여는 곁파일이 표에서 빠졌다 — 잡아야 한다.
    write(table, full[:1])
    assert manifest_check(str(table), found) == 1

    # 표에는 읽는다고 적혔는데 읽개가 안 여는 파일 — 잡아야 한다.
    write(table, full + [("sequences/00/extra.txt", b"d", "yes")])
    (root / "sequences" / "00" / "extra.txt").write_text("d")
    assert manifest_check(str(table), found) == 1


def test_a_segment_crosses_a_gap_only_when_the_gap_is_inside_it():
    # 구간은 양 끝 표본을 다 쓴다. 공백은 9 와 10 사이에 있다.
    g = [10]
    # 네 자리를 가른다 — 공백 뒤에서 시작, 공백 뒤에서 끝, 속에 둠, 멀리 둠.
    idx = [(10, 20, 1.0),   # a == g: 공백 뒤 첫 표본에서 시작한다. 안 문다.
           (5, 10, 1.0),    # b == g: 공백 뒤 첫 표본에서 끝난다. 문다.
           (5, 20, 1.0),    # 속에 둔다. 문다.
           (11, 20, 1.0),   # 공백보다 뒤에 있다. 안 문다.
           (0, 9, 1.0)]     # 공백보다 앞에 있다. 안 문다.
    assert list(crossing(g, idx)) == [False, True, True, False, False]

    # 공백이 하나도 없으면 아무 구간도 안 문다.
    assert not crossing([], idx).any()


def test_the_gap_index_feeds_crossing_as_the_sample_after_the_gap():
    # 두 함수를 따로 보면 색인 규약이 어긋나도 각자는 맞다. 검출부터
    # 판정까지 실제 호출로 한 번 꿰어 그 규약을 고정한다. 표본 넷에 공백은
    # 1 번과 2 번 사이 하나다.
    t = [0.0, 0.1, 10.1, 10.2]
    g = gap_index(t)
    # 검출은 공백 뒤 첫 표본을 가리킨다 — 왼쪽 색인 1 이 아니라 2 다.
    assert list(g) == [2]
    assert not crossing(g, [(0, 1, 1.0)])[0]
    assert crossing(g, [(1, 2, 1.0)])[0]


def test_an_endpoint_interpolated_across_a_gap_is_counted_apart():
    """공백이 점수에 닿는 두 길을 갈라 센다.

    본문은 구간 안쪽 깊숙한 공백이 공개 경로 길이를 거쳐 어떤 구간이
    존재하는지를 바꾸고, 끝점 보간이 쓰는 표본과 겹치는 공백은 끝점 자세
    자체를 바꾼다고 적는다. 그 둘이 실제로 다른 구간에 걸리는지를 고정한다.
    """
    import numpy as np
    # 공백은 9 번과 10 번 표본 사이 하나다. 간격이 어긋남보다 넉넉하면 색인 a
    # 의 끝점은 (a, a+1) 에서 보간된다 — 그 짝이 공백이 되는 것은 a 가 9 일
    # 때다. 지지 표본을 색인 그대로 준 것이 그 경우다.
    g = [10]
    idx = [(9, 20, 1.0),    # 시작 끝점이 공백 위에서 보간된다
           (0, 9, 1.0),     # 끝 끝점이 공백 위에서 보간된다 — 공백은 구간 밖
           (5, 20, 1.0),    # 공백을 속에 두지만 끝점은 공백에서 멀다
           (11, 20, 1.0)]   # 공백보다 뒤
    sup = np.arange(21)
    touch = endpoint_touched(g, idx, sup)
    assert list(touch) == [True, True, False, False]
    # 끝점이 걸린 둘 가운데 하나는 crossing 이 아예 안 잡는다. 공백이 구간
    # 안에 없는데도 끝점 자세가 그 공백을 건너 보간되는 자리다.
    assert list(crossing(g, idx)) == [True, False, True, False]
    inner = crossing(g, idx) & ~touch
    assert list(inner) == [False, False, True, False]


def test_a_short_interval_moves_the_support_past_the_next_sample():
    """어긋남이 그 자리 간격보다 크면 보간이 더 뒤의 짝을 읽는다.

    색인을 a+1 로 못박으면 이 경우가 통째로 틀린다 — 공백에 닿는 구간을
    안 닿는다고 찍고, 실제로는 안 닿는 구간을 닿는다고 찍는다. 중앙 간격이
    어긋남보다 훨씬 커도 짧은 자리 하나면 그렇게 된다.
    """
    import numpy as np
    # 0,1,2,3 번은 1 밀리초보다 촘촘하고 그 뒤는 성기다. 0 번 표본에서
    # 1 밀리초를 밀면 3 번과 4 번 사이로 떨어진다.
    t = np.array([0.0, 0.0003, 0.0006, 0.0009, 0.05, 0.10, 0.15])
    sup = support_start(t, t + 1e-3)
    assert list(sup[:1]) == [3]
    # 공백이 3 번과 4 번 사이면 0 번에서 시작하는 구간의 끝점이 그 위에서
    # 보간된다. 못박은 색인(a+1=1)으로는 안 걸린다.
    g = [4]
    idx = [(0, 2, 1.0)]
    assert list(endpoint_touched(g, idx, sup)) == [True]
    assert list(endpoint_touched(g, idx, np.arange(len(t)))) == [False]


def test_a_gap_just_past_the_window_still_reaches_the_last_endpoint():
    """창 밖 공백도 마지막 끝점의 보간에는 들어온다.

    공백을 창으로 자른 축에서 찾으면 그 공백이 목록에 없다. 그러면 밀린
    끝점이 실제로는 그 공백을 건너 보간되는데도 "공백과 무관" 으로 찍힌다.
    공백 검출을 보간이 읽는 축과 같게 두면 걸린다.
    """
    import numpy as np
    t = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 10.4])
    keep = 5                                   # 마지막 표본은 창에서 빠진다
    assert len(gap_index(t[:keep])) == 0        # 창 안에는 공백이 없다
    g = gap_index(t)
    assert list(g) == [5]
    sup = support_start(t, t[:keep] + 1e-3)
    # 색인 4 의 끝점은 4 와 5 사이에서 보간된다 — 그 짝이 공백이다.
    assert list(endpoint_touched(g, [(0, 4, 1.0)], sup)) == [True]


def test_a_class_score_reweighs_inside_the_class_and_is_nan_when_empty():
    import numpy as np
    w = np.array([1.0, 3.0])
    err = np.array([2.0, 6.0])
    # 묶음 안에서 무게를 다시 맞춘다. 전체 무게로 나누면 묶음의 평균이
    # 아니라 전체에 대한 기여가 되고, 본문이 그 수를 평균으로 읽는다.
    assert class_score(w, err, np.array([False, True])) == 6.0
    assert class_score(w, err, np.array([True, True])) == 5.0
    assert class_score(w, err, np.array([False, False])) != \
        class_score(w, err, np.array([False, False]))      # nan != nan


def test_the_class_shares_add_up_to_the_whole_bound():
    """두 길의 크기는 다시 맞춘 평균이 아니라 분해된 몫으로 센다.

    D = sum_k w_k d_k 라 묶음별 합이 그대로 D 가 된다. 다시 맞춘 평균은
    묶음마다 분모가 달라서 더해도 D 가 안 되고, 그래서 "어느 길이 더 큰가"
    를 그 수로 물으면 안 된다 — 끝점이 닿는 구간이 무게의 0.1 퍼센트인데
    다시 맞춘 평균은 전체보다 백 배 크게 나온다.
    """
    import numpy as np
    w = np.array([1.0, 3.0, 0.5])
    err = np.array([2.0, 6.0, 8.0])
    a = np.array([True, False, False])
    b = np.array([False, True, False])
    c = np.array([False, False, True])
    whole = float(np.dot(w, err))
    got = (class_share(w, err, a) + class_share(w, err, b)
           + class_share(w, err, c))
    assert abs(got - whole) < 1e-12, (got, whole)
    # 빈 묶음은 0 을 넣는다. nan 이면 합이 통째로 nan 이 된다.
    assert class_share(w, err, np.array([False, False, False])) == 0.0


def test_a_narrowed_run_cannot_overwrite_the_table_of_every_release(tmp_path):
    """릴리스 하나만 재고 덮으면 열 릴리스를 센 표가 사라진다.

    실제로 그랬다 — 공백 두 길을 엔시엘티에서만 다시 재려고 돌렸더니 결과표가
    1802 줄에서 55 줄이 됐다. 줄어든 표 자체는 멀쩡해 보이고, 그것을 읽은
    생성기가 내는 본문 수만 조용히 한 릴리스의 수가 된다.
    """
    import csv as _csv
    import pytest
    path = str(tmp_path / "score.csv")
    with open(path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["release", "x"])
        w.writeheader()
        w.writerows([{"release": "kitti", "x": "1"},
                     {"release": "nclt", "x": "2"}])
    with pytest.raises(SystemExit):
        refuse_to_shrink(path, [{"release": "nclt", "x": "3"}])
    # 같은 릴리스를 다 담고 있으면 통과한다 — 다시 재는 것 자체는 막지 않는다.
    refuse_to_shrink(path, [{"release": "nclt", "x": "3"},
                            {"release": "kitti", "x": "4"}])
    # 아직 없는 표는 그냥 쓴다.
    refuse_to_shrink(str(tmp_path / "없는표.csv"), [{"release": "nclt"}])


def test_the_sweep_table_never_names_a_sequence_the_bound_table_lacks():
    # 훑기표에만 있는 이름은 그 표가 낡았다는 뜻이다. 실제로 그랬다 — 중복
    # 제거 뒤 튬 알지비디 훑기표만 다시 안 나서 같은 주행이 두 이름으로
    # 남았고, 그 표를 읽는 묶음표가 그만큼 더 센 구간으로 D 를 냈다.
    #
    # 반대쪽은 비는 것이 정상이라 한 방향으로만 본다. 훑기는 밀린 변형이
    # 자료 밖으로 안 나가게 양 끝을 잘라 구간을 짓는데, 짧은 주행은 그러고
    # 나면 구간이 하나도 안 남아 표에서 빠진다(누씬즈 527 -> 518).
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(here, "results", "next")
    seen = 0
    for release in RELEASES:
        for tag in ("rpe1m", "segment"):
            a = os.path.join(out, "%s_%s_D.csv" % (release, tag))
            b = os.path.join(out, "%s_%s_sweep.csv" % (release, tag))
            if not (os.path.exists(a) and os.path.exists(b)):
                continue
            seen += 1
            extra = names_in(b) - names_in(a)
            assert not extra, "%s %s: %s" % (release, tag, sorted(extra))
    if not seen:
        import pytest
        pytest.skip("결과표가 없는 나무다")


def names_in(path):
    import csv
    with open(path) as f:
        return {r["sequence"] for r in csv.DictReader(f)}


def test_gap_macros_keep_the_two_rules_apart():
    # 릴리스 이름은 실제 키를 쓴다 — 매크로가 보기 좋은 이름으로 바꿔
    # 내보내므로 지어낸 이름은 그 자리에서 걸린다.
    rows = [
        # 1 m 간격 규칙 — 공백을 무는 구간이 소수라 뺀 나머지와 견줄 수 있다.
        {"release": "nclt", "rule": "rpe1m", "sequence": "x",
         "segments": "100", "segments_crossing": "4",
         "weight_crossing": "0.04",
         "D_rot_all": "1.0", "D_rot_gapfree": "1.1",
         "D_trans_all": "1.0", "D_trans_gapfree": "1.0"},
        # 100~800 m 규칙 — 전부 물어서 뺄 나머지가 없다.
        {"release": "nclt", "rule": "seg", "sequence": "x",
         "segments": "100", "segments_crossing": "100",
         "weight_crossing": "1.0",
         "D_rot_all": "1.0", "D_rot_gapfree": "nan",
         "D_trans_all": "1.0", "D_trans_gapfree": "nan"},
        {"release": "kitti", "rule": "rpe1m", "sequence": "y",
         "segments": "100", "segments_crossing": "0",
         "weight_crossing": "0.0",
         "D_rot_all": "1.0", "D_rot_gapfree": "1.0",
         "D_trans_all": "1.0", "D_trans_gapfree": "1.0"},
        {"release": "kitti", "rule": "seg", "sequence": "y",
         "segments": "100", "segments_crossing": "0",
         "weight_crossing": "0.0",
         "D_rot_all": "1.0", "D_rot_gapfree": "1.0",
         "D_trans_all": "1.0", "D_trans_gapfree": "1.0"},
    ]
    got = dict(gap_policy_macros(rows))
    assert got["GapCrossReleasesTotal"] == "2"
    assert got["GapCrossFreeReleases"] == "1"
    # 섞었다면 무게 최댓값이 seg 쪽 1.0 을 받아 100 이 됐을 것이다.
    assert float(got["GapRpeWeightMax"]) < 10.0
    assert got["GapSegCrossWorstPct"] == "100"
    # 공백을 무는 구간이 없는 시퀀스는 0 으로 세지 않는다 — 세면 묽어진다.
    assert float(got["GapRpeDShiftMax"]) >= 10.0
    # 남은 구간이 하나도 없으면 묶을 것이 없어 배율을 내지 않는다.
    assert got["GapSegDomEmptyReleases"] == "1"
    assert "GapSegSubTransRatio" not in got


def test_the_leftover_segments_are_pooled_by_their_own_count():
    # 공백을 뺀 쪽을 전체 구간 수로 묶으면 배율이 남은 구간의 몫만큼
    # 묽어진다. 남은 것이 1 퍼센트도 안 되는 자리라 그 차이가 배율을
    # 통째로 지운다 — 뺀 쪽은 뺀 구간 수로 센다.
    rows = [{"release": "nclt", "rule": "seg", "sequence": s,
             "segments": "1000", "segments_crossing": "990",
             "weight_crossing": "0.99",
             "D_rot_all": "1.0", "D_rot_gapfree": g,
             "D_trans_all": "1.0", "D_trans_gapfree": g}
            for s, g in (("a", "3.0"), ("b", "3.0"))]
    # 값이 없는 시퀀스는 묶음에서 빠진다 — 세면 분모만 늘어난다.
    rows.append({"release": "nclt", "rule": "seg", "sequence": "c",
                 "segments": "1000", "segments_crossing": "1000",
                 "weight_crossing": "1.0",
                 "D_rot_all": "1.0", "D_rot_gapfree": "nan",
                 "D_trans_all": "1.0", "D_trans_gapfree": "nan"})
    got = dict(gap_policy_macros(rows))
    assert got["GapSegSubSeqs"] == "2"
    assert got["GapSegSubSeqsAll"] == "3"
    assert got["GapSegSubFree"] == "20"
    # 세 자리마다 쉼표로 끊는다. 표가 이미 그렇게 적어서, 본문만 얇은 공백을
    # 쓰면 같은 종류의 수가 한 지면에서 두 꼴로 나온다.
    assert got["GapSegSubSegs"] == "2,000"
    assert got["GapSegSubTransRatio"] == "3"
    assert got["GapSegSubRotRatio"] == "3"
