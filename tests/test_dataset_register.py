"""자격 등록부가 쓴 릴리스를 빠짐없이 싣고, 안 읽은 것을 읽은 척 안 하는지 본다.

등록부가 비는 쪽으로 틀리면 조용하다. 릴리스를 하나 더해 놓고 등록부에 안
적으면 표의 수는 아홉인데 설명은 여덟이 되고, 그 어긋남은 아무 데서도 안
터진다. 그래서 측정 쪽 표(RELEASE_ROOTS)와 등록부를 서로 맞춰 본다.

이용 조건은 거꾸로 틀린다 — 빈칸을 싫어해서 그럴듯한 말을 채우면 그게 근거로
읽힌다. 못 읽은 네 자리는 "미확인" 으로 남아 있어야 하고, 그 자리에 왜 못
읽었는지가 같이 적혀 있어야 한다.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

from dataset_register import (FIELDS, KNOWN, PENDING,  # noqa: E402
                              read_verdicts)
from kitti_segment_D import RELEASE_ROOTS  # noqa: E402

# 그 릴리스가 시각을 공개하는가만 적는 칸이다. 우리가 어느 축에서 쟀는지는
# measured_axis 가 따로 든다 — 둘을 한 칸에 적으면 공식 시각이 있는데 다른
# 축에서 잰 자리(키티-360)가 표에서 안 보인다.
TIME_SOURCE_OK = ("공식", "없음", "미확인")


def test_every_measured_release_has_a_register_row():
    assert set(KNOWN) == set(RELEASE_ROOTS)


def test_no_row_leaves_a_column_empty():
    free = {"release", "frame_convention", "sequences", "frames",
            "measured_hz", "measured_hz_min", "measured_hz_max",
            "span_hours", "local_path"}
    for rel, row in KNOWN.items():
        assert set(row) == set(FIELDS) - free, rel
        for k, v in row.items():
            assert v.strip(), (rel, k)


def test_an_unread_licence_says_so_and_says_why():
    # 짐작으로 채우지 않는다. 미확인이면 왜 못 읽었는지가 같은 칸에 있어야
    # 하고, 읽은 것이면 그 자료 쪽 주소가 붙어 있어야 한다.
    for rel, row in list(KNOWN.items()) + [(p["release"], p)
                                           for p in PENDING]:
        t = row["terms"]
        if t.startswith("미확인"):
            assert len(t) > len("미확인"), rel
        else:
            assert row["terms_url"].startswith("http"), rel


def test_the_time_source_column_only_takes_known_words():
    for rel, row in list(KNOWN.items()) + [(p["release"], p)
                                           for p in PENDING]:
        head = row["time_source"].split("(")[0]
        assert head in TIME_SOURCE_OK, (rel, row["time_source"])


def test_the_register_says_which_axis_each_release_was_measured_on():
    # 키티는 공개 시각을 읽는다 — load_poses 가 sequences/<시퀀스>/times.txt 를
    # 같이 연다. 그래서 잰 주기가 10 이 아니라 9.64 다. 키티-360 은 반대로
    # 공식 시각이 있는데도 프레임 격자에서 재고(load_poses_kitti360 의
    # stamped 기본값이 거짓이고 _gather_raw 가 그 기본값으로 부른다), 그래서
    # 잰 주기가 정확히 10 이다. 이 둘이 표에서 갈려 보여야 한다.
    assert KNOWN["kitti"]["time_source"] == "공식"
    assert "times.txt" in KNOWN["kitti"]["measured_axis"]
    assert "times.txt" in KNOWN["kitti"]["file"]
    assert "프레임 격자" in KNOWN["kitti360"]["measured_axis"]
    assert "없음" in KNOWN["kitti360"]["time_source"]


def test_the_axis_the_register_claims_is_the_axis_the_loaders_read():
    # 표의 글과 적재 함수가 갈라지면 되돌리려는 사람이 다른 축으로 읽는다.
    # 글끼리 맞대지 말고 코드에서 직접 본다.
    import inspect

    import kitti_segment_D as K

    assert "times.txt" in inspect.getsource(K.load_poses)
    assert K.LOADER["kitti"] is K.load_poses
    sig = inspect.signature(K.load_poses_kitti360)
    assert sig.parameters["stamped"].default is False
    # _gather_raw 는 경로 하나만 넘긴다 — stamped 를 안 넘기므로 기본값이 쓰인다.
    src = inspect.getsource(K._gather_raw)
    assert "load(paths[0])" in src and "stamped" not in src


def test_a_single_transposed_row_is_not_swallowed(tmp_path):
    # 전치 줄이 하나만 섞여도 릴리스 판정이 넘어가야 한다. 다수결로 읽으면
    # 아홉 대 하나가 통과로 찍히는데, 규약은 릴리스의 성질이라 섞여 나온
    # 것 자체가 둘 중 하나는 틀렸다는 뜻이다.
    p = tmp_path / "probe.csv"
    p.write_text("release,verdict\n"
                 "a,as_read\na,transposed\na,transposed\n"
                 "b,indecisive\nb,indecisive\n"
                 "c,as_read\nc,as_read\nc,transposed\n"
                 "d,as_read\nd,indecisive\n")
    got = read_verdicts(str(p))
    assert got["a"].startswith("전치 의심")
    assert got["b"] == "못 가림"
    assert got["c"] == "전치 의심(줄 1)"
    assert got["d"] == "읽은 대로(가린 줄 1)"


def test_a_missing_probe_file_is_not_read_as_a_pass():
    assert read_verdicts("results/next/this-does-not-exist.csv") == {}
