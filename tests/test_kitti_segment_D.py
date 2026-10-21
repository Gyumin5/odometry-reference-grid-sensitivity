"""공식 지표 단위로 옮긴 D 가 같은 양을 재는지 본다.

이 측정의 값어치는 단위에 있다. 사람들이 표에 적는 수(도/미터)와 같은 단위라야
"두 방법의 차이가 2D 보다 작으면 그 순위는 기준 파일이 못 떠받친다" 가 성립한다.
그래서 구간 고르기·길이 정규화·각도 환산 셋을 각각 묶어 둔다.
"""

import json
import os
import pathlib
import sys
import tempfile

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

from kitti_segment_D import (  # noqa: E402
    LENGTHS, RELEASE_ROOTS, RELEASES, common_window, euclidean, exact,
    gather, geodesic, load_poses, load_poses_helipr, load_poses_kitti360,
    load_poses_nclt, load_poses_nuscenes, load_poses_pit30m, measure,
    relative, relative_translation, resample, rms_displacement,
    segment_weights, segments, ypr_to_rot_boreas,
)


def straight(n=1200, fs=10.0, speed=10.0, yaw_rate=0.0):
    """일정 속도 직진 + 일정 각속도. 공식 구간이 실제로 잡히게 충분히 길다."""
    t = np.arange(n) / fs
    pos = np.stack([speed * t, np.zeros(n), np.zeros(n)], axis=1)
    rot = Rotation.from_rotvec(np.outer(t, [0.0, yaw_rate, 0.0])).as_matrix()
    return t, rot, pos


def test_the_boreas_rotation_follows_the_official_construction():
    # 공식 pyboreas 는 roll(r) @ pitch(p) @ yaw(y) 로 짓고 세 기본 행렬이 전부
    # 능동 회전의 전치다. scipy 로 그냥 지으면 전치가 되는데, 그 잘못은 회전
    # 쪽에선 안 드러난다 — 전부 전치해도 상대 회전의 측지각이 그대로다.
    # 드러나는 곳은 구간 상대 병진 하나뿐이라 여기서 두 성질을 같이 못박는다.
    y, p, r = 0.7, -0.2, 0.1
    ours = ypr_to_rot_boreas(np.array([y]), np.array([p]), np.array([r]))[0]
    scipy_zyx = Rotation.from_euler("ZYX", [y, p, r]).as_matrix()
    assert np.allclose(ours, scipy_zyx.T)

    # 본체 좌표에서 본 전진 방향은 방위가 돌아도 한 자리에 머문다.
    n = 400
    yaw = np.linspace(0.0, 2.0, n)
    zeros = np.zeros(n)
    rot = ypr_to_rot_boreas(yaw, zeros, zeros)
    step = 2.0
    pos = np.cumsum(np.stack([step * np.cos(np.pi / 2 - yaw),
                              step * np.sin(np.pi / 2 - yaw), zeros], 1), 0)
    d = pos[1:] - pos[:-1]
    body = np.einsum("nji,nj->ni", rot[:-1], d)
    ang = np.degrees(np.arctan2(body[:, 1], body[:, 0]))
    assert ang.max() - ang.min() < 1.0
    # 전치하면 같은 양이 방위를 따라 돈다.
    bad = np.einsum("nij,nj->ni", rot[:-1], d)
    bang = np.degrees(np.arctan2(bad[:, 1], bad[:, 0]))
    assert bang.max() - bang.min() > 100.0


def test_a_segment_ends_at_the_first_frame_that_passes_the_length():
    # 공식 규칙은 "넘는 첫 프레임" 이다(dist[i] > dist[first] + len). 딱 닿는
    # 자리는 안 받는다. 더 가거나 덜 간 프레임을 고르면 길이로 나누는
    # 정규화가 어긋난다. 공식 원본과의 대조는 test_kitti_official_crosscheck.
    _, _, pos = straight(n=400, speed=10.0)   # 1 m/프레임
    segs = segments(pos, lengths=(100.0,), step=50)
    assert segs
    for i, j, L in segs:
        assert np.linalg.norm(pos[j] - pos[i]) > L
        assert np.linalg.norm(pos[j - 1] - pos[i]) <= L


def test_a_trajectory_that_never_travels_far_enough_yields_no_segments():
    t = np.arange(300) / 10.0
    pos = np.zeros((300, 3))
    rot = np.broadcast_to(np.eye(3), (300, 3, 3)).copy()
    assert segments(pos) == []
    assert measure(t, rot, pos) == (None, 0)


def test_the_weights_normalise_by_the_requested_length():
    # 길이가 두 배인 구간은 같은 각오차에 절반만 기여해야 한다. 이게 깨지면
    # 보고하는 수가 도/미터가 아니다.
    idx = [(0, 1, 100.0), (0, 1, 200.0)]
    w = segment_weights(idx)
    assert abs(w[0] / w[1] - 2.0) < 1e-12
    assert abs(w[0] - 1.0 / (2 * 100.0)) < 1e-12


def test_a_known_angle_comes_back_as_that_angle_over_the_length():
    # 모든 구간을 같은 각만큼 통째로 돌려 놓으면 D 는 그 각을 길이로 나눈 값이다.
    _, rot, pos = straight(n=900, yaw_rate=0.05)
    idx = [s for s in segments(pos) if s[2] == 100.0]
    assert len(idx) > 5
    q = relative(rot, idx)
    ang = 0.013
    qp = np.einsum("nab,bc->nac", q, Rotation.from_rotvec([0, 0, ang]).as_matrix())
    d = float(np.dot(segment_weights(idx), geodesic(q, qp)))
    assert abs(d - ang / 100.0) < 1e-9


def test_a_constant_turn_rate_is_unmoved_by_a_half_frame_shift():
    # 양 끝이 같은 만큼 밀리고 각속도가 일정하면 상대회전이 글자 그대로 같다.
    # 0 이 안 나오면 구간 쌍이 밀린 뒤에도 같은 프레임을 가리키는지가 틀린 것이다.
    d, nseg = measure(*straight(n=1500, yaw_rate=0.08))
    assert nseg > 50
    assert d["phase_half"] < 1e-10


def test_an_irregular_turn_actually_moves():
    # 위 검사가 0 만 보고 통과하지 않도록, 각속도가 변하면 움직이는지도 본다.
    # 흔들리는 주기는 구간 길이와 약분되면 안 된다 — 10 m/s 에서 100 m 구간은
    # 10 초라, 0.7 Hz 로 흔들면 양 끝 증분이 정확히 같아져 차가 0 으로 죽는다.
    # 속도도 한 프레임 1 m 로 두면 안 된다 — 구간 끝이 늘 짝수 번째 프레임에
    # 떨어져서 절반으로 솎아도 그 자리가 살아남고 decim2 가 0 으로 죽는다.
    t, _, pos = straight(n=1500, speed=9.0)
    ang = 0.4 * np.sin(2.0 * np.pi * 0.37 * t) + 0.05 * t
    rot = Rotation.from_rotvec(np.stack(
        [np.zeros_like(t), ang, np.zeros_like(t)], axis=1)).as_matrix()
    d, _ = measure(t, rot, pos)
    assert d["phase_half"] > 1e-7
    assert d["decim2"] > 1e-7


def test_the_same_instants_measure_the_same_from_any_clock_origin():
    """시각을 유닉스 기원으로 옮겨 적어도 D 가 그대로여야 한다.

    초당 네 장이면 간격 0.25 초라 이진으로 떨어지고, 기원 1.7e9 에 얹어도
    표본이 정확히 같은 순간을 가리킨다. 같은 질문이니 답도 같아야 한다.
    원점을 안 빼면 안 같다 — 이 안에서 만드는 질의 시각(반 표본 위상, 추정
    격자)이 기원 크기의 칸 2.4e-07 초 위로 눌린다.
    """
    t, _, pos = straight(n=1500, fs=4.0, speed=9.0)
    ang = 0.4 * np.sin(2.0 * np.pi * 0.37 * t) + 0.05 * t
    rot = Rotation.from_rotvec(np.stack(
        [np.zeros_like(t), ang, np.zeros_like(t)], axis=1)).as_matrix()
    assert np.array_equal(1.7e9 + t - 1.7e9, t)      # 같은 순간임을 못박는다
    near, n_near = measure(t, rot, pos)
    far, n_far = measure(1.7e9 + t, rot, pos)
    assert near is not None and n_near == n_far
    assert near == far


def test_a_denser_publication_moves_less_at_the_same_motion():
    t, _, pos = straight(n=1500, fs=10.0)
    tf, _, posf = straight(n=3000, fs=20.0)

    def wobble(tt):
        a = 0.4 * np.sin(2.0 * np.pi * 0.37 * tt)
        return Rotation.from_rotvec(np.stack(
            [np.zeros_like(tt), a, np.zeros_like(tt)], axis=1)).as_matrix()

    slow = measure(t, wobble(t), pos)[0]
    fast = measure(tf, wobble(tf), posf)[0]
    assert fast["phase_half"] < slow["phase_half"]


def test_every_protocol_reports_in_degrees_per_metre_not_radians():
    # 라디안을 도로 안 바꾸면 수가 57 배 작게 적히고, 읽는 사람이 자기 표와
    # 맞대면 조용히 틀린다. main 이 하는 환산을 여기서 묶어 둔다.
    d = measure(*straight(n=1500, yaw_rate=0.08))[0]
    for k in ("phase_half", "decim2", "interp"):
        assert k in d
        assert 0.0 <= d[k] < 1.0   # 라디안/미터는 작은 수다
        assert np.degrees(d[k]) / (d[k] + 1e-30) > 50.0


def test_the_pose_file_is_read_as_three_by_four_rows(tmp_path):
    p = tmp_path / "poses"
    p.mkdir()
    f = p / "07.txt"
    rows = []
    for k in range(5):
        r = Rotation.from_rotvec([0, 0.01 * k, 0]).as_matrix()
        rows.append(" ".join("%.9f" % v for v in
                             [r[0, 0], r[0, 1], r[0, 2], 1.0 * k,
                              r[1, 0], r[1, 1], r[1, 2], 0.0,
                              r[2, 0], r[2, 1], r[2, 2], 2.0 * k]))
    f.write_text("\n".join(rows) + "\n")
    t, rot, pos = load_poses(str(f))
    assert rot.shape == (5, 3, 3) and pos.shape == (5, 3)
    assert abs(pos[3, 0] - 3.0) < 1e-9 and abs(pos[3, 2] - 6.0) < 1e-9
    # 시각 파일이 없으면 명목 10 Hz 로 지어낸다.
    assert abs(t[1] - 0.1) < 1e-9


def _kitti360_file(tmp_path, frames):
    f = tmp_path / "poses.txt"
    rows = []
    for k in frames:
        r = Rotation.from_rotvec([0, 0.01 * k, 0]).as_matrix()
        rows.append(" ".join(["%d" % k] + ["%.9f" % v for v in
                                           [r[0, 0], r[0, 1], r[0, 2], 1.0 * k,
                                            r[1, 0], r[1, 1], r[1, 2], 0.0,
                                            r[2, 0], r[2, 1], r[2, 2], 2.0 * k]
                                           ]))
    f.write_text("\n".join(rows) + "\n")
    return f


def test_the_kitti360_file_has_a_frame_number_before_the_matrix(tmp_path):
    # 첫 칸을 행렬의 일부로 읽으면 자세가 통째로 밀린다. 조용히 틀리는 자리다.
    f = _kitti360_file(tmp_path, [1, 2, 3, 4])
    t, rot, pos = load_poses_kitti360(str(f))
    assert rot.shape == (4, 3, 3) and pos.shape == (4, 3)
    assert np.allclose(np.einsum("nab,ncb->nac", rot, rot), np.eye(3),
                       atol=1e-9)
    assert abs(pos[2, 0] - 3.0) < 1e-9 and abs(pos[2, 2] - 6.0) < 1e-9


def test_a_missing_frame_stays_a_hole_in_time(tmp_path):
    # 번호가 빈 자리를 행 번호로 메우면 0.3 초 걸린 이동이 0.1 초로 읽혀
    # 그만큼 없는 운동이 생긴다. 시각은 번호에서 와야 한다.
    f = _kitti360_file(tmp_path, [1, 2, 5, 6])
    t, _, _ = load_poses_kitti360(str(f))
    assert np.allclose(t, [0.1, 0.2, 0.5, 0.6])
    assert abs((t[2] - t[1]) - 0.3) < 1e-12


def test_a_plain_kitti_line_is_not_accepted_by_the_kitti360_reader(tmp_path):
    # 칸이 열둘인 줄은 키티 꼴이다. 받아들이면 첫 수를 번호로 먹고 나머지
    # 열하나로 3x4 를 짓다가 조용히 어긋난다.
    f = tmp_path / "poses.txt"
    f.write_text(" ".join(["1.0"] * 12) + "\n")
    t, rot, pos = load_poses_kitti360(str(f))
    assert len(t) == 0 and len(rot) == 0 and len(pos) == 0


def test_the_release_switch_picks_the_matching_reader(tmp_path):
    # 릴리스마다 파일 꼴이 다른데 읽는 쪽을 하나로 두면 한쪽이 조용히 비거나
    # 잘못 읽힌다. 고르는 자리를 검사로 묶는다.
    d = tmp_path / "2013_05_28_drive_0042_sync"
    d.mkdir()
    _kitti360_file(d, [1, 2, 3])
    got = list(gather("kitti360", str(tmp_path)))
    assert [name for name, _ in got] == ["0042"]
    assert len(got[0][1][0]) == 3
    # 같은 뿌리를 키티로 읽으면 비어야 한다 — 뿌리가 하나뿐이라 릴리스 이름이
    # 읽는 쪽을 고르는 유일한 값이고, 그 고르기가 틀리면 여기서 걸린다.
    assert list(gather("kitti", str(tmp_path))) == []


def test_every_registered_release_has_a_root_and_a_reader():
    # 릴리스를 더할 때 빠뜨리기 쉬운 자리가 둘이다 — 등록표에 이름만 넣고
    # 읽는 가지를 안 만들면 그 릴리스는 조용히 빈 줄이 되고, 가지만 만들고
    # 이름을 안 넣으면 어느 스크립트도 그 릴리스를 돌지 않는다. 비어 있는
    # 임시 뿌리로 부르면 둘 다 예외 없이 빈 결과여야 한다.
    assert set(RELEASES) == set(RELEASE_ROOTS)
    for release in RELEASES:
        assert RELEASE_ROOTS[release].startswith("/")
        with tempfile.TemporaryDirectory() as empty:
            assert list(gather(release, empty)) == []


def test_the_nclt_reader_drops_the_rows_the_release_left_empty():
    # 공개 파일은 모르는 시각을 -nan 으로 적는다. 0 으로 메우면 없는 자세가
    # 구간 끝점에 서서 그 구간의 변위를 통째로 지어낸다.
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "groundtruth_2012-01-08.csv")
        with open(p, "w") as f:
            f.write("1000000,-nan,-nan,-nan,-nan,-nan,-nan\n")
            f.write("1010000,1.0,2.0,3.0,0.0,0.0,0.0\n")
            f.write("1020000,1.5,2.5,3.5,0.0,0.0,0.5\n")
        t, rot, pos = load_poses_nclt(p)
    assert len(t) == 2
    assert t[0] == 1.01 and abs(t[1] - 1.02) < 1e-12
    assert list(pos[0]) == [1.0, 2.0, 3.0]
    # 방위 0.5 라디안이 Z 축 회전으로 들어갔는지 — 순서를 뒤집어 넣는 자리다.
    assert abs(np.arctan2(rot[1][1, 0], rot[1][0, 0]) - 0.5) < 1e-12


def test_the_official_length_set_is_the_one_the_metric_uses():
    assert LENGTHS == (100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0)


def _wobble(tt, amp=0.4, f=0.37):
    a = amp * np.sin(2.0 * np.pi * f * tt) + 0.05 * tt
    return Rotation.from_rotvec(np.stack(
        [np.zeros_like(tt), a, np.zeros_like(tt)], axis=1)).as_matrix()


def test_the_rms_form_bounds_an_rms_reported_score():
    # 표가 평균이 아니라 제곱평균제곱근으로 적혀 있으면 평균 꼴 상한은 안 선다.
    # 민코프스키 부등식으로 세운 쪽이 실제로 안 깨지는지 적대적으로 때린다.
    t, _, pos = straight(n=1500, speed=9.0)
    rot = _wobble(t)
    idx = segments(pos)
    L = np.asarray([p[2] for p in idx], dtype=float)
    dt = float(np.median(np.diff(t)))
    q = relative(rot, idx)
    half = resample(t, rot, pos, np.clip(t + 0.5 * dt, t[0], t[-1]), "slerp")[0]
    qp = relative(half, idx)
    d_rms = rms_displacement(geodesic(q, qp), idx)
    assert d_rms > 0

    rng = np.random.default_rng(5)
    for trial in range(40):
        scale = 10.0 ** rng.uniform(-4, 0)
        a = np.einsum("nab,nbc->nac", q, Rotation.from_rotvec(
            rng.normal(0.0, scale, size=(len(idx), 3))).as_matrix())
        e = float(np.sqrt(np.mean((geodesic(a, q) / L) ** 2)))
        ep = float(np.sqrt(np.mean((geodesic(a, qp) / L) ** 2)))
        assert abs(ep - e) <= d_rms + 1e-12, (trial, scale, abs(ep - e), d_rms)


def test_the_mean_form_does_not_cover_an_rms_table():
    # 두 꼴을 한 수로 섞어 쓰면 안 되는 이유. 같은 자료에서 값이 다르다.
    t, _, pos = straight(n=1500, speed=9.0)
    d, _ = measure(t, _wobble(t), pos)
    assert d["phase_half_rms"] > d["phase_half"]
    for k in ("phase_half", "decim2", "interp"):
        assert k + "_rms" in d


def _rand_pose(rng):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(rng.normal(0.0, 1.0, 3)).as_matrix()
    T[:3, 3] = rng.normal(0.0, 50.0, 3)
    return T


def test_the_translation_distance_is_what_the_official_error_pose_measures():
    # 공식 코드는 pose_error = inv(delta_result) * delta_gt 를 만들고 그 이동
    # 성분의 길이를 t_err 로 쓴다. 우리가 쓰는 ||t_g - t_r|| 이 같은 수인지
    # 4x4 로 직접 조립해 대 본다. 같아야 병진 칸에 상한이 그대로 선다.
    rng = np.random.default_rng(11)
    idx = [(0, 1, 100.0)]
    for trial in range(20):
        gi, gj, ri, rj = (_rand_pose(rng) for _ in range(4))
        dg = np.linalg.inv(gi) @ gj
        dr = np.linalg.inv(ri) @ rj
        official = float(np.linalg.norm((np.linalg.inv(dr) @ dg)[:3, 3]))
        tg = relative_translation(np.stack([gi[:3, :3], gj[:3, :3]]),
                                  np.stack([gi[:3, 3], gj[:3, 3]]), idx)
        tr = relative_translation(np.stack([ri[:3, :3], rj[:3, :3]]),
                                  np.stack([ri[:3, 3], rj[:3, 3]]), idx)
        got = float(euclidean(tg, tr)[0])
        assert abs(got - official) < 1e-9, (trial, got, official)


def test_a_constant_speed_straight_line_has_no_translation_displacement():
    # 등속 직진은 위치가 시각의 1차식이라 선형보간이 정확하다. 반 표본을 밀어도
    # 구간 이동 벡터가 그대로이므로 병진 쪽 D 는 0 이어야 한다. 0 이 아니면
    # 구간 고르기나 보간 쪽에 군더더기가 있다는 뜻이다.
    t, rot, pos = straight(n=1200)
    d, _ = measure(t, rot, pos)
    assert d["t_phase_half"] < 1e-12


def test_the_start_frame_attitude_leaks_into_the_translation_column():
    # 공식 t_err 은 시작 프레임에서 본 벡터의 차이다. 그래서 위치가 한 치도
    # 안 변해도 시작 자세가 재표집으로 흔들리면 병진 칸이 같이 움직인다.
    t, _, pos = straight(n=1500, speed=9.0)
    d, _ = measure(t, _wobble(t), pos)
    assert d["t_phase_half"] > 0


def test_the_translation_bound_is_not_broken_by_adversarial_estimates():
    t, _, pos = straight(n=1500, speed=9.0)
    rot = _wobble(t)
    idx = segments(pos)
    w = segment_weights(idx)
    dt = float(np.median(np.diff(t)))
    hr, hp = resample(t, rot, pos,
                      np.clip(t + 0.5 * dt, t[0], t[-1]), "slerp")
    qt = relative_translation(rot, pos, idx)
    qpt = relative_translation(hr, hp, idx)
    d_trans = float(np.dot(w, euclidean(qt, qpt)))
    assert d_trans > 0

    rng = np.random.default_rng(7)
    for trial in range(60):
        scale = 10.0 ** rng.uniform(-3, 1)
        a = qt + rng.normal(0.0, scale, size=qt.shape)
        e = float(np.dot(w, euclidean(a, qt)))
        ep = float(np.dot(w, euclidean(a, qpt)))
        assert abs(ep - e) <= d_trans + 1e-12, (trial, scale, abs(ep - e))


def test_every_protocol_reports_a_translation_column_too():
    t, _, pos = straight(n=1500, speed=9.0)
    d, _ = measure(t, _wobble(t), pos)
    for k in ("phase_half", "decim2", "interp"):
        assert "t_" + k in d
        assert "t_" + k + "_rms" in d


def test_the_segment_rule_can_be_switched_to_one_metre_spacing():
    # 키티-360 공식 순위표는 1 m 떨어진 두 자세 사이에서 상대오차를 잰다.
    # 구간 규칙을 넘겼는데 안 쓰이면 100~800 m 값이 1 m 값으로 보고된다.
    t, _, pos = straight(n=400, speed=10.0)
    raw = {}
    d, nseg = measure(t, _wobble(t), pos, raw=raw, lengths=(1.0,), step=1)
    # 구간은 공통 창 안에서 선다. 전체 배열로 세면 끝 프레임만큼 어긋난다.
    assert nseg == len(segments(pos[:common_window(t)], (1.0,), 1))
    assert np.all(raw["phase_half"][2] == 1.0)
    assert d is not None


def test_one_metre_spacing_and_the_official_lengths_do_not_agree():
    # 둘이 같은 수로 나오면 규칙 전환이 조용히 무시된 것이다. 짧은 구간일수록
    # 같은 재표집 흔들림이 길이로 나뉘며 커진다.
    t, _, pos = straight(n=1500, speed=9.0)
    rot = _wobble(t)
    short, _ = measure(t, rot, pos, lengths=(1.0,), step=1)
    official, _ = measure(t, rot, pos)
    assert short["t_phase_half"] > official["t_phase_half"]


def test_the_bound_still_holds_at_one_metre_spacing():
    # 증명은 가중평균 꼴에만 기대므로 구간 길이를 바꿔도 서야 한다. 여기서
    # 깨지면 1 m 순위표에 이 상한을 들이댈 수 없다.
    t, _, pos = straight(n=1200, speed=9.0)
    rot = _wobble(t)
    idx = segments(pos, (1.0,), 1)
    w = segment_weights(idx)
    dt = float(np.median(np.diff(t)))
    hr, hp = resample(t, rot, pos,
                      np.clip(t + 0.5 * dt, t[0], t[-1]), "slerp")
    qt = relative_translation(rot, pos, idx)
    qpt = relative_translation(hr, hp, idx)
    d_trans = float(np.dot(w, euclidean(qt, qpt)))
    assert d_trans > 0

    rng = np.random.default_rng(11)
    for trial in range(40):
        scale = 10.0 ** rng.uniform(-4, 0)
        a = qt + rng.normal(0.0, scale, size=qt.shape)
        e = float(np.dot(w, euclidean(a, qt)))
        ep = float(np.dot(w, euclidean(a, qpt)))
        assert abs(ep - e) <= d_trans + 1e-12, (trial, scale, abs(ep - e))


def test_a_trajectory_timed_in_satellite_seconds_still_measures():
    # 시각이 1e9 자리로 실린 릴리스(헬리퍼·누씬즈·핏서티엠·보레아스)에서
    # 보간 격자가 끝점을 아주 조금 넘겨 짚으면 구면 선형보간이 거절한다.
    # 그러면 interp 규약 하나가 통째로 사라지는데, 0 으로 나오는 게 아니라
    # 그 릴리스의 측정 자체가 안 돈다.
    t, _, pos = straight(n=1500, speed=9.0)
    d, _ = measure(t + 1.69e9, _wobble(t), pos)
    assert "interp" in d and d["interp"] > 0


def test_the_big_epoch_result_matches_the_zero_based_one():
    # 시각 원점은 상대자세를 안 바꾼다. 두 값이 어긋나면 큰 수에서 정밀도가
    # 깎인 것이고, 그 릴리스들의 수치를 믿을 수 없다.
    t, _, pos = straight(n=1500, speed=9.0)
    rot = _wobble(t)
    a, _ = measure(t, rot, pos)
    b, _ = measure(t + 1.69e9, rot, pos)
    for k in ("phase_half", "decim2", "interp"):
        assert abs(a[k] - b[k]) <= 1e-9 + 1e-6 * abs(a[k]), k


def test_the_helipr_reader_takes_the_quaternion_in_xyzw_order(tmp_path):
    # 순서를 섞으면 자세가 조용히 다른 회전이 된다. 90 도 요 회전 하나를
    # 적어 두고 그대로 돌아오는지 본다.
    q = Rotation.from_euler("z", 90.0, degrees=True).as_quat()  # xyzw
    p = tmp_path / "Ouster_gt.txt"
    p.write_text("1689534551843447420 1.0 2.0 3.0 %.17g %.17g %.17g %.17g\n"
                 % tuple(q))
    t, rot, pos = load_poses_helipr(str(p))
    assert abs(t[0] - 1689534551.84344742) < 1e-6
    assert np.allclose(pos[0], [1.0, 2.0, 3.0])
    assert np.allclose(rot[0], Rotation.from_euler("z", 90.0,
                                                   degrees=True).as_matrix())


def test_the_nuscenes_reader_takes_the_quaternion_in_wxyz_order(tmp_path):
    # 누씬즈만 w 가 앞이다. 다른 릴리스 규약을 그대로 쓰면 회전이 틀리는데
    # 값의 크기가 그럴듯해서 눈으로는 안 잡힌다.
    r = Rotation.from_euler("z", 30.0, degrees=True)
    x, y, z, w = r.as_quat()
    p = tmp_path / "scene-0001_pose.json"
    p.write_text(json.dumps([{"utime": 1531883530467511,
                              "orientation": [w, x, y, z],
                              "pos": [10.0, 20.0, 0.0]}]))
    t, rot, pos = load_poses_nuscenes(str(p))
    assert abs(t[0] - 1531883530.467511) < 1e-6
    assert np.allclose(rot[0], r.as_matrix())
    assert np.allclose(pos[0], [10.0, 20.0, 0.0])


def _pit30m_file(tmp_path, yaw_deg, valid):
    cont = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8"),
                     ("yaw", "<f8"), ("pitch", "<f8"), ("roll", "<f8"),
                     ("valid", "?")])
    top = np.dtype([("capture_time", "<f8"),
                    ("poses_and_differentials_valid", "?"),
                    ("continuous", cont)])
    d = np.zeros(len(yaw_deg), dtype=top)
    d["capture_time"] = np.arange(len(yaw_deg)) * 0.01
    d["poses_and_differentials_valid"] = True
    d["continuous"]["yaw"] = yaw_deg
    d["continuous"]["x"] = np.arange(len(yaw_deg), dtype=float)
    d["continuous"]["valid"] = valid
    p = tmp_path / "all_poses.npz"
    np.savez(str(p), data=d)
    return str(p)


def test_the_pit30m_angles_are_read_as_degrees():
    # 라디안으로 읽으면 방위 120 이 열아홉 바퀴가 된다. 값은 여전히 회전행렬
    # 이라 아무 경고 없이 통과한다.
    with tempfile.TemporaryDirectory() as d:
        p = _pit30m_file(pathlib.Path(d), [120.0, 120.0], [True, True])
        _, rot, _ = load_poses_pit30m(p)
    assert np.allclose(rot[0], Rotation.from_euler("Z", 120.0,
                                                   degrees=True).as_matrix())


def test_the_pit30m_reader_drops_the_invalid_rows():
    # 유효 표시가 안 선 자세는 자리만 채운 값이다. 그대로 쓰면 구간 길이가
    # 엉뚱한 자리에서 쌓인다.
    with tempfile.TemporaryDirectory() as d:
        p = _pit30m_file(pathlib.Path(d), [0.0, 10.0, 20.0],
                         [True, False, True])
        t, rot, pos = load_poses_pit30m(p)
    assert len(t) == 2 and len(rot) == 2 and len(pos) == 2
    assert np.allclose(pos[:, 0], [0.0, 2.0])


def test_the_window_leaves_both_conventions_inside_the_released_samples():
    t = np.arange(11, dtype=float) / 10.0
    keep = common_window(t)
    # 반 표본 위상은 t+0.05 를, 반 주기 솎기는 솎은 격자(끝이 t[-1]) 위에서
    # t 를 묻는다. 둘 다 공개 표본 안에 들어야 한다.
    assert t[keep - 1] + 0.05 <= t[-1]
    assert t[keep - 1] <= t[::2][-1]
    # 마지막 프레임은 반 표본 위상이 범위를 넘기므로 빠진다.
    assert keep == len(t) - 1


def test_the_window_also_respects_the_decimated_grid():
    # 표본이 짝수 개면 솎은 격자의 끝이 t[-2] 라 거기서 한 번 더 좁혀진다.
    t = np.arange(10, dtype=float) / 10.0
    keep = common_window(t)
    assert t[keep - 1] <= t[::2][-1]
    assert keep == len(t) - 1


def test_a_number_written_to_be_read_back_survives_the_round_trip():
    """문턱과 견주려고 적는 수는 깎지 않는다.

    세 자리로 깎으면 1.002 도 0.9996 도 "1" 이 되어, 넘었는지 세는 쪽이
    경계 바로 위아래에서 틀린다. 그 경계가 그 수를 적는 이유다.
    """
    for x in (1.0023456, 0.99961234, 3.1853e-07, 0.0):
        assert float(exact(x)) == x
        assert (float(exact(x)) > 1.0) == (x > 1.0)
    assert float("%.3g" % 1.0023456) == 1.0        # 깎으면 이렇게 된다.


def test_the_nearest_rule_takes_the_earlier_sample_at_a_tie():
    """동률에서 어느 표본을 고르는지를 못박는다.

    이 자리는 실제로 밟힌다. 성긴 격자가 표본을 하나 걸러 고르면(비율 0.5)
    남은 질의가 정확히 두 표본 한가운데에 놓여 정확한 동률이 된다. 그래서
    규칙은 적는 것으로 끝나지 않고 수를 가른다 — 핏서티엠에서 규칙을 뒤집으면
    묶음 D 가 상대 0.23 퍼센트까지 달라진다(키티는 동률이 한 번도 없어 0).
    """
    t = np.array([0.0, 1.0, 2.0])
    rot = Rotation.from_rotvec([[0.0, 0.0, 0.0],
                                [0.0, 0.1, 0.0],
                                [0.0, 0.2, 0.0]]).as_matrix()
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    gr, gp = resample(t, rot, pos, np.array([0.5, 1.5]), "nearest")
    assert gp == pytest.approx(pos[[0, 1]])
    assert gr == pytest.approx(rot[[0, 1]])


def test_the_measurement_does_not_clip_the_last_frame():
    """끝을 잘라 붙이면 마지막 프레임이 한 표본 통째로 어긋나 값이 부푼다."""
    t, rot, pos = straight(yaw_rate=0.02)
    got, _ = measure(t, rot, pos)
    dt = float(np.median(np.diff(t)))
    keep = common_window(t, dt)
    idx = segments(pos[:keep])
    w = segment_weights(idx)
    dr, dp = resample(t[::2], rot[::2], pos[::2], t[:keep], "slerp")
    want = float(np.dot(w, euclidean(
        relative_translation(rot[:keep], pos[:keep], idx),
        relative_translation(dr, dp, idx))))
    assert got["t_decim2"] == pytest.approx(want, rel=1e-12)
