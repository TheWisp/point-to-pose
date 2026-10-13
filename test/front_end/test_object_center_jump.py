"""A pose's translation column is not object displacement away from the origin."""

import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from test.front_end.test_pose_jump_guard import make_front_end, stats


def run(candidate, previous, points, enabled=True):
    fe = make_front_end(
        pose_jump_guard_object_center=enabled,
        pose_jump_guard_trans_thres=0.03,
        pose_jump_guard_rot_deg_thres=15,
    )
    obj = SimpleNamespace(key_points=points, lost=False, lost_streak=0)
    return fe._apply_pose_jump_guard(candidate, previous, stats(11, 11, 1), obj)


def rotating_about(center, degrees):
    t = np.eye(4)
    t[:3, :3] = Rotation.from_euler("y", degrees, degrees=True).as_matrix()
    t[:3, 3] = center - t[:3, :3] @ center
    return t


def test_small_rotation_around_distant_object_is_not_large_translation():
    center = np.array([0.06, -0.03, 0.45])
    t = rotating_about(center, 5)
    points = center + np.array([[0.01, 0, 0], [-0.01, 0, 0]])
    _, reject, info = run(t, np.eye(4), points)
    assert not reject
    assert info["origin_dt"] > 0.03
    assert info["dt"] == pytest.approx(0, abs=1e-12)
    assert run(t, np.eye(4), points, enabled=False)[1]


@pytest.mark.parametrize("translation,degrees", [(0.04, 5), (0, 20)])
def test_real_displacement_or_large_rotation_still_requires_support(
    translation, degrees
):
    center = np.array([0.06, -0.03, 0.45])
    t = rotating_about(center, degrees)
    t[0, 3] += translation
    assert run(t, np.eye(4), center[None])[1]


def test_decision_and_distance_are_invariant_to_camera_coordinate_origin():
    center = np.array([0.06, -0.03, 0.45])
    previous = rotating_about(center, -3)
    candidate = rotating_about(center, 5)
    candidate[0, 3] += 0.01
    q = rotating_about(np.array([2.0, 3.0, -1.0]), 40)
    q[:3, 3] += [1.0, -2.0, 3.0]
    before = run(candidate, previous, center[None])
    after = run(
        q @ candidate @ np.linalg.inv(q),
        q @ previous @ np.linalg.inv(q),
        (q[:3, :3] @ center + q[:3, 3])[None],
    )
    assert before[1] == after[1]
    assert before[2]["dt"] == pytest.approx(after[2]["dt"], abs=1e-12)
    assert before[2]["ddeg"] == pytest.approx(after[2]["ddeg"], abs=1e-10)


@pytest.mark.parametrize("points", [[], [[np.nan, 0, 0]], [[1, 2]]])
def test_missing_or_unusable_model_retains_origin_check(points):
    t = np.eye(4)
    t[0, 3] = 0.04
    _, rejected, info = run(t, np.eye(4), points)
    assert rejected and info["translation_reference"] == "origin"


def test_recorded_partial_view_rejection_is_a_rotation_lever_arm():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/partial_jump.json").read_text()
    )
    t, prev, points = (
        np.asarray(fixture[k]) for k in ["candidate", "previous", "points"]
    )
    _, rejected, info = run(t, prev, points)
    assert not rejected
    assert info["origin_dt"] == pytest.approx(0.034481213856, abs=1e-10)
    assert info["dt"] < 0.001 and info["ddeg"] < 5
    assert run(t, prev, points, enabled=False)[1]
