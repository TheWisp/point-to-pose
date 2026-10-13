"""A delayed map update must not replace today's pose with a historical pose.

Exercise the real pipeline step with known rigid motion. Neural tracking and
batch graph solving are controlled; state application and SDF routing are real.
"""

from types import SimpleNamespace as NS

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from point2pose.pipeline.modular_pipeline import ModularPipeline


def pose(angles, xyz):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
    result[:3, 3] = xyz
    return result


def run_step(current, keyframes, optimized, *, limit=3.0, preserve=True):
    pipe = ModularPipeline.__new__(ModularPipeline)
    pipe._initialized = True
    pipe.pose_log_files = []
    pipe.use_depth_estimate = False
    pipe.num_obj = len(current)
    pipe.objects = [NS(pose=p.copy()) for p in current]
    pipe.use_local_graph = False
    pipe.use_key_frame_graph = True
    pipe.preserve_delayed_keyframe_motion = preserve
    pipe.max_rel_translation = limit
    pipe.max_rel_rotation_deg = 1000.0
    fe = NS(
        tracks=None, track_3d=None, track_valid=None, uncertainties=None, visibles=None
    )
    pipe.frontend = NS(step=lambda *a: fe, tracker=None)
    pipe._update_object_from_frontend = lambda *a: None
    pipe.track_table = NS(update_track_table=lambda *a: None)
    pipe._update_history_frames = lambda *a: None
    pipe.hist_frames = pipe.hist_fe_results = pipe.hist_track_tables = []
    pipe.kf_manager = NS(update=lambda **kw: keyframes)
    pipe.local_optimizer = NS(reset=lambda *a: None)
    pipe._log_step = lambda *a: None
    fused = []

    def graph_update(frames):
        # Real KeyFrameGraph mutates these same keyframe objects before returning.
        for k in frames:
            k.pose = optimized[(k.obj_id, k.kf_idx)].copy()
        return optimized, {}

    def integrate(obj, keyframe):
        fused.append((keyframe.frame_id, keyframe.pose.copy()))
        return False

    pipe.kf_graph = NS(update=graph_update)
    pipe.sdf_builder = NS(integrate_keyframe=integrate)
    result = pipe.step(NS(id=54))
    return result, fused


def kf(old, *, frame=51, obj=0, index=1):
    return NS(pose=old.copy(), frame_id=frame, obj_id=obj, kf_idx=index)


@pytest.mark.parametrize(
    "motion",
    [
        pose([0, 0, 0], [0.02, -0.01, 0.03]),
        pose([15, -7, 11], [0.02, -0.01, 0.03]),
    ],
)
def test_unchanged_historical_graph_cannot_erase_known_intervening_motion(motion):
    old = pose([20, 5, -30], [0.1, -0.05, 0.4])
    current = motion @ old
    result, fused = run_step([current], [kf(old)], {(0, 1): old})
    np.testing.assert_allclose(result[0], current, atol=1e-12)
    np.testing.assert_allclose(fused[0][1], old, atol=1e-12)


def test_graph_correction_preserves_motion_and_keeps_sdf_at_observation_time():
    # Different rotations prevent an incorrect multiplication order from passing.
    old = pose([20, 5, -30], [0.1, -0.05, 0.4])
    motion = pose([15, -7, 11], [0.02, -0.01, 0.03])
    corrected = pose([18, 7, -32], [0.102, -0.049, 0.403])
    result, fused = run_step([motion @ old], [kf(old)], {(0, 1): corrected})
    points = np.array([[0.01, 0.02, 0.03, 1], [-0.1, 0.2, -0.05, 1]]).T
    np.testing.assert_allclose(
        result[0] @ points, motion @ corrected @ points, atol=1e-12
    )
    np.testing.assert_allclose(fused[0][1], corrected, atol=1e-12)


def test_current_frame_retains_direct_graph_pose_behavior():
    old = pose([20, 5, -30], [0.1, -0.05, 0.4])
    corrected = pose([18, 7, -32], [0.102, -0.049, 0.403])
    result, fused = run_step([old], [kf(old, frame=54)], {(0, 1): corrected})
    np.testing.assert_allclose(result[0], corrected, atol=1e-12)
    np.testing.assert_allclose(fused[0][1], corrected, atol=1e-12)


def test_out_of_order_promotions_apply_only_latest_correction_per_object():
    old = pose([20, 5, -30], [0.1, -0.05, 0.4])
    recent = pose([21, 3, -25], [0.11, -0.04, 0.42])
    current = pose([28, 0, -24], [0.12, -0.01, 0.43])
    correction = pose([2, -1, 3], [0.002, -0.001, 0.003])
    updated_recent = recent @ correction
    second = pose([-15, 6, 12], [-0.2, 0.02, 0.5])
    frames = [kf(recent, frame=52, index=2), kf(second, obj=1), kf(old)]
    values = {(0, 2): updated_recent, (0, 1): old @ correction, (1, 1): second}
    result, fused = run_step([current, second], frames, values)
    np.testing.assert_allclose(result[0], current @ correction, atol=1e-12)
    np.testing.assert_allclose(result[1], second, atol=1e-12)
    assert len(fused) == 3


def test_jump_guard_checks_correction_not_time_elapsed_since_keyframe():
    old = np.eye(4)
    current = pose([0, 0, 0], [0.2, 0, 0])
    correction = pose([0, 0, 0], [0.001, 0, 0])
    result, _ = run_step([current], [kf(old)], {(0, 1): correction}, limit=0.01)
    np.testing.assert_allclose(result[0][:3, 3], [0.201, 0, 0], atol=1e-12)
    result, _ = run_step([current], [kf(old)], {(0, 1): current}, limit=0.01)
    np.testing.assert_allclose(result[0], current, atol=1e-12)


def test_explicit_legacy_mode_reproduces_historical_pose_assignment():
    old = np.eye(4)
    current = pose([15, -7, 11], [0.02, -0.01, 0.03])
    result, _ = run_step([current], [kf(old)], {(0, 1): old}, preserve=False)
    np.testing.assert_array_equal(result[0], old)
