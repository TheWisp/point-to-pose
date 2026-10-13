"""The returned pose, support statistics and history must describe one estimate."""

import numpy as np
import pytest

from point2pose.modules.register.svd_cluster_ransac_register import (
    SVDClusterRANSACRegister,
)


def transform(pose, points):
    return points @ pose[:3, :3].T + pose[:3, 3]


@pytest.mark.parametrize(
    "case", ["accepted", "rotated", "unsupported", "skipped", "disabled"]
)
def test_refinement_result_and_support_describe_the_returned_pose(monkeypatch, case):
    # Control the two estimates to exercise return routing and the real final
    # support gate independently of RANSAC randomness and SDF convergence.
    source = np.array([[x, y, 0.4] for x in [-0.01, 0.0, 0.01] for y in [-0.01, 0.01]])
    seed = np.eye(4)
    refined = np.eye(4)
    seed[0, 3] = 0.02
    if case == "rotated":
        angle = np.deg2rad(25)
        refined[:3, :3] = [
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)],
        ]
        centre = source.mean(axis=0)
        refined[:3, 3] = centre - refined[:3, :3] @ centre + [0.01, -0.02, 0.03]
    target = transform(refined, source)
    if case == "unsupported":
        target = transform(seed, source)
    if case in ("skipped", "disabled"):
        refined = seed.copy()
        target = transform(seed, source)

    reg = SVDClusterRANSACRegister(
        {
            "max_clusters": 1,
            "select_method": "inlier_count",
            "inlier_thres": 0.005,
            "min_inliers": 5,
            "enable_sdf_refine": case != "disabled",
        }
    )
    monkeypatch.setattr(
        reg,
        "_RANSAC",
        lambda **kw: {"T": seed.copy(), "ninliers": len(source), "mean_res": 0.0},
    )

    def refine(**kw):
        assert case != "disabled", "disabled refinement must not run"
        return refined.copy(), {"enabled": True, "applied": case != "skipped"}

    monkeypatch.setattr(reg, "_maybe_refine_with_sdf", refine)
    pose, stats = reg.register(source, target, init_pose=np.eye(4))
    expected = refined if case in ("accepted", "rotated") else seed
    actual_residuals = np.linalg.norm(transform(pose, source) - target, axis=1)
    np.testing.assert_allclose(pose, expected, atol=1e-12)
    np.testing.assert_allclose(stats["residuals"], actual_residuals, atol=1e-12)
    np.testing.assert_array_equal(
        stats["inliers"], actual_residuals <= reg._inlier_thres
    )
    np.testing.assert_array_equal(reg._last_selected_T, pose)
    assert stats["final_inlier_gate"]["applied"] == (case == "unsupported")


def test_no_registration_candidate_keeps_the_prior(monkeypatch):
    reg = SVDClusterRANSACRegister({"enable_sdf_refine": True})
    monkeypatch.setattr(reg, "_RANSAC", lambda **kw: None)
    prior = np.eye(4)
    prior[0, 3] = 0.03
    points = np.zeros((6, 3))
    pose, stats = reg.register(points, points, init_pose=prior)
    np.testing.assert_array_equal(pose, prior)
    assert stats["clusters"] == []
    assert not np.any(stats["inliers"])


def test_explicit_legacy_return_mode_reproduces_the_old_pose_statistics_mismatch(
    monkeypatch,
):
    source = np.array([[x, y, 0.4] for x in [-0.01, 0, 0.01] for y in [-0.01, 0.01]])
    seed = np.eye(4)
    seed[0, 3] = 0.02
    reg = SVDClusterRANSACRegister(
        {
            "max_clusters": 1,
            "select_method": "inlier_count",
            "inlier_thres": 0.005,
            "min_inliers": 5,
            "enable_sdf_refine": True,
            "return_refined_pose": False,
        }
    )
    monkeypatch.setattr(
        reg,
        "_RANSAC",
        lambda **kw: {"T": seed.copy(), "ninliers": len(source), "mean_res": 0.0},
    )
    monkeypatch.setattr(
        reg,
        "_maybe_refine_with_sdf",
        lambda **kw: (np.eye(4), {"enabled": True, "applied": True}),
    )
    pose, stats = reg.register(source, source, init_pose=np.eye(4))
    np.testing.assert_array_equal(pose, seed)
    np.testing.assert_allclose(stats["residuals"], 0, atol=1e-12)
    assert np.linalg.norm(transform(pose, source) - source, axis=1).min() > 0.019
