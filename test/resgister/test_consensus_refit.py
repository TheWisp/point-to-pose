"""Rejecting a depth-edge match must also remove its influence from the fit."""

import json
from pathlib import Path

import numpy as np
import pytest

from point2pose.modules.register.svd_cluster_ransac_register import (
    SVDClusterRANSACRegister,
)
from point2pose.utils.transform import transform_pts


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("with_prior", [False, True])
def test_rejected_depth_edge_does_not_remain_in_final_fit(weighted, with_prior):
    data = json.loads(
        (Path(__file__).parent / "fixtures/depth_edge_correspondences.json").read_text()
    )
    src, dst = np.asarray(data["src_pcd"]), np.asarray(data["tgt_pcd"])
    prior = np.eye(4)
    if with_prior:
        angle = np.radians(12)
        prior[:3, :3] = [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
        prior[:3, 3] = [0.03, -0.01, 0.02]
    reg = SVDClusterRANSACRegister(
        {
            "ransac_iters": 100,
            "sample_size": 4,
            "min_inliers": 5,
            "inlier_thres": 0.005,
            "max_iter": 15,
        }
    )
    weights = (
        reg._build_weights(len(src), None, np.asarray(data["sigma_tgt"]), None)
        if weighted
        else None
    )
    remaining = np.ones(len(src), dtype=bool)
    np.random.seed(17)
    candidate = reg._RANSAC(transform_pts(prior, src), dst, weights, remaining, prior)
    assert candidate is not None
    ids = candidate["inliers"]
    assert 1 not in ids, "The observed depth-edge correspondence is inconsistent"
    expected = (
        reg._weighted_svd_fit(src[ids], dst[ids], weights[ids])
        if weighted
        else reg._svd_fit(src[ids], dst[ids])
    )
    # This fails on the original one-pass refinement even though that same
    # implementation already labels point 1 an outlier.
    np.testing.assert_allclose(candidate["T"], expected, atol=1e-9)
    residuals = np.linalg.norm(transform_pts(candidate["T"], src) - dst, axis=1)
    np.testing.assert_array_equal(ids, np.flatnonzero(residuals <= 0.005))
    np.testing.assert_array_equal(~remaining, residuals <= 0.005)


def test_clean_moving_object_preserves_the_known_rigid_transform():
    rng = np.random.default_rng(3)
    src = rng.uniform(-0.03, 0.03, (30, 3)) + [0, 0, 0.4]
    angle = np.radians(24)
    expected = np.eye(4)
    expected[:3, :3] = [
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle), np.cos(angle)],
    ]
    expected[:3, 3] = [0.01, -0.02, 0.04]
    dst = transform_pts(expected, src)
    reg = SVDClusterRANSACRegister(
        {
            "ransac_iters": 100,
            "sample_size": 4,
            "min_inliers": 5,
            "inlier_thres": 0.005,
            "max_iter": 15,
        }
    )
    candidate = reg._RANSAC(src, dst, None, np.ones(len(src), bool), None)
    np.testing.assert_allclose(candidate["T"], expected, atol=1e-12)
    assert candidate["ninliers"] == len(src)


def test_explicit_one_pass_mode_retains_the_recorded_rejected_point_influence():
    data = json.loads(
        (Path(__file__).parent / "fixtures/depth_edge_correspondences.json").read_text()
    )
    src, dst = np.asarray(data["src_pcd"]), np.asarray(data["tgt_pcd"])
    reg = SVDClusterRANSACRegister(
        {
            "ransac_iters": 100,
            "sample_size": 4,
            "min_inliers": 5,
            "inlier_thres": 0.005,
            "consensus_refit": False,
        }
    )
    np.random.seed(17)
    candidate = reg._RANSAC(src, dst, None, np.ones(len(src), bool), None)
    ids = candidate["inliers"]
    assert 1 not in ids
    refitted = reg._svd_fit(src[ids], dst[ids])
    assert np.max(np.abs(candidate["T"] - refitted)) > 1e-4
