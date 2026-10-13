"""Depth edges expose biased pixel lookup; integer dense clouds stay unchanged."""

import numpy as np
import pytest

from point2pose.utils.camera import convert_pixel_to_world


@pytest.mark.parametrize("remove_invalid", [False, True])
def test_fractional_track_samples_nearest_surface_and_matching_ray(remove_invalid):
    # Recorded edge neighbourhood: floor selects 405 mm, nearest selects
    # 414 mm. Shift the real pixel (354.62, 309.59) into this small fixture.
    depth = np.array([[399, 402, 405, 414, 415]] * 5, dtype=np.uint16)
    k = np.array([[500.0, 0, 2], [0, 500.0, 2], [0, 0, 1]])
    point, valid = convert_pixel_to_world(
        [[2.619873, 2.5869446]],
        depth,
        k,
        depth_factor=1000,
        remove_invalid=remove_invalid,
    )
    np.testing.assert_array_equal(valid, [True])
    np.testing.assert_allclose(point, [[0.414 / 500, 0.414 / 500, 0.414]], atol=1e-8)


def test_integer_queries_and_world_transform_preserve_known_geometry():
    depth = np.array([[400, 450], [500, 550]], dtype=np.uint16)
    k = np.array([[100.0, 0, 0], [0, 200.0, 0], [0, 0, 1]])
    world = np.eye(4)
    world[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    world[:3, 3] = [0.1, -0.2, 0.3]
    points, valid = convert_pixel_to_world(
        [[0, 0], [1, 0], [0, 1], [1, 1]],
        depth,
        k,
        depth_factor=1000,
        cam2world=world,
    )
    camera = np.array(
        [[0, 0, 0.4], [0.0045, 0, 0.45], [0, 0.0025, 0.5], [0.0055, 0.00275, 0.55]]
    )
    expected = camera @ world[:3, :3].T + world[:3, 3]
    np.testing.assert_array_equal(valid, [True] * 4)
    np.testing.assert_allclose(points, expected, atol=1e-7)


def test_nearest_pixel_outside_image_does_not_wrap_or_clip_to_another_surface():
    depth = np.full((3, 3), 0.4, dtype=np.float32)
    points, valid = convert_pixel_to_world([[-0.6, 1], [2.6, 1]], depth, np.eye(3))
    np.testing.assert_array_equal(valid, [False, False])
    assert np.isnan(points).all()


def test_invalid_depth_and_uncertainty_keep_original_return_contract():
    depth = np.full((5, 5), 0.4, dtype=np.float32)
    depth[2, 2] = 0
    points, valid, sigma = convert_pixel_to_world(
        [[1.1, 1.1], [1.8, 1.8]],
        depth,
        np.eye(3),
        compute_depth_uncertainty=True,
        remove_invalid=True,
    )
    np.testing.assert_array_equal(valid, [True, False])
    assert points.shape == (1, 3)
    assert sigma.shape == (1,)
    assert np.isfinite(points).all() and np.isfinite(sigma).all()
