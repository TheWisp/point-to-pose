"""Ambiguous surfaces must not become sparse rigid-fit correspondences."""

import numpy as np

from point2pose.utils.tracked_depth import convert_tracked_pixel_to_world as convert


def test_depth_edge_is_invalid_without_averaging_two_surfaces():
    depth = np.full((7, 7), 0.4, dtype=np.float32)
    depth[:, 4:] = 0.45
    points, valid = convert([[3.2, 3.2]], depth, np.eye(3))
    assert not valid[0] and np.isnan(points).all()


def test_isolated_bad_pixel_does_not_move_a_flat_surface():
    depth = np.full((7, 7), 0.4, dtype=np.float32)
    depth[3, 3] = 0.405
    points, valid = convert([[3.1, 3.1]], depth, np.eye(3))
    assert valid[0]
    np.testing.assert_allclose(points, [[1.2, 1.2, 0.4]], atol=1e-7)


def test_outside_image_is_not_counted_as_repeated_support():
    depth = np.zeros((5, 5), dtype=np.float32)
    depth[0, 0] = 0.4
    points, valid = convert([[0, 0]], depth, np.eye(3))
    assert not valid[0] and np.isnan(points).all()


def test_metric_units_world_transform_and_invalid_output_indices():
    depth = np.full((7, 7), 400, dtype=np.uint16)
    transform = np.eye(4)
    transform[:3, 3] = [0.1, -0.2, 0.3]
    points, valid = convert(
        [[3, 3], [-1, 3], [float("nan"), 3]],
        depth,
        np.eye(3),
        depth_factor=1000,
        cam2world=transform,
        remove_invalid=True,
    )
    np.testing.assert_array_equal(valid, [True, False, False])
    np.testing.assert_allclose(points, [[1.3, 1.0, 0.7]], atol=1e-7)
