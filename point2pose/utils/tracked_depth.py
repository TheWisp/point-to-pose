"""Experimental group-style depth lookup for sparse tracks, never dense surfaces."""

import warnings

import numpy as np


def convert_tracked_pixel_to_world(
    pixel,
    depth_image,
    cam_intrinsics,
    depth_factor=1.0,
    cam2world=np.eye(4),
    remove_invalid=False,
    min_depth=0.05,
    max_depth=1.0,
    fill_missing_depth=False,
    window_size=5,
    min_neighbors=1,
    compute_depth_uncertainty=False,
    sigma_min=0.002,
    sigma_max=0.05,
    sigma_base_a=0.003,
    sigma_base_b=0.0,
    edge_alpha=5.0,
    *,
    edge_m=0.012,
):
    """Median XYZ of a local patch; discard ambiguous depth edges.

    This experiment reproduces LeRobot groups_scene.lookup_3d's rule: nearest
    pixel, 5x5 window, at least three valid samples, 10th–90th percentile depth
    spread at most 12 mm. Those existing comparison settings are not claimed
    to be calibrated for every sensor/object. The caller's window and depth
    range are respected. Initialization, live tracks and new model points all
    use this path; dense reconstruction continues to use the original helper.

    The median is used regardless of fill_missing_depth. This candidate does
    not calculate depth uncertainty; the active pipeline requests none.
    """
    if compute_depth_uncertainty:
        raise ValueError(
            "experimental median lookup does not estimate depth uncertainty"
        )
    if (
        window_size < 1
        or window_size % 2 != 1
        or not np.isfinite(edge_m)
        or edge_m <= 0
    ):
        raise ValueError(
            "require an odd positive window size and positive finite edge tolerance"
        )
    uv = np.asarray(pixel, dtype=float).reshape(-1, 2)
    h, w = depth_image.shape[:2]
    valid = np.zeros(len(uv), dtype=bool)
    points = np.full((len(uv), 3), np.nan)
    inside = (
        np.isfinite(uv).all(axis=1)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < w)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < h)
    )
    ids = np.flatnonzero(inside)
    if len(ids):
        rounded = np.rint(uv[ids]).astype(int)
        offsets = np.arange(-(window_size // 2), window_size // 2 + 1)
        xs, ys = np.broadcast_arrays(
            rounded[:, 0, None, None] + offsets[None, None, :],
            rounded[:, 1, None, None] + offsets[None, :, None],
        )
        within = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        z = depth_image[np.clip(ys, 0, h - 1), np.clip(xs, 0, w - 1)].astype(
            np.float32
        ) / float(depth_factor)
        finite = within & np.isfinite(z) & (z > min_depth) & (z < max_depth)
        z = np.where(finite, z, np.nan).astype(float)
        k = np.asarray(cam_intrinsics)
        xyz = np.stack(
            ((xs - k[0, 2]) * z / k[0, 0], (ys - k[1, 2]) * z / k[1, 1], z), axis=-1
        ).reshape(len(ids), -1, 3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            lo, hi = np.nanpercentile(z.reshape(len(ids), -1), [10, 90], axis=1)
            median = np.nanmedian(xyz, axis=1)
        ok = (finite.reshape(len(ids), -1).sum(axis=1) >= max(3, min_neighbors)) & (
            hi - lo <= edge_m
        )
        transform = np.asarray(cam2world)
        points[ids[ok]] = median[ok] @ transform[:3, :3].T + transform[:3, 3]
        valid[ids[ok]] = True
    return (points[valid] if remove_invalid else points), valid
