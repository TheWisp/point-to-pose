"""A recorded depth edge makes the selectable association rules distinguishable."""

import numpy as np
import pytest

from point2pose.utils.tracked_depth import lookup_tracked_depth


@pytest.mark.parametrize(
    "mode,depth_m,valid",
    [
        ("legacy", 0.405, True),
        ("nearest", 0.414, True),
        ("edge", np.nan, False),
    ],
)
def test_lookup_mode_selects_the_requested_surface_rule(mode, depth_m, valid):
    depth = np.array([[399, 402, 405, 414, 415]] * 5, dtype=np.uint16)
    points, accepted = lookup_tracked_depth(
        [[2.619873, 2.5869446]],
        depth,
        np.eye(3),
        depth_factor=1000,
        mode=mode,
    )
    assert bool(accepted[0]) == valid
    np.testing.assert_allclose(points[0, 2], depth_m, atol=1e-7)


def test_default_rounds_but_unknown_modes_cannot_silently_choose_a_rule():
    depth = np.array([[399, 402, 405, 414, 415]] * 5, dtype=np.uint16)
    points, _ = lookup_tracked_depth(
        [[2.619873, 2.5869446]], depth, np.eye(3), depth_factor=1000
    )
    np.testing.assert_allclose(points[0, 2], 0.414, atol=1e-7)
    with pytest.raises(ValueError, match="Unknown tracked_depth_mode"):
        lookup_tracked_depth([[1, 1]], depth, np.eye(3), mode="unknown")
