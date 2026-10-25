# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Accepted periodic frames must survive LAMMPS restart remapping."""
from copy import deepcopy

import numpy as np
import pytest
from test_cli import FrameSource, accepted_frame

from qmdpa.runtime import _canonical_periodic_frame, _compare_restart, _frame

BOX = ([0, 0, 0], [6, 6, 6], 0.0, 0.0, 0.0, [1, 1, 1], 0)


@pytest.mark.parametrize("x,image,expected_x,expected_image", [
    ([6.25, 2, 3], [0, 0, 0], [0.25, 2, 3], [1, 0, 0]),
    ([-0.25, 2, 3], [0, 0, 0], [5.75, 2, 3], [-1, 0, 0]),
    ([12.25, -6.5, 18.75], [-2, 3, -1], [0.25, 5.5, 0.75], [0, 1, 2]),
    ([6, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0]),
])
def test_same_unwrapped_coordinates_have_identical_checkpoint_representation(x, image, expected_x, expected_image):
    frame = accepted_frame()
    frame["atoms"][0].update(x=x, image=image)
    original = deepcopy(frame)
    restored = deepcopy(frame)
    restored["atoms"][0].update(x=expected_x, image=expected_image)
    _canonical_periodic_frame(frame, BOX)
    _compare_restart(restored, {"frame": frame})
    assert frame["atoms"][0]["x"] == expected_x
    assert frame["atoms"][0]["image"] == expected_image
    for old, new in zip(original["atoms"], frame["atoms"], strict=True):
        np.testing.assert_array_equal(np.array(old["x"]) + 6*np.array(old["image"]),
                                      np.array(new["x"]) + 6*np.array(new["image"]))
        assert old["f"] == new["f"] and old["v"] == new["v"] and old["id"] == new["id"]
    # Canonicalization does not weaken the strict restart corruption guard.
    restored["atoms"][0]["image"][0] += 1
    with pytest.raises(ValueError, match="image"):
        _compare_restart(restored, {"frame": frame})


def test_native_frame_records_wrapped_image_pair_without_mutating_native_arrays():
    source = FrameSource()
    source.arrays["x"][1] = [6.25, 2, 3]
    source.extract_box = lambda: BOX
    original = source.arrays["x"].copy()
    raw = _frame(source, None)
    wrapped = _frame(source, None, periodic=True)
    assert raw["atoms"][0]["x"] == [6.25, 2, 3]
    assert wrapped["atoms"][0]["x"] == [0.25, 2, 3]
    assert wrapped["atoms"][0]["image"] == [1, 0, 0]
    np.testing.assert_array_equal(source.arrays["x"], original)
    np.testing.assert_array_equal(source.arrays["image"], [0, 0])


def test_nonzero_origin_and_tilt_preserve_physical_position():
    frame = accepted_frame()
    box = ([-2, 1, 3], [4, 7, 9], 1.0, 0.0, 0.0, [1, 1, 1], 0)
    frame["atoms"][0].update(x=[11.5, 13.25, -2.0], image=[-1, 2, 0])
    from qmdpa.region import box_matrix
    cell = box_matrix(box)
    original = deepcopy(frame)
    _canonical_periodic_frame(frame, box)
    for a, b in zip(original["atoms"], frame["atoms"], strict=True):
        np.testing.assert_allclose(np.array(a["x"]) + np.array(a["image"]) @ cell,
                                   np.array(b["x"]) + np.array(b["image"]) @ cell, atol=1e-14)
        fractional = np.linalg.solve(cell.T, np.array(b["x"]) - box[0])
        assert np.all(fractional >= 0) and np.all(fractional < 1)


def test_periodic_frame_rejects_partial_boundary():
    source = FrameSource()
    box = list(BOX)
    box[5] = [1, 1, 0]
    source.extract_box = lambda: box
    with pytest.raises(RuntimeError, match="three-dimensional PBC"):
        _frame(source, None, periodic=True)
