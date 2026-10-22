# Copyright (2026) Pengchao Zhang <pczhang1997@gmail.com>
# SPDX-License-Identifier: GPL-2.0-only
"""Regression coverage for atom migration, large tags, and image continuity."""

import numpy as np
import pytest

from qmdpa.region import FixedRegion, box_matrix, unwrap


def test_migration_sorting_empty_rank_and_large_tags():
    region = FixedRegion((2**40 + 1, 9), ("O", "H"))
    packets = [([], []), ([9], [[1, 0, 0]]), ([2**40 + 1], [[0, 0, 0]])]
    geometry = region.assemble(packets)
    np.testing.assert_array_equal(geometry.positions, [[0, 0, 0], [1, 0, 0]])
    delta = np.array([[1, 2, 3], [4, 5, 6]])
    np.testing.assert_array_equal(
        region.local_forces([9, 10, 2**40 + 1], delta), [[4, 5, 6], [0, 0, 0], [1, 2, 3]]
    )
    assert region.local_forces([], delta).shape == (0, 3)


@pytest.mark.parametrize("packets", [[], [([7, 7], [[0, 0, 0], [1, 0, 0]])], [([8], [[0, 0, 0]])]])
def test_missing_duplicate_or_unknown_region_ids_fail(packets):
    with pytest.raises(ValueError):
        FixedRegion((7,), ("H",)).assemble(packets)


def test_triclinic_crossing_preserves_continuous_coordinate():
    box = ([0, 0, 0], [10, 11, 12], 2.0, 3.0, 4.0, [1, 1, 1], 0)
    cell = box_matrix(box)
    before = unwrap([[9.9, 2, 3]], [[0, 0, 0]], cell)
    after = unwrap([[0.1, 2, 3]], [[1, 0, 0]], cell)
    np.testing.assert_allclose(after - before, [[0.2, 0, 0]], atol=1e-14)
    np.testing.assert_allclose(unwrap([[0, 0, 0]], [[0, 1, 1]], cell), [[6, 14, 12]])


def test_fractional_image_is_rejected():
    with pytest.raises(ValueError, match="integers"):
        unwrap([[0, 0, 0]], [[0.5, 0, 0]], np.eye(3))
