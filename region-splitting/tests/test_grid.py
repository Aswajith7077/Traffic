"""Tests for static grid-based (manual) partitioning (``services/grid.py``).

Grid partitioning is a pure function of node coordinates, so these tests
exercise the standalone toolkit (``compute_grid_shape``, ``assign_grid_cells``)
rather than the SUMO-bound ``GridService`` class.
"""

from __future__ import annotations

import numpy as np
import pytest
from services.grid import assign_grid_cells, compute_grid_shape


def _square_lattice(n_per_side: int) -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(n_per_side), np.arange(n_per_side))
    return np.column_stack([xs.ravel(), ys.ravel()]).astype(float)


def test_compute_grid_shape_targets_mean_band():
    coords = _square_lattice(20)  # 400 nodes, square bounding box
    rows, cols = compute_grid_shape(coords, mean_min=10, mean_max=50)

    n_nodes = len(coords)
    mean_size = n_nodes / (rows * cols)
    assert 1 <= mean_size <= 2 * 50  # generous band; exact target isn't guaranteed after rounding
    assert rows >= 1 and cols >= 1


def test_compute_grid_shape_matches_aspect_ratio():
    # Wide, short bounding box (width >> height) should yield more columns than rows.
    xs = np.linspace(0, 100, 200)
    ys = np.linspace(0, 5, 200)
    coords = np.column_stack([xs, ys])

    rows, cols = compute_grid_shape(coords, mean_min=2, mean_max=10)
    assert cols >= rows


def test_compute_grid_shape_handles_degenerate_bbox():
    # All points share the same x (zero width) — must not divide by zero.
    coords = np.column_stack([np.zeros(10), np.arange(10, dtype=float)])
    rows, cols = compute_grid_shape(coords, mean_min=2, mean_max=5)
    assert rows >= 1 and cols >= 1


def test_assign_grid_cells_covers_all_points():
    coords = _square_lattice(10)
    cell_ids = assign_grid_cells(coords, rows=3, cols=3)

    assert len(cell_ids) == len(coords)
    assert cell_ids.min() >= 0
    assert cell_ids.max() < 9


def test_assign_grid_cells_separates_corners():
    # Four extreme corners of a unit square must land in four distinct 2x2 cells.
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    cell_ids = assign_grid_cells(coords, rows=2, cols=2)
    assert len(set(cell_ids.tolist())) == 4


def test_assign_grid_cells_single_point_no_crash():
    coords = np.array([[5.0, 5.0]])
    cell_ids = assign_grid_cells(coords, rows=4, cols=4)
    assert cell_ids.shape == (1,)
    assert 0 <= cell_ids[0] < 16


@pytest.mark.parametrize("rows,cols", [(1, 1), (5, 1), (1, 5)])
def test_assign_grid_cells_degenerate_grid_shapes(rows, cols):
    coords = _square_lattice(6)
    cell_ids = assign_grid_cells(coords, rows=rows, cols=cols)
    assert cell_ids.max() < rows * cols
