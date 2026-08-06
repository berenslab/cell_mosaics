"""Tests for cell_mosaics.plot_metrics."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
from matplotlib import pyplot as plt

from cell_mosaics.plot_metrics import plot_voronoi_tessellation


class TestPlotVoronoiTessellation:
    def test_no_interior_mask_draws_single_scatter_group(self):
        rng = np.random.default_rng(0)
        positions = rng.uniform(0, 100, (20, 2))

        fig, ax = plot_voronoi_tessellation(positions)
        scatters = [c for c in ax.collections if hasattr(c, "get_offsets")]
        assert len(scatters[-1].get_offsets()) == len(positions)
        plt.close(fig)

    def test_interior_mask_splits_points_by_color(self):
        rng = np.random.default_rng(1)
        positions = rng.uniform(0, 100, (20, 2))
        mask = np.zeros(len(positions), dtype=bool)
        mask[:12] = True
        rng.shuffle(mask)

        fig, ax = plot_voronoi_tessellation(
            positions, interior_mask=mask, point_color="red", exterior_color="gray",
        )
        scatters = [c for c in ax.collections if hasattr(c, "get_offsets")]
        exterior_scatter, interior_scatter = scatters[-2], scatters[-1]

        assert len(exterior_scatter.get_offsets()) == int((~mask).sum())
        assert len(interior_scatter.get_offsets()) == int(mask.sum())
        np.testing.assert_allclose(sorted(exterior_scatter.get_offsets().tolist()),
                                    sorted(positions[~mask].tolist()))
        np.testing.assert_allclose(sorted(interior_scatter.get_offsets().tolist()),
                                    sorted(positions[mask].tolist()))
        plt.close(fig)

    def test_all_interior_mask_puts_all_points_in_point_color_group(self):
        rng = np.random.default_rng(2)
        positions = rng.uniform(0, 100, (15, 2))
        mask = np.ones(len(positions), dtype=bool)

        fig, ax = plot_voronoi_tessellation(positions, interior_mask=mask)
        scatters = [c for c in ax.collections if hasattr(c, "get_offsets")]
        exterior_scatter, interior_scatter = scatters[-2], scatters[-1]

        assert len(exterior_scatter.get_offsets()) == 0
        assert len(interior_scatter.get_offsets()) == len(positions)
        plt.close(fig)
