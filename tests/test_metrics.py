"""Tests for cell_mosaics.metrics."""

import numpy as np
import pytest

from cell_mosaics.metrics import (
    compute_centroids,
    density_recovery_profile,
    drp_effective_radius,
    nearest_neighbor_distances,
    nnd_statistics,
    voronoi_analysis,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def regular_grid(n: int, spacing: float) -> np.ndarray:
    """Return a regular n×n grid of 2D positions."""
    x, y = np.meshgrid(np.arange(n) * spacing, np.arange(n) * spacing)
    return np.column_stack([x.ravel(), y.ravel()])


# ---------------------------------------------------------------------------
# compute_centroids
# ---------------------------------------------------------------------------

class TestComputeCentroids:
    def test_single_square(self):
        square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        centroids = compute_centroids([square])
        np.testing.assert_allclose(centroids[0], [0.5, 0.5])

    def test_multiple_outlines_shape(self):
        outlines = [
            np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float),
            np.array([[10, 10], [12, 10], [11, 12]], dtype=float),
        ]
        centroids = compute_centroids(outlines)
        assert centroids.shape == (2, 2)

    def test_centroid_of_axis_aligned_rectangle(self):
        rect = np.array([[0, 0], [4, 0], [4, 2], [0, 2]], dtype=float)
        centroids = compute_centroids([rect])
        np.testing.assert_allclose(centroids[0], [2.0, 1.0])


# ---------------------------------------------------------------------------
# nearest_neighbor_distances
# ---------------------------------------------------------------------------

class TestNearestNeighborDistances:
    def test_regular_grid_spacing(self):
        # All NNDs in a regular grid equal the grid spacing
        positions = regular_grid(5, spacing=10.0)
        nnd = nearest_neighbor_distances(positions)
        np.testing.assert_allclose(nnd, 10.0)

    def test_two_points(self):
        positions = np.array([[0.0, 0.0], [3.0, 4.0]])
        nnd = nearest_neighbor_distances(positions)
        np.testing.assert_allclose(nnd, [5.0, 5.0])

    def test_always_positive(self):
        rng = np.random.default_rng(0)
        positions = rng.uniform(0, 100, (50, 2))
        nnd = nearest_neighbor_distances(positions)
        assert np.all(nnd > 0)

    def test_output_shape(self):
        positions = np.random.default_rng(1).uniform(0, 100, (30, 2))
        nnd = nearest_neighbor_distances(positions)
        assert nnd.shape == (30,)


# ---------------------------------------------------------------------------
# nnd_statistics
# ---------------------------------------------------------------------------

class TestNndStatistics:
    def test_required_keys(self):
        positions = regular_grid(4, 10.0)
        stats = nnd_statistics(positions)
        assert {"nnd", "mean_nnd", "std_nnd", "regularity_index", "cv"} <= set(stats)

    def test_regular_grid_high_ri(self):
        # A perfect grid has zero std → RI should be very large
        positions = regular_grid(6, 10.0)
        stats = nnd_statistics(positions)
        assert stats["regularity_index"] > 10

    def test_random_low_ri(self):
        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 200, (200, 2))
        stats = nnd_statistics(positions)
        # Random Poisson process: RI ≈ 1.9 — well below the regular-grid value
        assert stats["regularity_index"] < 5

    def test_mean_matches_nnd(self):
        positions = regular_grid(4, 10.0)
        stats = nnd_statistics(positions)
        np.testing.assert_allclose(stats["mean_nnd"], np.mean(stats["nnd"]))

    def test_cv_and_ri_reciprocal(self):
        rng = np.random.default_rng(7)
        positions = rng.uniform(0, 100, (40, 2))
        stats = nnd_statistics(positions)
        np.testing.assert_allclose(
            stats["cv"], 1.0 / stats["regularity_index"], rtol=1e-10
        )


# ---------------------------------------------------------------------------
# density_recovery_profile
# ---------------------------------------------------------------------------

class TestDensityRecoveryProfile:
    def test_output_shape(self):
        rng = np.random.default_rng(0)
        positions = rng.uniform(0, 500, (100, 2))
        centers, drp = density_recovery_profile(
            positions, (0, 500, 0, 500), n_bins=15
        )
        assert centers.shape == drp.shape == (15,)

    def test_bin_centers_positive_and_ordered(self):
        rng = np.random.default_rng(1)
        positions = rng.uniform(0, 500, (80, 2))
        centers, _ = density_recovery_profile(positions, (0, 500, 0, 500))
        assert np.all(centers > 0)
        assert np.all(np.diff(centers) > 0)

    def test_random_points_drp_near_one_at_large_distance(self):
        # For a random process the DRP should average ≈ 1 at large distances
        rng = np.random.default_rng(2)
        positions = rng.uniform(0, 1000, (500, 2))
        centers, drp = density_recovery_profile(
            positions, (0, 1000, 0, 1000), n_bins=20, max_radius=200
        )
        np.testing.assert_allclose(drp[-5:].mean(), 1.0, atol=0.3)

    def test_regular_grid_exclusion_zone(self):
        # Regular grid with spacing 50: first distance bin (< 5) must be empty
        positions = regular_grid(10, spacing=50.0)
        centers, drp = density_recovery_profile(
            positions,
            field_bounds=(0.0, 450.0, 0.0, 450.0),
            n_bins=20,
            max_radius=100.0,
        )
        # bin width = 5 µm < spacing of 50 µm → no neighbors in first bin
        assert drp[0] == 0.0

    def test_drp_nonnegative(self):
        rng = np.random.default_rng(3)
        positions = rng.uniform(0, 300, (60, 2))
        _, drp = density_recovery_profile(positions, (0, 300, 0, 300))
        assert np.all(drp >= 0)

    def test_interior_mask_matches_unmasked_when_all_true(self):
        rng = np.random.default_rng(5)
        positions = rng.uniform(0, 300, (60, 2))
        mask = np.ones(len(positions), dtype=bool)
        _, drp_unmasked = density_recovery_profile(positions, (0, 300, 0, 300))
        _, drp_masked = density_recovery_profile(
            positions, (0, 300, 0, 300), interior_mask=mask
        )
        np.testing.assert_allclose(drp_masked, drp_unmasked)

    def test_interior_mask_restricts_reference_cells(self):
        # A grid with one corner cell removed from the reference set should
        # still use it as a neighbour, but never as a reference point.
        positions = regular_grid(6, spacing=50.0)
        mask = np.ones(len(positions), dtype=bool)
        mask[0] = False
        centers_all, drp_all = density_recovery_profile(
            positions, (0.0, 250.0, 0.0, 250.0), n_bins=10, max_radius=100.0
        )
        centers_masked, drp_masked = density_recovery_profile(
            positions,
            (0.0, 250.0, 0.0, 250.0),
            n_bins=10,
            max_radius=100.0,
            interior_mask=mask,
        )
        np.testing.assert_allclose(centers_masked, centers_all)
        assert not np.allclose(drp_masked, drp_all)


# ---------------------------------------------------------------------------
# drp_effective_radius
# ---------------------------------------------------------------------------

class TestDrpEffectiveRadius:
    def test_required_keys(self):
        rng = np.random.default_rng(6)
        positions = rng.uniform(0, 300, (80, 2))
        result = drp_effective_radius(positions, (0, 300, 0, 300))
        expected = {
            "effective_radius",
            "missing_count",
            "mean_density",
            "bin_centers",
            "cumulative_observed",
            "cumulative_expected",
        }
        assert expected <= set(result)

    def test_effective_radius_nonnegative(self):
        rng = np.random.default_rng(7)
        positions = rng.uniform(0, 300, (80, 2))
        result = drp_effective_radius(positions, (0, 300, 0, 300))
        assert result["effective_radius"] >= 0.0
        assert result["missing_count"] >= 0.0

    def test_regular_grid_has_larger_effective_radius_than_random(self):
        # A regular grid has a genuine exclusion zone; a random field of
        # matching density should not.
        n, spacing = 10, 50.0
        field = (0.0, (n - 1) * spacing, 0.0, (n - 1) * spacing)
        regular = regular_grid(n, spacing)

        rng = np.random.default_rng(8)
        random_pos = rng.uniform(field[0], field[1], (n * n, 2))

        re_regular = drp_effective_radius(
            regular, field, n_bins=30, max_radius=100.0
        )["effective_radius"]
        re_random = drp_effective_radius(
            random_pos, field, n_bins=30, max_radius=100.0
        )["effective_radius"]
        assert re_regular > re_random

    def test_effective_radius_roughly_matches_grid_spacing(self):
        # For a regular grid the exclusion zone should be on the order of
        # the grid spacing (within a factor of ~2).
        positions = regular_grid(10, spacing=50.0)
        field = (0.0, 450.0, 0.0, 450.0)
        result = drp_effective_radius(positions, field, n_bins=30, max_radius=100.0)
        assert 25.0 < result["effective_radius"] < 100.0

    def test_cumulative_arrays_shape(self):
        rng = np.random.default_rng(9)
        positions = rng.uniform(0, 300, (50, 2))
        result = drp_effective_radius(positions, (0, 300, 0, 300), n_bins=12)
        assert result["bin_centers"].shape == (12,)
        assert result["cumulative_observed"].shape == (12,)
        assert result["cumulative_expected"].shape == (12,)
        # Both cumulative sequences must be monotonically non-decreasing.
        assert np.all(np.diff(result["cumulative_observed"]) >= 0.0)
        assert np.all(np.diff(result["cumulative_expected"]) > 0.0)


# ---------------------------------------------------------------------------
# voronoi_analysis
# ---------------------------------------------------------------------------

class TestVoronoiAnalysis:
    def test_required_keys(self):
        positions = regular_grid(5, 20.0)
        result = voronoi_analysis(positions)
        expected = {"areas", "mean_area", "std_area", "cv_area", "regularity_index", "n_interior"}
        assert expected <= set(result)

    def test_positive_areas(self):
        rng = np.random.default_rng(4)
        positions = rng.uniform(0, 100, (50, 2))
        result = voronoi_analysis(positions, field_bounds=(0, 100, 0, 100))
        assert np.all(result["areas"] > 0)

    def test_regular_grid_low_cv(self):
        # All interior Voronoi cells of a regular grid are congruent → CV ≈ 0
        positions = regular_grid(8, 20.0)
        result = voronoi_analysis(positions, field_bounds=(0.0, 140.0, 0.0, 140.0))
        assert result["cv_area"] < 0.01

    def test_random_higher_cv_than_regular(self):
        n = 7
        spacing = 20.0
        field = (0.0, (n - 1) * spacing, 0.0, (n - 1) * spacing)
        regular = regular_grid(n, spacing)

        rng = np.random.default_rng(42)
        random_pos = rng.uniform(0, (n - 1) * spacing, (n * n, 2))

        regular_cv = voronoi_analysis(regular, field_bounds=field)["cv_area"]
        random_cv = voronoi_analysis(random_pos, field_bounds=field)["cv_area"]
        assert regular_cv < random_cv

    def test_n_interior_not_exceeds_total(self):
        positions = regular_grid(6, 15.0)
        result = voronoi_analysis(positions, field_bounds=(0, 75, 0, 75))
        assert result["n_interior"] <= len(positions)

    def test_without_field_bounds(self):
        # Should still work; only infinite Voronoi regions are dropped
        positions = regular_grid(5, 10.0)
        result = voronoi_analysis(positions)
        assert result["n_interior"] > 0
        assert np.all(result["areas"] > 0)

    def test_margin_factor_default_matches_factor_one(self):
        rng = np.random.default_rng(11)
        positions = rng.uniform(0, 300, (60, 2))
        field = (0, 300, 0, 300)
        default = voronoi_analysis(positions, field_bounds=field)
        explicit = voronoi_analysis(positions, field_bounds=field, margin_factor=1.0)
        assert default["n_interior"] == explicit["n_interior"]
        np.testing.assert_allclose(np.sort(default["areas"]), np.sort(explicit["areas"]))

    def test_larger_margin_factor_excludes_more_cells(self):
        rng = np.random.default_rng(12)
        positions = rng.uniform(0, 500, (80, 2))
        field = (0, 500, 0, 500)
        loose = voronoi_analysis(positions, field_bounds=field, margin_factor=1.0)
        strict = voronoi_analysis(positions, field_bounds=field, margin_factor=2.5)
        assert strict["n_interior"] < loose["n_interior"]

    def test_larger_margin_factor_caps_extreme_areas(self):
        # A boundary-adjacent cell in a sparse patch can have a Voronoi area
        # many times the median; a stricter margin should exclude it.
        rng = np.random.default_rng(0)
        positions = rng.uniform(0, 1000, (80, 2))
        field = (0, 1000, 0, 1000)
        loose = voronoi_analysis(positions, field_bounds=field, margin_factor=1.0)
        strict = voronoi_analysis(positions, field_bounds=field, margin_factor=2.5)
        assert strict["areas"].max() < loose["areas"].max()
