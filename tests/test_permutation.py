"""Tests for cell_mosaics.permutation."""

import numpy as np

from cell_mosaics.permutation import label_permutation_test


def regular_grid(n: int, spacing: float) -> np.ndarray:
    """Return a regular n×n grid of 2D positions."""
    x, y = np.meshgrid(np.arange(n) * spacing, np.arange(n) * spacing)
    return np.column_stack([x.ravel(), y.ravel()])


class TestLabelPermutationTest:
    def test_two_interleaved_regular_mosaics_score_significant(self):
        # Two independent regular grids, offset so they interleave -- a
        # textbook "two independent mosaics" pattern. Each type's own
        # regularity index should stand out against a random-subset null.
        rng = np.random.default_rng(0)
        grid_a = regular_grid(6, 20.0) + rng.normal(scale=0.5, size=(36, 2))
        grid_b = regular_grid(6, 20.0) + np.array([10.0, 10.0]) + rng.normal(scale=0.5, size=(36, 2))

        positions = np.vstack([grid_a, grid_b])
        labels = np.array(["A"] * len(grid_a) + ["B"] * len(grid_b))

        result = label_permutation_test(positions, labels, n_permutations=500, seed=1)

        assert set(result) == {"A", "B"}
        for group in ("A", "B"):
            assert result[group]["observed"] > 5  # clearly regular
            assert result[group]["p_value"] < 0.05

    def test_random_labels_on_single_poisson_process_not_significant(self):
        # A single random point process arbitrarily split into two labels:
        # neither "group" is a real distinct mosaic, so p-values should be
        # unremarkable (not reliably small).
        rng = np.random.default_rng(2)
        positions = rng.uniform(0, 200, (150, 2))
        labels = rng.choice(["X", "Y"], size=150)

        result = label_permutation_test(positions, labels, n_permutations=500, seed=3)

        for group in ("X", "Y"):
            assert result[group]["p_value"] > 0.05

    def test_groups_param_restricts_output(self):
        positions = regular_grid(6, 10.0)
        labels = np.tile(["A", "B"], 18)

        result = label_permutation_test(positions, labels, groups=["A"], n_permutations=50, seed=4)

        assert set(result) == {"A"}

    def test_interior_mask_used_for_summary_not_neighbor_search(self):
        # A regular grid with one far-flung outlier point. Without an
        # interior mask, the grid point nearest the outlier gets a huge NND
        # and the regularity index collapses. Masking that boundary point out
        # of the *summary* (while still letting it act as a neighbor) should
        # restore a high regularity index.
        grid = regular_grid(5, 10.0)
        outlier = np.array([[1000.0, 1000.0]])
        positions = np.vstack([grid, outlier])
        labels = np.array(["A"] * len(grid) + ["B"])

        interior_mask = np.array([True] * len(grid) + [False])

        result = label_permutation_test(
            positions, labels, groups=["A"], interior_mask=interior_mask,
            n_permutations=50, seed=5,
        )
        assert result["A"]["observed"] > 5

    def test_cv_area_statistic_lower_for_regular_group(self):
        rng = np.random.default_rng(6)
        regular = regular_grid(6, 20.0)
        random_pts = rng.uniform(0, 100, (36, 2))

        positions = np.vstack([regular, random_pts])
        labels = np.array(["regular"] * len(regular) + ["random"] * len(random_pts))
        field_bounds = (positions[:, 0].min(), positions[:, 0].max(),
                        positions[:, 1].min(), positions[:, 1].max())

        result = label_permutation_test(
            positions, labels, statistic="cv_area", field_bounds=field_bounds,
            n_permutations=200, seed=7,
        )
        assert result["regular"]["p_value"] < 0.1

    def test_unknown_statistic_raises(self):
        positions = regular_grid(4, 10.0)
        labels = np.array(["A"] * len(positions))
        try:
            label_permutation_test(positions, labels, statistic="not_a_stat")
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_small_group_returns_nan(self):
        positions = regular_grid(4, 10.0)
        labels = np.array(["A"] * (len(positions) - 1) + ["B"])

        result = label_permutation_test(positions, labels, groups=["B"], n_permutations=20, seed=8)
        assert np.isnan(result["B"]["observed"])
        assert np.isnan(result["B"]["p_value"])
