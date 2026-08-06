"""Tests for cell_mosaics.permutation."""

import numpy as np
import pytest

from cell_mosaics.permutation import cross_nnd_permutation_test, label_permutation_test


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

    def test_strata_restricts_permutation_to_blocks(self):
        # Two spatially separated blocks; each group lives entirely in one.
        # Under a stratified null the labels can never cross blocks, so every
        # permutation reproduces the observed grouping exactly.
        left = regular_grid(5, 10.0)
        right = regular_grid(5, 10.0) + np.array([500.0, 0.0])
        positions = np.vstack([left, right])
        labels = np.array(["L"] * len(left) + ["R"] * len(right))
        strata = np.array([0] * len(left) + [1] * len(right))

        result = label_permutation_test(
            positions, labels, groups=["L"], n_permutations=30, seed=0, strata=strata,
        )
        null = result["L"]["null"]
        np.testing.assert_allclose(null, result["L"]["observed"])

    def test_strata_length_validated(self):
        positions = regular_grid(4, 10.0)
        labels = np.array(["A"] * len(positions))
        with pytest.raises(ValueError):
            label_permutation_test(positions, labels, strata=np.zeros(3), n_permutations=5)

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


class TestCrossNndPermutationTest:
    def test_subset_of_one_mosaic_is_not_flagged_as_independent(self):
        # Take ONE regular mosaic and arbitrarily call a tenth of its cells
        # "focal". They are genuinely members of the parent mosaic, so the
        # observed mean nearest-parent distance should sit mid-null.
        rng = np.random.default_rng(0)
        mosaic = regular_grid(12, 20.0) + rng.normal(scale=1.0, size=(144, 2))

        labels = np.array(["parent"] * len(mosaic), dtype=object)
        labels[rng.choice(len(mosaic), size=14, replace=False)] = "focal"

        result = cross_nnd_permutation_test(
            mosaic, labels, focal_group="focal", parent_group="parent",
            n_permutations=500, seed=1,
        )
        assert result["p_closer"] > 0.05

    def test_independent_type_is_flagged_as_closer_than_chance(self):
        # A regular parent mosaic plus an *independent* sprinkling of focal
        # cells that ignore its exclusion zone. Those focal cells land much
        # closer to parent somas than a relabelled mosaic subset would.
        rng = np.random.default_rng(2)
        parent = regular_grid(12, 20.0) + rng.normal(scale=1.0, size=(144, 2))
        focal = rng.uniform(parent.min(), parent.max(), (14, 2))

        positions = np.vstack([parent, focal])
        labels = np.array(["parent"] * len(parent) + ["focal"] * len(focal), dtype=object)

        result = cross_nnd_permutation_test(
            positions, labels, focal_group="focal", parent_group="parent",
            n_permutations=500, seed=3,
        )
        assert result["p_closer"] < 0.05

    def test_interior_mask_limits_evaluated_focal_cells(self):
        rng = np.random.default_rng(4)
        parent = regular_grid(10, 20.0)
        focal = rng.uniform(0, 180, (20, 2))
        positions = np.vstack([parent, focal])
        labels = np.array(["parent"] * len(parent) + ["focal"] * len(focal), dtype=object)

        interior_mask = np.ones(len(positions), dtype=bool)
        interior_mask[len(parent) + 5:] = False  # keep only 5 focal cells

        result = cross_nnd_permutation_test(
            positions, labels, focal_group="focal", parent_group="parent",
            interior_mask=interior_mask, n_permutations=50, seed=5,
        )
        assert result["n_focal_evaluated"] == 5

    def test_missing_group_raises(self):
        positions = regular_grid(5, 10.0)
        labels = np.array(["parent"] * len(positions), dtype=object)

        with pytest.raises(ValueError):
            cross_nnd_permutation_test(positions, labels, focal_group="focal", parent_group="parent")

    def test_stratified_null_removes_density_inhomogeneity_artifact(self):
        # A densely sampled left half and a sparse right half. The focal cells
        # are a random subset of the SAME population (so they are genuine
        # "members"), but they all live in the sparse half. A global null draws
        # pseudo-focal cells mostly from the dense half, making the observed
        # distance look extreme for a reason that has nothing to do with
        # membership. Stratifying by half must fix that.
        rng = np.random.default_rng(0)
        dense = rng.uniform(0, 100, (300, 2))
        sparse = rng.uniform(0, 100, (30, 2)) + np.array([200.0, 0.0])

        positions = np.vstack([dense, sparse])
        labels = np.array(["parent"] * len(positions), dtype=object)
        focal_idx = len(dense) + rng.choice(len(sparse), size=12, replace=False)
        labels[focal_idx] = "focal"

        strata = np.where(positions[:, 0] < 150, 0, 1)

        global_null = cross_nnd_permutation_test(
            positions, labels, focal_group="focal", parent_group="parent",
            n_permutations=400, seed=1,
        )
        stratified = cross_nnd_permutation_test(
            positions, labels, focal_group="focal", parent_group="parent",
            n_permutations=400, seed=1, strata=strata,
        )

        # Global null: observed sits far out in the tail purely from density.
        assert global_null["p_farther"] < 0.01
        # Stratified null: the same genuine members now look unremarkable.
        assert stratified["p_farther"] > 0.05
        assert stratified["p_closer"] > 0.05

    def test_strata_length_validated(self):
        positions = regular_grid(6, 10.0)
        labels = np.array(["parent"] * (len(positions) - 6) + ["focal"] * 6, dtype=object)

        with pytest.raises(ValueError):
            cross_nnd_permutation_test(
                positions, labels, focal_group="focal", parent_group="parent",
                strata=np.zeros(3), n_permutations=10,
            )

    def test_p_values_are_complementary_directions(self):
        rng = np.random.default_rng(6)
        parent = regular_grid(8, 20.0)
        focal = rng.uniform(0, 140, (10, 2))
        positions = np.vstack([parent, focal])
        labels = np.array(["parent"] * len(parent) + ["focal"] * len(focal), dtype=object)

        result = cross_nnd_permutation_test(
            positions, labels, focal_group="focal", parent_group="parent",
            n_permutations=100, seed=7,
        )
        assert result["p_closer"] + result["p_farther"] >= 1.0  # overlap only at ties
