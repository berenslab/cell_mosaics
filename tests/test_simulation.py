"""Tests for cell_mosaics.simulation."""

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

from cell_mosaics.metrics import nnd_border_mask, nnd_statistics
from cell_mosaics.simulation import simulate_random_mosaic

BOUNDS = (0.0, 250.0, 0.0, 250.0)


class TestSimulateRandomMosaic:
    def test_count_and_bounds(self):
        positions, diameters = simulate_random_mosaic(
            100, BOUNDS, soma_diameter_mean=10.0, soma_diameter_sd=1.0, seed=0)
        assert positions.shape == (100, 2)
        assert diameters.shape == (100,)
        assert np.all((positions >= 0) & (positions <= 250))

    def test_somata_never_overlap(self):
        positions, diameters = simulate_random_mosaic(
            200, BOUNDS, soma_diameter_mean=10.0, soma_diameter_sd=1.0, seed=1)
        dist = squareform(pdist(positions))
        min_allowed = 0.5 * (diameters[:, None] + diameters[None, :])
        off_diagonal = ~np.eye(len(positions), dtype=bool)
        assert np.all(dist[off_diagonal] >= min_allowed[off_diagonal])

    def test_diameters_follow_requested_distribution(self):
        _, diameters = simulate_random_mosaic(
            300, BOUNDS, soma_diameter_mean=8.0, soma_diameter_sd=0.5, seed=2)
        assert diameters.mean() == pytest.approx(8.0, abs=0.1)
        assert diameters.std() == pytest.approx(0.5, abs=0.1)
        assert np.all(diameters > 0)

    def test_fixed_dmin_is_respected(self):
        positions, diameters = simulate_random_mosaic(80, BOUNDS, dmin_mean=15.0, seed=3)
        assert pdist(positions).min() >= 15.0
        np.testing.assert_array_equal(diameters, 0.0)

    def test_no_constraint_is_csr(self):
        # Mean NN regularity index of a Poisson pattern is ~1.91 (Cook 1996).
        ris = []
        for seed in range(100):
            positions, _ = simulate_random_mosaic(150, (0, 300, 0, 300), seed=seed)
            mask = nnd_border_mask(positions, (0, 300, 0, 300))
            ris.append(nnd_statistics(positions, interior_mask=mask)["regularity_index"])
        assert np.mean(ris) == pytest.approx(1.91, abs=0.1)

    def test_soma_size_raises_regularity_with_density(self):
        # Keeley et al. (2020, Fig. 2g): 10 um somata in a 250 x 250 um field
        # give an NNRI of ~3 at 1,000 cells/mm2, rising to ~9 at 5,000.
        def median_ri(n_cells):
            ris = []
            for seed in range(10):
                positions, _ = simulate_random_mosaic(
                    n_cells, BOUNDS, soma_diameter_mean=10.0, soma_diameter_sd=0.5, seed=seed)
                mask = nnd_border_mask(positions, BOUNDS)
                ris.append(nnd_statistics(positions, interior_mask=mask)["regularity_index"])
            return np.median(ris)

        low, high = median_ri(62), median_ri(312)
        assert 2.5 < low < 4.0
        assert 7.0 < high < 11.0

    def test_seed_reproducible(self):
        a, _ = simulate_random_mosaic(50, BOUNDS, soma_diameter_mean=10.0, seed=7)
        b, _ = simulate_random_mosaic(50, BOUNDS, soma_diameter_mean=10.0, seed=7)
        np.testing.assert_array_equal(a, b)

    def test_impossible_density_raises(self):
        # 100 disks of diameter 30 in a 100 x 100 field: packing intensity ~7.
        with pytest.raises(RuntimeError):
            simulate_random_mosaic(100, (0, 100, 0, 100), soma_diameter_mean=30.0,
                                   seed=0, max_attempts_per_cell=2000)

    def test_zero_cells(self):
        positions, diameters = simulate_random_mosaic(0, BOUNDS, soma_diameter_mean=10.0)
        assert positions.shape == (0, 2)
        assert diameters.shape == (0,)
