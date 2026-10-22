"""Tests for cell_mosaics.monte_carlo."""

import numpy as np
import pytest

from cell_mosaics.monte_carlo import (
    REGULARITY_STATISTICS,
    compare_to_random_simulations,
    envelope_test,
    regularity_statistics,
)
from cell_mosaics.simulation import simulate_random_mosaic

BOUNDS = (0.0, 300.0, 0.0, 300.0)
DRP_KWARGS = {"n_bins": 20, "max_radius": 60.0}
SOMA = {"soma_diameter_mean": 10.0, "soma_diameter_sd": 1.0}


@pytest.fixture(scope="module")
def regular_mosaic():
    # Horizontal-cell-like: a spacing rule well beyond soma size.
    positions, _ = simulate_random_mosaic(130, BOUNDS, dmin_mean=20.0, dmin_sd=2.0, seed=1, **SOMA)
    return positions


@pytest.fixture(scope="module")
def soma_only_mosaic():
    # AII-amacrine-like: spaced only by soma size.
    positions, _ = simulate_random_mosaic(130, BOUNDS, seed=2, **SOMA)
    return positions


class TestRegularityStatistics:
    def test_keys_and_counts(self, soma_only_mosaic):
        stats = regularity_statistics(soma_only_mosaic, BOUNDS, DRP_KWARGS)
        for key in REGULARITY_STATISTICS:
            assert np.isfinite(stats[key])
        assert stats["n_cells"] == 130
        assert 0 < stats["n_voronoi_interior"] < stats["n_nnd_interior"] <= 130
        assert stats["density"] == pytest.approx(130 / 300.0 ** 2)

    def test_random_effective_radius_approximates_soma_size(self):
        # Keeley et al. (2020): for soma-constrained random simulations the
        # effective radius "returns an estimate close to the soma size".
        ers = [
            regularity_statistics(
                simulate_random_mosaic(312, (0, 250, 0, 250), seed=s, soma_diameter_mean=10.0,
                                       soma_diameter_sd=0.5)[0],
                (0, 250, 0, 250), {"n_bins": 25, "max_radius": 50.0},
            )["effective_radius"]
            for s in range(10)
        ]
        assert np.median(ers) == pytest.approx(10.0, abs=1.5)


class TestCompareToRandomSimulations:
    def test_regular_mosaic_exceeds_every_simulation(self, regular_mosaic):
        result = compare_to_random_simulations(
            regular_mosaic, BOUNDS, n_simulations=49, drp_kwargs=DRP_KWARGS, seed=0, **SOMA)
        for key in REGULARITY_STATISTICS:
            assert result[key]["p_greater"] == pytest.approx(1 / 50)
            assert result[key]["ratio"] > 1.3

    def test_soma_only_mosaic_is_unremarkable(self, soma_only_mosaic):
        result = compare_to_random_simulations(
            soma_only_mosaic, BOUNDS, n_simulations=49, drp_kwargs=DRP_KWARGS, seed=0, **SOMA)
        for key in ("nnd_regularity_index", "voronoi_regularity_index"):
            assert result[key]["p_greater"] > 0.05
            assert result[key]["ratio"] == pytest.approx(1.0, abs=0.25)

    def test_output_shapes_and_simulations(self, soma_only_mosaic):
        result = compare_to_random_simulations(
            soma_only_mosaic, BOUNDS, n_simulations=5, drp_kwargs=DRP_KWARGS, seed=0,
            return_simulations=True, **SOMA)
        for key in REGULARITY_STATISTICS:
            assert result[key]["simulated"].shape == (5,)
        assert len(result["simulations"]) == 5
        sim_positions, sim_diameters = result["simulations"][0]
        assert sim_positions.shape == soma_only_mosaic.shape
        assert sim_diameters.shape == (len(soma_only_mosaic),)

    def test_seed_reproducible(self, soma_only_mosaic):
        a = compare_to_random_simulations(soma_only_mosaic, BOUNDS, n_simulations=3, seed=4, **SOMA)
        b = compare_to_random_simulations(soma_only_mosaic, BOUNDS, n_simulations=3, seed=4, **SOMA)
        np.testing.assert_array_equal(a["voronoi_regularity_index"]["simulated"],
                                      b["voronoi_regularity_index"]["simulated"])


class TestEnvelopeTest:
    @pytest.mark.parametrize("function", ["G", "L"])
    def test_soma_model_rejected_for_regular_mosaic(self, regular_mosaic, function):
        result = envelope_test(regular_mosaic, BOUNDS, function, n_simulations=49, seed=0, **SOMA)
        assert result["p_value"] == pytest.approx(1 / 50)

    @pytest.mark.parametrize("function", ["G", "L"])
    def test_true_model_fits(self, regular_mosaic, function):
        result = envelope_test(regular_mosaic, BOUNDS, function, n_simulations=49, seed=0,
                               dmin_mean=20.0, dmin_sd=2.0, **SOMA)
        assert result["p_value"] > 0.05

    def test_output_shapes(self, soma_only_mosaic):
        r = np.linspace(0, 50, 26)
        result = envelope_test(soma_only_mosaic, BOUNDS, "L", r, n_simulations=7, seed=0, **SOMA)
        assert result["simulated"].shape == (7, 26)
        for key in ("observed", "mean", "lower", "upper"):
            assert result[key].shape == (26,)
        assert np.all(result["lower"] <= result["mean"])
        assert np.all(result["mean"] <= result["upper"])
        assert result["u_simulated"].shape == (7,)

    def test_unknown_function_raises(self, soma_only_mosaic):
        with pytest.raises(ValueError):
            envelope_test(soma_only_mosaic, BOUNDS, "K", n_simulations=2)
