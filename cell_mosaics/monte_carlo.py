"""Compare a real mosaic with matched random simulations (Monte Carlo testing).

Implements the analysis advocated by Keeley, Eglen & Reese (2020), "From
random to regular: Variation in the patterning of retinal mosaics":

1. Simulate random fields matched to the real one in cell number and field
   geometry, and constrained by soma size (`simulation.simulate_random_mosaic`).
2. Compute the same trio of spatial statistics for the real and every
   simulated field -- the nearest-neighbor regularity index (NNRI), the
   Voronoi domain regularity index (VDRI), and the DRP effective radius with
   its packing factor -- excluding border cells identically in each.
3. Rank the real field among the simulations (99 by default, so that a real
   field beyond all of them has p = 0.01), and express each statistic as a
   ratio to its random expectation (the "regularity ratio", Keeley & Reese
   2014), which removes the regularity that soma size alone imposes at a
   given density.

`envelope_test` adds the cumulative G and L function test of Raven et al.
(2003), suited to low-density mosaics and to testing whether a d_min spacing
rule is sufficient to explain the observed patterning.
"""

from __future__ import annotations

import numpy as np

from .metrics import (
    drp_effective_radius,
    g_function,
    l_function,
    nnd_border_mask,
    nnd_statistics,
    packing_factor,
    voronoi_analysis,
    voronoi_border_mask,
)
from .simulation import simulate_random_mosaic

REGULARITY_STATISTICS = (
    "nnd_regularity_index",
    "voronoi_regularity_index",
    "effective_radius",
    "packing_factor",
)

DRP_DEFAULTS = {"method": "rodieck", "edge_correction": True}


def regularity_statistics(
    positions: np.ndarray,
    field_bounds: tuple[float, float, float, float],
    drp_kwargs: dict | None = None,
) -> dict:
    """The NNRI, VDRI, effective radius and packing factor of one field.

    Border cells are excluded by the rules of Keeley et al. (2020): from the
    NNRI, cells closer to the border than to their nearest neighbor
    (`metrics.nnd_border_mask`); from the VDRI, cells whose Voronoi domain is
    not fully inside the field (`metrics.voronoi_border_mask`).

    Parameters
    ----------
    positions : np.ndarray of shape (n_cells, 2)
    field_bounds : (xmin, xmax, ymin, ymax)
    drp_kwargs : dict, optional
        Forwarded to `metrics.drp_effective_radius` (e.g. `n_bins`,
        `max_radius`), on top of the defaults `DRP_DEFAULTS`: Rodieck's
        effective radius with translation edge correction, so that a random
        simulation's effective radius approximates its soma size, as in the
        paper. Fix `n_bins` and `max_radius` explicitly when comparing
        fields, since the default `max_radius` scales with the field size.

    Returns
    -------
    dict with keys
        nnd_regularity_index, voronoi_regularity_index, effective_radius,
        packing_factor : float — NaN where too few interior cells remain
        n_cells, n_nnd_interior, n_voronoi_interior : int
        density : float — cells per unit area
    """
    positions = np.asarray(positions, dtype=float)
    xmin, xmax, ymin, ymax = field_bounds
    n_cells = len(positions)
    density = n_cells / ((xmax - xmin) * (ymax - ymin))

    nnd_mask = nnd_border_mask(positions, field_bounds)
    nnri = (nnd_statistics(positions, interior_mask=nnd_mask)["regularity_index"]
            if nnd_mask.sum() >= 3 else np.nan)

    vor_mask = voronoi_border_mask(positions, field_bounds)
    vdri = (voronoi_analysis(positions, interior_mask=vor_mask)["regularity_index"]
            if vor_mask.sum() >= 3 else np.nan)

    er = drp_effective_radius(
        positions, field_bounds, **{**DRP_DEFAULTS, **(drp_kwargs or {})})["effective_radius"]

    return {
        "nnd_regularity_index": float(nnri),
        "voronoi_regularity_index": float(vdri),
        "effective_radius": float(er),
        "packing_factor": packing_factor(er, density),
        "n_cells": n_cells,
        "n_nnd_interior": int(nnd_mask.sum()),
        "n_voronoi_interior": int(vor_mask.sum()),
        "density": float(density),
    }


def _rank_p_values(observed: float, simulated: np.ndarray) -> tuple[float, float]:
    """One-sided Monte Carlo p-values, counting the observed field as one draw."""
    valid = simulated[~np.isnan(simulated)]
    if np.isnan(observed) or len(valid) == 0:
        return np.nan, np.nan
    n = len(valid)
    p_greater = (1 + np.sum(valid >= observed)) / (n + 1)
    p_less = (1 + np.sum(valid <= observed)) / (n + 1)
    return float(p_greater), float(p_less)


def compare_to_random_simulations(
    positions: np.ndarray,
    field_bounds: tuple[float, float, float, float],
    *,
    soma_diameter_mean: float = 0.0,
    soma_diameter_sd: float = 0.0,
    dmin_mean: float = 0.0,
    dmin_sd: float = 0.0,
    n_simulations: int = 99,
    drp_kwargs: dict | None = None,
    seed: int | None = None,
    return_simulations: bool = False,
) -> dict:
    """Rank a real mosaic's regularity statistics among matched random simulations.

    Each simulation has the same number of cells and field as the real one,
    placed by `simulation.simulate_random_mosaic` with the given soma-size
    and/or d_min constraint (Keeley et al. 2020, Figs. 3, 8, 11). Leaving
    every constraint at 0 compares against CSR, which the paper argues is
    "largely pointless": always pass at least the real soma size.

    Parameters
    ----------
    positions : np.ndarray of shape (n_cells, 2)
        The real mosaic.
    field_bounds : (xmin, xmax, ymin, ymax)
        The real sampled field.
    soma_diameter_mean, soma_diameter_sd, dmin_mean, dmin_sd : float
        Forwarded to `simulation.simulate_random_mosaic`.
    n_simulations : int, optional
        Default 99, the convention in the paper.
    drp_kwargs : dict, optional
        Forwarded to `metrics.drp_effective_radius` for the real and every
        simulated field alike.
    seed : int, optional
    return_simulations : bool, optional
        Also return the simulated positions and diameters. Default False.

    Returns
    -------
    dict with keys
        observed : dict — `regularity_statistics` of the real field
        <statistic> : dict, for each of `REGULARITY_STATISTICS`, with
            observed : float
            simulated : np.ndarray of shape (n_simulations,)
            simulated_mean, simulated_sd : float
            ratio : float — observed / simulated_mean; for the regularity
                indices this is the regularity ratio (Keeley & Reese 2014)
            p_greater : float — Monte Carlo p-value that the real field is
                larger than random, (1 + #{sim >= obs}) / (n + 1)
            p_less : float — likewise for smaller than random (e.g. AII
                amacrine cells packed closer than their mean soma size)
        simulations : list of (positions, diameters), if `return_simulations`
    """
    positions = np.asarray(positions, dtype=float)
    rng = np.random.default_rng(seed)
    observed = regularity_statistics(positions, field_bounds, drp_kwargs)

    simulated = {key: np.empty(n_simulations) for key in REGULARITY_STATISTICS}
    simulations = []
    for s in range(n_simulations):
        sim_positions, sim_diameters = simulate_random_mosaic(
            len(positions), field_bounds,
            soma_diameter_mean=soma_diameter_mean, soma_diameter_sd=soma_diameter_sd,
            dmin_mean=dmin_mean, dmin_sd=dmin_sd, seed=rng,
        )
        stats = regularity_statistics(sim_positions, field_bounds, drp_kwargs)
        for key in REGULARITY_STATISTICS:
            simulated[key][s] = stats[key]
        if return_simulations:
            simulations.append((sim_positions, sim_diameters))

    result: dict = {"observed": observed}
    for key in REGULARITY_STATISTICS:
        sims = simulated[key]
        sim_mean = float(np.nanmean(sims)) if np.any(~np.isnan(sims)) else np.nan
        p_greater, p_less = _rank_p_values(observed[key], sims)
        result[key] = {
            "observed": observed[key],
            "simulated": sims,
            "simulated_mean": sim_mean,
            "simulated_sd": float(np.nanstd(sims)) if np.any(~np.isnan(sims)) else np.nan,
            "ratio": observed[key] / sim_mean if sim_mean else np.nan,
            "p_greater": p_greater,
            "p_less": p_less,
        }
    if return_simulations:
        result["simulations"] = simulations
    return result


def _evaluate_function(function, positions, field_bounds, r):
    if function == "G":
        return g_function(positions, r, interior_mask=nnd_border_mask(positions, field_bounds))
    if function == "L":
        return l_function(positions, field_bounds, r)
    raise ValueError(f"Unknown function {function!r}; choose 'G' or 'L'")


def envelope_test(
    positions: np.ndarray,
    field_bounds: tuple[float, float, float, float],
    function: str = "G",
    r: np.ndarray | None = None,
    *,
    soma_diameter_mean: float = 0.0,
    soma_diameter_sd: float = 0.0,
    dmin_mean: float = 0.0,
    dmin_sd: float = 0.0,
    n_simulations: int = 99,
    seed: int | None = None,
) -> dict:
    """Monte Carlo goodness-of-fit of a spacing model via the G or L function.

    Following Raven et al. (2003) and Keeley et al. (2020, Fig. 5), the real
    field's cumulative function is compared with those of `n_simulations`
    fields from the model (soma size and/or d_min rule). The min/max
    envelope is an informal guide; the formal test is Diggle's (2002)
    ranking: for every pattern i (real and simulated),

        u_i = sum_r (f_i(r) - mean_{j != i} f_j(r))**2 * dr

    measures its departure from the average of the others, and the real
    field's rank among all u gives the p-value. A small p (e.g. < 0.05,
    i.e. the real field ranks among the 5 most deviant of 100) rejects the
    model; otherwise the model is an acceptable fit.

    Parameters
    ----------
    positions : np.ndarray of shape (n_cells, 2)
    field_bounds : (xmin, xmax, ymin, ymax)
    function : {'G', 'L'}, optional
        'G' -- cumulative NND distribution (`metrics.g_function`, border
        cells excluded by `metrics.nnd_border_mask`). 'L' -- scaled
        cumulative DRP (`metrics.l_function`). Default 'G'.
    r : np.ndarray, optional
        Evenly spaced distances. Defaults to 101 points from 0 to a quarter
        of the shorter field side.
    soma_diameter_mean, soma_diameter_sd, dmin_mean, dmin_sd : float
        The model, forwarded to `simulation.simulate_random_mosaic`.
    n_simulations : int, optional
        Default 99.
    seed : int, optional

    Returns
    -------
    dict with keys
        r : np.ndarray
        observed : np.ndarray — f(r) for the real field
        simulated : np.ndarray of shape (n_simulations, len(r))
        mean, lower, upper : np.ndarray — mean and min/max envelope of the
            simulations
        u_observed : float
        u_simulated : np.ndarray of shape (n_simulations,)
        p_value : float — (1 + #{u_sim >= u_obs}) / (n_simulations + 1)
    """
    positions = np.asarray(positions, dtype=float)
    xmin, xmax, ymin, ymax = field_bounds
    if r is None:
        r = np.linspace(0.0, min(xmax - xmin, ymax - ymin) / 4.0, 101)
    r = np.asarray(r, dtype=float)
    dr = float(r[1] - r[0]) if len(r) > 1 else 1.0

    rng = np.random.default_rng(seed)
    observed = _evaluate_function(function, positions, field_bounds, r)
    simulated = np.empty((n_simulations, len(r)))
    for s in range(n_simulations):
        sim_positions, _ = simulate_random_mosaic(
            len(positions), field_bounds,
            soma_diameter_mean=soma_diameter_mean, soma_diameter_sd=soma_diameter_sd,
            dmin_mean=dmin_mean, dmin_sd=dmin_sd, seed=rng,
        )
        simulated[s] = _evaluate_function(function, sim_positions, field_bounds, r)

    curves = np.vstack([observed, simulated])
    n_curves = len(curves)
    mean_of_others = (curves.sum(axis=0) - curves) / (n_curves - 1)
    u = np.nansum((curves - mean_of_others) ** 2, axis=1) * dr
    u_observed, u_simulated = float(u[0]), u[1:]

    return {
        "r": r,
        "observed": observed,
        "simulated": simulated,
        "mean": simulated.mean(axis=0),
        "lower": simulated.min(axis=0),
        "upper": simulated.max(axis=0),
        "u_observed": u_observed,
        "u_simulated": u_simulated,
        "p_value": float((1 + np.sum(u_simulated >= u_observed)) / (n_simulations + 1)),
    }
