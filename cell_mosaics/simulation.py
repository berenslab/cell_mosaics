"""Random simulations of retinal mosaics, for comparison with real fields.

Keeley, Eglen & Reese (2020) argue that a real mosaic should be compared not
with complete spatial randomness (CSR) of dimensionless points, but with
random distributions of *cells*: matched in number and field geometry, and
constrained by soma size so that no two somata overlap. Soma size alone
raises the regularity index well above the CSR value of ~1.9, increasingly
so at high density, so CSR sets a very low bar for calling a mosaic
"regular". A minimal-distance (d_min) spacing rule larger than soma size
(Galli-Resta et al. 1997) then tests whether a local exclusion rule is
sufficient to reproduce the observed patterning.
"""

from __future__ import annotations

import numpy as np

# Upper bound on candidate-to-cell distances evaluated per batch (~16 MB of float64).
_MAX_DISTANCE_ENTRIES = 2_000_000


def _positive_normal(rng: np.random.Generator, mean: float, sd: float, size: int) -> np.ndarray:
    """Draw from N(mean, sd), redrawing any non-positive values."""
    if sd <= 0:
        return np.full(size, float(mean))
    values = rng.normal(mean, sd, size)
    bad = values <= 0
    while bad.any():
        values[bad] = rng.normal(mean, sd, int(bad.sum()))
        bad = values <= 0
    return values


def simulate_random_mosaic(
    n_cells: int,
    field_bounds: tuple[float, float, float, float],
    *,
    soma_diameter_mean: float = 0.0,
    soma_diameter_sd: float = 0.0,
    dmin_mean: float = 0.0,
    dmin_sd: float = 0.0,
    seed: int | np.random.Generator | None = None,
    max_attempts_per_cell: int = 100_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly insert cells into a field, rejecting any that break a spacing rule.

    Cells are inserted one at a time at a uniformly random position. A
    candidate is rejected, and replaced by a new random candidate, if it
    lies closer to any previously placed cell than allowed by either rule:

    - **Soma size** (Keeley et al. 2020, Fig. 2): each cell has a diameter
      drawn from N(`soma_diameter_mean`, `soma_diameter_sd`), and two cells
      may not overlap, i.e. dist(i, j) >= (d_i + d_j) / 2. Use the mean and
      SD of the real population's equivalent-circle diameters,
      2 * sqrt(soma_area / pi).
    - **d_min spacing rule** (Galli-Resta et al. 1997; Raven et al. 2003):
      each candidate draws an exclusion distance from N(`dmin_mean`,
      `dmin_sd`) and must be at least that far from every placed cell.

    With both means at 0 this is complete spatial randomness (CSR). Both
    distributions are redrawn until positive. Diameters are assigned to
    cells up front and kept on rejection (only the position is redrawn), so
    the simulated sizes follow the requested distribution exactly; the d_min
    is redrawn with each candidate, as in the original d_min model.

    Parameters
    ----------
    n_cells : int
        Number of cells to place, typically matched to the real field.
    field_bounds : (xmin, xmax, ymin, ymax)
        Field in which cell centres are placed, matched to the real field.
    soma_diameter_mean, soma_diameter_sd : float, optional
        Soma-size constraint. Default 0 (none).
    dmin_mean, dmin_sd : float, optional
        Minimal-distance spacing rule. Default 0 (none).
    seed : int, np.random.Generator or None, optional
    max_attempts_per_cell : int, optional
        Give up if this many consecutive candidates are rejected for one
        cell. This happens when the field is too crowded for the requested
        rule: random sequential insertion jams at a packing intensity of
        about 0.55 (see `metrics.packing_intensity`). Default 100,000.

    Returns
    -------
    positions : np.ndarray of shape (n_cells, 2)
    diameters : np.ndarray of shape (n_cells,)
        Soma diameter per cell (all 0 if no soma-size constraint).

    Raises
    ------
    RuntimeError
        If a cell cannot be placed within `max_attempts_per_cell` candidates.
    """
    rng = np.random.default_rng(seed)
    xmin, xmax, ymin, ymax = field_bounds
    diameters = (_positive_normal(rng, soma_diameter_mean, soma_diameter_sd, n_cells)
                 if soma_diameter_mean > 0 else np.zeros(n_cells))
    use_dmin = dmin_mean > 0

    positions = np.empty((n_cells, 2))
    if n_cells == 0 or (not use_dmin and soma_diameter_mean <= 0):
        positions[:, 0] = rng.uniform(xmin, xmax, n_cells)
        positions[:, 1] = rng.uniform(ymin, ymax, n_cells)
        return positions, diameters

    positions[0] = rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)

    # Candidates are drawn in batches and the first valid one is accepted.
    # That is the same process as drawing them one at a time (the first
    # success in an i.i.d. sequence), just vectorised. The batch grows while
    # a cell keeps failing, so a crowded field doesn't cost a numpy call per
    # candidate, but is capped to keep the (batch x i) distance matrix small.
    for i in range(1, n_cells):
        batch_cap = max(16, _MAX_DISTANCE_ENTRIES // i)
        batch = 16
        attempts = 0
        while True:
            if attempts >= max_attempts_per_cell:
                raise RuntimeError(
                    f"Could not place cell {i + 1} of {n_cells} after {attempts} "
                    "attempts; the spacing rule is too large for this density "
                    "(random insertion jams at a packing intensity of ~0.55)."
                )
            batch = min(batch, batch_cap, max_attempts_per_cell - attempts)
            candidates = np.column_stack([rng.uniform(xmin, xmax, batch),
                                          rng.uniform(ymin, ymax, batch)])
            dist = np.linalg.norm(candidates[:, None, :] - positions[None, :i, :], axis=2)
            threshold = np.broadcast_to(0.5 * (diameters[i] + diameters[:i]), dist.shape)
            if use_dmin:
                dmin = _positive_normal(rng, dmin_mean, dmin_sd, batch)
                threshold = np.maximum(threshold, dmin[:, None])
            hit = np.flatnonzero(np.all(dist >= threshold, axis=1))
            if hit.size:
                positions[i] = candidates[hit[0]]
                break
            attempts += batch
            batch *= 2

    return positions, diameters
