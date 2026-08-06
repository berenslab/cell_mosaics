"""Spatial statistics metrics for cell mosaics.

Includes nearest-neighbor distance (NND), density recovery profile (DRP),
and Voronoi domain analysis — standard measures of mosaic regularity.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree, Voronoi
from scipy.spatial.distance import cdist


def compute_centroids(cell_outlines: list[np.ndarray]) -> np.ndarray:
    """Compute centroids of cell polygon outlines.

    Parameters
    ----------
    cell_outlines : list of np.ndarray
        Each array has shape (N, 2) with XY coordinates.

    Returns
    -------
    np.ndarray of shape (n_cells, 2)
    """
    return np.array([outline.mean(axis=0) for outline in cell_outlines])


def nearest_neighbor_distances(positions: np.ndarray) -> np.ndarray:
    """Compute the nearest-neighbor distance (NND) for each cell.

    Parameters
    ----------
    positions : np.ndarray of shape (n_cells, 2)

    Returns
    -------
    np.ndarray of shape (n_cells,)
    """
    tree = KDTree(positions)
    distances, _ = tree.query(positions, k=2)  # k=1 is the point itself
    return distances[:, 1]


def nnd_statistics(positions: np.ndarray, interior_mask: np.ndarray | None = None) -> dict:
    """Compute NND summary statistics including the regularity index.

    The regularity index (RI) is defined as mean_NND / std_NND.
    Higher RI indicates a more regular (non-random) mosaic.
    A Poisson process typically gives RI ≈ 1.9; regular mosaics often exceed 4.

    Parameters
    ----------
    positions : np.ndarray of shape (n_cells, 2)
    interior_mask : np.ndarray of bool, shape (n_cells,), optional
        Restrict the summary statistics (mean/std/RI/cv, and `nnd` in the
        returned dict) to this subset of cells, while still computing every
        cell's nearest neighbor against the *full* `positions` array. This
        avoids underestimating NND for cells near a crop boundary whose true
        nearest neighbor lies just outside it -- compute over the full field,
        report only the interior. Defaults to using every position.

    Returns
    -------
    dict with keys
        nnd : np.ndarray — per-cell nearest-neighbor distances (interior only,
            if `interior_mask` is given)
        mean_nnd : float
        std_nnd : float
        regularity_index : float — mean / std
        cv : float — coefficient of variation (std / mean)
    """
    nnd = nearest_neighbor_distances(positions)
    if interior_mask is not None:
        nnd = nnd[interior_mask]
    mean = float(np.mean(nnd))
    std = float(np.std(nnd))
    return {
        "nnd": nnd,
        "mean_nnd": mean,
        "std_nnd": std,
        "regularity_index": mean / std if std > 0 else np.inf,
        "cv": std / mean if mean > 0 else np.inf,
    }


def _drp_bin_counts(
    positions: np.ndarray,
    bin_edges: np.ndarray,
    interior_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Average per-reference-cell neighbor count in each DRP annulus.

    Neighbor distances are always measured against the full `positions`
    array; `interior_mask` only restricts which cells act as reference
    points (the denominator/rows of the average), so a masked-out cell can
    still be counted as someone else's neighbor.
    """
    dists = cdist(positions, positions)
    np.fill_diagonal(dists, np.inf)

    if interior_mask is not None:
        dists = dists[np.asarray(interior_mask, dtype=bool)]
    n_ref = dists.shape[0]

    n_bins = len(bin_edges) - 1
    counts = np.zeros(n_bins)
    for j in range(n_bins):
        in_ring = (dists >= bin_edges[j]) & (dists < bin_edges[j + 1])
        counts[j] = in_ring.sum() / n_ref
    return counts


def density_recovery_profile(
    positions: np.ndarray,
    field_bounds: tuple[float, float, float, float],
    n_bins: int = 20,
    max_radius: float | None = None,
    interior_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Density Recovery Profile (DRP).

    The DRP measures local cell density as a function of distance from each
    cell, normalized to the global density. Values below 1 at small distances
    indicate an exclusion zone. Based on Rodieck (1991).

    Note: cells near the field edge underestimate neighbor counts at large
    radii, since their true neighbors may lie outside `field_bounds`. Pass
    `interior_mask` to exclude such cells from acting as reference points
    (see below); this does not by itself correct the more subtle bias from
    counting neighbors *of* an interior cell that happen to fall outside
    the field entirely (no edge-weighting correction is applied).

    Parameters
    ----------
    positions : np.ndarray of shape (n_cells, 2)
    field_bounds : (xmin, xmax, ymin, ymax)
    n_bins : int
        Number of distance bins. Default 20.
    max_radius : float or None
        Maximum distance to analyse. Defaults to one quarter of the shorter
        field dimension.
    interior_mask : np.ndarray of bool, shape (n_cells,), optional
        Restrict which cells act as reference points to this subset, while
        neighbor distances are still measured against the *full* `positions`
        array. This avoids the systematic under-count for reference cells
        near a crop boundary whose true nearest neighbors lie just outside
        it -- mirrors the `interior_mask` parameter of `nnd_statistics` and
        `voronoi_analysis`. Defaults to using every position as a reference
        point (the original, uncorrected behaviour).

    Returns
    -------
    bin_centers : np.ndarray of shape (n_bins,)
    drp : np.ndarray of shape (n_bins,) — normalized density (1 = random)
    """
    xmin, xmax, ymin, ymax = field_bounds
    field_area = (xmax - xmin) * (ymax - ymin)
    n_cells = len(positions)
    global_density = n_cells / field_area

    if max_radius is None:
        max_radius = min(xmax - xmin, ymax - ymin) / 4.0

    bin_edges = np.linspace(0.0, max_radius, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_areas = np.pi * (bin_edges[1:] ** 2 - bin_edges[:-1] ** 2)

    counts = _drp_bin_counts(positions, bin_edges, interior_mask)
    expected = global_density * bin_areas
    drp = counts / expected

    return bin_centers, drp


def drp_effective_radius(
    positions: np.ndarray,
    field_bounds: tuple[float, float, float, float],
    n_bins: int = 20,
    max_radius: float | None = None,
    interior_mask: np.ndarray | None = None,
) -> dict:
    """Quantify the DRP exclusion zone via Rodieck's (1991) effective radius.

    Companion statistic to `density_recovery_profile`: the effective radius
    r_e is the radius of a disc that, if completely empty out to r_e and at
    the mean field density beyond, would account for the same number of
    "missing" neighbors as the observed profile. It is computed from the
    cumulative (not per-annulus) observed and expected neighbor counts:

        N_obs(r)      = mean number of neighbors within radius r per
                        reference cell
        N_exp(r)      = mean_density * pi * r**2   (expected count under CSR)
        N_missing     = max_r [N_exp(r) - N_obs(r)]
        r_e           = sqrt(N_missing / (pi * mean_density))

    Taking the maximum of the deficit over r (rather than reading it off at
    a fixed radius) is robust to bin noise beyond the true exclusion zone,
    where the deficit should fluctuate around zero rather than keep growing.

    Note: because it is a maximum over a noisy sequence, this estimator has
    a systematic upward bias for small `n_cells` even under CSR (a handful
    of positive fluctuations in the deficit will be picked up as if they
    were a real exclusion zone). The bias shrinks as `n_cells` grows at
    fixed density; for fields with only a few dozen cells, treat
    `effective_radius` as an upper bound rather than a precise estimate.

    Parameters
    ----------
    positions, field_bounds, n_bins, max_radius, interior_mask
        As per `density_recovery_profile`.

    Returns
    -------
    dict with keys
        effective_radius : float — r_e, in the same units as `positions`
        missing_count : float — N_missing, the equivalent number of
            neighbors "missing" per reference cell due to the exclusion zone
        mean_density : float — the global field density used as the CSR
            baseline (the DRP's horizontal reference line)
        bin_centers : np.ndarray of shape (n_bins,)
        cumulative_observed : np.ndarray of shape (n_bins,) — N_obs(r) at
            each bin's outer edge
        cumulative_expected : np.ndarray of shape (n_bins,) — N_exp(r) at
            each bin's outer edge
    """
    xmin, xmax, ymin, ymax = field_bounds
    field_area = (xmax - xmin) * (ymax - ymin)
    n_cells = len(positions)
    mean_density = n_cells / field_area

    if max_radius is None:
        max_radius = min(xmax - xmin, ymax - ymin) / 4.0

    bin_edges = np.linspace(0.0, max_radius, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    counts = _drp_bin_counts(positions, bin_edges, interior_mask)
    cumulative_observed = np.cumsum(counts)
    cumulative_expected = mean_density * np.pi * bin_edges[1:] ** 2

    deficit = cumulative_expected - cumulative_observed
    missing_count = float(max(deficit.max(), 0.0))
    effective_radius = (
        float(np.sqrt(missing_count / (np.pi * mean_density)))
        if missing_count > 0
        else 0.0
    )

    return {
        "effective_radius": effective_radius,
        "missing_count": missing_count,
        "mean_density": float(mean_density),
        "bin_centers": bin_centers,
        "cumulative_observed": cumulative_observed,
        "cumulative_expected": cumulative_expected,
    }


def voronoi_analysis(
    positions: np.ndarray,
    field_bounds: tuple[float, float, float, float] | None = None,
    interior_mask: np.ndarray | None = None,
) -> dict:
    """Compute Voronoi tessellation and summarise domain areas.

    Cells with infinite Voronoi regions (on the convex hull) and, optionally,
    cells within one cell-length of the field boundary are excluded to reduce
    boundary artefacts.

    Parameters
    ----------
    positions : np.ndarray of shape (n_cells, 2)
    field_bounds : (xmin, xmax, ymin, ymax) or None
        If provided (and `interior_mask` is not), cells within one
        mean-cell-length of each boundary edge are excluded.
    interior_mask : np.ndarray of bool, shape (n_cells,), optional
        Explicit interior selection (e.g. a fixed stable-sampling crop box),
        used instead of the automatic `field_bounds` margin. The Voronoi
        tessellation itself still uses the *full* `positions` array, so
        interior cells get correct neighbors/areas from cells just outside
        the mask -- only the summary statistics are restricted. Cells with an
        infinite (unbounded) region are always excluded regardless.

    Returns
    -------
    dict with keys
        areas : np.ndarray — Voronoi domain areas for interior cells
        mean_area : float
        std_area : float
        cv_area : float — coefficient of variation (std / mean)
        regularity_index : float — mean / std
        n_interior : int — number of interior cells used
    """
    vor = Voronoi(positions)

    # Compute area for each point; NaN for infinite regions
    areas = np.full(len(positions), np.nan)
    for i, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if -1 not in region and len(region) > 0:
            verts = vor.vertices[region]
            areas[i] = _polygon_area(verts)

    if interior_mask is not None:
        resolved_interior_mask = np.asarray(interior_mask, dtype=bool) & ~np.isnan(areas)
    elif field_bounds is not None:
        xmin, xmax, ymin, ymax = field_bounds
        field_area = (xmax - xmin) * (ymax - ymin)
        margin = np.sqrt(field_area / len(positions))  # ≈ one cell length
        resolved_interior_mask = (
            ~np.isnan(areas)
            & (positions[:, 0] > xmin + margin)
            & (positions[:, 0] < xmax - margin)
            & (positions[:, 1] > ymin + margin)
            & (positions[:, 1] < ymax - margin)
        )
    else:
        resolved_interior_mask = ~np.isnan(areas)

    interior_areas = areas[resolved_interior_mask]
    n = len(interior_areas)
    mean = float(np.mean(interior_areas)) if n > 0 else np.nan
    std = float(np.std(interior_areas)) if n > 0 else np.nan

    return {
        "areas": interior_areas,
        "mean_area": mean,
        "std_area": std,
        "cv_area": std / mean if mean > 0 else np.nan,
        "regularity_index": mean / std if std > 0 else np.inf,
        "n_interior": int(np.sum(resolved_interior_mask)),
    }


def voronoi_domain_polygons(
    positions: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Finite Voronoi domain polygons for `positions`, for plotting.

    Companion to `voronoi_analysis`, which returns only summary statistics.
    Points whose Voronoi region is unbounded (those on the convex hull) have
    no finite polygon and are omitted, so use the returned `point_indices` to
    map polygons back onto `positions`.

    Parameters
    ----------
    positions : np.ndarray of shape (n_cells, 2)

    Returns
    -------
    polygons : list of np.ndarray
        Each an (m, 2) array of polygon vertices, ordered around the domain.
    areas : np.ndarray of shape (len(polygons),)
        Domain area for each returned polygon.
    point_indices : np.ndarray of shape (len(polygons),)
        Index into `positions` of the point each polygon belongs to.
    """
    vor = Voronoi(positions)

    polygons: list[np.ndarray] = []
    areas: list[float] = []
    point_indices: list[int] = []
    for i, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if len(region) > 0 and -1 not in region:
            verts = vor.vertices[region]
            polygons.append(verts)
            areas.append(_polygon_area(verts))
            point_indices.append(i)

    return polygons, np.asarray(areas), np.asarray(point_indices, dtype=int)


def _polygon_area(vertices: np.ndarray) -> float:
    """Signed area of a polygon via the shoelace formula."""
    x, y = vertices[:, 0], vertices[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
