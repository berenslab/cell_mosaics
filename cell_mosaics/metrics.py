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


def nnd_statistics(positions: np.ndarray) -> dict:
    """Compute NND summary statistics including the regularity index.

    The regularity index (RI) is defined as mean_NND / std_NND.
    Higher RI indicates a more regular (non-random) mosaic.
    A Poisson process typically gives RI ≈ 1.9; regular mosaics often exceed 4.

    Parameters
    ----------
    positions : np.ndarray of shape (n_cells, 2)

    Returns
    -------
    dict with keys
        nnd : np.ndarray — per-cell nearest-neighbor distances
        mean_nnd : float
        std_nnd : float
        regularity_index : float — mean / std
        cv : float — coefficient of variation (std / mean)
    """
    nnd = nearest_neighbor_distances(positions)
    mean = float(np.mean(nnd))
    std = float(np.std(nnd))
    return {
        "nnd": nnd,
        "mean_nnd": mean,
        "std_nnd": std,
        "regularity_index": mean / std if std > 0 else np.inf,
        "cv": std / mean if mean > 0 else np.inf,
    }


def density_recovery_profile(
    positions: np.ndarray,
    field_bounds: tuple[float, float, float, float],
    n_bins: int = 20,
    max_radius: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Density Recovery Profile (DRP).

    The DRP measures local cell density as a function of distance from each
    cell, normalized to the global density. Values below 1 at small distances
    indicate an exclusion zone. Based on Rodieck (1991).

    Note: boundary effects are not corrected; cells near the field edge will
    underestimate neighbor counts at large radii.

    Parameters
    ----------
    positions : np.ndarray of shape (n_cells, 2)
    field_bounds : (xmin, xmax, ymin, ymax)
    n_bins : int
        Number of distance bins. Default 20.
    max_radius : float or None
        Maximum distance to analyse. Defaults to one quarter of the shorter
        field dimension.

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

    # All pairwise distances; self-distance set to inf to exclude self-pairs
    dists = cdist(positions, positions)
    np.fill_diagonal(dists, np.inf)

    # Average neighbor count per cell in each ring
    counts = np.zeros(n_bins)
    for j in range(n_bins):
        in_ring = (dists >= bin_edges[j]) & (dists < bin_edges[j + 1])
        counts[j] = in_ring.sum() / n_cells

    expected = global_density * bin_areas
    drp = counts / expected

    return bin_centers, drp


def voronoi_analysis(
    positions: np.ndarray,
    field_bounds: tuple[float, float, float, float] | None = None,
) -> dict:
    """Compute Voronoi tessellation and summarise domain areas.

    Cells with infinite Voronoi regions (on the convex hull) and, optionally,
    cells within one cell-length of the field boundary are excluded to reduce
    boundary artefacts.

    Parameters
    ----------
    positions : np.ndarray of shape (n_cells, 2)
    field_bounds : (xmin, xmax, ymin, ymax) or None
        If provided, cells within one mean-cell-length of each boundary edge
        are excluded.

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

    if field_bounds is not None:
        xmin, xmax, ymin, ymax = field_bounds
        field_area = (xmax - xmin) * (ymax - ymin)
        margin = np.sqrt(field_area / len(positions))  # ≈ one cell length
        interior_mask = (
            ~np.isnan(areas)
            & (positions[:, 0] > xmin + margin)
            & (positions[:, 0] < xmax - margin)
            & (positions[:, 1] > ymin + margin)
            & (positions[:, 1] < ymax - margin)
        )
    else:
        interior_mask = ~np.isnan(areas)

    interior_areas = areas[interior_mask]
    n = len(interior_areas)
    mean = float(np.mean(interior_areas)) if n > 0 else np.nan
    std = float(np.std(interior_areas)) if n > 0 else np.nan

    return {
        "areas": interior_areas,
        "mean_area": mean,
        "std_area": std,
        "cv_area": std / mean if mean > 0 else np.nan,
        "regularity_index": mean / std if std > 0 else np.inf,
        "n_interior": int(np.sum(interior_mask)),
    }


def _polygon_area(vertices: np.ndarray) -> float:
    """Signed area of a polygon via the shoelace formula."""
    x, y = vertices[:, 0], vertices[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
