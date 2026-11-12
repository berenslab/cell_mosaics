"""Spatial statistics metrics for cell mosaics.

Includes nearest-neighbor distance (NND), density recovery profile (DRP),
and Voronoi domain analysis — standard measures of mosaic regularity — plus
the packing factor, the cumulative G and L functions, and the border-cell
exclusion rules of Keeley, Eglen & Reese (2020).
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
    edge_correction_bounds: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    """Average per-reference-cell neighbor count in each DRP annulus.

    Neighbor distances are always measured against the full `positions`
    array; `interior_mask` only restricts which cells act as reference
    points (the denominator/rows of the average), so a masked-out cell can
    still be counted as someone else's neighbor.

    With `edge_correction_bounds`, each pair is weighted by the translation
    edge correction A / ((W - |dx|) * (H - |dy|)) for that rectangle, which
    compensates on average for neighbors lost outside the field.
    """
    dists = cdist(positions, positions)
    np.fill_diagonal(dists, np.inf)
    if edge_correction_bounds is not None:
        xmin, xmax, ymin, ymax = edge_correction_bounds
        width, height = xmax - xmin, ymax - ymin
        dx = np.abs(positions[:, None, 0] - positions[None, :, 0])
        dy = np.abs(positions[:, None, 1] - positions[None, :, 1])
        weights = width * height / ((width - dx) * (height - dy))
    else:
        weights = np.ones_like(dists)

    if interior_mask is not None:
        interior_mask = np.asarray(interior_mask, dtype=bool)
        dists, weights = dists[interior_mask], weights[interior_mask]
    n_ref = dists.shape[0]

    n_bins = len(bin_edges) - 1
    counts = np.zeros(n_bins)
    for j in range(n_bins):
        in_ring = (dists >= bin_edges[j]) & (dists < bin_edges[j + 1])
        counts[j] = weights[in_ring].sum() / n_ref
    return counts


def density_recovery_profile(
    positions: np.ndarray,
    field_bounds: tuple[float, float, float, float],
    n_bins: int = 20,
    max_radius: float | None = None,
    interior_mask: np.ndarray | None = None,
    mean_density: float | None = None,
    edge_correction: bool = False,
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
    the field entirely, unless `edge_correction` is set.

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
    mean_density : float or None, optional
        Density (cells per unit area) used as the CSR baseline the profile is
        normalized against. Defaults to `len(positions) / field_area`.

        Override this when `positions` and the reference cells cover regions
        of *different* density, which is exactly the case when `positions` is
        padded beyond `interior_mask` to give reference cells complete
        neighbourhoods: the padding contributes to `len(positions)` and to
        `field_area`, but the profile describes neighbourhoods around the
        interior cells only. If the interior is denser than the padded region
        (a well-sampled core inside a sparse periphery), the default baseline
        is too low, so the profile settles above 1 instead of recovering to
        it and the exclusion zone is understated. Passing the interior's own
        density fixes both.
    edge_correction : bool, optional
        Weight each pair by the translation edge correction for the
        rectangular `field_bounds`, A / ((W - |dx|) * (H - |dy|)), so that
        neighbors lost beyond the field edge are compensated for on average
        and every cell can act as a reference point. An alternative to
        `interior_mask` (combining the two over-corrects). Default False.

    Returns
    -------
    bin_centers : np.ndarray of shape (n_bins,)
    drp : np.ndarray of shape (n_bins,) — normalized density (1 = random)
    """
    xmin, xmax, ymin, ymax = field_bounds
    field_area = (xmax - xmin) * (ymax - ymin)
    n_cells = len(positions)
    global_density = n_cells / field_area if mean_density is None else mean_density

    if max_radius is None:
        max_radius = min(xmax - xmin, ymax - ymin) / 4.0

    bin_edges = np.linspace(0.0, max_radius, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_areas = np.pi * (bin_edges[1:] ** 2 - bin_edges[:-1] ** 2)

    counts = _drp_bin_counts(positions, bin_edges, interior_mask,
                             field_bounds if edge_correction else None)
    expected = global_density * bin_areas
    drp = counts / expected

    return bin_centers, drp


def drp_effective_radius(
    positions: np.ndarray,
    field_bounds: tuple[float, float, float, float],
    n_bins: int = 20,
    max_radius: float | None = None,
    interior_mask: np.ndarray | None = None,
    mean_density: float | None = None,
    edge_correction: bool = False,
    method: str = "max_deficit",
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
    `effective_radius` as an upper bound rather than a precise estimate. It
    also overshoots for near-lattice mosaics, whose deficit peaks again just
    inside the second ring of neighbors.

    `method='rodieck'` instead follows Rodieck's (1991) definition, as used
    by Keeley, Eglen & Reese (2020): N_missing is the deficit accumulated
    over the bins from the origin up to (not including) the first bin whose
    density reaches `mean_density`, i.e. only the contiguous low-density
    region around the origin. It is unbiased for a lattice and does not pick
    up noise beyond the exclusion zone, but becomes unreliable when a
    low-density field puts a chance close pair in an early bin.

    Parameters
    ----------
    positions, field_bounds, n_bins, max_radius, interior_mask, mean_density, edge_correction
        As per `density_recovery_profile`. `mean_density` sets the CSR
        baseline that the deficit -- and hence r_e -- is measured against, so
        override it whenever you would override it there.
    method : {'max_deficit', 'rodieck'}, optional
        How N_missing is read off the profile (see above). Default
        'max_deficit'.

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
    if mean_density is None:
        mean_density = n_cells / field_area

    if max_radius is None:
        max_radius = min(xmax - xmin, ymax - ymin) / 4.0

    bin_edges = np.linspace(0.0, max_radius, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    counts = _drp_bin_counts(positions, bin_edges, interior_mask,
                             field_bounds if edge_correction else None)
    cumulative_observed = np.cumsum(counts)
    cumulative_expected = mean_density * np.pi * bin_edges[1:] ** 2

    deficit = cumulative_expected - cumulative_observed
    if method == "max_deficit":
        missing_count = float(max(deficit.max(), 0.0))
    elif method == "rodieck":
        bin_expected = np.diff(mean_density * np.pi * bin_edges ** 2)
        recovered = np.flatnonzero(counts >= bin_expected)
        n_below = recovered[0] if recovered.size else n_bins
        missing_count = float(deficit[n_below - 1]) if n_below > 0 else 0.0
    else:
        raise ValueError(f"Unknown method {method!r}; choose 'max_deficit' or 'rodieck'")
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
    margin_factor: float = 1.0,
) -> dict:
    """Compute Voronoi tessellation and summarise domain areas.

    Cells with infinite Voronoi regions (on the convex hull) and, optionally,
    cells within one cell-length of the field boundary are excluded to reduce
    boundary artefacts.

    Note: a Voronoi cell's area blows up whenever its true neighbor lies
    outside the observed field, not only right at the convex hull -- a
    sparse patch of a random (CSR) process near the edge can leave a
    "just inside the margin" cell with a domain many times the median, since
    nothing constrains it on the missing side. One cell-length (the default
    margin) does not reliably exclude these; if `mean_area`/`cv_area` are
    dominated by one or two extreme areas, increase `margin_factor` (at the
    cost of fewer interior cells) rather than trusting the default margin.

    Parameters
    ----------
    positions : np.ndarray of shape (n_cells, 2)
    field_bounds : (xmin, xmax, ymin, ymax) or None
        If provided (and `interior_mask` is not), cells within
        `margin_factor` mean-cell-lengths of each boundary edge are excluded.
    interior_mask : np.ndarray of bool, shape (n_cells,), optional
        Explicit interior selection (e.g. a fixed stable-sampling crop box),
        used instead of the automatic `field_bounds` margin. The Voronoi
        tessellation itself still uses the *full* `positions` array, so
        interior cells get correct neighbors/areas from cells just outside
        the mask -- only the summary statistics are restricted. Cells with an
        infinite (unbounded) region are always excluded regardless.
    margin_factor : float, optional
        Scales the automatic `field_bounds` margin (ignored when
        `interior_mask` is given). Default 1.0 (one mean-cell-length).
        Increase to exclude more of the boundary, trading away interior
        cells for less bias from cells whose true neighbor lies outside
        the field. Default 1.0.

    Returns
    -------
    dict with keys
        areas : np.ndarray — Voronoi domain areas for interior cells
        mean_area : float
        std_area : float
        cv_area : float — coefficient of variation (std / mean)
        regularity_index : float — mean / std
        n_interior : int — number of interior cells used
        interior_mask : np.ndarray of bool, shape (n_cells,) — the resolved
            interior selection over *all* input `positions` (not just the
            interior ones), for reuse elsewhere, e.g. to color excluded
            cells differently in `plot_metrics.plot_voronoi_tessellation`.
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
        margin = margin_factor * np.sqrt(field_area / len(positions))  # ≈ one cell length
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
        "interior_mask": resolved_interior_mask,
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


def nnd_border_mask(
    positions: np.ndarray,
    field_bounds: tuple[float, float, float, float],
) -> np.ndarray:
    """Cells whose nearest neighbor is certain, for border-cell exclusion.

    A cell lying closer to the field border than to its nearest neighbor
    could have its true nearest neighbor just outside the field, so its NND
    is uncertain. Excluding such cells is the border rule of Keeley, Eglen &
    Reese (2020, Fig. 1); pass the result as `interior_mask` to
    `nnd_statistics` or `g_function`.

    Parameters
    ----------
    positions : np.ndarray of shape (n_cells, 2)
    field_bounds : (xmin, xmax, ymin, ymax)

    Returns
    -------
    np.ndarray of bool, shape (n_cells,) — True where NND < distance to border
    """
    xmin, xmax, ymin, ymax = field_bounds
    x, y = positions[:, 0], positions[:, 1]
    border_distance = np.minimum.reduce([x - xmin, xmax - x, y - ymin, ymax - y])
    return nearest_neighbor_distances(positions) < border_distance


def voronoi_border_mask(
    positions: np.ndarray,
    field_bounds: tuple[float, float, float, float],
) -> np.ndarray:
    """Cells whose Voronoi domain is certain, for border-cell exclusion.

    A domain is uncertain if it is unbounded or any of its vertices lies
    outside the field, since an unobserved cell beyond the border could then
    cut it. Unlike the fixed margin of `voronoi_analysis(field_bounds=...)`,
    this adapts to local density, and is the rule used by Keeley, Eglen &
    Reese (2020, Fig. 7). Pass the result as `interior_mask` to
    `voronoi_analysis`.

    Parameters
    ----------
    positions : np.ndarray of shape (n_cells, 2)
    field_bounds : (xmin, xmax, ymin, ymax)

    Returns
    -------
    np.ndarray of bool, shape (n_cells,)
    """
    xmin, xmax, ymin, ymax = field_bounds
    vor = Voronoi(positions)
    mask = np.zeros(len(positions), dtype=bool)
    for i, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if len(region) == 0 or -1 in region:
            continue
        verts = vor.vertices[region]
        mask[i] = bool(np.all(
            (verts[:, 0] >= xmin) & (verts[:, 0] <= xmax)
            & (verts[:, 1] >= ymin) & (verts[:, 1] <= ymax)
        ))
    return mask


def packing_factor(effective_radius: float, density: float) -> float:
    """Rodieck's (1991) packing factor, (ER / MR)**2.

    MR is the maximal radius: the nearest-neighbor spacing of a perfect
    hexagonal lattice at the same density, sqrt(2 / (sqrt(3) * density)).
    The packing factor runs from 0 (complete spatial randomness) to 1 (a
    hexagonal lattice). It is scale-invariant but measures efficient packing
    rather than regularity per se, and inherits the unreliability of the
    effective radius at low density -- compare it with random simulations
    (`monte_carlo.compare_to_random_simulations`) rather than reading it
    absolutely.

    Parameters
    ----------
    effective_radius : float
        E.g. `drp_effective_radius(...)["effective_radius"]`.
    density : float
        Cells per unit area, in the units of `effective_radius`.
    """
    maximal_radius = np.sqrt(2.0 / (np.sqrt(3.0) * density))
    return float((effective_radius / maximal_radius) ** 2)


def packing_intensity(diameter: float | np.ndarray, density: float) -> float:
    """Fraction of the field covered by disks of `diameter` (Eglen & Willshaw 2002).

    Ranges from 0 (infinitesimal points) to ~0.91 (hexagonally packed disks).
    Random sequential placement of non-overlapping disks jams at ~0.55, so
    `simulation.simulate_random_mosaic` cannot reach a soma-size or d_min
    constraint whose packing intensity is much above that.

    Parameters
    ----------
    diameter : float or np.ndarray
        A single disk diameter, or one per cell (the mean squared diameter
        is used).
    density : float
        Cells per unit area.
    """
    mean_sq_diameter = float(np.mean(np.asarray(diameter, dtype=float) ** 2))
    return density * np.pi * mean_sq_diameter / 4.0


def g_function(
    positions: np.ndarray,
    r: np.ndarray,
    interior_mask: np.ndarray | None = None,
) -> np.ndarray:
    """The G function: the cumulative distribution of nearest-neighbor distances.

    G(r) is the fraction of cells whose nearest neighbor lies within r
    (Diggle 2002). Being cumulative, it avoids the bin-width choices of an
    NND histogram, which matters for low-density mosaics.

    Parameters
    ----------
    positions : np.ndarray of shape (n_cells, 2)
    r : np.ndarray
        Distances at which to evaluate G.
    interior_mask : np.ndarray of bool, shape (n_cells,), optional
        Cells to include, e.g. from `nnd_border_mask`. Nearest neighbors
        are still searched among all `positions`. Defaults to every cell.

    Returns
    -------
    np.ndarray of shape r.shape
    """
    nnd = nearest_neighbor_distances(positions)
    if interior_mask is not None:
        nnd = nnd[np.asarray(interior_mask, dtype=bool)]
    nnd = np.sort(nnd)
    if len(nnd) == 0:
        return np.full(np.shape(r), np.nan)
    return np.searchsorted(nnd, np.asarray(r, dtype=float), side="right") / len(nnd)


def l_function(
    positions: np.ndarray,
    field_bounds: tuple[float, float, float, float],
    r: np.ndarray,
) -> np.ndarray:
    """Besag's L function, sqrt(K(r) / pi), with translation edge correction.

    K(r) is Ripley's (1976) K function: the expected number of further cells
    within r of a typical cell, divided by the density. It is the cumulative
    counterpart of the density recovery profile, and L(r) = r for complete
    spatial randomness; an exclusion zone shows as L(r) < r at small r.

    Pairs are weighted by the translation edge correction for a rectangular
    window, A / ((W - |dx|) * (H - |dy|)), which compensates for pairs that
    would fall outside the field.

    Parameters
    ----------
    positions : np.ndarray of shape (n_cells, 2)
    field_bounds : (xmin, xmax, ymin, ymax)
    r : np.ndarray
        Distances at which to evaluate L.

    Returns
    -------
    np.ndarray of shape r.shape
    """
    xmin, xmax, ymin, ymax = field_bounds
    width, height = xmax - xmin, ymax - ymin
    area = width * height
    n = len(positions)
    r = np.asarray(r, dtype=float)
    if n < 2:
        return np.full(r.shape, np.nan)

    pairs = KDTree(positions).query_pairs(float(r.max()), output_type="ndarray")
    delta = np.abs(positions[pairs[:, 0]] - positions[pairs[:, 1]])
    dist = np.hypot(delta[:, 0], delta[:, 1])
    weights = area / ((width - delta[:, 0]) * (height - delta[:, 1]))

    order = np.argsort(dist)
    cumulative_weight = np.concatenate([[0.0], np.cumsum(weights[order])])
    within = cumulative_weight[np.searchsorted(dist[order], r, side="right")]
    # Each unordered pair counts once from each of its two cells.
    k = area * 2.0 * within / (n * (n - 1))
    return np.sqrt(k / np.pi)


def _polygon_area(vertices: np.ndarray) -> float:
    """Signed area of a polygon via the shoelace formula."""
    x, y = vertices[:, 0], vertices[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
