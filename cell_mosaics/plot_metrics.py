"""Plotting utilities for cell_mosaics spatial statistics.

One function per metric in `cell_mosaics.metrics`: nearest-neighbour
distance (NND), the density recovery profile (DRP), and Voronoi domain
analysis. Each takes the metric's own result (not raw positions or
outlines) and draws it onto an axes, following the same `(fig, ax)`
return convention as `plotting.plot_coverage` (fig is None when `ax` is
supplied). `color`/`label`/`show_mean` let several calls be overlaid on
one `ax` to compare mosaics, e.g. one polygon type per call.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.spatial import Voronoi, voronoi_plot_2d

# matplotlib color spec: a name/hex string, or an RGB(A) tuple (e.g. from a colormap).
ColorLike = str | tuple[float, float, float] | tuple[float, float, float, float]


def _add_legend_if_labeled(ax: Axes) -> None:
    _, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend()


def plot_nnd(
    nnd_stats: dict,
    *,
    ax: Axes | None = None,
    figsize: tuple[int, int] = (6, 4),
    bins: int = 15,
    color: ColorLike = "steelblue",
    label: str | None = None,
    show_mean: bool = True,
) -> tuple[Figure | None, Axes]:
    """Plot the nearest-neighbour distance (NND) distribution.

    Parameters
    ----------
    nnd_stats : dict
        Output of `metrics.nnd_statistics`.
    ax : matplotlib.axes.Axes or None, optional
        Draw on this axes; otherwise create a new figure. Default None.
    figsize : tuple[int, int], optional
        Figure size if a new figure is created. Default (6, 4).
    bins : int, optional
        Histogram bin count. Default 15.
    color : ColorLike, optional
        Histogram and mean-line color. Default 'steelblue'.
    label : str or None, optional
        Legend label for this series, e.g. a mosaic/polygon-type name.
        Pass this (and call the function once per series on a shared
        `ax`) to compare several mosaics; leave as None for a single,
        titled plot. Default None.
    show_mean : bool, optional
        Draw a vertical line at the mean NND. Default True.

    Returns
    -------
    (fig, ax)
        fig may be None if `ax` was provided.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None

    overlay = label is not None
    ax.hist(
        nnd_stats["nnd"],
        bins=bins,
        color=color,
        alpha=0.6 if overlay else 1.0,
        edgecolor="none" if overlay else "white",
        linewidth=0.5,
        label=label,
    )
    if show_mean:
        mean_label = f"{label} mean" if overlay else f"mean = {nnd_stats['mean_nnd']:.1f}"
        ax.axvline(nnd_stats["mean_nnd"], color=color if overlay else "tomato",
                   linewidth=2, linestyle="--" if overlay else "-", label=mean_label)

    ax.set_xlabel("Nearest-neighbour distance")
    ax.set_ylabel("Count")
    if not overlay:
        ax.set_title(f"NND distribution (RI = {nnd_stats['regularity_index']:.2f})")
    _add_legend_if_labeled(ax)
    return fig, ax


def plot_drp(
    bin_centers: np.ndarray,
    drp: np.ndarray,
    *,
    ax: Axes | None = None,
    figsize: tuple[int, int] = (6, 4),
    color: ColorLike = "steelblue",
    label: str | None = None,
    show_baseline: bool = True,
    effective_radius: float | None = None,
) -> tuple[Figure | None, Axes]:
    """Plot a Density Recovery Profile (DRP) curve.

    Parameters
    ----------
    bin_centers, drp : np.ndarray
        Output of `metrics.density_recovery_profile`.
    ax : matplotlib.axes.Axes or None, optional
        Draw on this axes; otherwise create a new figure. Default None.
    figsize : tuple[int, int], optional
        Figure size if a new figure is created. Default (6, 4).
    color : ColorLike, optional
        Line color. Default 'steelblue'.
    label : str or None, optional
        Legend label for this series, e.g. a mosaic/polygon-type name.
        Pass this (and call the function once per series on a shared
        `ax`) to compare several mosaics; leave as None for a single,
        titled plot. Default None.
    show_baseline : bool, optional
        Draw the horizontal "random" (DRP = 1) reference line. Skipped if
        it has already been drawn on `ax` by an earlier call. Default True.
    effective_radius : float or None, optional
        If given (e.g. from `metrics.drp_effective_radius`), draw a
        vertical line marking the exclusion-zone radius. Default None.

    Returns
    -------
    (fig, ax)
        fig may be None if `ax` was provided.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None

    overlay = label is not None
    ax.plot(bin_centers, drp, color=color, linewidth=2, label=label)

    _, existing_labels = ax.get_legend_handles_labels()
    if show_baseline and "random" not in existing_labels:
        ax.axhline(1.0, color="gray", linewidth=1, linestyle="--", label="random")

    if effective_radius is not None:
        re_label = f"{label} r_e" if overlay else f"excl. zone ≈ {effective_radius:.0f}"
        ax.axvline(effective_radius, color=color if overlay else "tomato",
                   linewidth=1.5, linestyle=":", label=re_label)

    ax.set_xlabel("Distance")
    ax.set_ylabel("Normalised density")
    if not overlay:
        ax.set_title("Density Recovery Profile")
    _add_legend_if_labeled(ax)
    return fig, ax


def plot_voronoi_tessellation(
    positions: np.ndarray,
    *,
    ax: Axes | None = None,
    figsize: tuple[int, int] = (6, 6),
    field_bounds: tuple[float, float, float, float] | None = None,
    interior_mask: np.ndarray | None = None,
    point_color: ColorLike = "tomato",
    line_color: ColorLike = "steelblue",
    exterior_color: ColorLike = "gray",
) -> tuple[Figure | None, Axes]:
    """Plot the Voronoi tessellation of cell positions.

    Parameters
    ----------
    positions : np.ndarray of shape (n_cells, 2)
    ax : matplotlib.axes.Axes or None, optional
        Draw on this axes; otherwise create a new figure. Default None.
    figsize : tuple[int, int], optional
        Figure size if a new figure is created. Default (6, 6).
    field_bounds : (xmin, xmax, ymin, ymax) or None, optional
        If given, sets the axes limits to the field rather than the
        (looser) default from the Voronoi diagram. Default None.
    interior_mask : np.ndarray of bool, shape (n_cells,), optional
        Which somas were used for the summary statistics, e.g. the
        `interior_mask` returned by `metrics.voronoi_analysis`. Somas where
        this is False (excluded boundary cells) are drawn in
        `exterior_color` instead of `point_color`. Default None (all somas
        drawn in `point_color`).
    point_color : ColorLike, optional
        Color for interior cell positions. Default 'tomato'.
    line_color : ColorLike, optional
        Color for the Voronoi ridge lines. Default 'steelblue'.
    exterior_color : ColorLike, optional
        Color for somas excluded by `interior_mask`. Default 'gray'.

    Returns
    -------
    (fig, ax)
        fig may be None if `ax` was provided.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None

    vor = Voronoi(positions)
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors=line_color,
                     line_width=1, point_size=3)

    if interior_mask is None:
        ax.scatter(positions[:, 0], positions[:, 1], s=15, color=point_color, zorder=3)
    else:
        interior_mask = np.asarray(interior_mask, dtype=bool)
        ax.scatter(positions[~interior_mask, 0], positions[~interior_mask, 1],
                   s=15, color=exterior_color, zorder=3)
        ax.scatter(positions[interior_mask, 0], positions[interior_mask, 1],
                   s=15, color=point_color, zorder=4)

    if field_bounds is not None:
        xmin, xmax, ymin, ymax = field_bounds
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_title("Voronoi tessellation")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return fig, ax


def plot_voronoi_areas(
    vor_stats: dict,
    *,
    ax: Axes | None = None,
    figsize: tuple[int, int] = (6, 4),
    bins: int = 15,
    color: ColorLike = "steelblue",
    label: str | None = None,
    show_mean: bool = True,
) -> tuple[Figure | None, Axes]:
    """Plot the Voronoi domain area distribution.

    Parameters
    ----------
    vor_stats : dict
        Output of `metrics.voronoi_analysis`.
    ax : matplotlib.axes.Axes or None, optional
        Draw on this axes; otherwise create a new figure. Default None.
    figsize : tuple[int, int], optional
        Figure size if a new figure is created. Default (6, 4).
    bins : int, optional
        Histogram bin count. Default 15.
    color : ColorLike, optional
        Histogram and mean-line color. Default 'steelblue'.
    label : str or None, optional
        Legend label for this series, e.g. a mosaic/polygon-type name.
        Pass this (and call the function once per series on a shared
        `ax`) to compare several mosaics; leave as None for a single,
        titled plot. Default None.
    show_mean : bool, optional
        Draw a vertical line at the mean domain area. Default True.

    Returns
    -------
    (fig, ax)
        fig may be None if `ax` was provided.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None

    overlay = label is not None
    ax.hist(
        vor_stats["areas"],
        bins=bins,
        color=color,
        alpha=0.6 if overlay else 1.0,
        edgecolor="none" if overlay else "white",
        linewidth=0.5,
        label=label,
    )
    if show_mean:
        mean_label = f"{label} mean" if overlay else f"mean = {vor_stats['mean_area']:.0f}"
        ax.axvline(vor_stats["mean_area"], color=color if overlay else "tomato",
                   linewidth=2, linestyle="--" if overlay else "-", label=mean_label)

    ax.set_xlabel("Voronoi domain area")
    ax.set_ylabel("Count")
    if not overlay:
        ax.set_title(f"Domain area distribution (CV = {vor_stats['cv_area']:.3f})")
    _add_legend_if_labeled(ax)
    return fig, ax
