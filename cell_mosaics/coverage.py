from __future__ import annotations

"""Tools for computing and visualizing coverage density maps.

This module provides a CoverageDensityMapper class that can rasterize polygonal
cell outlines (or their convex hulls) into a 2D integer grid representing how
many cell_outlines cover each pixel. Use plotting.plot_coverage to visualize the
result.
"""

import numpy as np
from skimage.draw import polygon
from scipy.spatial import ConvexHull
from typing import Iterable

from cell_mosaics import plotting


class CoverageDensityMapper:
    """Rasterize cell_outlines into a coverage count image.

    Parameters
    ----------
    field_bounds : tuple[float, float, float, float]
        Bounds of the field as (xmin, xmax, ymin, ymax).
    resolution : int, optional
        Grid resolution along each axis (number of pixels). Default is 1000.

    Attributes
    ----------
    coverage_count : np.ndarray
        2D array of shape (resolution, resolution) with coverage counts.
    x_grid, y_grid : np.ndarray
        Coordinate grids used for rasterization.
    dx, dy : float
        Pixel size along x and y.
    cell_outlines : list[np.ndarray]
        List of added cell_outlines (each as an (N, 2) array).
    """

    def __init__(self, field_bounds: tuple[float, float, float, float], resolution: int = 1000) -> None:
        self.xmin, self.xmax, self.ymin, self.ymax = field_bounds
        self.resolution = int(resolution)

        self.x_grid = np.linspace(self.xmin, self.xmax, self.resolution)
        self.y_grid = np.linspace(self.ymin, self.ymax, self.resolution)
        self.dx = (self.xmax - self.xmin) / self.resolution
        self.dy = (self.ymax - self.ymin) / self.resolution

        self.cell_outlines = []

        self.coverage_count: np.ndarray = np.zeros((self.resolution, self.resolution), dtype=int)

    def _rasterize(self, pts: np.ndarray) -> None:
        """Fill the coverage grid for a single polygon given by points.

        Parameters
        ----------
        pts : np.ndarray
            Array of shape (N, 2) with polygon vertices as (x, y).
        """
        # convert to grid coordinates and clip
        x = np.clip((pts[:, 0] - self.xmin) / self.dx, 0, self.resolution - 1)
        y = np.clip((pts[:, 1] - self.ymin) / self.dy, 0, self.resolution - 1)
        rr, cc = polygon(y, x, self.coverage_count.shape)
        self.coverage_count[rr, cc] += 1

    def add_polygon(self, points: np.ndarray | Iterable[Iterable[float]]) -> None:
        """Add a polygon to the coverage map.

        Parameters
        ----------
        points : array-like of shape (N, 2)
            Polygon vertices as (x, y). Must have N >= 3.
        """
        points = np.asarray(points, float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
            raise ValueError("points must be (N,2) with N>=3")
        self.cell_outlines.append(points)
        self._rasterize(points)

    def add_multiple_polygons(self, cell_outlines: Iterable[np.ndarray]) -> None:
        """Add multiple cell_outlines.

        Parameters
        ----------
        cell_outlines : Iterable[np.ndarray]
            Iterable of arrays, each of shape (N, 2).
        """
        for pts in cell_outlines:
            self.add_polygon(pts)

    def add_convex_hull(self, points: np.ndarray | Iterable[Iterable[float]]) -> None:
        """Add the convex hull of the given points.

        If fewer than 3 points are provided or hull fails, this is a no-op.

        Parameters
        ----------
        points : array-like of shape (N, 2)
            Points whose convex hull will be rasterized.
        """
        points = np.asarray(points, float)
        if len(points) < 3:
            return
        try:
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
        except Exception:
            return
        self.add_polygon(hull_points)

    def add_multiple_hulls(self, cell_outlines: Iterable[np.ndarray]) -> None:
        """Add convex hulls for multiple point sets."""
        for pts in cell_outlines:
            self.add_convex_hull(pts)

    def plot_coverage(self, plot_cell_outlines, **kwargs) -> tuple[object | None, object, object]:
        """Plot the coverage map using cell_mosaics.plotting.plot_coverage.

        Parameters
        ----------
        **kwargs : dict
            Forwarded to plotting.plot_coverage.

        Returns
        -------
        tuple
            (fig, ax, im) as returned by plotting.plot_coverage. fig may be None
            if an axes was provided.
        """
        fig, ax, im = plotting.plot_coverage(
            self.coverage_count, extent=(self.xmin, self.xmax, self.ymin, self.ymax),
            cell_outlines=self.cell_outlines if plot_cell_outlines else None,
            **kwargs)
        return fig, ax, im

    def get_coverage_statistics(self) -> dict:
        """Compute summary statistics of the coverage grid.

        Returns
        -------
        dict
            Dictionary with keys: 'max_coverage', 'mean_coverage', 'coverage_std',
            'area_covered_fraction', 'total_pixels', 'covered_pixels'.
        """
        non_zero = self.coverage_count[self.coverage_count > 0]
        return {
            'max_coverage': int(np.max(self.coverage_count)) if self.coverage_count.size else 0,
            'mean_coverage': float(np.mean(non_zero)) if non_zero.size else 0.0,
            'coverage_std': float(np.std(non_zero)) if non_zero.size else 0.0,
            'area_covered_fraction': float(
                np.sum(self.coverage_count > 0) / self.coverage_count.size) if self.coverage_count.size else 0.0,
            'total_pixels': int(self.coverage_count.size),
            'covered_pixels': int(np.sum(self.coverage_count > 0)),
        }
