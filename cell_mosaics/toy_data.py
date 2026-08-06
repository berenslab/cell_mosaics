"""Utility functions to generate toy polygonal data for demos and tests."""

from __future__ import annotations

import numpy as np


def _generate_centroids(
        n_cells: int,
        field_size: int,
        rng: np.random.Generator,
        arrangement: str,
        jitter: float,
        margin: float = 100.0,
) -> np.ndarray:
    """Place `n_cells` centroids either at random (CSR) or on a jittered grid."""
    if arrangement == "random":
        return rng.uniform(margin, field_size - margin, (n_cells, 2))

    if arrangement == "regular":
        n_side = int(np.ceil(np.sqrt(n_cells)))
        span = field_size - 2 * margin
        spacing = span / max(n_side - 1, 1)
        xs = margin + spacing * np.arange(n_side)
        ys = margin + spacing * np.arange(n_side)
        grid_x, grid_y = np.meshgrid(xs, ys)
        points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        rng.shuffle(points)
        points = points[:n_cells].copy()
        points += rng.uniform(-jitter * spacing, jitter * spacing, points.shape)
        return points

    raise ValueError(f"Unknown arrangement: {arrangement!r} (expected 'random' or 'regular')")


def generate_example_neurons(
        n_cells: int = 50,
        field_size: int = 1000,
        cell_size_range: tuple[float, float] = (50, 150),
        polygon_type: str = "irregular",
        arrangement: str = "random",
        jitter: float = 0.15,
        seed: int = 42,
) -> list[np.ndarray]:
    """Generate synthetic polygonal neuron outlines.

    Parameters
    ----------
    n_cells : int
        Number of neuron outlines to generate.
    field_size : int
        Spatial extent for both x and y (i.e., field is [0, field_size]²).
    cell_size_range : tuple[float, float]
        Approximate radius range for cell_outlines.
    polygon_type : {"irregular", "concave", "convex"}
        Shape style of the generated cell_outlines.
    arrangement : {"random", "regular"}
        Spatial placement of cell centroids. "random" places them uniformly
        at random (complete spatial randomness). "regular" places them on a
        square grid with positional jitter, approximating a real, mostly
        regular retinal mosaic.
    jitter : float
        Positional jitter for `arrangement="regular"`, as a fraction of the
        grid spacing. Ignored when `arrangement="random"`. Default 0.15.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[np.ndarray]
        Each entry is an array of shape (N, 2) with XY polygon coordinates.
    """
    rng = np.random.default_rng(seed)
    centroids = _generate_centroids(n_cells, field_size, rng, arrangement, jitter)
    cell_outlines: list[np.ndarray] = []

    for cx, cy in centroids:
        radius = rng.uniform(*cell_size_range)

        if polygon_type == "irregular":
            n_points = rng.integers(8, 16)
            angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
            radii = radius * (0.7 + 0.6 * rng.random(n_points))
            x = cx + radii * np.cos(angles)
            y = cy + radii * np.sin(angles)
        elif polygon_type == "concave":
            n_points = rng.integers(10, 20)
            angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
            radii = []
            for j in range(n_points):
                if j % 3 == 1:
                    radii.append(radius * 0.3)
                else:
                    radii.append(radius * (0.8 + 0.4 * rng.random()))
            radii = np.asarray(radii)
            x = cx + radii * np.cos(angles)
            y = cy + radii * np.sin(angles)
        else:  # convex
            n_points = rng.integers(6, 12)
            angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
            radii = radius * (0.8 + 0.3 * rng.random(n_points))
            x = cx + radii * np.cos(angles)
            y = cy + radii * np.sin(angles)

        cell_outlines.append(np.column_stack([x, y]))

    return cell_outlines


def bounds_from_field_size(field_size: int) -> tuple[float, float, float, float]:
    """Return (xmin, xmax, ymin, ymax) for a square field of given size."""
    return 0.0, float(field_size), 0.0, float(field_size)
