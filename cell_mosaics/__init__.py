"""
cell‑mosaics
================
Utilities to compute and plot retinal mosaic coverage maps.

Quick start
-----------
>>> from cell_mosaics import CoverageDensityMapper
>>> mapper = CoverageDensityMapper(field_bounds=(0, 1000, 0, 1000), resolution=500)
>>> # add cell_outlines or convex hulls (arrays shaped (N,2))
>>> # mapper.add_polygon(points)
>>> # fig, ax, _ = mapper.plot_coverage()
"""
from .coverage import CoverageDensityMapper
from .metrics import (
    compute_centroids,
    density_recovery_profile,
    drp_effective_radius,
    nearest_neighbor_distances,
    nnd_statistics,
    voronoi_analysis,
    voronoi_domain_polygons,
)
from .permutation import cross_nnd_permutation_test, label_permutation_test
from .plot_metrics import (
    plot_drp,
    plot_nnd,
    plot_voronoi_areas,
    plot_voronoi_tessellation,
)
from .plotting import plot_polygon
from .toy_data import generate_example_neurons

__version__ = "0.1.0"

__all__ = [
    "CoverageDensityMapper",
    "compute_centroids",
    "cross_nnd_permutation_test",
    "density_recovery_profile",
    "drp_effective_radius",
    "generate_example_neurons",
    "label_permutation_test",
    "nearest_neighbor_distances",
    "nnd_statistics",
    "plot_drp",
    "plot_nnd",
    "plot_polygon",
    "plot_voronoi_areas",
    "plot_voronoi_tessellation",
    "voronoi_analysis",
    "voronoi_domain_polygons",
]
