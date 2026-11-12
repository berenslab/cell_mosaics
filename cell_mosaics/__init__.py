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
    g_function,
    l_function,
    nearest_neighbor_distances,
    nnd_border_mask,
    nnd_statistics,
    packing_factor,
    packing_intensity,
    voronoi_analysis,
    voronoi_border_mask,
    voronoi_domain_polygons,
)
from .monte_carlo import compare_to_random_simulations, envelope_test, regularity_statistics
from .permutation import cross_nnd_permutation_test, label_permutation_test
from .plot_metrics import (
    plot_drp,
    plot_envelope,
    plot_nnd,
    plot_simulation_comparison,
    plot_voronoi_areas,
    plot_voronoi_tessellation,
)
from .plotting import plot_polygon
from .simulation import simulate_random_mosaic
from .toy_data import generate_example_neurons

__version__ = "0.1.0"

__all__ = [
    "CoverageDensityMapper",
    "compare_to_random_simulations",
    "compute_centroids",
    "cross_nnd_permutation_test",
    "density_recovery_profile",
    "drp_effective_radius",
    "envelope_test",
    "g_function",
    "generate_example_neurons",
    "l_function",
    "label_permutation_test",
    "nearest_neighbor_distances",
    "nnd_border_mask",
    "nnd_statistics",
    "packing_factor",
    "packing_intensity",
    "plot_drp",
    "plot_envelope",
    "plot_nnd",
    "plot_polygon",
    "plot_simulation_comparison",
    "plot_voronoi_areas",
    "plot_voronoi_tessellation",
    "regularity_statistics",
    "simulate_random_mosaic",
    "voronoi_analysis",
    "voronoi_border_mask",
    "voronoi_domain_polygons",
]
