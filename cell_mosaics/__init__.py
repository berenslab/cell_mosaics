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
from .plotting import plot_polygon
from .toy_data import generate_example_neurons

__all__ = [
    "CoverageDensityMapper",
    "generate_example_neurons",
    "plot_polygon",
]
