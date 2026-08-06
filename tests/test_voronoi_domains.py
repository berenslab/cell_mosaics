"""Tests for cell_mosaics.metrics.voronoi_domain_polygons."""

import numpy as np

from cell_mosaics.metrics import voronoi_analysis, voronoi_domain_polygons


def regular_grid(n: int, spacing: float) -> np.ndarray:
    x, y = np.meshgrid(np.arange(n) * spacing, np.arange(n) * spacing)
    return np.column_stack([x.ravel(), y.ravel()])


class TestVoronoiDomainPolygons:
    def test_shapes_agree(self):
        positions = regular_grid(6, 10.0)
        polygons, areas, point_indices = voronoi_domain_polygons(positions)
        assert len(polygons) == len(areas) == len(point_indices)
        assert all(p.ndim == 2 and p.shape[1] == 2 for p in polygons)

    def test_only_interior_points_have_finite_domains(self):
        # A 5×5 grid has 16 hull points with unbounded regions, leaving 9.
        positions = regular_grid(5, 10.0)
        _, _, point_indices = voronoi_domain_polygons(positions)
        assert len(point_indices) == 9
        assert np.all(point_indices < len(positions))

    def test_regular_grid_domains_are_unit_cells(self):
        spacing = 10.0
        positions = regular_grid(6, spacing)
        _, areas, _ = voronoi_domain_polygons(positions)
        np.testing.assert_allclose(areas, spacing**2, rtol=1e-9)

    def test_areas_match_voronoi_analysis(self):
        rng = np.random.default_rng(0)
        positions = rng.uniform(0, 100, (60, 2))

        _, areas, _ = voronoi_domain_polygons(positions)
        stats = voronoi_analysis(positions)  # no bounds -> all finite regions

        np.testing.assert_allclose(np.sort(areas), np.sort(stats["areas"]))

    def test_point_indices_select_containing_point(self):
        # Each returned polygon must actually contain its own point.
        from matplotlib.path import Path

        rng = np.random.default_rng(1)
        positions = rng.uniform(0, 100, (40, 2))
        polygons, _, point_indices = voronoi_domain_polygons(positions)

        for poly, idx in zip(polygons, point_indices):
            assert Path(poly).contains_point(positions[idx])
