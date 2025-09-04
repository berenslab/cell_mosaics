import numpy as np

from cell_mosaics.coverage import CoverageDensityMapper


def test_boundary_coverage_non_zero_on_edges():
    # Define a simple square field
    bounds = (0.0, 100.0, 0.0, 100.0)
    mapper = CoverageDensityMapper(bounds, resolution=100)

    # Polygon crossing the LEFT boundary (x < 0)
    poly_left = np.array([
        [-10.0, 50.0],  # outside to the left
        [20.0, 80.0],
        [20.0, 20.0],
    ], dtype=float)
    mapper.add_polygon(poly_left)

    # Polygon crossing the TOP boundary (y > ymax)
    poly_top = np.array([
        [40.0, 90.0],
        [60.0, 110.0],  # outside above the top
        [80.0, 90.0],
    ], dtype=float)
    mapper.add_polygon(poly_top)

    # Sanity: stats can be computed
    stats = mapper.get_coverage_statistics()
    assert isinstance(stats, dict)
    assert stats["total_pixels"] == mapper.coverage_count.size

    # Check coverage along the leftmost column and top row
    left_col_sum = mapper.coverage_count[:, 0].sum()
    top_row_sum = mapper.coverage_count[-1, :].sum()  # origin='lower' -> last row is top

    assert left_col_sum > 0, "Expected non-zero coverage along the left boundary column."
    assert top_row_sum > 0, "Expected non-zero coverage along the top boundary row."