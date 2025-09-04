# cell-mosaics

Tools for computing and visualizing coverage density maps for cell outlines.

## Installation

Once published to PyPI:

```bash
pip install cell-mosaics
```

For development, clone the repo and install with dev extras:

```bash
pip install -e .[dev]
```

## Quick start

```python
from cell_mosaics import CoverageDensityMapper

mapper = CoverageDensityMapper(field_bounds=(0, 1000, 0, 1000), resolution=500)
# Add polygons (Nx2 arrays of x,y) via mapper.add_polygon(...)
# Then visualize using plotting utilities in the package.
```

## License

MIT
