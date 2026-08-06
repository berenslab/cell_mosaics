# cell-mosaics

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Tools for computing and visualizing **retinal cell mosaics**: coverage density
maps built from cell outlines, and the standard spatial-statistics measures
used to quantify how regularly cells of a given type tile the retina.

## Features

- **Coverage density maps** — rasterize cell outlines (or their convex hulls)
  onto a pixel grid and summarize how many cells cover each pixel.
- **Mosaic regularity metrics**
  - Nearest-neighbour distance (NND) and the regularity index
  - Density recovery profile (DRP) and its effective radius (exclusion-zone size)
  - Voronoi domain analysis, with boundary-cell exclusion
- **Plotting helpers** for every metric above, plus polygon/coverage-map
  visualization, sharing a consistent `(fig, ax)` API so plots can be
  composed or overlaid for comparison.
- **Permutation tests** for whether a labeled subgroup of cells is more (or
  less) spatially regular than chance.
- **Synthetic mosaic generator** for testing and demos, spanning different
  outline shapes and spatial arrangements (random vs. mostly regular).

## Installation

```bash
pip install cell-mosaics
```

For development (tests, notebooks):

```bash
uv sync --group dev
```

## Quick start

```python
import cell_mosaics as cm

# Synthetic outlines for a demo mosaic
outlines = cm.generate_example_neurons(n_cells=80, field_size=1000, seed=42)
centroids = cm.compute_centroids(outlines)
bounds = cm.toy_data.bounds_from_field_size(1000)

# Regularity via nearest-neighbour distance
nnd_stats = cm.nnd_statistics(centroids)
print(f"Regularity index: {nnd_stats['regularity_index']:.2f}")

fig, ax = cm.plot_nnd(nnd_stats)
```

See [`notebooks/demo.ipynb`](notebooks/demo.ipynb) for a full walkthrough,
including the density recovery profile, Voronoi domain analysis, and
side-by-side comparisons across mosaic types.

## Statistical background

The regularity metrics implemented here — the regularity index, the density
recovery profile, and Voronoi domain analysis, along with the standard
boundary-effect corrections applied to each — follow the methodology
described in:

> Eglen, S. J. (2012). Cellular spacing: analysis and modelling of retinal
> mosaics. In N. Le Novère (Ed.), *Computational Systems Neurobiology*
> (Chapter 12, pp. 365–385). Springer.

## Development

```bash
uv sync --group dev
pytest
```

## License

MIT — see [LICENSE](LICENSE).
