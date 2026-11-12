"""Demonstration figures mimicking Keeley, Eglen & Reese (2020).

"From random to regular: Variation in the patterning of retinal mosaics",
J Comp Neurol 528:2135-2160. Each figure here re-creates the logic of one of
the paper's figures with `cell_mosaics`, using *synthetic* mosaics whose
ground truth is known. The "real" fields are drawn from a soma-size plus
d_min spacing model whose parameters are loosely modelled on the paper's
cell types (hence "HC-like" etc.); the matched random simulations are
constrained by soma size only, as in the paper. Numbers will therefore
resemble, but not reproduce, the paper's.

    fig01  CSR vs a regular mosaic; NNRI of CSR vs number of points
    fig02  soma size alone raises the NNRI of random simulations
    fig03  AII-like mosaics vs 99 soma-constrained random simulations
    fig04  autocorrelogram, DRP and effective radius: regular vs random
    fig05  sparse DA-like mosaic: G/L envelopes for two spacing models
    fig06  uniform expansion changes the effective radius, not the RI
    fig07  displaced (GCL-like) mosaic: NNRI vs VDRI
    fig08  from random to regular: six cell types side by side
    fig09  spatial statistics as a function of density

Usage:
    python scripts/keeley2020_figures.py [--out DIR] [--fast] [--only 1 2 8]
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import EllipseCollection, LineCollection
from scipy.stats import mannwhitneyu

import cell_mosaics as cm

# --- Palette (validated with the dataviz skill's palette checker) -----------
# Cell types use categorical slots; random simulations are always neutral
# gray, as in the paper. Soma size, an ordered magnitude, uses a blue ramp.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
RANDOM_FILL = "#c4c3bf"
RANDOM_INK = "#5c5b57"
SIZE_RAMP = ["#86b6ef", "#2a78d6", "#104281"]

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": INK_2,
    "axes.labelcolor": INK,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "text.color": INK,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "savefig.dpi": 150,
    "lines.linewidth": 2,
})


# --- Synthetic cell types -----------------------------------------------------

@dataclass(frozen=True)
class CellType:
    """A synthetic mosaic: somata of N(soma_mean, soma_sd) µm under a d_min rule."""

    name: str
    color: str
    density: float  # cells / mm^2
    soma_mean: float  # µm
    soma_sd: float
    dmin_mean: float = 0.0  # µm; 0 = spaced by soma size only
    dmin_sd: float = 0.0
    field: float = 300.0  # µm, side of the square field
    displacement_sd: float = 0.0  # µm, passive lateral displacement after placement

    @property
    def bounds(self):
        return (0.0, self.field, 0.0, self.field)

    @property
    def soma(self):
        return {"soma_diameter_mean": self.soma_mean, "soma_diameter_sd": self.soma_sd}

    @property
    def spacing(self):
        """Mean intercell spacing, 1 / sqrt(density), in µm."""
        return 1000.0 / np.sqrt(self.density)

    @property
    def drp_kwargs(self):
        return {"n_bins": 20, "max_radius": min(2.5 * self.spacing, self.field / 4)}

    def n_cells(self, density=None):
        return int(round((density or self.density) * (self.field / 1000.0) ** 2))

    def simulate_real(self, rng, density=None):
        density = density or self.density
        # Regular mosaics keep their regularity across density by scaling the
        # spacing rule with the intercell spacing (Keeley et al. 2020, Fig. 9c).
        scale = np.sqrt(self.density / density)
        positions, diameters = cm.simulate_random_mosaic(
            self.n_cells(density), self.bounds, dmin_mean=self.dmin_mean * scale,
            dmin_sd=self.dmin_sd * scale, seed=rng, **self.soma)
        if self.displacement_sd:
            positions = displace(positions, diameters, self.bounds, self.displacement_sd, rng)
        return positions, diameters

    def simulate_random(self, n_cells, rng):
        return cm.simulate_random_mosaic(n_cells, self.bounds, seed=rng, **self.soma)


def displace(positions, diameters, bounds, sd, rng, tries=30):
    """Jitter each soma by N(0, sd), keeping it in the field and non-overlapping."""
    xmin, xmax, ymin, ymax = bounds
    positions = positions.copy()
    for i in rng.permutation(len(positions)):
        for _ in range(tries):
            candidate = positions[i] + rng.normal(0.0, sd, 2)
            if not (xmin <= candidate[0] <= xmax and ymin <= candidate[1] <= ymax):
                continue
            dist = np.linalg.norm(positions - candidate, axis=1)
            dist[i] = np.inf
            if np.all(dist >= 0.5 * (diameters + diameters[i])):
                positions[i] = candidate
                break
    return positions


HC = CellType("HC-like", "#2a78d6", 1450, 10.0, 1.0, 20.0, 2.0)
SAC = CellType("SAC-like", "#eb6834", 1000, 9.0, 0.8, 23.0, 3.0)
VGLUT3 = CellType("VGluT3-like", "#1baf7a", 800, 9.0, 0.8, 23.0, 8.0)
DA = CellType("DA-like", "#008300", 40, 12.0, 1.5, 70.0, 50.0, field=1500.0)
AII = CellType("AII-like", "#e87ba4", 3900, 7.2, 0.4, field=250.0)
BC = CellType("BC-like", "#eda100", 3000, 8.0, 0.7, field=250.0)
SAC_GCL = CellType("SAC-GCL-like", SAC.color, 1000, 9.0, 0.8, 23.0, 3.0, displacement_sd=10.0)

# Row order of Figure 8, from regular to random, as in the paper.
TYPES = [HC, SAC, VGLUT3, DA, AII, BC]


# --- Analysis helpers ---------------------------------------------------------

def analyse(positions, ct):
    """Every per-field quantity the figures need, with the paper's border rules."""
    stats = cm.regularity_statistics(positions, ct.bounds, ct.drp_kwargs)
    nnd = cm.nearest_neighbor_distances(positions)[cm.nnd_border_mask(positions, ct.bounds)]
    areas = cm.voronoi_analysis(
        positions, interior_mask=cm.voronoi_border_mask(positions, ct.bounds))["areas"]
    drp_r, drp = cm.density_recovery_profile(positions, ct.bounds, edge_correction=True,
                                             **ct.drp_kwargs)
    return {"stats": stats, "nnd": nnd, "areas": areas, "drp_r": drp_r,
            "drp_density": drp * stats["density"] * 1e6}  # cells / mm^2


def sample_fields(ct, n_fields, rng, density_spread=0.0):
    """`n_fields` real fields of `ct`, each with one matched random simulation."""
    real, random = [], []
    for _ in range(n_fields):
        density = ct.density * rng.uniform(1 - density_spread, 1 + density_spread)
        positions, diameters = ct.simulate_real(rng, density)
        sim_positions, sim_diameters = ct.simulate_random(len(positions), rng)
        real.append({"positions": positions, "diameters": diameters, **analyse(positions, ct)})
        random.append({"positions": sim_positions, "diameters": sim_diameters,
                       **analyse(sim_positions, ct)})
    return real, random


def stat_values(fields, key):
    return np.array([f["stats"][key] for f in fields])


def frequency_curves(samples, bins):
    """Per-field percentage frequency distribution, one row per field."""
    return np.array([np.histogram(s, bins)[0] / max(len(s), 1) * 100.0 for s in samples])


def mean_sem(values, axis=0):
    values = np.asarray(values, dtype=float)
    mean = np.nanmean(values, axis=axis)
    n = np.sum(~np.isnan(values), axis=axis)
    if np.min(n) < 2:
        return mean, np.zeros_like(mean)
    return mean, np.nanstd(values, axis=axis, ddof=1) / np.sqrt(n)


# --- Drawing helpers ----------------------------------------------------------

def nice_length(limit):
    """Largest round scale-bar length (µm) not exceeding `limit`."""
    return max(v for v in (10, 20, 25, 50, 100, 200, 250, 500, 1000) if v <= limit)


def panel_label(ax, letter):
    ax.text(-0.02, 1.02, letter, transform=ax.transAxes, fontsize=11, fontweight="bold",
            ha="right", va="bottom")


def draw_field(ax, positions, diameters, bounds, color, *, voronoi=False, faded=None,
               scale_bar=None, title=None):
    """Somata at their size (dots if dimensionless), optional Voronoi domains.

    Voronoi domains are drawn only for cells whose domain is certain
    (`voronoi_border_mask`), the cells the analysis keeps. `faded` marks
    excluded border cells, drawn lighter.
    """
    xmin, xmax, ymin, ymax = bounds
    if voronoi:
        keep = cm.voronoi_border_mask(positions, bounds)
        polygons, _, index = cm.voronoi_domain_polygons(positions)
        segments = [np.vstack([p, p[:1]]) for p, i in zip(polygons, index) if keep[i]]
        ax.add_collection(LineCollection(segments, colors=color, linewidths=0.6, alpha=0.7))

    alpha = np.ones(len(positions)) if faded is None else np.where(faded, 0.3, 1.0)
    colors = [matplotlib.colors.to_rgba(color, a) for a in alpha]
    if np.all(diameters == 0):
        ax.scatter(positions[:, 0], positions[:, 1], s=6, c=colors, linewidths=0)
    else:
        ax.add_collection(EllipseCollection(
            diameters, diameters, np.zeros(len(diameters)), units="xy", offsets=positions,
            offset_transform=ax.transData, facecolors=colors, edgecolors="none"))

    if scale_bar:
        span = xmax - xmin
        x1, y = xmax - 0.04 * span, ymin - 0.05 * span
        ax.plot([x1 - scale_bar, x1], [y, y], color=INK, linewidth=2, solid_capstyle="butt",
                clip_on=False)
        ax.text(x1 - scale_bar / 2, y - 0.015 * span, f"{scale_bar:g} µm", ha="center",
                va="top", fontsize=7.5, color=INK_2)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(RANDOM_FILL)
        spine.set_linewidth(0.8)
    if title:
        ax.set_title(title)


def draw_mean_sem(ax, x, curves, color, label, fill=None):
    mean, sem = mean_sem(curves)
    ax.fill_between(x, mean - sem, mean + sem, color=fill or color, alpha=0.45, linewidth=0)
    ax.plot(x, mean, color=color, label=label)


def draw_distribution_pair(ax, real_samples, random_samples, bins, color, xlabel):
    """Mean (± SEM) frequency distributions of real fields vs their random simulations."""
    centers = 0.5 * (bins[:-1] + bins[1:])
    draw_mean_sem(ax, centers, frequency_curves(random_samples, bins), RANDOM_INK, "random",
                  fill=RANDOM_FILL)
    draw_mean_sem(ax, centers, frequency_curves(real_samples, bins), color, "real")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency (%)")
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(bottom=0)


def draw_bar_pair(ax, real, random, color, ylabel, annotate=True):
    """Mean ± SEM bars for real vs random, with each field as a dot."""
    rng = np.random.default_rng(0)
    for x, values, fill in [(0, real, color), (1, random, RANDOM_FILL)]:
        values = np.asarray(values, dtype=float)
        values = values[~np.isnan(values)]
        mean, sem = mean_sem(values)
        ax.bar(x, mean, width=0.62, color=fill, yerr=sem, capsize=3,
               error_kw={"elinewidth": 1, "ecolor": INK_2, "capthick": 1})
        ax.scatter(x + rng.uniform(-0.16, 0.16, len(values)), values, s=12, color=SURFACE,
                   edgecolors=INK_2, linewidths=0.7, zorder=3)
    ax.set_xticks([0, 1], ["real", "random"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel(ylabel)
    if annotate:
        real_v, random_v = (np.asarray(v, float)[~np.isnan(v)] for v in (real, random))
        if len(real_v) > 1 and len(random_v) > 1:
            p = mannwhitneyu(real_v, random_v).pvalue
            ax.set_title(f"ratio {real_v.mean() / random_v.mean():.2f}, p = {p:.2g}",
                         fontsize=8, color=INK_2)


def draw_drp_bars(ax, fields, color, density_line=True):
    """DRP as bars (mean ± SEM over fields), dashed mean density, ER marked."""
    r = fields[0]["drp_r"]
    width = r[1] - r[0]
    mean, sem = mean_sem([f["drp_density"] for f in fields])
    ax.bar(r, mean, width=width * 0.9, color=color, yerr=sem,
           error_kw={"elinewidth": 0.8, "ecolor": INK_2})
    if density_line:
        ax.axhline(np.mean(stat_values(fields, "density")) * 1e6, color=INK, linewidth=1,
                   linestyle="--", label="mean density")
    er = np.nanmean(stat_values(fields, "effective_radius"))
    ax.axvline(er, color=INK_2, linewidth=1.2, linestyle=":", label=f"ER = {er:.1f} µm")
    ax.set_xlabel("Distance (µm)")
    ax.set_ylabel("Density (cells/mm²)")
    ax.set_xlim(0, r[-1] + width / 2)
    ax.legend(loc="lower right")


def draw_autocorrelogram(ax, positions, radius, color, title=None):
    delta = positions[None, :, :] - positions[:, None, :]
    dist = np.linalg.norm(delta, axis=2)
    keep = (dist <= radius) & (dist > 0)
    ax.scatter(delta[..., 0][keep], delta[..., 1][keep], s=2.5, color=color, linewidths=0,
               alpha=0.6)
    ax.add_patch(plt.Circle((0, 0), radius, fill=False, color=RANDOM_FILL, linewidth=0.8))
    ax.set_xlim(-radius * 1.02, radius * 1.02)
    ax.set_ylim(-radius * 1.02, radius * 1.02)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title)


def save(fig, out, name):
    path = out / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# --- Figures ------------------------------------------------------------------

def fig01(out, fast):
    """Paper Fig. 1: a regular mosaic vs CSR, and the NNRI of CSR vs point count."""
    rng = np.random.default_rng(1)
    n_fields = 8 if fast else 32
    real, _ = sample_fields(HC, n_fields, rng)
    csr = [cm.simulate_random_mosaic(len(f["positions"]), HC.bounds, seed=rng)[0] for f in real]
    csr_analysis = [{"nnd": cm.nearest_neighbor_distances(p)[cm.nnd_border_mask(p, HC.bounds)],
                     "ri": cm.regularity_statistics(p, HC.bounds)["nnd_regularity_index"]}
                    for p in csr]

    fig, axes = plt.subplot_mosaic([["a", "b", "c", "d"], ["e", "e", "f", "f"]],
                                   figsize=(14, 7.2), gridspec_kw={"height_ratios": [1.15, 1]})
    p0, d0 = real[0]["positions"], real[0]["diameters"]
    draw_field(axes["a"], p0, d0, HC.bounds, HC.color, faded=~cm.nnd_border_mask(p0, HC.bounds),
               scale_bar=50, title=f"{HC.name} mosaic ({len(p0)} cells)")
    draw_field(axes["b"], csr[0], np.zeros(len(csr[0])), HC.bounds, RANDOM_INK,
               faded=~cm.nnd_border_mask(csr[0], HC.bounds), title="CSR, same density")

    bins = np.arange(0, 47.5, 2.5)
    draw_distribution_pair(axes["c"], [f["nnd"] for f in real], [c["nnd"] for c in csr_analysis],
                           bins, HC.color, "Nearest-neighbour distance (µm)")
    handles, _ = axes["c"].get_legend_handles_labels()
    axes["c"].legend(handles, ["CSR", HC.name])
    axes["c"].set_title(f"NND, {n_fields} fields (mean ± SEM)")

    draw_bar_pair(axes["d"], stat_values(real, "nnd_regularity_index"),
                  [c["ri"] for c in csr_analysis], HC.color, "NN regularity index")
    axes["d"].set_xticks([0, 1], ["real", "CSR"])

    counts = [10, 20, 40, 80, 160, 320]
    n_sims = 40 if fast else 100
    unit = (0.0, 1.0, 0.0, 1.0)
    excluded, included = [], []
    for n in counts:
        ex, inc = [], []
        for _ in range(n_sims):
            p, _ = cm.simulate_random_mosaic(n, unit, seed=rng)
            mask = cm.nnd_border_mask(p, unit)
            if mask.sum() >= 3:
                ex.append(cm.nnd_statistics(p, interior_mask=mask)["regularity_index"])
            inc.append(cm.nnd_statistics(p)["regularity_index"])
        excluded.append(ex)
        included.append(inc)

    for key, data, title in [("e", excluded, "Border cells excluded"),
                             ("f", included, "Border cells included")]:
        ax = axes[key]
        ax.boxplot(data, tick_labels=[str(c) for c in counts], widths=0.55, patch_artist=True,
                   boxprops={"facecolor": RANDOM_FILL, "edgecolor": RANDOM_INK},
                   medianprops={"color": INK, "linewidth": 1.5},
                   whiskerprops={"color": RANDOM_INK}, capprops={"color": RANDOM_INK},
                   flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "none",
                               "markeredgecolor": RANDOM_INK})
        ax.axhline(1.91, color=INK_2, linestyle="--", linewidth=1)
        ax.text(0.55, 1.95, "CSR theory 1.91", va="bottom", ha="left", fontsize=8,
                color=INK_2)
        means = [np.mean(d) for d in data]
        ax.plot(np.arange(1, len(counts) + 1), means, "o", color=INK, markersize=4,
                label="mean")
        ax.set_xlabel("Number of points in the field")
        ax.set_ylabel("NN regularity index")
        ax.set_ylim(0, 7)
        n_clipped = sum(np.sum(np.asarray(d) > 7) for d in data)
        ax.set_title(f"{title}: {n_sims} CSR simulations per count"
                     + (f" ({n_clipped} outliers > 7 not shown)" if n_clipped else ""))
        ax.legend(loc="upper right")
    axes["f"].sharey(axes["e"])

    for k in "abcdef":
        panel_label(axes[k], k)
    fig.suptitle("Fig. 1 — A regular mosaic vs complete spatial randomness (CSR)", x=0.02,
                 ha="left", fontsize=11)
    fig.tight_layout()
    save(fig, out, "fig01_csr_regularity_index")


def fig02(out, fast):
    """Paper Fig. 2: random simulations constrained by soma size, at two densities."""
    rng = np.random.default_rng(2)
    bounds = (0.0, 250.0, 0.0, 250.0)
    sizes = [5.0, 7.5, 10.0]
    fig = plt.figure(figsize=(14, 6.6))
    grid = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 0.15, 2.2])
    letters = iter("abcdef")
    for col, size in enumerate(sizes):
        for row, density in enumerate([2000, 4000]):
            ax = fig.add_subplot(grid[row, col])
            n = int(density * 0.0625)
            p, d = cm.simulate_random_mosaic(n, bounds, soma_diameter_mean=size,
                                             soma_diameter_sd=0.5, seed=rng)
            draw_field(ax, p, d, bounds, SIZE_RAMP[col],
                       scale_bar=50 if (row, col) == (1, 2) else None,
                       title=f"{size:g} µm, {density:,} cells/mm²")
            panel_label(ax, next(letters))

    ax = fig.add_subplot(grid[:, 4])
    densities = np.array([1000, 2000, 3000, 4000, 5000])
    n_sims = 20 if fast else 100
    for col, size in enumerate(sizes):
        quartiles = []
        for density in densities:
            ris = []
            for _ in range(n_sims):
                p, _ = cm.simulate_random_mosaic(int(density * 0.0625), bounds,
                                                 soma_diameter_mean=size, soma_diameter_sd=0.5,
                                                 seed=rng)
                mask = cm.nnd_border_mask(p, bounds)
                ris.append(cm.nnd_statistics(p, interior_mask=mask)["regularity_index"])
            quartiles.append(np.percentile(ris, [25, 50, 75]))
        quartiles = np.array(quartiles)
        ax.fill_between(densities, quartiles[:, 0], quartiles[:, 2], color=SIZE_RAMP[col],
                        alpha=0.25, linewidth=0)
        ax.plot(densities, quartiles[:, 1], "o-", color=SIZE_RAMP[col], markersize=5,
                label=f"{size:g} µm somata")
        ax.text(densities[-1] + 120, quartiles[-1, 1], f"{size:g} µm", va="center",
                fontsize=8.5, color=INK)
    ax.axhline(1.91, color=INK_2, linestyle="--", linewidth=1)
    ax.text(densities[0], 1.95, "CSR 1.91", fontsize=8, color=INK_2, va="bottom")
    ax.set_xlim(800, 5600)
    ax.set_xticks(densities)
    ax.set_xlabel("Density (cells/mm²)")
    ax.set_ylabel("NN regularity index (median, IQR)")
    ax.set_title(f"{n_sims} random simulations per point, 250 × 250 µm field")
    ax.legend(loc="upper left")
    panel_label(ax, "g")
    fig.suptitle("Fig. 2 — Soma size alone raises the regularity index of random mosaics, "
                 "increasingly with density", x=0.02, ha="left", fontsize=11)
    fig.tight_layout()
    save(fig, out, "fig02_soma_size_random_simulations")


def fig03(out, fast):
    """Paper Figs. 3 / 11: single fields vs 99 matched random simulations."""
    rng = np.random.default_rng(3)
    n_sims = 49 if fast else 99
    # Density and soma size of the four AII fields in the paper's Fig. 3.
    species = [("mouse", 3936, 7.2, 0.4), ("bat", 2221, 8.9, 0.7),
               ("rabbit", 1932, 8.8, 0.7), ("monkey", 1143, 8.5, 0.7)]
    bounds = (0.0, 200.0, 0.0, 200.0)

    fig = plt.figure(figsize=(14, 6.4))
    grid = fig.add_gridspec(2, 6, width_ratios=[1, 1, 1, 1, 0.1, 2.0])
    results = []
    for col, (label, density, size, sd) in enumerate(species):
        soma = {"soma_diameter_mean": size, "soma_diameter_sd": sd}
        n = int(density * 0.04)
        p, d = cm.simulate_random_mosaic(n, bounds, seed=rng, **soma)
        sim_p, sim_d = cm.simulate_random_mosaic(n, bounds, seed=rng, **soma)
        result = cm.compare_to_random_simulations(p, bounds, n_simulations=n_sims, seed=rng,
                                                  **soma)
        results.append((label, result))
        draw_field(fig.add_subplot(grid[0, col]), p, d, bounds, AII.color, voronoi=True,
                   scale_bar=50 if col == 3 else None,
                   title=f"{AII.name}, {label}\n{density:,}/mm², soma {size} ± {sd} µm")
        draw_field(fig.add_subplot(grid[1, col]), sim_p, sim_d, bounds, RANDOM_INK,
                   voronoi=True, title="matched random simulation")

    for row, (stat, ylabel) in enumerate([("nnd_regularity_index", "NN regularity index"),
                                          ("voronoi_regularity_index",
                                           "Voronoi regularity index")]):
        ax = fig.add_subplot(grid[row, 5])
        for x, (label, result) in enumerate(results):
            sims = result[stat]["simulated"]
            ax.scatter(x + rng.uniform(-0.18, 0.18, len(sims)), sims, s=10, color=RANDOM_FILL,
                       linewidths=0, label=f"{n_sims} random" if x == 0 else None)
            ax.scatter(x, result[stat]["observed"], s=70, marker="D", color=AII.color,
                       edgecolors=INK, linewidths=0.8, zorder=3,
                       label="real field" if x == 0 else None)
            ax.text(x, np.nanmax(sims) * 1.04, f"p = {result[stat]['p_greater']:.2f}",
                    ha="center", fontsize=7.5, color=INK_2)
        ax.set_xticks(range(len(results)), [r[0] for r in results])
        ax.set_ylabel(ylabel)
        ax.legend(loc="lower left")
        panel_label(ax, "i" if row == 0 else "j")
    fig.suptitle("Fig. 3 — Mosaics spaced only by soma size are indistinguishable from "
                 "soma-constrained random simulations", x=0.02, ha="left", fontsize=11)
    fig.tight_layout()
    save(fig, out, "fig03_monte_carlo_single_fields")


def fig04(out, fast):
    """Paper Fig. 4: autocorrelogram and density recovery profile."""
    rng = np.random.default_rng(4)
    n_fields = 8 if fast else 32
    real, random = sample_fields(HC, n_fields, rng)
    radius = HC.drp_kwargs["max_radius"]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.8))
    for row, (fields, color, label) in enumerate([(real, HC.color, HC.name),
                                                  (random, RANDOM_INK, "random")]):
        p, d = fields[0]["positions"], fields[0]["diameters"]
        ax = axes[row, 0]
        draw_field(ax, p, d, HC.bounds, color, scale_bar=50 if row else None,
                   title=f"{label} mosaic")
        centre = p[np.argmin(np.linalg.norm(p - 150.0, axis=1))]
        ax.add_patch(plt.Circle(centre, radius, fill=False, color=INK, linewidth=1))
        draw_autocorrelogram(axes[row, 1], p, radius, color,
                             title=f"autocorrelogram (radius {radius:.0f} µm)")
        draw_drp_bars(axes[row, 2], fields, color if row == 0 else RANDOM_FILL)
        axes[row, 2].set_title(f"DRP, {n_fields} fields, mean ± SEM")
    axes[1, 2].sharey(axes[0, 2])
    for ax, k in zip(axes.flat, "abcdef"):
        panel_label(ax, k)
    fig.suptitle("Fig. 4 — The exclusion zone of a regular mosaic exceeds the one soma size "
                 "imposes", x=0.02, ha="left", fontsize=11)
    fig.tight_layout()
    save(fig, out, "fig04_autocorrelogram_drp")


def fig05(out, fast):
    """Paper Fig. 5: a sparse mosaic, and G/L envelope tests of two spacing rules."""
    rng = np.random.default_rng(5)
    n_fields = 6 if fast else 16
    n_sims = 49 if fast else 99
    real, random = sample_fields(DA, n_fields, rng)
    models = {
        f"soma size {DA.soma_mean:g} ± {DA.soma_sd:g} µm": dict(DA.soma),
        f"d_min {DA.dmin_mean:g} ± {DA.dmin_sd:g} µm": dict(dmin_mean=DA.dmin_mean,
                                                           dmin_sd=DA.dmin_sd, **DA.soma),
    }
    grids = {"G": np.linspace(0, 300, 61), "L": np.linspace(0, 375, 76)}

    p_values = {(f, m): [] for f in grids for m in models}
    examples = {}
    for i, field in enumerate(real):
        for function, r in grids.items():
            for model_name, model in models.items():
                env = cm.envelope_test(field["positions"], DA.bounds, function, r,
                                       n_simulations=n_sims, seed=rng, **model)
                p_values[(function, model_name)].append(env["p_value"])
                if i == 0:
                    examples[(function, model_name)] = env

    fig, axes = plt.subplot_mosaic(
        [["a", "b", "c", "d"], ["e", "f", "i", "i"], ["g", "h", "j", "j"]],
        figsize=(14, 11.5))
    draw_field(axes["a"], real[0]["positions"], real[0]["diameters"], DA.bounds, DA.color,
               title=f"{DA.name} mosaic ({DA.density:g} cells/mm²)")
    draw_field(axes["b"], random[0]["positions"], random[0]["diameters"], DA.bounds,
               RANDOM_INK, scale_bar=250, title="matched random (soma size)")
    # Somata are tiny at this scale; overdraw positions so they stay visible.
    for key, fields, color in [("a", real, DA.color), ("b", random, RANDOM_INK)]:
        axes[key].scatter(*fields[0]["positions"].T, s=9, color=color, linewidths=0)

    draw_distribution_pair(axes["c"], [f["nnd"] for f in real], [f["nnd"] for f in random],
                           np.arange(0, 425, 25), DA.color, "Nearest-neighbour distance (µm)")
    axes["c"].legend(loc="lower right")
    inset = axes["c"].inset_axes([0.62, 0.45, 0.36, 0.5])
    draw_bar_pair(inset, stat_values(real, "nnd_regularity_index"),
                  stat_values(random, "nnd_regularity_index"), DA.color, "NNRI", annotate=False)
    inset.tick_params(labelsize=7)
    inset.yaxis.label.set_size(7)

    r = real[0]["drp_r"]
    draw_mean_sem(axes["d"], r, [f["drp_density"] for f in random], RANDOM_INK, "random",
                  fill=RANDOM_FILL)
    draw_mean_sem(axes["d"], r, [f["drp_density"] for f in real], DA.color, "real")
    axes["d"].axhline(DA.density, color=INK, linewidth=1, linestyle="--", label="mean density")
    axes["d"].set_xlabel("Distance (µm)")
    axes["d"].set_ylabel("Density (cells/mm²)")
    axes["d"].set_title(f"DRP, {n_fields} fields, mean ± SEM")
    axes["d"].legend(loc="lower right")

    for function, keys in [("G", "ef"), ("L", "gh")]:
        for key, model_name in zip(keys, models):
            ax = axes[key]
            env = examples[(function, model_name)]
            cm.plot_envelope(env, ax=ax, color=DA.color, envelope_color=RANDOM_INK)
            ax.collections[0].set_color(RANDOM_FILL)
            ax.collections[0].set_alpha(0.7)
            ax.set_xlabel("Distance (µm)")
            ax.set_ylabel(f"{function}(r)")
            verdict = "fits" if env["p_value"] > 0.05 else "rejected"
            ax.set_title(f"{function}, model: {model_name}\np = {env['p_value']:.2f} "
                         f"({verdict}; {n_sims} simulations)")
            ax.legend(loc="lower right" if function == "G" else "upper left")
        axes[keys[1]].sharey(axes[keys[0]])

    for function, key in [("G", "i"), ("L", "j")]:
        ax = axes[key]
        fractions = [np.mean(np.array(p_values[(function, m)]) > 0.05) for m in models]
        ax.bar(range(len(models)), fractions, width=0.45, color=DA.color)
        for x, frac in enumerate(fractions):
            ax.text(x, frac + 0.02, f"{frac:.0%}", ha="center", va="bottom", fontsize=8.5)
        ax.set_xticks(range(len(models)), list(models))
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Fraction of fields fit (p > 0.05)")
        ax.set_title(f"{function} function: {n_fields} fields, {n_sims} simulations each")

    for k in "abcdefghij":
        panel_label(axes[k], k)
    fig.suptitle("Fig. 5 — A sparse mosaic is fit by a spacing rule far larger than soma size",
                 x=0.02, ha="left", fontsize=11)
    fig.tight_layout()
    save(fig, out, "fig05_sparse_mosaic_envelopes")


def fig06(out, fast):
    """Paper Fig. 6: uniform expansion enlarges the exclusion zone, not the regularity."""
    rng = np.random.default_rng(6)
    bounds = (0.0, 400.0, 0.0, 400.0)
    p, d = cm.simulate_random_mosaic(480, bounds, soma_diameter_mean=10.0, soma_diameter_sd=1.0,
                                     seed=rng)
    factor = 1.5
    expanded_bounds = (0.0, 400.0 * factor, 0.0, 400.0 * factor)
    expanded = p * factor
    drp_kwargs = {"n_bins": 20, "max_radius": 50.0}

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.8))
    rows = [(p, d, bounds, "random mosaic, somata 10 ± 1 µm"),
            (expanded, d, expanded_bounds, f"same mosaic, expanded ×{factor:g}")]
    for row, (pos, diam, bnds, title) in enumerate(rows):
        stats = cm.regularity_statistics(pos, bnds, drp_kwargs)
        r, drp = cm.density_recovery_profile(pos, bnds, edge_correction=True, **drp_kwargs)
        view = (bnds[1] / 2 - 200, bnds[1] / 2 + 200, bnds[3] / 2 - 200, bnds[3] / 2 + 200)
        draw_field(axes[row, 0], pos, diam, view, RANDOM_INK, scale_bar=50,
                   title=title + (" (centre)" if row else ""))
        draw_autocorrelogram(axes[row, 1], pos, 50.0, RANDOM_INK, title="autocorrelogram (50 µm)")
        ax = axes[row, 2]
        ax.bar(r, drp * stats["density"] * 1e6, width=(r[1] - r[0]) * 0.9, color=RANDOM_FILL)
        ax.axhline(stats["density"] * 1e6, color=INK, linewidth=1, linestyle="--",
                   label="mean density")
        ax.axvline(stats["effective_radius"], color=INK_2, linewidth=1.2, linestyle=":",
                   label=f"ER = {stats['effective_radius']:.1f} µm")
        ax.set_xlabel("Distance (µm)")
        ax.set_ylabel("Density (cells/mm²)")
        ax.set_title(f"NNRI {stats['nnd_regularity_index']:.2f}, "
                     f"VDRI {stats['voronoi_regularity_index']:.2f}, "
                     f"PF {stats['packing_factor']:.2f}")
        ax.legend(loc="lower right")
    for ax, k in zip(axes.flat, "abcdef"):
        panel_label(ax, k)
    fig.suptitle("Fig. 6 — Identical patterning, different exclusion zones: the effective "
                 "radius is not a regularity measure", x=0.02, ha="left", fontsize=11)
    fig.tight_layout()
    save(fig, out, "fig06_expansion_effective_radius")


def fig07(out, fast):
    """Paper Fig. 7: a displaced mosaic loses NN regularity before Voronoi regularity."""
    rng = np.random.default_rng(7)
    n_fields = 8 if fast else 16
    data = {ct.name: sample_fields(ct, n_fields, rng) for ct in (SAC, SAC_GCL)}
    labels = {SAC.name: "INL (in place)", SAC_GCL.name: f"GCL (displaced {SAC_GCL.displacement_sd:g} µm)"}

    fig, axes = plt.subplots(3, 4, figsize=(14, 10.5))
    for col, ct in enumerate((SAC, SAC_GCL)):
        real, random = data[ct.name]
        draw_field(axes[0, col], real[0]["positions"], real[0]["diameters"], ct.bounds,
                   ct.color, voronoi=True, title=f"{SAC.name}, {labels[ct.name]}")
        draw_field(axes[0, col + 2], random[0]["positions"], random[0]["diameters"], ct.bounds,
                   RANDOM_INK, voronoi=True, scale_bar=50 if col else None,
                   title=f"matched random for {labels[ct.name].split(' ')[0]}")
        draw_bar_pair(axes[1, col], stat_values(real, "nnd_regularity_index"),
                      stat_values(random, "nnd_regularity_index"), ct.color,
                      f"NNRI — {labels[ct.name].split(' ')[0]}")
        draw_bar_pair(axes[1, col + 2], stat_values(real, "voronoi_regularity_index"),
                      stat_values(random, "voronoi_regularity_index"), ct.color,
                      f"VDRI — {labels[ct.name].split(' ')[0]}")
        draw_distribution_pair(axes[2, col], [f["nnd"] for f in real],
                               [f["nnd"] for f in random], np.arange(0, 47.5, 2.5), ct.color,
                               "Nearest-neighbour distance (µm)")
        draw_distribution_pair(axes[2, col + 2], [f["areas"] for f in real],
                               [f["areas"] for f in random], np.arange(0, 2550, 150), ct.color,
                               "Voronoi domain area (µm²)")
    for ax in axes[2]:
        ax.legend(loc="upper right")
    axes[1, 1].sharey(axes[1, 0])
    axes[1, 3].sharey(axes[1, 2])
    for ax, k in zip(axes.flat, "abcdefghijkl"):
        panel_label(ax, k)
    fig.suptitle("Fig. 7 — Lateral displacement erodes the nearest-neighbour regularity "
                 "before the Voronoi regularity", x=0.02, ha="left", fontsize=11)
    fig.tight_layout()
    save(fig, out, "fig07_nnri_vs_vdri")


def fig08(out, fast, cache):
    """Paper Fig. 8: from random to regular, six cell types."""
    n_fields = 4 if fast else 8
    columns = ["field", "random", "NND", "NNRI", "VD", "VDRI", "DRP", "ER", "PF"]
    fig, axes = plt.subplots(len(TYPES), len(columns), figsize=(27, 3.0 * len(TYPES)),
                             gridspec_kw={"width_ratios": [1, 1, 1.25, 0.8, 1.25, 0.8, 1.25,
                                                           0.8, 0.8]})
    for row, ct in enumerate(TYPES):
        real, random = cache.setdefault(
            ct.name, sample_fields(ct, n_fields, np.random.default_rng(80 + row), 0.25))
        draw_field(axes[row, 0], real[0]["positions"], real[0]["diameters"], ct.bounds,
                   ct.color, voronoi=True, scale_bar=nice_length(ct.field / 5))
        draw_field(axes[row, 1], random[0]["positions"], random[0]["diameters"], ct.bounds,
                   RANDOM_INK, voronoi=True)
        if ct is DA:
            for col, fields, color in [(0, real, ct.color), (1, random, RANDOM_INK)]:
                axes[row, col].scatter(*fields[0]["positions"].T, s=7, color=color, linewidths=0)
        axes[row, 0].set_ylabel(ct.name, fontsize=11, color=INK, labelpad=8)

        spacing = ct.spacing
        draw_distribution_pair(axes[row, 2], [f["nnd"] for f in real],
                               [f["nnd"] for f in random], np.linspace(0, 1.4 * spacing, 21),
                               ct.color, "NND (µm)")
        mean_area = 1e6 / ct.density
        draw_distribution_pair(axes[row, 4], [f["areas"] for f in real],
                               [f["areas"] for f in random], np.linspace(0, 2.2 * mean_area, 21),
                               ct.color, "Voronoi area (µm²)")
        for col, stat, label in [(3, "nnd_regularity_index", "NNRI"),
                                 (5, "voronoi_regularity_index", "VDRI"),
                                 (7, "effective_radius", "ER (µm)"),
                                 (8, "packing_factor", "Packing factor")]:
            draw_bar_pair(axes[row, col], stat_values(real, stat), stat_values(random, stat),
                          ct.color, label)
        r = real[0]["drp_r"]
        ax = axes[row, 6]
        draw_mean_sem(ax, r, [f["drp_density"] for f in random], RANDOM_INK, "random",
                      fill=RANDOM_FILL)
        draw_mean_sem(ax, r, [f["drp_density"] for f in real], ct.color, "real")
        ax.axhline(np.mean(stat_values(real, "density")) * 1e6, color=INK, linewidth=1,
                   linestyle="--")
        ax.set_xlabel("Distance (µm)")
        ax.set_ylabel("cells/mm²")
        if row == 0:
            for col in (2, 4, 6):
                axes[row, col].legend(loc="upper right")

    titles = ["Real field + Voronoi", "Matched random", "NND distribution", "NN reg. index",
              "Voronoi area distribution", "Voronoi reg. index", "Density recovery profile",
              "Effective radius", "Packing factor"]
    for col, title in enumerate(titles):
        axes[0, col].annotate(title, (0.5, 1.22), xycoords="axes fraction", ha="center",
                              fontsize=10, fontweight="bold")
    fig.suptitle(f"Fig. 8 — From random to regular: six synthetic cell types, {n_fields} fields "
                 "each, every field with one matched soma-constrained random simulation "
                 "(mean ± SEM; dots = fields)", x=0.01, ha="left", fontsize=12, y=1.0)
    fig.tight_layout()
    save(fig, out, "fig08_random_to_regular")


def fig09(out, fast, cache):
    """Paper Fig. 9: each statistic against field density, per cell type."""
    n_fields = 4 if fast else 8
    for row, ct in enumerate(TYPES):
        cache.setdefault(ct.name, sample_fields(ct, n_fields, np.random.default_rng(80 + row), 0.25))

    # DA-like is left out: at ~40 cells/mm2 it would compress the density axis,
    # and its effective radius is unreliable anyway (see Fig. 8).
    groups = [("Spacing rule beyond soma size", [HC, SAC, VGLUT3]),
              ("Spaced by soma size only", [AII, BC])]
    stats = [("nnd_regularity_index", "NN regularity index"),
             ("voronoi_regularity_index", "Voronoi regularity index"),
             ("effective_radius", "Effective radius (µm)"),
             ("packing_factor", "Packing factor")]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for row, (group_title, types) in enumerate(groups):
        for col, (stat, ylabel) in enumerate(stats):
            ax = axes[row, col]
            for ct in types:
                real, _ = cache[ct.name]
                density = stat_values(real, "density") * 1e6
                values = stat_values(real, stat)
                ax.scatter(density, values, s=30, color=ct.color, edgecolors=SURFACE,
                           linewidths=0.8, label=ct.name, zorder=3)
                if len(density) > 2:
                    slope, intercept = np.polyfit(density, values, 1)
                    xs = np.linspace(density.min(), density.max(), 10)
                    ax.plot(xs, slope * xs + intercept, color=ct.color, linewidth=1.5)
            ax.set_xlabel("Density (cells/mm²)")
            ax.set_ylabel(ylabel)
            if col == 0:
                ax.set_title(group_title, loc="left", fontweight="bold")
        axes[row, 0].legend(loc="best")
    for ax, k in zip(axes.flat, "abcdefgh"):
        panel_label(ax, k)
    fig.suptitle("Fig. 9 — Regular mosaics hold regularity and packing constant across density "
                 "by scaling their spacing; soma-limited mosaics do the opposite",
                 x=0.02, ha="left", fontsize=11)
    fig.tight_layout()
    save(fig, out, "fig09_statistics_vs_density")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parent.parent / "figures" / "keeley2020")
    parser.add_argument("--fast", action="store_true",
                        help="fewer fields and simulations, for a quick look")
    parser.add_argument("--only", type=int, nargs="+", metavar="N",
                        help="figure numbers to make (default: all)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cache: dict = {}
    figures = {1: fig01, 2: fig02, 3: fig03, 4: fig04, 5: fig05, 6: fig06, 7: fig07,
               8: lambda out, fast: fig08(out, fast, cache),
               9: lambda out, fast: fig09(out, fast, cache)}
    for number in args.only or sorted(figures):
        start = time.time()
        print(f"Fig. {number}...")
        figures[number](args.out, args.fast)
        print(f"  {time.time() - start:.1f} s")


if __name__ == "__main__":
    main()
