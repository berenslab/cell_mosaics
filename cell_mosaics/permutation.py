"""Label-permutation tests for mosaic independence.

Tests whether subgroups defined by a label array are more spatially regular
than a random same-size subset of the pooled population would be -- the
standard evidence (Rockhill et al. 2000; Cook 1996) that two putative
cell-type mosaics are each independently, separately regular rather than an
arbitrary split of one combined mosaic. The same machinery, applied to a
relabeled array that merges two candidate groups into one, tests whether a
third, ambiguous group (e.g. a "displaced" variant) is more consistent with
belonging to one parent mosaic than the other.
"""

from __future__ import annotations

import numpy as np

from .metrics import nnd_statistics, voronoi_analysis

STATISTICS = {
    "regularity_index": {"higher_is_more_regular": True, "min_n": 3},
    "cv_area": {"higher_is_more_regular": False, "min_n": 4},
}


def _group_statistic(positions, interior_mask, statistic, field_bounds):
    min_n = STATISTICS[statistic]["min_n"]
    if interior_mask.sum() < min_n or len(positions) < min_n:
        return np.nan
    if statistic == "regularity_index":
        return nnd_statistics(positions, interior_mask=interior_mask)["regularity_index"]
    if statistic == "cv_area":
        return voronoi_analysis(positions, field_bounds=field_bounds, interior_mask=interior_mask)["cv_area"]
    raise ValueError(f"Unknown statistic {statistic!r}; choose from {list(STATISTICS)}")


def label_permutation_test(
    positions: np.ndarray,
    labels: np.ndarray,
    groups: list | None = None,
    statistic: str = "regularity_index",
    interior_mask: np.ndarray | None = None,
    n_permutations: int = 2000,
    field_bounds: tuple[float, float, float, float] | None = None,
    seed: int | None = None,
) -> dict:
    """Test whether real subgroups in `labels` are more regular than chance.

    The null model shuffles `labels` across all `positions` (preserving each
    group's size), so a group's null distribution is what its regularity
    statistic would look like if its members were an arbitrary random subset
    of the full pooled population rather than a real, distinct mosaic.

    To test whether an ambiguous group belongs with one candidate parent
    mosaic rather than another, relabel positions before calling this
    (e.g. give the ambiguous cells the same label as candidate parent A in
    one call, candidate parent B in another, both drawn from the same pooled
    universe) and compare the two results.

    Parameters
    ----------
    positions : np.ndarray of shape (n, 2)
        All candidate soma positions in the pooled universe being shuffled
        across (e.g. every soma of every subtype under consideration).
    labels : np.ndarray of shape (n,)
        Group label per position.
    groups : list, optional
        Which label values to test. Defaults to every unique label in
        `labels`.
    statistic : {'regularity_index', 'cv_area'}
        'regularity_index' -- NND mean/std (higher = more regular).
        'cv_area' -- Voronoi domain-area CV (lower = more regular).
    interior_mask : np.ndarray of bool, shape (n,), optional
        Restrict the summary statistic to this subset of `positions` (e.g. a
        stable-sampling crop box) while still computing nearest
        neighbors/Voronoi adjacency against every position in `positions`,
        so boundary cells aren't penalized for neighbors excluded by the
        mask. Defaults to using every position.
    n_permutations : int
    field_bounds : tuple, optional
        Forwarded to `voronoi_analysis`; unused (and unnecessary) when an
        explicit `interior_mask` is given.
    seed : int, optional

    Returns
    -------
    dict
        `{group: {'observed': float, 'null': np.ndarray, 'p_value': float}}`.
        `p_value` is one-sided: the fraction of permutations at least as
        regular as the observed value (>= for `regularity_index`, <= for
        `cv_area`) -- a small p-value means the real group is more regular
        than chance. `np.nan` where a group (real or permuted) has fewer
        interior cells than the statistic requires.
    """
    if statistic not in STATISTICS:
        raise ValueError(f"Unknown statistic {statistic!r}; choose from {list(STATISTICS)}")

    positions = np.asarray(positions, dtype=float)
    labels = np.asarray(labels)
    if len(positions) != len(labels):
        raise ValueError("positions and labels must have the same length")

    if interior_mask is None:
        interior_mask = np.ones(len(positions), dtype=bool)
    else:
        interior_mask = np.asarray(interior_mask, dtype=bool)

    if groups is None:
        groups = list(np.unique(labels))

    rng = np.random.default_rng(seed)
    higher_is_better = STATISTICS[statistic]["higher_is_more_regular"]

    results = {}
    for group in groups:
        mask = labels == group
        observed = _group_statistic(positions[mask], interior_mask[mask], statistic, field_bounds)

        null = np.empty(n_permutations)
        for p in range(n_permutations):
            perm_mask = rng.permutation(mask)
            null[p] = _group_statistic(positions[perm_mask], interior_mask[perm_mask], statistic, field_bounds)

        valid = ~np.isnan(null)
        if np.isnan(observed) or not valid.any():
            p_value = np.nan
        elif higher_is_better:
            p_value = float(np.mean(null[valid] >= observed))
        else:
            p_value = float(np.mean(null[valid] <= observed))

        results[group] = {"observed": observed, "null": null, "p_value": p_value}

    return results
