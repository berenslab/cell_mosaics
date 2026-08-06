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
from scipy.spatial import KDTree

from .metrics import nnd_statistics, voronoi_analysis

STATISTICS = {
    "regularity_index": {"higher_is_more_regular": True, "min_n": 3},
    "cv_area": {"higher_is_more_regular": False, "min_n": 4},
}


def _permute(rng, mask, strata=None):
    """Permute `mask`, independently within each stratum if `strata` is given.

    A global permutation assumes the cells are spatially homogeneous. When
    sampling density varies across the field (e.g. a densely proofread centre
    and a sparse periphery), it draws permuted group members from wherever
    cells are densest, so the null describes a different neighbourhood than
    the real group occupies. Permuting only within strata (e.g. blocks of a
    coarse spatial grid) holds each group's spatial distribution roughly
    fixed, isolating spatial *arrangement* from where the cells happen to lie.
    """
    if strata is None:
        return rng.permutation(mask)

    permuted = mask.copy()
    for stratum in np.unique(strata):
        in_stratum = strata == stratum
        permuted[in_stratum] = rng.permutation(mask[in_stratum])
    return permuted


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
    strata: np.ndarray | None = None,
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
    strata : np.ndarray of shape (n,), optional
        Stratum id per position. Labels are then permuted only *within* each
        stratum, so the null preserves each group's coarse spatial
        distribution -- use this when sampling density varies across the
        field, otherwise the null is drawn from wherever cells are densest.
        Defaults to permuting globally.

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

    if strata is not None:
        strata = np.asarray(strata)
        if len(strata) != len(positions):
            raise ValueError("strata must have the same length as positions")

    rng = np.random.default_rng(seed)
    higher_is_better = STATISTICS[statistic]["higher_is_more_regular"]

    results = {}
    for group in groups:
        mask = labels == group
        observed = _group_statistic(positions[mask], interior_mask[mask], statistic, field_bounds)

        null = np.empty(n_permutations)
        for p in range(n_permutations):
            perm_mask = _permute(rng, mask, strata)
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


def cross_nnd_permutation_test(
    positions: np.ndarray,
    labels: np.ndarray,
    focal_group,
    parent_group,
    interior_mask: np.ndarray | None = None,
    n_permutations: int = 2000,
    seed: int | None = None,
    strata: np.ndarray | None = None,
) -> dict:
    """Test whether `focal_group` cells respect `parent_group`'s exclusion zone.

    Statistic: the mean distance from each focal soma to the *nearest* parent
    soma. The null shuffles which pooled (focal + parent) cells carry the
    focal label, preserving the focal count.

    This asks directly whether the focal cells behave like members of the
    parent mosaic, and -- unlike testing a merged group's overall regularity
    -- its answer is driven by the focal cells themselves rather than by
    however many parent cells they are pooled with. That makes it the
    appropriate test when the focal group is much smaller than the parent
    (where a merged-group regularity test just reports the parent's own
    regularity).

    Interpretation:

    - **Same mosaic.** If the focal cells are members of the parent's mosaic,
      they sit in its gaps and keep its characteristic spacing, exactly like
      a randomly relabelled subset of the pooled mosaic would. Observed
      lands mid-null, so `p_closer` is unremarkable (≈0.5).
    - **Independent types.** If the focal cells belong to a different type,
      nothing stops them sitting arbitrarily close to parent somas, so the
      observed mean lands below the null: `p_closer` is small.

    So a *small* `p_closer` is evidence **against** the focal cells belonging
    to the parent mosaic. Note that an unremarkable `p_closer` is a failure
    to reject rather than positive proof of membership.

    Parameters
    ----------
    positions : np.ndarray of shape (n, 2)
    labels : np.ndarray of shape (n,)
    focal_group, parent_group
        Label values selecting the two sets. Only cells carrying one of these
        two labels take part; the rest of `positions` is ignored.
    interior_mask : np.ndarray of bool, shape (n,), optional
        Restricts which *focal* cells are evaluated (e.g. to a
        stable-sampling crop). Nearest-parent lookups always search the full
        parent set, so a focal cell near the crop edge can still match a
        parent soma outside it. Defaults to using every focal cell.
    n_permutations : int
    seed : int, optional
    strata : np.ndarray of shape (n,), optional
        Stratum id per position; the focal label is then permuted only within
        each stratum. Strongly recommended here whenever sampling density is
        uneven across the field: with a global null, a sparsely sampled focal
        group is compared against pseudo-focal cells drawn from the dense
        centre, which inflates the observed distance relative to the null for
        *both* candidate parents and buries the membership signal.

    Returns
    -------
    dict with keys
        observed : float — mean nearest-parent distance for the real labels
        null : np.ndarray of shape (n_permutations,)
        p_closer : float — fraction of null <= observed (small = focal cells
            sit closer to parent cells than chance, i.e. independent types)
        p_farther : float — fraction of null >= observed
        n_focal_evaluated : int
    """
    positions = np.asarray(positions, dtype=float)
    labels = np.asarray(labels)
    if len(positions) != len(labels):
        raise ValueError("positions and labels must have the same length")

    if interior_mask is None:
        interior_mask = np.ones(len(positions), dtype=bool)
    else:
        interior_mask = np.asarray(interior_mask, dtype=bool)

    pool = (labels == focal_group) | (labels == parent_group)
    pool_positions = positions[pool]
    pool_interior = interior_mask[pool]
    is_focal = labels[pool] == focal_group

    if strata is not None:
        strata = np.asarray(strata)
        if len(strata) != len(positions):
            raise ValueError("strata must have the same length as positions")
        strata = strata[pool]

    if not is_focal.any():
        raise ValueError(f"no cells carry focal_group label {focal_group!r}")
    if is_focal.all():
        raise ValueError(f"no cells carry parent_group label {parent_group!r}")

    def _mean_nearest_parent(focal_selection):
        focal_pts = pool_positions[focal_selection & pool_interior]
        parent_pts = pool_positions[~focal_selection]
        if len(focal_pts) == 0 or len(parent_pts) == 0:
            return np.nan
        distances, _ = KDTree(parent_pts).query(focal_pts, k=1)
        return float(np.mean(distances))

    observed = _mean_nearest_parent(is_focal)

    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations)
    for p in range(n_permutations):
        null[p] = _mean_nearest_parent(_permute(rng, is_focal, strata))

    valid = ~np.isnan(null)
    if np.isnan(observed) or not valid.any():
        p_closer = p_farther = np.nan
    else:
        p_closer = float(np.mean(null[valid] <= observed))
        p_farther = float(np.mean(null[valid] >= observed))

    return {
        "observed": observed,
        "null": null,
        "p_closer": p_closer,
        "p_farther": p_farther,
        "n_focal_evaluated": int((is_focal & pool_interior).sum()),
    }
