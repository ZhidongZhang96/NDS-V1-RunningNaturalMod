"""Audit / diagnostic tooling for the binary running-modulation index.

This module is intentionally decoupled from the formal analysis in
``utils.py``. The formal :class:`utils.BinaryModulation` exposes a single
modulation index — the **sign-safe** MI stored in ``analysis.mi`` — and the
formal notebook (``visual_coding_B.ipynb``) never touches the raw or
denominator-thresholded ("robust") variants.

Everything that motivated *choosing* the sign-safe metric lives here instead:

- the raw MI ``(R_run - R_still) / (R_run + R_still)`` and its instabilities
  (denominator near zero / negative, |MI| > 1, sign reversal);
- the "robust" raw MI, i.e. raw MI restricted to cells whose denominator
  clears a magnitude threshold;
- side-by-side raw / robust / sign-safe comparisons and the paired
  grating-vs-natural and Allen-metadata versions built on raw/robust MI;
- the negative-response structure diagnostics (example traces, PCA over
  trial traces, trace heatmap).

Reproduce a batch of fitted analyses with
``utils.run_binary_modulation_analysis`` and pass the resulting ``results``
dict to the functions below.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, wilcoxon
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ==============================================================================
# Raw and robust modulation index (diagnostic only)
# ==============================================================================


def compute_raw_mi(analysis):
    """Recompute the raw modulation index from an analysis' mean responses.

    .. math::
        MI_{\\text{raw}} = \\frac{R_{\\text{run}} - R_{\\text{still}}}
                                 {R_{\\text{run}} + R_{\\text{still}}}

    The formal :meth:`utils.BinaryModulation.compute_mi` now stores the
    sign-safe MI in ``analysis.mi``; this helper reconstructs the historical
    raw MI from the (unchanged) ``r_run`` / ``r_still`` so the audit can study
    its instabilities. A cell is left as ``NaN`` where the denominator is
    non-finite or within ``1e-12`` of zero, matching the original definition.

    Parameters
    ----------
    analysis : utils.BinaryModulation
        Analysis with :meth:`compute_mi` already run.

    Returns
    -------
    mi_raw : np.ndarray, shape (n_cells,)
    """
    r_run = np.asarray(analysis.r_run, dtype=float)
    r_still = np.asarray(analysis.r_still, dtype=float)

    denom = r_run + r_still
    mi_raw = np.full(r_run.shape, np.nan, dtype=float)
    valid = np.isfinite(denom) & (np.abs(denom) > 1e-12)
    mi_raw[valid] = (r_run[valid] - r_still[valid]) / denom[valid]
    return mi_raw


def get_robust_mi(analysis, denom_threshold: float = 1e-3):
    """Return raw MI together with a robust-cell mask.

    A cell is "robust" if its raw MI is finite, its denominator
    (:math:`R_{\\text{run}} + R_{\\text{still}}`) is finite, and
    :math:`|R_{\\text{run}} + R_{\\text{still}}| > \\text{denom\\_threshold}`.
    This is a *filter*, not a normalization of MI itself.

    Parameters
    ----------
    analysis : utils.BinaryModulation
        Analysis with :meth:`compute_mi` already run.
    denom_threshold : float, optional
        Minimum allowed ``|R_run + R_still|``, by default ``1e-3``.

    Returns
    -------
    mi : np.ndarray, shape (n_cells,)
        Raw modulation index.
    robust : np.ndarray of bool, shape (n_cells,)
        Mask of cells passing the robustness filter.
    """
    mi = compute_raw_mi(analysis)
    denom = np.asarray(analysis.r_run, dtype=float) + np.asarray(analysis.r_still, dtype=float)

    robust = (
        np.isfinite(mi)
        & np.isfinite(denom)
        & (np.abs(denom) > denom_threshold)
    )
    return mi, robust


def summarize_mi_by_stimulus(results: dict, denom_threshold: float = 1e-3) -> pd.DataFrame:
    """Per-stimulus summary of raw vs. robust modulation index.

    Parameters
    ----------
    results : dict
        Mapping stimulus -> :class:`utils.BinaryModulation`.
    denom_threshold : float, optional
        Passed to :func:`get_robust_mi`, by default ``1e-3``.

    Returns
    -------
    pandas.DataFrame
        Columns: stimulus, n_cells, n_robust, n_excluded, median_MI_raw,
        median_MI_robust, mean_MI_robust, median_delta_R,
        frac_positive_MI_robust.
    """
    rows = []
    for stimulus, analysis in results.items():
        mi, robust = get_robust_mi(analysis, denom_threshold)
        finite = np.isfinite(mi)
        delta = np.asarray(analysis.r_run, dtype=float) - np.asarray(analysis.r_still, dtype=float)

        rows.append({
            "stimulus": stimulus,
            "n_cells": len(mi),
            "n_robust": int(robust.sum()),
            "n_excluded": int(finite.sum() - robust.sum()),
            "median_MI_raw": float(np.nanmedian(mi[finite])),
            "median_MI_robust": float(np.nanmedian(mi[robust])),
            "mean_MI_robust": float(np.nanmean(mi[robust])),
            "median_delta_R": float(np.nanmedian(delta[robust])),
            "frac_positive_MI_robust": float(np.mean(mi[robust] > 0)),
        })
    return pd.DataFrame(rows)


def summarize_mi_versions(results: dict, epsilon: float = 1e-12) -> pd.DataFrame:
    """Per-stimulus comparison of raw MI, sign-safe MI, and delta_R.

    Raw MI is reconstructed via :func:`compute_raw_mi`; the sign-safe MI is
    read directly from the formal ``analysis.mi``; ``delta_R`` is
    ``r_run - r_still``. This table quantifies exactly the failure modes that
    motivated dropping raw MI as the formal metric.

    Parameters
    ----------
    results : dict
        Mapping stimulus -> :class:`utils.BinaryModulation`.
    epsilon : float, optional
        Retained for signature compatibility with the formal sign-safe
        constant, by default ``1e-12``.

    Returns
    -------
    pandas.DataFrame
        Columns: stimulus, n_cells, n_negative_raw_denominator,
        n_raw_sign_reversal, n_abs_raw_mi_gt_1, median_raw_mi,
        median_safe_mi, median_delta_R, fraction_positive_raw_mi,
        fraction_positive_safe_mi, fraction_positive_delta_R.
    """
    rows = []
    for stimulus, analysis in results.items():
        r_run = np.asarray(analysis.r_run, dtype=float)
        r_still = np.asarray(analysis.r_still, dtype=float)
        mi_raw = compute_raw_mi(analysis)
        mi_safe = np.asarray(analysis.mi, dtype=float)
        delta_r = r_run - r_still

        raw_denom = r_run + r_still
        negative_raw_denominator = np.isfinite(raw_denom) & (raw_denom < 0)

        finite_raw = np.isfinite(mi_raw) & np.isfinite(delta_r)
        sign_reversal = np.zeros(len(mi_raw), dtype=bool)
        sign_reversal[finite_raw] = (
            np.sign(mi_raw[finite_raw]) != np.sign(delta_r[finite_raw])
        )

        abs_mi_gt_1 = np.isfinite(mi_raw) & (np.abs(mi_raw) > 1)

        finite_mi_raw = mi_raw[np.isfinite(mi_raw)]
        finite_mi_safe = mi_safe[np.isfinite(mi_safe)]
        finite_delta_r = delta_r[np.isfinite(delta_r)]

        rows.append({
            "stimulus": stimulus,
            "n_cells": len(mi_raw),
            "n_negative_raw_denominator": int(negative_raw_denominator.sum()),
            "n_raw_sign_reversal": int(sign_reversal.sum()),
            "n_abs_raw_mi_gt_1": int(abs_mi_gt_1.sum()),
            "median_raw_mi": float(np.nanmedian(finite_mi_raw)) if finite_mi_raw.size else float("nan"),
            "median_safe_mi": float(np.nanmedian(finite_mi_safe)) if finite_mi_safe.size else float("nan"),
            "median_delta_R": float(np.nanmedian(finite_delta_r)) if finite_delta_r.size else float("nan"),
            "fraction_positive_raw_mi": float(np.mean(finite_mi_raw > 0)) if finite_mi_raw.size else float("nan"),
            "fraction_positive_safe_mi": float(np.mean(finite_mi_safe > 0)) if finite_mi_safe.size else float("nan"),
            "fraction_positive_delta_R": float(np.mean(finite_delta_r > 0)) if finite_delta_r.size else float("nan"),
        })

    return pd.DataFrame(rows)


def compare_gratings_vs_natural(results: dict, denom_threshold: float = 1e-3):
    """Paired grating-vs-natural-scenes comparison of **robust raw** MI.

    Historical/diagnostic counterpart of
    :func:`utils.compare_gratings_vs_natural`. For each matched cell,
    grating MI is the mean of the (robust) drifting- and static-grating raw
    MI, compared against natural-scenes raw MI with a paired Wilcoxon
    signed-rank test.

    Returns
    -------
    result_df : pandas.DataFrame
    valid : np.ndarray of bool, shape (n_cells,)
    grating_values : np.ndarray
    natural_values : np.ndarray
    """
    mi_dg, robust_dg = get_robust_mi(results["drifting_gratings"], denom_threshold)
    mi_sg, robust_sg = get_robust_mi(results["static_gratings"], denom_threshold)
    mi_ns, robust_ns = get_robust_mi(results["natural_scenes"], denom_threshold)

    mi_grating = np.nanmean(np.vstack([mi_dg, mi_sg]), axis=0)
    robust_grating = robust_dg & robust_sg & np.isfinite(mi_grating)

    valid = robust_grating & robust_ns & np.isfinite(mi_grating) & np.isfinite(mi_ns)

    grating_values = mi_grating[valid]
    natural_values = mi_ns[valid]

    stat, p = wilcoxon(grating_values, natural_values)

    result_df = pd.DataFrame([{
        "comparison": "gratings_vs_natural_scenes",
        "n_cells": int(valid.sum()),
        "median_grating_MI": float(np.nanmedian(grating_values)),
        "median_natural_scene_MI": float(np.nanmedian(natural_values)),
        "median_difference_NS_minus_grating": float(np.nanmedian(natural_values - grating_values)),
        "wilcoxon_stat": float(stat),
        "p_value": float(p),
        "frac_NS_greater_than_grating": float(np.mean(natural_values > grating_values)),
    }])

    return result_df, valid, grating_values, natural_values


def validate_mi_against_metadata(
    results: dict,
    metadata: pd.DataFrame,
    matched_cell_ids,
    denom_threshold: float = 1e-3,
    robust: bool = True,
):
    """Validate raw/robust MI against Allen's running-modulation metrics.

    Diagnostic counterpart of
    :func:`utils.validate_mi_against_metadata`. Aligns ``metadata``
    to ``matched_cell_ids`` and compares, per stimulus, raw (or robust) MI to
    the Allen ``run_mod_*`` column via Spearman correlation.

    Parameters
    ----------
    results : dict
        Must contain ``"drifting_gratings"``, ``"static_gratings"``,
        ``"natural_scenes"``.
    metadata : pandas.DataFrame
        Table with ``cell_specimen_id``, ``run_mod_dg``, ``run_mod_sg``,
        ``run_mod_ns``.
    matched_cell_ids : array-like
        Cell IDs defining row order.
    denom_threshold : float, optional
        Passed to :func:`get_robust_mi`, by default ``1e-3``.
    robust : bool, optional
        If True (default), filter to robust cells; if False, use raw MI with
        no denominator-based filtering.

    Returns
    -------
    validation_df : pandas.DataFrame
    aligned : dict
        Mapping stimulus -> {"mi": array, "ref": array}.
    """
    meta = metadata.set_index("cell_specimen_id").loc[matched_cell_ids]

    mapping = {
        "drifting_gratings": "run_mod_dg",
        "static_gratings": "run_mod_sg",
        "natural_scenes": "run_mod_ns",
    }

    rows = []
    aligned = {}
    for stimulus, meta_col in mapping.items():
        analysis = results[stimulus]

        if robust:
            mi, robust_mask = get_robust_mi(analysis, denom_threshold)
        else:
            mi = compute_raw_mi(analysis)
            robust_mask = np.ones_like(mi, dtype=bool)

        ref = meta[meta_col].to_numpy()
        valid = robust_mask & np.isfinite(mi) & np.isfinite(ref)

        rho, pval = spearmanr(mi[valid], ref[valid])

        rows.append({
            "stimulus": stimulus,
            "metadata_col": meta_col,
            "n_cells": int(valid.sum()),
            "spearman_rho": float(rho),
            "p_value": float(pval),
            "median_our_MI": float(np.nanmedian(mi[valid])),
            "median_metadata_run_mod": float(np.nanmedian(ref[valid])),
        })
        aligned[stimulus] = {"mi": mi[valid], "ref": ref[valid]}

    return pd.DataFrame(rows), aligned


def get_negative_response_cells(results: dict, denom_threshold: float = 0.0) -> pd.DataFrame:
    """Flag cells whose raw-MI denominator is negative or whose MI sign disagrees with delta_R.

    For every stimulus and cell this reports the raw quantities behind the
    raw MI so sign-ambiguous or out-of-range cells can be inspected directly,
    without altering the formal sign-safe metric.

    Parameters
    ----------
    results : dict
        Mapping stimulus -> :class:`utils.BinaryModulation`.
    denom_threshold : float, optional
        A cell's ``negative_denominator`` flag is
        ``(r_run + r_still) < denom_threshold``, by default ``0.0``.

    Returns
    -------
    pandas.DataFrame
        Columns: stimulus, cell, r_run, r_still, denominator, delta_r, mi,
        negative_denominator, sign_reversal, abs_mi_gt_1.
    """
    rows = []
    for stimulus, analysis in results.items():
        r_run = np.asarray(analysis.r_run, dtype=float)
        r_still = np.asarray(analysis.r_still, dtype=float)
        mi = compute_raw_mi(analysis)

        denom = r_run + r_still
        delta_r = r_run - r_still

        finite = np.isfinite(mi) & np.isfinite(delta_r)
        sign_reversal = np.zeros(len(mi), dtype=bool)
        sign_reversal[finite] = np.sign(mi[finite]) != np.sign(delta_r[finite])

        negative_denominator = denom < denom_threshold
        abs_mi_gt_1 = np.abs(mi) > 1

        for cell in range(len(mi)):
            rows.append({
                "stimulus": stimulus,
                "cell": cell,
                "r_run": float(r_run[cell]),
                "r_still": float(r_still[cell]),
                "denominator": float(denom[cell]),
                "delta_r": float(delta_r[cell]),
                "mi": float(mi[cell]),
                "negative_denominator": bool(negative_denominator[cell]),
                "sign_reversal": bool(sign_reversal[cell]),
                "abs_mi_gt_1": bool(abs_mi_gt_1[cell]),
            })

    return pd.DataFrame(rows)


# ==============================================================================
# Negative-response structure (exploratory)
# ==============================================================================


def extract_negative_trial_traces(
    analysis,
    cell_indices,
    include_states=("running", "still"),
) -> pd.DataFrame:
    """Return individual (un-averaged) trial response traces for selected cells.

    Parameters
    ----------
    analysis : utils.BinaryModulation
        Analysis with :meth:`classify_trials` already run.
    cell_indices : iterable of int
        Cells to extract traces for.
    include_states : tuple of str, optional
        Which trial states to include, from ``{"running", "still"}``,
        by default both.

    Returns
    -------
    pandas.DataFrame
        One row per (cell, trial) pair, columns: cell, trial, state, trace,
        condition.
    """
    responses = np.asarray(analysis._td.responses)  # (n_cells, n_trials, duration)
    state_masks = {
        "running": analysis.run_mask,
        "still": analysis.still_mask,
    }
    labels = analysis.get_condition_labels()

    rows = []
    for cell in cell_indices:
        for state in include_states:
            mask = state_masks[state]
            for trial in np.where(mask)[0]:
                rows.append({
                    "cell": int(cell),
                    "trial": int(trial),
                    "state": state,
                    "trace": responses[cell, trial, :].copy(),
                    "condition": labels[trial] if labels is not None else None,
                })

    return pd.DataFrame(rows)


def run_negative_trace_pca(
    traces: pd.DataFrame,
    n_components: int = 3,
    center_each_trace: bool = False,
    scale_features: bool = True,
):
    """Exploratory PCA over trial-level response traces.

    Rows are individual trial traces, columns are time points. This is a
    pattern-detection tool only — it cannot establish that a negative
    deflection reflects biological inhibition.

    Parameters
    ----------
    traces : pandas.DataFrame
        Output of :func:`extract_negative_trial_traces`.
    n_components : int, optional
        Number of principal components to keep, by default 3.
    center_each_trace : bool, optional
        If True, subtract each trace's own mean before PCA. Off by default,
        because that would remove sustained negative offsets.
    scale_features : bool, optional
        If True (default), standardize each time-point column before PCA.

    Returns
    -------
    scores : np.ndarray, shape (n_kept, n_components)
    components : np.ndarray, shape (n_components, n_timepoints)
    explained_variance_ratio : np.ndarray, shape (n_components,)
    metadata : pandas.DataFrame
        Rows of ``traces`` corresponding to ``scores`` (non-finite dropped).
    """
    trace_matrix = np.vstack(traces["trace"].to_numpy())  # (n_traces, n_timepoints)

    finite_rows = np.all(np.isfinite(trace_matrix), axis=1)
    trace_matrix = trace_matrix[finite_rows]
    metadata = traces.loc[finite_rows].reset_index(drop=True)

    x = trace_matrix.copy()
    if center_each_trace:
        x = x - x.mean(axis=1, keepdims=True)

    if scale_features:
        x = StandardScaler().fit_transform(x)

    n_components = min(n_components, x.shape[0], x.shape[1])
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(x)

    return scores, pca.components_, pca.explained_variance_ratio_, metadata


# ==============================================================================
# Plotting (diagnostic)
# ==============================================================================


def plot_raw_vs_safe_scatter(results: dict, epsilon: float = 1e-12, ax=None):
    """Scatter raw MI (symlog x) against sign-safe MI, all stimuli overlaid.

    Off-diagonal / out-of-band points make the raw-MI failure modes visible:
    unbounded magnitudes and sign reversals that the bounded sign-safe MI
    avoids.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.8, 5.8), constrained_layout=True)
    else:
        fig = ax.figure

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, stimulus in enumerate(results):
        mi_raw = compute_raw_mi(results[stimulus])
        mi_safe = np.asarray(results[stimulus].mi, dtype=float)
        valid = np.isfinite(mi_raw) & np.isfinite(mi_safe)
        ax.scatter(mi_raw[valid], mi_safe[valid], alpha=0.7, s=28,
                   label=f"{stimulus} (n={int(valid.sum())})", color=colors[i % len(colors)])

    for y in (-1, 1):
        ax.axhline(y, linestyle="--", color="firebrick", linewidth=0.9)
    for x in (-1, 1):
        ax.axvline(x, linestyle="--", color="firebrick", linewidth=0.5, alpha=0.6)
    ax.axhline(0, linestyle=":", color="gray", linewidth=0.7)
    ax.axvline(0, linestyle=":", color="gray", linewidth=0.7)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("Raw MI  (symlog scale; unbounded)")
    ax.set_ylabel("Sign-safe MI  (bounded in [-1, 1])")
    ax.set_title("Raw vs sign-safe MI\noff-diagonal points = sign reversals")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    return fig, ax


def plot_raw_mi_histograms(results: dict):
    """Plot the raw MI histogram for each stimulus in ``results``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray of matplotlib.axes.Axes
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)

    for ax, stimulus in zip(axes, results):
        mi = compute_raw_mi(results[stimulus])
        valid_mi = mi[np.isfinite(mi)]
        median_mi = np.nanmedian(valid_mi)

        ax.hist(valid_mi, bins=20, alpha=0.8)
        ax.axvline(median_mi, linestyle="--", color="black", label=f"median={median_mi:.3f}")
        ax.axvline(0, linestyle=":", color="gray")
        ax.set_title(stimulus)
        ax.set_xlabel("Raw MI")
        ax.legend(frameon=False)

    axes[0].set_ylabel("Number of cells")
    return fig, axes


def plot_robust_mi_histograms(results: dict, denom_threshold: float = 1e-3):
    """Plot the robust raw-MI histogram for each stimulus in ``results``.

    Robust cells satisfy ``|R_run + R_still| > denom_threshold`` (see
    :func:`get_robust_mi`); this is a filter, not a normalization.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray of matplotlib.axes.Axes
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, stimulus in zip(axes, results):
        mi, robust = get_robust_mi(results[stimulus], denom_threshold)
        mi_robust = mi[robust]
        median_mi = np.nanmedian(mi_robust)

        ax.hist(mi_robust, bins=20, alpha=0.8)
        ax.axvline(median_mi, linestyle="--", color="black", label=f"median={median_mi:.3f}")
        ax.axvline(0, linestyle=":", color="gray")

        ax.set_xlim(-1, 1)
        ax.set_title(stimulus)
        ax.set_xlabel("Robust MI")
        ax.legend(frameon=False)

    axes[0].set_ylabel("Number of cells")
    return fig, axes


def plot_population_response_scatter(results: dict):
    """Scatter running- vs. still-trial population responses per stimulus.

    One point is one neuron; x = mean still-trial response, y = mean
    running-trial response. Descriptive, not a significance test on its own.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray of matplotlib.axes.Axes
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), constrained_layout=True)
    axes = np.atleast_1d(axes)

    for ax, stimulus in zip(axes, results):
        results[stimulus].plot_scatter(cell=None, ax=ax)

    return fig, axes


def plot_grating_natural_paired_distribution(grating_values, natural_values, p_value=None, ax=None):
    """Paired boxplot + per-cell connecting lines for grating vs. NS MI.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    grating_values = np.asarray(grating_values, dtype=float)
    natural_values = np.asarray(natural_values, dtype=float)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.figure

    ax.boxplot(
        [grating_values, natural_values],
        tick_labels=["Gratings\nmean(DG, SG)", "Natural\nscenes"],
        showfliers=False,
    )

    for g, n in zip(grating_values, natural_values):
        ax.plot([1, 2], [g, n], color="gray", alpha=0.25, linewidth=0.8)

    ax.axhline(0, linestyle=":", color="gray")
    ax.set_ylabel("MI")
    title = "Paired MI comparison"
    if p_value is not None:
        title += f"\nWilcoxon p={p_value:.3g}"
    ax.set_title(title)

    return fig, ax


def plot_negative_trace_examples(traces: pd.DataFrame, cell: int, n_examples: int = 6, ax=None):
    """Plot example running/still trial traces plus mean +/- std band, for one cell.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3.5))
    else:
        fig = ax.figure

    cell_df = traces[traces["cell"] == cell]
    state_colors = {"running": "C1", "still": "C0"}
    rng = np.random.default_rng(0)

    for state in cell_df["state"].unique():
        state_df = cell_df[cell_df["state"] == state]
        trace_matrix = np.vstack(state_df["trace"].to_numpy())
        t = np.arange(trace_matrix.shape[1])
        color = state_colors.get(state, "gray")

        n_show = min(n_examples, trace_matrix.shape[0])
        idx = rng.choice(trace_matrix.shape[0], size=n_show, replace=False)
        for i in idx:
            ax.plot(t, trace_matrix[i], color=color, alpha=0.25, linewidth=0.8)

        mean_trace = np.nanmean(trace_matrix, axis=0)
        std_trace = np.nanstd(trace_matrix, axis=0)
        ax.plot(t, mean_trace, color=color, linewidth=2.5, label=f"{state} mean (n={trace_matrix.shape[0]})")
        ax.fill_between(t, mean_trace - std_trace, mean_trace + std_trace, color=color, alpha=0.2)

    ax.axhline(0, linestyle=":", color="black", linewidth=0.8)
    ax.set_xlabel("Frame within response window")
    ax.set_ylabel("ΔF/F")
    ax.set_title(f"Cell {cell}")
    ax.legend(frameon=False, fontsize=8)

    return fig, ax


def plot_negative_trace_heatmap(traces: pd.DataFrame, pca_scores=None, ax=None):
    """Heatmap of trial traces, ordered by cell, state, and optionally PCA score.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    df = traces.reset_index(drop=True).copy()
    sort_cols = ["cell", "state"]
    if pca_scores is not None:
        df["_pc1"] = np.asarray(pca_scores)[:, 0]
        sort_cols = sort_cols + ["_pc1"]

    order = df.sort_values(sort_cols).index.to_numpy()
    sorted_df = df.loc[order].reset_index(drop=True)
    matrix = np.vstack(sorted_df["trace"].to_numpy())

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, max(2, 0.15 * len(order))))
    else:
        fig = ax.figure

    vmax = np.nanpercentile(np.abs(matrix), 95)
    vmax = vmax if vmax > 0 else 1.0
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax, label="ΔF/F")
    ax.set_xlabel("Frame within response window")
    ax.set_ylabel("Trial")

    group_sizes = sorted_df.groupby(["cell", "state"], sort=False).size()
    yticks, yticklabels = [], []
    boundary = 0
    for (cell, state), size in group_sizes.items():
        yticks.append(boundary + size / 2 - 0.5)
        yticklabels.append(f"cell {cell}\n({state})")
        boundary += size
        if boundary < len(sorted_df):
            ax.axhline(boundary - 0.5, color="black", linewidth=0.6, alpha=0.6)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=7)

    pc1_note = " — rows ordered by PC1 score within each cell/state group" if pca_scores is not None else ""
    ax.set_title(f"Trial traces, grouped by cell and state{pc1_note}", fontsize=9)

    return fig, ax


def plot_negative_trace_pca_scores(scores, metadata: pd.DataFrame, pc_x: int = 0, pc_y: int = 1, ax=None):
    """Scatter PCA scores, colored by behavioral state, one marker shape per cell.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure

    state_colors = {"running": "C1", "still": "C0"}
    markers = ["o", "s", "^", "D", "P", "X", "v"]
    cells = sorted(metadata["cell"].unique())
    marker_map = {c: markers[i % len(markers)] for i, c in enumerate(cells)}

    for cell in cells:
        for state, color in state_colors.items():
            m = ((metadata["cell"] == cell) & (metadata["state"] == state)).to_numpy()
            if not m.any():
                continue
            ax.scatter(
                scores[m, pc_x],
                scores[m, pc_y],
                color=color,
                marker=marker_map[cell],
                alpha=0.7,
                label=f"cell {cell} ({state})",
            )

    ax.axhline(0, linestyle=":", color="gray", linewidth=0.8)
    ax.axvline(0, linestyle=":", color="gray", linewidth=0.8)
    ax.set_xlabel(f"PC{pc_x + 1}")
    ax.set_ylabel(f"PC{pc_y + 1}")
    ax.legend(frameon=False, fontsize=7, ncol=2)

    return fig, ax


def plot_negative_trace_pcs(components, explained_variance_ratio, n_show: int = 3, ax=None):
    """Plot the temporal loadings of the first ``n_show`` principal components.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3.5))
    else:
        fig = ax.figure

    n_show = min(n_show, components.shape[0])
    t = np.arange(components.shape[1])
    for i in range(n_show):
        ax.plot(t, components[i], label=f"PC{i + 1} ({100 * explained_variance_ratio[i]:.1f}% var)")

    ax.axhline(0, linestyle=":", color="gray", linewidth=0.8)
    ax.set_xlabel("Frame within response window")
    ax.set_ylabel("Loading")
    ax.set_title("Principal-component temporal loadings")
    ax.legend(frameon=False)

    return fig, ax
