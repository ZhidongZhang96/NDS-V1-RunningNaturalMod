import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Rectangle
from pathlib import Path
import textwrap
from dataclasses import dataclass
from scipy.stats import f_oneway, mannwhitneyu, spearmanr, ttest_1samp, wilcoxon

STIMULI = ['drifting_gratings', 'static_gratings', 'natural_scenes', 'spontaneous']
SHORT_STIM = ['DG', 'SG', 'NS', 'Spont']

_STIM_TO_SHORT = dict(zip(STIMULI, SHORT_STIM))
_SHORT_TO_STIM = dict(zip(SHORT_STIM, STIMULI))

def stim_to_short(stim: str) -> str:
    """Map full stimulus name (e.g. 'drifting_gratings') to short label (e.g. 'DG')."""
    return _STIM_TO_SHORT.get(stim, stim)

def short_to_stim(s: str) -> str:
    """Map short label (e.g. 'DG') to full stimulus name (e.g. 'drifting_gratings')."""
    return _SHORT_TO_STIM.get(s, s)

def check_stim(stimulus:str):
    assert stimulus in STIMULI, f"You must choose one of the stimulus type: {STIMULI}"

RESPONSE_WINDOWS = {
    "drifting_gratings": (10, 60),
    "static_gratings": (5, 7),
    "natural_scenes": (5, 7),
    "spontaneous": (0, 15),
}


# ==============================================================================
# Data Loading
# ==============================================================================


def load_data(path="data", fname="visual_coding_data"):
    raw = dict(np.load(Path(path) / f"{fname}.npz", allow_pickle=True))
    data = {
        "matched_cell_ids": raw["matched_cell_ids"],
        "templates": {},
        "sessions": {},
    }

    def session(L):
        return data["sessions"].setdefault(L, {"stim_tables": {}})

    for key, val in raw.items():
        if key == "matched_cell_ids":
            continue
        parts = key.split("__")
        if parts[0] == "tmpl":
            data["templates"][parts[1]] = val
            continue
        L = parts[0]
        s = session(L)
        if parts[1] == "stim" and parts[3] == "values":
            stim = parts[2]
            cols = list(raw[f"{L}__stim__{stim}__cols"])
            s["stim_tables"][stim] = pd.DataFrame(val, columns=cols)
        elif parts[1] == "epoch" and parts[2] == "values":
            cols = list(raw[f"{L}__epoch__cols"])
            s["stim_epoch_table"] = pd.DataFrame(val, columns=cols)
        elif parts[1] in ("session_type",):
            s["session_type"] = val.item() if hasattr(val, "item") else val
        elif parts[1] in ("t", "dff", "roi_masks", "max_projection", "running_speed"):
            s[parts[1]] = val
    return data


def concat_per_cell(*arrays):
    """Concatenate per-cell metric arrays from multiple containers.

    Each input is an ``(n_cells_i,)`` or ``(n_cells_i, ...)`` array from one
    container's analysis.  Returns them concatenated along axis 0 — the
    pooled population vector.

    Example::

        # pool MI across containers for 'drifting_gratings':
        pooled_mi = concat_per_cell(*[mr["drifting_gratings"].mi for mr in bm_list])
    """
    return np.concatenate([np.asarray(a) for a in arrays if a is not None], axis=0)


def pool_binary_modulation(all_results):
    """Pool BinaryModulation per-cell results from multiple containers.

    ``all_results`` is a list of dicts ``{stimulus: BinaryModulation}`` as
    returned by :func:`run_binary_modulation_analysis`. Returns a new dict
    of wrapper objects whose per-cell attributes are concatenated across
    containers, compatible with the plotting functions in this module.

    Example::

        bm_list = [run_binary_modulation_analysis(d, RESPONSE_WINDOWS) for d in data_list]
        pooled = pool_binary_modulation(bm_list)
        fig = plot_metric_comparison(pooled, metric="mi")
    """
    from types import SimpleNamespace
    if not all_results:
        return {}
    stims = list(all_results[0].keys())
    _PER_CELL_ATTRS = [
        "mi", "delta_r", "r_run", "r_still",
        "gain_a", "gain_b", "gain_r2", "gain_valid", "n_gain_conditions",
        "modulated_mask", "p_values", "significant",
    ]
    pooled = {}
    for stim in stims:
        insts = [r[stim] for r in all_results]
        kwargs = {}
        for attr in _PER_CELL_ATTRS:
            vals = [getattr(inst, attr, None) for inst in insts]
            if all(v is not None for v in vals):
                kwargs[attr] = np.concatenate([np.asarray(v) for v in vals], axis=0)
            else:
                # Fill with NaN for stimuli where this attr doesn't exist (e.g. gain for spontaneous)
                n_cells = sum(len(np.asarray(getattr(inst, "mi", []))) for inst in insts)
                kwargs[attr] = np.full(n_cells, np.nan)
        kwargs["_td"] = SimpleNamespace(stimulus=stim)
        pooled[stim] = SimpleNamespace(**kwargs)
    return pooled


def load_containers(path="data"):
    """Load all ``container_*.npz`` files in a directory.

    Returns a list of data dicts, each compatible with :func:`load_data` format.
    Also returns the sorted list of container IDs.

    Usage:
        data_list, cids = load_containers("data")
        for data in data_list:
            td = extract_trials(data, "drifting_gratings")
            # ...
    """
    p = Path(path)
    files = sorted(p.glob("container_*.npz"))
    if not files:
        raise FileNotFoundError(f"No container_*.npz files found in {p}")
    data_list = []
    cids = []
    for f in files:
        cid = int(f.stem.split("_")[1])
        cids.append(cid)
        data_list.append(load_data(path, f.stem))
    return data_list, cids


def print_info(data):
    print(f"matched cells: {len(data['matched_cell_ids'])}")
    key_templates = list(data["templates"])
    print(
        f"templates: {key_templates} --- {data['templates'][key_templates[0]].shape}, {data['templates'][key_templates[1]].shape}"
    )
    for L, s in sorted(data["sessions"].items()):
        print(f"\nsession {L} ({s.get('session_type')})")
        print(
            f"  t: {s['t'].shape}, dff: {s['dff'].shape}, roi_masks: {s['roi_masks'].shape}"
        )
        for name, df in s["stim_tables"].items():
            print(f"  stim '{name}': {df.shape} cols={list(df.columns)}")


# ==============================================================================
# Helper
# ==============================================================================


def get_condition_intervals(epoch_table: pd.DataFrame, conditions=None) -> dict:
    """Group stimulus epoch intervals by visual condition.

    Parameters
    ----------
    epoch_table : pandas.DataFrame
        Table containing at least ``stimulus``, ``start``, and ``end`` columns.
    conditions : str, list, tuple, set, or None, optional
        Grouping strategy for intervals:
        - ``None``: all intervals are grouped under ``"All"``.
        - ``"all"``: keep each unique stimulus as its own condition.
        - iterable of stimulus names: matching stimuli are kept, and all other
          stimuli are grouped under ``"Others"``.

    Returns
    -------
    dict
        Mapping from condition name to a list of ``(start, end)`` tuples, e.g.,
        ```
        {
            'Others': [(19709, 37767), ...],
            'drifting_gratings': [(747, 18775), ...]
        }
        ```
    """
    df = epoch_table.copy()

    if conditions is None:
        df["visual_condition"] = "All"
    elif conditions == "all":
        df["visual_condition"] = df["stimulus"]
    else:
        assert isinstance(
            conditions, (list, tuple, set)
        ), "conditions must be a list, 'all', or None"
        df["visual_condition"] = df["stimulus"].apply(
            lambda x: x if x in conditions else "Others"
        )

    intervals_dict = {}
    for cond, group in df.groupby("visual_condition"):
        intervals_dict[cond] = [
            (int(row.start), int(row.end)) for row in group.itertuples()
        ]

    return intervals_dict

def plot_mean_sem(ax, x, data, color, alpha=0.5, label=None,
                   linestyle='-', marker_facecolor='black',
                   marker_edgecolor='black'):
    """Plot mean ± SEM as fill_between + line with markers (NaN-safe)."""
    m = np.nanmean(data, axis=0)
    n_valid = np.sum(~np.isnan(data), axis=0)
    s = np.nanstd(data, axis=0, ddof=0) / np.sqrt(n_valid)
    ax.fill_between(x, m - s, m + s, color=color, alpha=alpha,
                    edgecolor='none', label=label)
    ax.plot(x, m, color=color, marker='o', markersize=3,
            markerfacecolor=marker_facecolor, markeredgecolor=marker_edgecolor,
            linestyle=linestyle)


# ==============================================================================
# Trial Extraction
# ==============================================================================


@dataclass
class TrialData:
    """Per-trial data extracted for a single stimulus type.

    Attributes
    ----------
    stimulus: str
        Name of the stimulus, one of :var:`STIMULI`
    params: dict
        Parameters used to extract trials data, e.g., `offset` after stimuli onset and `duration` of each trial.
    responses : np.ndarray
        ΔF/F within the response window (with offset after stimuli onset), shape ``(n_cells, n_trials, len_windows)``.
    running_speed : np.ndarray
        Running speed during each trial, shape ``(n_trials, duration)``.
    time : np.ndarray
        Real time points for each trial, in seconds, shape ``(n_trials, duration)``.
    stimulus_params : dict[str, np.ndarray]
        Trial-level stimulus parameters. Each entry has shape ``(n_trials,)``.
        Keys correspond to the stimulus table columns (excluding ``start``/``end``),
        e.g. ``"orientation"``, ``"temporal_frequency"``, ``"frame"``.
    """
    stimulus: str
    params: dict
    responses: np.ndarray
    running_speed: np.ndarray
    time: np.ndarray
    stimulus_params: dict | None = None

    def __post_init__(self):
        self.n_neurons = self.responses.shape[0]


def extract_trials(
    data,
    stimulus: str,
    response_window: tuple | None = None,
) -> TrialData:
    """Extract responses and running speed for a given stimulus (from all sessions).

    For each trial listed in the stimulus table, this function slices the
    ΔF/F and running-speed arrays at the indicated start/end indices.

    Parameters
    ----------
    data : dict
        Data dictionary returned by :func:`load_data`.
    stimulus : str
        Stimulus name, must be one of the :obj:`STIMULI`, i.e. ``"drifting_gratings"``, ``"static_gratings"``,
        ``"natural_scenes"`` and ``"spontaneous"``.
    response_window : tuple of (int, int)
        ``(offset, duration)`` in time points relative to trial start:
        - ``offset``: frames to skip at the beginning of each trial.
        - ``duration``: frames to include; ``None`` means use the full trial
          length (minus offset).

    Returns
    -------
    TrialData
    """
    
    check_stim(stimulus)

    if not response_window:
        response_window = RESPONSE_WINDOWS[stimulus]
    offset, duration = response_window
    params: dict = {"offset": offset, "duration": duration}

    # ----- 1. Generate windows (branch-specific) -----
    if stimulus == "spontaneous":
        windows = []  # list of (session_key, start, end)
        for session_key in ("A", "B"):
            s = data["sessions"][session_key]
            et = s["stim_epoch_table"]
            spon = et[et["stimulus"] == "spontaneous"]
            if len(spon) == 0:
                # Build spontaneous windows from inter-epoch gaps
                epochs = et.sort_values("start")
                gap_starts = epochs["end"].values[:-1]
                gap_ends = epochs["start"].values[1:]
                for gs, ge in zip(gap_starts, gap_ends):
                    if ge > gs:
                        n_frames = ge - gs
                        n_trials = (n_frames - offset) // duration
                        for i in range(n_trials):
                            windows.append((session_key, int(gs) + offset + i * duration,
                                           int(gs) + offset + (i + 1) * duration))
                continue
            epoch_start = int(spon["start"].values[0])
            epoch_end = int(spon["end"].values[0])

            n_frames = epoch_end - epoch_start
            n_trials = (n_frames - offset) // duration
            for i in range(n_trials):
                t_start = epoch_start + offset + i * duration
                t_end = t_start + duration
                windows.append((session_key, t_start, t_end))
        stimulus_params = None
    else:
        s_key = "A" if stimulus == "drifting_gratings" else "B"
        s = data["sessions"][s_key]
        stim_tables = s["stim_tables"][stimulus]

        # remove the blank-sweep trial, where the params are all NaN for gratings
        if stimulus == 'drifting_gratings':
            stim_tables = stim_tables[stim_tables['blank_sweep'] != 1].drop(columns=['blank_sweep'])
        elif stimulus == 'natural_scenes':
            stim_tables = stim_tables[stim_tables['frame'] != -1]
        elif stimulus == 'static_gratings':
            stim_tables = stim_tables.dropna(subset=['orientation'])

        trial_starts = np.array(stim_tables["start"]) + offset
        windows = [
            (s_key, int(start), int(start + duration))
            for start in trial_starts
        ]
        stimulus_params = {
            col: stim_tables[col].to_numpy()
            for col in stim_tables.columns
            if col not in ["start", "end"]
        }

    # ----- 2. Slice & stack (shared) -----
    all_responses = []
    all_speeds = []
    all_times = []
    for session_key, start, end in windows:
        s = data["sessions"][session_key]
        all_responses.append(s["dff"][:, start:end])            # (n_cells, duration)
        all_speeds.append(s["running_speed"][0, start:end])     # (duration,)
        all_times.append(s["t"][start:end])                      # (duration,)

    responses = np.stack(all_responses, axis=1)     # (n_cells, n_trials, duration)
    running_speed = np.stack(all_speeds, axis=0)    # (n_trials, duration)
    running_speed[running_speed < 0] = 0            # non-negative
    time = np.stack(all_times, axis=0)               # (n_trials, duration)

    return TrialData(
        stimulus,
        params,
        responses,
        running_speed,
        time,
        stimulus_params,
    )


# ==============================================================================
# Trial-level condition helpers (shared)
# ==============================================================================

def _get_condition_labels(td: TrialData):
    """Build one visual-stimulus condition label for each trial.

    Returns None for spontaneous activity.
    """
    stimulus = td.stimulus
    params = td.stimulus_params

    if stimulus == "spontaneous" or params is None:
        return None

    n_trials = td.responses.shape[1]

    if stimulus == "natural_scenes":
        return np.asarray(params["frame"])

    if stimulus == "drifting_gratings":
        condition_keys = ["orientation", "temporal_frequency"]
    elif stimulus == "static_gratings":
        condition_keys = ["orientation", "spatial_frequency", "phase"]
    else:
        raise ValueError(f"Unsupported stimulus: {stimulus}")

    labels = np.empty(n_trials, dtype=object)
    for i in range(n_trials):
        labels[i] = tuple(params[key][i] for key in condition_keys)
    return labels


def find_preferred_conditions(td: TrialData, top_frac: float = 0.5):
    """Find each neuron's preferred stimulus condition(s).

    Groups trials by visual condition, computes the mean response per
    condition (averaged over all trials regardless of running speed), and
    identifies which conditions are the most responsive for each neuron.

    Parameters
    ----------
    td : TrialData
    top_frac : float
        Fraction of top conditions (by mean response) to consider as
        "preferred" for each neuron.  ``0.5`` = the top half, ``1.0``
        = all conditions, ``0`` = none.  Default ``0.5``.

    Returns
    -------
    preferred_labels : list of tuple or None
        Length ``(n_cells,)`` — each element is a tuple of condition
        labels whose mean response is in the top *top_frac*.
        Empty tuple for neurons with no responsive condition.  None for
        spontaneous.
    preferred_trial_mask : np.ndarray or None
        Shape ``(n_cells, n_trials)`` bool — ``True`` for trials belonging
        to any preferred condition of that neuron.  None for spontaneous.
    condition_mean_responses : np.ndarray or None
        Shape ``(n_cells, n_conditions)`` — mean ΔF/F per condition.
        None for spontaneous.
    """
    labels = _get_condition_labels(td)
    if labels is None:
        return None, None, None

    responses = td.responses                     # (n_cells, n_trials, duration)
    trial_mean = responses.mean(axis=-1)         # (n_cells, n_trials)

    unique_conditions = pd.unique(labels)
    n_cells = td.n_neurons
    n_conditions = len(unique_conditions)
    n_top = max(1, int(round(n_conditions * top_frac)))  # at least 1

    condition_masks = []  # one bool mask per condition
    for cond in unique_conditions:
        mask = np.array([label == cond for label in labels], dtype=bool)
        condition_masks.append(mask)

    condition_mean_responses = np.full((n_cells, n_conditions), np.nan)
    for idx in range(n_conditions):
        condition_mean_responses[:, idx] = trial_mean[:, condition_masks[idx]].mean(axis=1)

    # top-k conditions per neuron
    top_indices = np.argsort(-condition_mean_responses, axis=1)[:, :n_top]  # (n_cells, n_top)

    preferred_labels = []
    preferred_trial_mask = np.zeros((n_cells, len(labels)), dtype=bool)
    for i in range(n_cells):
        conds = tuple(unique_conditions[idx] for idx in top_indices[i])
        preferred_labels.append(conds)
        for idx in top_indices[i]:
            preferred_trial_mask[i] |= condition_masks[idx]

    return preferred_labels, preferred_trial_mask, condition_mean_responses


# ==============================================================================
# Plotting (overview)
# ==============================================================================


class Plotter:
    """Plotting utilities for visual coding data.

    Parameters
    ----------
    data : dict
        Data dictionary returned by :func:`load_data`.
    """

    def __init__(self, data):
        self._data = data.copy()

    # ------------- helpers -------------

    def _session(self, session="A"): 
        return self._data["sessions"][session]

    def _get_intervals_and_colors(self, session, conditions):
        """Get condition intervals and build a color map.

        Parameters
        ----------
        session : str
            Session identifier ('A', 'B', 'C').
        conditions : str, list, tuple, set, or None
            Passed through to :func:`get_condition_intervals` to define
            how stimulus epochs are grouped and coloured.

        Returns
        -------
        intervals_dict : dict
            Mapping condition name -> list of (start, end) intervals.
        color_map : dict
            Mapping condition name -> matplotlib color.
        """
        s = self._session(session)
        epoch_table = s["stim_epoch_table"].copy()
        intervals_dict = get_condition_intervals(epoch_table, conditions)

        is_single_color = list(intervals_dict.keys()) == ["All"]
        if is_single_color:
            color_map = {"All": "C0"}
        else:
            color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            color_map = {}
            color_idx = 0
            for cond in intervals_dict.keys():
                if cond == "Others":
                    color_map[cond] = OTHER_COLOR
                else:
                    color_map[cond] = color_cycle[color_idx % len(color_cycle)]
                    color_idx += 1

        return intervals_dict, color_map

    # ------------- public plots -------------

    def field_of_view(self, session="A", ax=None):
        """Plot the field of view with ROI mask (red)."""
        s = self._session(session)
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(s["max_projection"], cmap="gray")
        ax.imshow(
            np.ma.masked_where(s["roi_masks"].sum(0) == 0, s["roi_masks"].sum(0)),
            cmap="autumn",
            alpha=0.5,
        )
        ax.set_title(f"session {session}: field of view + ROIs")
        ax.axis("off")
        return ax

    def stimulus_examples(self, num=3):
        """Plot example stimuli for each template type."""
        templates = self._data["templates"]
        fig, axes = plt.subplots(num, len(templates), layout="constrained")
        for idx, (name, tmpl) in enumerate(templates.items()):
            axs = axes[:, idx] if num > 1 else axes[idx]
            for i in range(num):
                ax = axs[i] if num > 1 else axs
                if i == 0:
                    ax.set_title(f"{name}\n({num} out of {tmpl.shape[0]})")
                ax.imshow(tmpl[i], cmap="gray")
                ax.axis("off")
        return fig

    def traces(self, session="A", cells=(0, 1, 2), conditions=None, fig_ax=None):
        """Plot calcium traces (ΔF/F) for multiple cells over time, grouped by visual conditions.

        Parameters
        ----------
        session : str, optional
            Session identifier ('A', 'B', 'C'), by default "A".
        cells : tuple, optional
            Cell indices to plot, by default (0, 1, 2).
        conditions : str, list, tuple, set, or None, optional
            Passed through to :func:`get_condition_intervals`.
        fig_ax : (fig, axes), optional
            Existing figure and axes to draw into.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        s = self._session(session)

        if fig_ax is None:
            fig, axes = plt.subplots(
                len(cells),
                1,
                figsize=(10, 1.6 * len(cells)),
                sharex=True,
                constrained_layout=True,
            )
        else:
            fig, axes = fig_ax

        axes = np.atleast_1d(axes)

        intervals_dict, color_map = self._get_intervals_and_colors(session, conditions)
        is_single_color = list(intervals_dict.keys()) == ["All"]

        for cell_idx, (ax, c) in enumerate(zip(axes, cells)):
            for cond, intervals in intervals_dict.items():
                current_color = color_map[cond]
                for i, interval in enumerate(intervals):
                    slc = slice(*interval)
                    t_segment = s["t"][slc]
                    dff_segment = s["dff"][c][slc]

                    if len(t_segment) == 0:
                        continue
                    if cell_idx == 0 and i == 0 and not is_single_color:
                        plot_label = cond
                    else:
                        plot_label = None
                    ax.plot(
                        t_segment,
                        dff_segment,
                        lw=0.5,
                        color=current_color,
                        label=plot_label,
                    )

            ax.set_ylabel(f"cell {c}\nΔF/F")

        if not is_single_color:
            axes[0].legend(
                loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False
            )

        axes[-1].set_xlabel("time (s)")
        fig.suptitle(f"session {session}: example traces")
        return fig

    def running_speed(self, session="A", conditions=None, ax=None):
        """Plot running speed over time, grouped by visual conditions.

        Parameters
        ----------
        session : str, default="A"
            Session identifier.
        conditions : str, list, tuple, set, or None, default=None
            How to color/group the trace — see :func:`get_condition_intervals`.
        ax : matplotlib.axes.Axes, optional
            Axis to draw into. Creates a new one if None.

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        s = self._session(session)
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 2))

        intervals_dict, color_map = self._get_intervals_and_colors(session, conditions)
        is_single_color = list(intervals_dict.keys()) == ["All"]

        for cond, intervals in intervals_dict.items():
            current_color = color_map[cond]
            for i, interval in enumerate(intervals):
                slc = slice(*interval)
                t_segment = s["t"][slc]
                speed_segment = s["running_speed"][0, slc]
                if len(t_segment) == 0:
                    continue

                speed_segment[speed_segment < 0] = 0

                plot_label = cond if (i == 0 and not is_single_color) else None
                ax.plot(
                    t_segment, speed_segment, lw=0.5, color=current_color, label=plot_label
                )

        if not is_single_color:
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)

        ax.set_xlabel("time (s)")
        ax.set_ylabel("running speed\n(cm/s)")
        ax.set_title(f"session {session}: running speed")
        return ax


# ==============================================================================
# Analysis 1 — Binned Speed Tuning
# ==============================================================================

# colour scheme for monotonicity categories
POS_COLOR = "#E7402D"                        # positive  (red)
NEG_COLOR = "#2492DC"                        # negative  (blue)
NM_COLOR  = "#71AD77"                        # non-monotonic (teal)
NTM_COLOR = "#B0A0A0"                        # non-tuned modulated (warm gray)
OTHER_COLOR = '#7f7f7f'                      # "Others" in condition plots


def _ensure_ax(ax=None, **subplots_kw):
    """Return ``(fig, ax)``, creating a new figure if ``ax`` is None."""
    if ax is None:
        fig, ax = plt.subplots(**subplots_kw)
    else:
        fig = ax.figure
    return fig, ax


def _hex_to_rgb(h: str) -> tuple[float, float, float]:
    """Convert hex colour '#RRGGBB' to (R, G, B) in [0, 1]."""
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))


POS_RGB = _hex_to_rgb(POS_COLOR)
NEG_RGB = _hex_to_rgb(NEG_COLOR)
NM_RGB  = _hex_to_rgb(NM_COLOR)


def _pvalue_stars(p: float) -> str:
    """Return significance stars for a p-value: '***' / '**' / '*' / ''."""
    if p <= 0.0001:
        return '***'
    elif p <= 0.001:
        return '**'
    elif p <= 0.01:
        return '*'
    return ''


def _add_identity_line(ax, x, y, label="", **kwargs):
    """Draw diagonal y=x line spanning the data range and set axis limits."""
    lo = min(np.nanmin(x), np.nanmin(y))
    hi = max(np.nanmax(x), np.nanmax(y))
    m = 0.05 * max(hi - lo, 1e-3)
    lo, hi = lo - m, hi + m
    line_kwargs = dict(linestyle="--", linewidth=1, color="gray", label=label)
    line_kwargs.update(kwargs)
    ax.plot([lo, hi], [lo, hi], **line_kwargs)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)



class SpeedTuning:
    """Running-speed binned tuning curves and significance tests.

    Bins trials by running speed, computes average response ± std per bin
    (tuning curve), then tests each neuron's selectivity via a shuffle test
    and reports Spearman correlation between response and running speed.

    Each instance corresponds to **one analysis scenario** — a single
    stimulus, or multiple stimuli pooled together (e.g. all visual stimuli
    vs. spontaneous).

    Inspired by (Christensen & Pillow, 2022)

    Parameters
    ----------
    trial_data : TrialData | dict[str, TrialData]
        Single :class:`TrialData` for one stimulus, or a dict mapping
        stimulus names to :class:`TrialData` (pooled together).
        If empty, auto-generated from dict keys.
    mode : str, optional
        The way to bin the data, 'equal_size' or 'equal_counts'. By default 'equal_size'
    n_bins : int, optional
        Number of speed bins, by default 20.
    neuron_mask : np.ndarray or None, optional
        Boolean mask ``(n_cells_picked,)`` — only these neurons are analysed.
        Typically from :attr:`BinaryModulation.tuned_mask`. None = all neurons.
    pref_trial_mask : np.ndarray or None, optional
        Boolean mask ``(n_cells_picked, n_trials_pref)`` — per-neuron trial selection
        (preferred-condition trials from :func:`find_preferred_conditions`).
        Trials where the mask is False are set to NaN. None = all trials.
    """

    def __init__(self, trial_data: TrialData | dict[str, TrialData],
                 mode='equal_size', n_bins: int = 20,
                 neuron_mask=None, pref_trial_mask=None):
        self._td = trial_data
        self.n_bins = n_bins
        self.mode = mode
        self.neuron_mask = neuron_mask  # if None, n_cells_picked = n_cells
        self.pref_trial_mask = pref_trial_mask  # if None, n_trials_pref = n_trials_total

        # by compute_tuning()
        self.responses: np.ndarray | None = None             # (n_cells_picked, n_trials_pref), neuron-masked only
        self.masked_responses: np.ndarray | None = None      # (n_cells_picked, n_trials_pref), trial-masked (NaN'd)
        self.speeds: np.ndarray | None = None                # (n_trials_pref,)

        self.bins_edges: np.ndarray | None = None            # (n_bins+1,)
        self.bins_centers: np.ndarray | None = None          # (n_bins,)
        self.bins_ids: np.ndarray | None = None              # (n_trials_pref,) starts from `1`
        self.bins_sub_ids: np.ndarray | None = None          # (n_trials_pref,) with ignored ones `-1`

        self.mean_all_responses: np.ndarray | None = None    # (n_cells_picked, n_bins)
        self.mean_responses: np.ndarray | None = None        # (n_bins,)
        self.std_responses: np.ndarray | None = None         # (n_bins,)

        # by significance_test()
        self.anova_p_values: np.ndarray | None = None       # (n_cells_picked,)
        self.significant_mask: np.ndarray | None = None      # bool, (n_cells_picked,)

        # by compute_spearman()
        self.rho: np.ndarray | None = None                   # (n_cells_picked,)
        self.rho_p_values: np.ndarray | None = None          # (n_cells_picked,)
        self.monotonic_mask : dict[str, np.ndarray] | None = None


    # ------------- helpers -------------

    def _pooled(self):
        """Pool trials across all stored TrialData.

        Averages the response and running-speed time windows to produce
        one value per trial, then concatenates trials from all stimuli.

        Returns
        -------
        responses : np.ndarray, shape ``(n_cells_picked, n_trials_pref)``
            Mean ΔF/F per trial (averaged over the response window).
        speed : np.ndarray, shape ``(n_trials_pref,)``
            Mean running speed per trial (averaged over the trial window).
        """
        if isinstance(self._td, TrialData):
            return self._td.responses.mean(axis=-1), self._td.running_speed.mean(axis=-1)
        all_r, all_v = [], []
        for td in self._td.values():
            all_r.append(td.responses.mean(axis=-1))         # (n_cells, n_trials)
            all_v.append(td.running_speed.mean(axis=-1))     # (n_trials,)
        return np.concatenate(all_r, axis=1), np.concatenate(all_v, axis=0)

    def _binned_responses(self, bins_ids=None):
        """Bin trials by ``bins_ids`` and average responses per bin.

        Parameters
        ----------
        bins_ids : np.ndarray or None
            Bin assignments for each trial, shape ``(n_trials_pref,)``.
            Uses ``self.bins_ids`` if None.
        mode : str, optional
            The way to bin the data
        Returns
        -------
        bins_edges : np.ndarray, shape ``(n_bins+1,)``
        bins_ids : np.ndarray, shape ``(n_trials_pref,)``
        mean_all_responses : np.ndarray, shape ``(n_cells_picked, n_bins)``
        """
        assert self.speeds is not None, "call compute_tuning() first"
        assert self.masked_responses is not None, "call compute_tuning() first"

        # bin the speed
        if self.bins_edges is None:
            if self.mode == 'equal_size':
                bins_edges = np.linspace(self.speeds.min(), self.speeds.max()+1e-6, num=self.n_bins+1)
                self.bins_centers = (bins_edges[:-1] + bins_edges[1:]) / 2
            elif self.mode == 'equal_counts':
                # Quantile-based binning: each bin gets ~equal number of trials
                percentiles = np.linspace(0, 100, self.n_bins + 1)
                bins_edges = np.percentile(self.speeds, percentiles)
                # Ensure strictly increasing edges (handle duplicate speed values
                # at boundaries, common with short sg/ns trials)
                for i in range(1, len(bins_edges)):
                    if bins_edges[i] <= bins_edges[i-1]:
                        bins_edges[i] = bins_edges[i-1] + 1e-8
                bins_edges[-1] = self.speeds.max() + 1e-6
                self.bins_centers = (bins_edges[:-1] + bins_edges[1:]) / 2

        else:
            bins_edges = self.bins_edges
        if bins_ids is None:
            bins_ids = np.digitize(self.speeds, bins_edges) if self.bins_ids is None else self.bins_ids

        # compute the mean response per bin (handle empty bins)
        mean_all_responses = []
        for b in range(1, self.n_bins + 1):
            mask = bins_ids == b
            if not np.any(mask):
                mean_all_responses.append(np.full(self.masked_responses.shape[0], np.nan))
                continue
            res_bin = self.masked_responses[:, mask]   # (n_cells_picked, n_trials_in_bin)
            mean_all_responses.append(np.nanmean(res_bin, axis=1))
        mean_all_responses = np.array(mean_all_responses).T  # (n_cells_picked, n_bins)

        return bins_edges, bins_ids, mean_all_responses

    def _subsample(self, max_per_bin=None, seed=42):
        """Subsample high-count bins to reduce speed-distribution imbalance.

        Marks dropped trials by setting their bin id to -1 in ``self.bins_sub_ids``.
        """
        assert self.bins_ids is not None
        bins_ids = self.bins_ids.copy()  # (n_trials_pref,)

        ids, counts = np.unique(bins_ids, return_counts=True)
        if max_per_bin is None:
            # If there's only one occupied bin, subsampling is undefined; keep all trials.
            if len(counts) < 2:
                self.bins_sub_ids = bins_ids
                return
            max_per_bin = np.partition(counts, -2)[-2]  # 2nd-largest occupied-bin count

        rng = np.random.default_rng(seed)
        count_map = dict(zip(ids, counts))
        for bid in ids:
            if count_map[bid] > max_per_bin:
                trials = np.where(bins_ids == bid)[0]
                selected = rng.choice(trials, size=max_per_bin, replace=False)
                unselected = np.setdiff1d(trials, selected)
                bins_ids[unselected] = -1

        self.bins_sub_ids = bins_ids


    # ------------- mask application -------------

    def _apply_masks(self, responses, speeds):
        """Apply neuron_mask and pref_trial_mask in order.

        Note: still trials are NOT excluded — speed-distribution imbalance is
        handled by :meth:`_subsample` instead.

        Parameters
        ----------
        responses : np.ndarray, shape ``(n_cells, n_trials)``
            Raw mean ΔF/F per trial from :meth:`_pooled`.
        speeds : np.ndarray, shape ``(n_trials,)``
            Raw mean running speed per trial from :meth:`_pooled`.

        Returns
        -------
        responses : np.ndarray, shape ``(n_cells_picked, n_trials_pref)``
            Neuron-masked responses.
        speeds : np.ndarray, shape ``(n_trials_pref,)``
            Speeds (aligned to neuron-masked responses).
        masked_responses : np.ndarray, shape ``(n_cells_picked, n_trials_pref)``
            Trial-masked responses (NaN where ``pref_trial_mask`` is False).
        """
        # apply neuron_mask
        if self.neuron_mask is not None:
            responses = responses[self.neuron_mask]

        # apply pref_trial_mask (NaN-masked per-neuron trial selection)
        if self.pref_trial_mask is not None:
            mask = self.pref_trial_mask
            if self.neuron_mask is not None:
                mask = mask[self.neuron_mask]
            masked_responses = responses.copy()
            masked_responses[~mask] = np.nan
        else:
            masked_responses = responses

        return responses, speeds, masked_responses


    # ------------- core computation -------------

    def run(self):
        """All in one function to compute tuning, test significance and monoticity"""
        self.compute_tuning()
        self.significance_test()
        self.compute_spearman()

    def compute_tuning(self):
        """Bin trials by running speed and compute tuning curves.
        """
        responses, speeds = self._pooled()
        self.responses, self.speeds, self.masked_responses = \
            self._apply_masks(responses, speeds)

        self.bins_edges, self.bins_ids, self.mean_all_responses = \
            self._binned_responses()

        # compute the mean and std across cells (ignore empty bins)
        self.mean_responses = np.nanmean(self.mean_all_responses, axis=0)
        self.std_responses = np.nanstd(self.mean_all_responses, axis=0)

        # subsample, for significant test
        self.bins_sub_ids = self.bins_ids
        self._subsample()

    def significance_test(self, threshold: float = 0.05):
        """One-way ANOVA across speed bins to assess tuning significance.

        For each cell, groups trial responses by their running-speed bin
        (using :attr:`bins_sub_ids`), excludes NaN responses and
        subsampled-out (-1) trials, then runs ``scipy.stats.f_oneway``
        across bins. A significant p-value means the cell's mean response
        differs across at least one speed bin (i.e., it is speed-tuned).

        Parameters
        ----------
        threshold : float, optional
            Significance threshold, by default 0.05.

        Stores
        ------
        anova_p_values : np.ndarray, shape ``(n_cells_picked,)``
            p-values from one-way ANOVA across speed bins.
        significant_mask : np.ndarray of bool, shape ``(n_cells_picked,)``
            True where p < threshold.
        """
        assert self.masked_responses is not None, "call compute_tuning() first"
        assert self.bins_sub_ids is not None, "call compute_tuning() first"

        n_cells = self.masked_responses.shape[0]
        p_values = np.full(n_cells, np.nan)

        # bin IDs are 1..n_bins; -1 marks subsampled-out trials
        valid_bins = np.arange(1, self.n_bins + 1)

        for i in range(n_cells):
            cell_resp = self.masked_responses[i]  # (n_trials_pref,)
            groups = []
            for bid in valid_bins:
                mask = self.bins_sub_ids == bid
                vals = cell_resp[mask]
                vals = vals[~np.isnan(vals)]
                if len(vals) >= 2:      # ANOVA needs ≥2 values per group
                    groups.append(vals)
            if len(groups) >= 2:        # need at least 2 groups to compare
                _, p_val = f_oneway(*groups)
                p_values[i] = p_val
            else:
                p_values[i] = 1.0       # insufficient data

        # NaN protection: cells with all-NaN responses are not significant
        if self.mean_all_responses is not None:
            nan_rows = np.all(np.isnan(self.mean_all_responses), axis=1)
            p_values[nan_rows] = 1.0

        significant_mask = p_values < threshold
        significant_mask[np.isnan(p_values)] = False

        self.anova_p_values = p_values
        self.significant_mask = significant_mask

    def compute_spearman(self, rho_threshold=0, p_threshold = 0.05):
        """Spearman rank correlation between response and running speed per cell, to test monotonicity of tuning.

        Note that only those neurons significantly tuned tested by :func:`significance_test` will be tested.

        Stores
        ------
        rho : np.ndarray, shape ``(n_cells_picked,)``
        rho_p_values : np.ndarray, shape ``(n_cells_picked,)``
        monotonic_mask : dict[str, np.array], with elements shape ``(n_cells_picked,)``
        """

        assert self.masked_responses is not None, "call compute_tuning() first"
        assert self.bins_sub_ids is not None, "call compute_tuning() first"
        assert self.significant_mask is not None, "call significance_test() first"

        seq_speed_ids = self.bins_sub_ids   # (n_trials_pref)
        seq_responses = self.masked_responses  # (n_cells_picked, n_trials_pref)

        # drop excluded trials (-1) before Spearman correlation
        valid_mask = seq_speed_ids != -1
        seq_speed_ids = seq_speed_ids[valid_mask]
        seq_responses = seq_responses[:, valid_mask]

        # per-cell Spearman: handle NaN from pref_trial_mask
        n_cells = seq_responses.shape[0]
        rho = np.full(n_cells, np.nan)
        rho_p_values = np.full(n_cells, np.nan)
        for i in range(n_cells):
            cell_valid = ~np.isnan(seq_responses[i])
            n_valid = cell_valid.sum()
            if n_valid <= 2:  # need >= 3 points for a meaningful correlation
                continue
            res = spearmanr(seq_speed_ids[cell_valid], seq_responses[i, cell_valid])
            rho[i] = res.statistic
            rho_p_values[i] = res.pvalue

        # NaN protection: cells with no valid responses should not be significant
        all_nan = np.all(np.isnan(seq_responses), axis=1)
        self.significant_mask[all_nan] = False

        # categorize monotonicity: positive, negative, or non-monotonic but tuned
        masking = (rho_p_values < p_threshold) & (np.abs(rho) >= rho_threshold)

        monotonic_mask = {
            'positive': (rho > 0) & masking & self.significant_mask,
            'negative': (rho < 0) & masking & self.significant_mask,
            'non-monotonic': (rho_p_values > p_threshold) & self.significant_mask
        }

        self.rho = rho
        self.rho_p_values = rho_p_values
        self.monotonic_mask = monotonic_mask


    # ------------- plotting & printing -------------

    def print_tuned_cells(self):
        assert self.significant_mask is not None, "call significance_test() first"
        assert self.monotonic_mask is not None, "call compute_spearman() first"
        print(f"Significantly tuned neurons: #{self.significant_mask.sum()} \n {np.where(self.significant_mask)[0]}")

        for key in ('positive', 'negative', 'non-monotonic'):
            mask = self.monotonic_mask[key]
            print(f"{key} tuned neurons: #{mask.sum()} \n {np.where(mask)[0]}")
            print(self.rho[mask].round(3))

    def _plot_tuning_curve(self, responses, ylabel, figsize, semcolor, label, ax):
        """Shared core: plot mean ± SEM tuning curve from a response matrix.

        Parameters
        ----------
        responses : np.ndarray, shape (n_cells_picked, n_bins)
            Response values to average and plot.
        ylabel : str
            Y-axis label.
        figsize : tuple
        semcolor : str or None
        label : str or None
        ax : plt.Axes or None

        Returns
        -------
        plt.Axes
        """
        assert self.bins_centers is not None, "call compute_tuning() first"

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        plot_mean_sem(ax, self.bins_centers, responses, semcolor,
                       label=label)
        ax.set_xlabel('running speed (cm/s)')
        ax.set_ylabel(ylabel)
        return ax


    def plot_tuning_curve(self, cells: list[int] | None = None, figsize=(5,3), semcolor=None, label=None, ax=None) -> plt.Axes:
        """Plot speed tuning curve with Mean and SEM over the given cells.

        Parameters
        ----------
        cell : list[int] or None, optional
            If given, plot the average of the given cells. Otherwise the average of all cells
            as a heatmap.
        ax : matplotlib.axes.Axes, optional
            Axes to draw into. Creates a new one if None.

        Returns
        -------
        plt.Axes
            The axes that were drawn into.
        """
        assert self.mean_all_responses is not None, "call compute_tuning() first"
        res_cells = self.mean_all_responses[cells] if cells else self.mean_all_responses
        return self._plot_tuning_curve(res_cells, 'average $\\Delta$F/F',
                                        figsize, semcolor, label, ax)

    def plot_tuning_curve_zscore(self, cells: list[int] | None = None, figsize=(5,3), semcolor=None, label=None, ax=None) -> plt.Axes:
        """Per-cell z-scored speed tuning curve with Mean and SEM.

        Z-scores each cell's tuning curve before averaging, so the figure
        shows relative modulation rather than absolute ΔF/F units.

        Parameters
        ----------
        cells : list[int] or None, optional
            Subset of cells to average. None = all cells.
        figsize : tuple, optional
        semcolor : str or None, optional
            Colour for the mean line and SEM shading.
        label : str or None, optional
            Legend label for the shaded region.
        ax : plt.Axes, optional
            Axes to draw into. Creates a new one if None.

        Returns
        -------
        plt.Axes
        """
        assert self.mean_all_responses is not None, "call compute_tuning() first"

        res_cells = self.mean_all_responses[cells] if cells else self.mean_all_responses
        mu = res_cells.mean(axis=1, keepdims=True)
        sd = res_cells.std(axis=1, keepdims=True)
        sd = np.where(sd == 0, 1.0, sd)
        res_z = (res_cells - mu) / sd

        return self._plot_tuning_curve(res_z, 'z-score',
                                        figsize, semcolor, label, ax)


    def _plot_tuning_by_monotonicity(self, responses, ylabel, axes, figsize,
                                     cells=None, spont_responses=None,
                                     mi=None, non_tuned_modulated_mask=None):
        """Shared core: plot monotonicity figure from a response matrix.

        Parameters
        ----------
        responses : np.ndarray, shape (n_cells_picked, n_bins)
            Response values to plot (raw mean or z-scored).
        ylabel : str
            Y-axis label for the leftmost subplot.
        axes : array-like of 3 or 4 Axes or None
        figsize : tuple
        cells : list[int] or None, optional
            Subset of cells to plot. None = all cells.
        spont_responses : np.ndarray or None, optional
            Spontaneous tuning responses of the same cells, shape (n_cells_picked, n_bins).
            When provided, each subplot shows the spontaneous tuning of its category
            as a baseline (same type of color) instead of non-significant cells (gray).
        mi : np.ndarray or None, optional
            Per-cell modulation index ``(n_cells_picked,)``. When provided, the mean MI
            of each category is shown in the subplot title.
        non_tuned_modulated_mask : np.ndarray or None, optional
            Boolean mask ``(n_cells_picked,)`` — cells that are running-modulated
            (from BinaryModulation) but not significantly speed-tuned. When provided,
            a 4th panel is drawn showing their average tuning curve.

        Returns
        -------
        plt.Figure
        """
        assert self.bins_centers is not None, "call compute_tuning() first"
        assert self.significant_mask is not None, "call significance_test() first"
        assert self.monotonic_mask is not None and self.rho is not None, "call compute_spearman() first"

        if cells is not None:
            cells_arr = list(cells) if not isinstance(cells, int) else [cells]
            responses = responses[cells_arr]
            if spont_responses is not None:
                spont_responses = spont_responses[cells_arr]
            bg = ~self.significant_mask[cells_arr]
            rho = self.rho[cells_arr]
            masks = {k: self.monotonic_mask[k][cells_arr]
                     for k in ('positive', 'negative', 'non-monotonic')}
            _mi = mi[cells_arr] if mi is not None else None
            if non_tuned_modulated_mask is not None:
                non_tuned_modulated_mask = non_tuned_modulated_mask[cells_arr]
        else:
            bg = ~self.significant_mask
            rho = self.rho
            masks = self.monotonic_mask
            _mi = mi

        cats = [
            ('positive',      POS_COLOR),
            ('negative',      NEG_COLOR),
            ('non-monotonic', NM_COLOR),
        ]
        n_cols = 4 if non_tuned_modulated_mask is not None else 3

        if axes is None:
            fig, axes = plt.subplots(1, n_cols, figsize=figsize)
        else:
            fig = axes.flat[0].figure if hasattr(axes, 'flat') else axes[0].figure

        for ax, (key, color) in zip(axes[:3], cats):
            mask = masks[key]
            n = mask.sum()
            label = 'non-mono' if key == 'non-monotonic' else key

            if spont_responses is not None:
                # spontaneous tuning of the same category cells with hollow markers as baseline
                if n > 0:
                    plot_mean_sem(ax, self.bins_centers, spont_responses[mask],
                                   color, alpha=0.15, linestyle='--',
                                   marker_facecolor='none')

            # category cells in colour
            if n > 0:
                plot_mean_sem(ax, self.bins_centers, responses[mask],
                               color, alpha=0.5)

            rho_mean = rho[mask].mean() if n > 0 else float('nan')
            # build title with optional MI
            title_parts = [f'{label} (n={n}']
            if not np.isnan(rho_mean):
                rho_str = f'{np.abs(rho_mean):.3f}'.replace('0.', '.', 1)
                title_parts.append(f', |$\\bar{{\\rho}}$|={rho_str}')
            if _mi is not None and n > 0:
                mi_mean = np.nanmean(_mi[mask])
                if np.isfinite(mi_mean):
                    mi_str = f'{mi_mean:.3f}'.replace('0.', '.', 1)
                    title_parts.append(f', MI={mi_str}')
            title_parts.append(')')
            ax.set_title(''.join(title_parts))
            ax.set_ylabel('')

        # 4th panel: non-tuned modulated cells
        if non_tuned_modulated_mask is not None and len(axes) > 3:
            ax4 = axes[3]
            nm_mask = non_tuned_modulated_mask
            n = nm_mask.sum()

            if spont_responses is not None and n > 0:
                plot_mean_sem(ax4, self.bins_centers, spont_responses[nm_mask],
                               NTM_COLOR, alpha=0.15, linestyle='--',
                               marker_facecolor='none')

            if n > 0:
                plot_mean_sem(ax4, self.bins_centers, responses[nm_mask],
                               NTM_COLOR, alpha=0.5)
            ax4.set_title(f'non-tuned mod. (n={n})')
            ax4.set_ylabel('')

        axes[0].set_ylabel(ylabel)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fig.tight_layout(rect=(0, 0, 1, 0.94))

        # legend: evoked vs spont line styles
        legend_handles = [
            Line2D([], [], color='black', marker='o', markersize=3,
                   markerfacecolor='black', markeredgecolor='black'),
            Line2D([], [], color='black', marker='o', markersize=3,
                   markerfacecolor='none', markeredgecolor='black',
                   linestyle='--'),
        ]
        legend_labels = ['Evoked', 'Spont']
        fig.legend(legend_handles, legend_labels, loc='upper center',
                   ncol=2, fontsize=12, frameon=False,
                   bbox_to_anchor=(0.5, 0.96))

        return fig

    def plot_tuning_by_monotonicity(self, axes=None, figsize=(10, 3.5),
                                    cells=None, spontaneous: 'SpeedTuning | None' = None,
                                    mi: np.ndarray | None = None,
                                    modulated_mask: np.ndarray | None = None) -> plt.Figure:
        """Subplots: tuning curves for positive / negative / non-monotonic cells separately,
        plus an optional 4th panel for non-tuned modulated cells.

        Parameters
        ----------
        axes : array-like of 3 or 4 Axes or None, optional
        figsize : tuple, optional
        cells : list[int] or None, optional
        spontaneous : SpeedTuning or None, optional
            A SpeedTuning instance for spontaneous activity. When provided, each
            subplot shows the spontaneous tuning of its category as a gray baseline.
        mi : np.ndarray or None, optional
            Per-cell modulation index ``(n_cells_picked,)`` from
            :class:`BinaryModulation`. When provided, each subplot title
            includes the mean MI of that category.
        modulated_mask : np.ndarray or None, optional
            Boolean mask ``(n_cells_picked,)`` for running-modulated cells
            (from :attr:`BinaryModulation.tuned_mask`). When provided, a 4th
            panel shows the average tuning of modulated but non-tuned cells.

        Returns
        -------
        plt.Figure
        """
        assert self.mean_all_responses is not None, "call compute_tuning() first"

        # align spontaneous responses: index by self.neuron_mask so cell count matches
        if spontaneous is not None:
            spont_resps = spontaneous.mean_all_responses
            if self.neuron_mask is not None:
                spont_resps = spont_resps[self.neuron_mask]
        else:
            spont_resps = None

        # align modulated_mask: index by self.neuron_mask if subsetting was applied
        if modulated_mask is not None and self.neuron_mask is not None:
            modulated_mask = modulated_mask[self.neuron_mask]

        non_tuned_mod = None
        if modulated_mask is not None:
            non_tuned_mod = modulated_mask & ~self.significant_mask

        return self._plot_tuning_by_monotonicity(
            self.mean_all_responses, 'average $\\Delta$F/F', axes, figsize, cells,
            spont_responses=spont_resps, mi=mi,
            non_tuned_modulated_mask=non_tuned_mod)

    def plot_tuning_by_monotonicity_zscore(self, axes=None, figsize=(10, 3.5),
                                           cells=None, spontaneous: 'SpeedTuning | None' = None,
                                           mi: np.ndarray | None = None,
                                           modulated_mask: np.ndarray | None = None) -> plt.Figure:
        """Subplots with per-cell z-scored tuning curves for each monotonicity category,
        plus an optional 4th panel for non-tuned modulated cells.

        Parameters
        ----------
        axes : array-like of 3 or 4 Axes or None, optional
        figsize : tuple, optional
        cells : list[int] or None, optional
        spontaneous : SpeedTuning or None, optional
            A SpeedTuning instance for spontaneous activity. When provided, each
            subplot shows the spontaneous tuning of its category as a gray baseline
            (also z-scored).
        mi : np.ndarray or None, optional
            Per-cell modulation index ``(n_cells_picked,)`` from
            :class:`BinaryModulation`. When provided, each subplot title
            includes the mean MI of that category.
        modulated_mask : np.ndarray or None, optional
            Boolean mask ``(n_cells_picked,)`` for running-modulated cells
            (from :attr:`BinaryModulation.tuned_mask`). When provided, a 4th
            panel shows the average tuning of modulated but non-tuned cells.

        Returns
        -------
        plt.Figure
        """
        assert self.mean_all_responses is not None, "call compute_tuning() first"

        mu = self.mean_all_responses.mean(axis=1, keepdims=True)
        sd = self.mean_all_responses.std(axis=1, keepdims=True)
        sd = np.where(sd == 0, 1.0, sd)  # avoid division by zero for flat cells
        responses_z = (self.mean_all_responses - mu) / sd

        # z-score spontaneous data using the same mean/sd
        if spontaneous is not None:
            assert spontaneous.mean_all_responses is not None, \
                "spontaneous SpeedTuning: call compute_tuning() first"
            # align spontaneous responses to match this instance's cells
            spont_resps = spontaneous.mean_all_responses
            if self.neuron_mask is not None:
                spont_resps = spont_resps[self.neuron_mask]
            spont_z = (spont_resps - mu) / sd
        else:
            spont_z = None

        # align modulated_mask: index by self.neuron_mask if subsetting was applied
        if modulated_mask is not None and self.neuron_mask is not None:
            modulated_mask = modulated_mask[self.neuron_mask]

        non_tuned_mod = None
        if modulated_mask is not None:
            non_tuned_mod = modulated_mask & ~self.significant_mask

        return self._plot_tuning_by_monotonicity(
            responses_z, 'z-scored', axes, figsize, cells,
            spont_responses=spont_z, mi=mi,
            non_tuned_modulated_mask=non_tuned_mod)


def plot_tuning_curves_grid(tunings: dict[str, SpeedTuning],
                             labels: list[str] | None = None,
                             cells: int | list[int] | None = None,
                             figsize=(8, 6),
                             show_rho: bool = False, zscore=False,
                             modulated_mask: dict[str, np.ndarray] | None = None) -> plt.Figure:
    """Plot tuning curves for all stimuli in a 2x2 grid.

    Parameters
    ----------
    tunings : dict[str, SpeedTuning]
        Mapping from stimulus label to SpeedTuning.
    labels : list[str] or None
        Labels to use (and order). Defaults to tunings keys.
    cells : int or list[int] or None
        Cell index(es) to plot (0-based).
        Single int → one cell; list → average of those cells;
        None → all cells.
    figsize : tuple, optional
        Figure size, by default (8, 6).
    show_rho : bool, optional
        If True, display the mean Spearman rho of plotted cells in each subplot title.
    modulated_mask : dict[str, np.ndarray] or None, optional
        Per-stimulus boolean masks ``(n_cells,)`` for running-modulated cells.
        When provided, only modulated cells are plotted (using SpeedTuning's
        ``neuron_mask`` for non-spontaneous stimuli; for spontaneous, the
        mask is used directly since its SpeedTuning has ``neuron_mask=None``).

    Returns
    -------
    plt.Figure
    """
    if labels is None:
        labels = list(tunings.keys())
    assert len(labels) <= 4, "max 4 stimuli for 2x2 grid"

    if isinstance(cells, int):
        cells = [cells]

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)

    for i, lbl in enumerate(labels):
        ax = axes.flat[i]
        t = tunings[lbl]

        # Determine cells to plot: restrict to modulated cells when mask given
        plot_cells = cells
        if modulated_mask is not None:
            mm = modulated_mask.get(lbl)
            if mm is not None:
                if t.neuron_mask is not None:
                    # SpeedTuning already filtered by neuron_mask (= modulated cells)
                    # No need to re-index — all cells are modulated
                    plot_cells = None
                else:
                    # neuron_mask is None (spontaneous) — filter to modulated cells
                    mod_idx = np.where(mm)[0]
                    plot_cells = mod_idx.tolist() if len(mod_idx) > 0 else []

                if plot_cells is not None and len(plot_cells) == 0:
                    ax.set_visible(False)
                    continue

        if zscore:
            t.plot_tuning_curve_zscore(cells=plot_cells, ax=ax, semcolor='gray')
        else:
            t.plot_tuning_curve(cells=plot_cells, ax=ax, semcolor='gray')
            # ax.set_ylim(bottom=0)
        title = stim_to_short(lbl)
        if show_rho:
            assert t.rho is not None, "call compute_spearman() first"
            rho_vals = t.rho[plot_cells] if plot_cells is not None else t.rho
            title += f'  ($\\bar{{\\rho}}$={rho_vals.mean():.3f})'
        ax.set_title(title)

    # x label only on bottom row
    for ax in axes[1, :]:
        ax.set_xlabel('running speed (cm/s)')
    for ax in axes[0, :]:
        ax.set_xlabel('')
    # y label only on left column
    for ax in axes[:, 1]:
        ax.set_ylabel('')

    # hide unused subplots
    for i in range(len(labels), 4):
        axes.flat[i].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig, axes


def plot_modulated_counts(modulated_mask: dict[str, np.ndarray],
                           total_cells: int | None = None,
                           ax: plt.Axes = None,
                           figsize=(4, 3)) -> plt.Axes:
    """Bar chart of modulated neuron counts per stimulus.

    Parameters
    ----------
    modulated_mask : dict[str, np.ndarray]
        Per-stimulus boolean masks ``(n_cells,)`` for running-modulated cells.
    total_cells : int or None, optional
        If given, a dashed horizontal reference line is drawn at this height.
    ax : plt.Axes, optional
    figsize : tuple, optional

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    labels = list(modulated_mask.keys())
    counts = [int(modulated_mask[lbl].sum()) for lbl in labels]
    short_labels = [stim_to_short(lbl) for lbl in labels]

    x = np.arange(len(labels))
    ax.bar(x, counts, 0.55, color='lightgray', edgecolor='gray', linewidth=0.8)

    for i, count in enumerate(counts):
        ax.text(i, count + 0.3, str(count), ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    if total_cells is not None:
        ax.axhline(y=total_cells, color='gray', linestyle='--',
                   linewidth=0.8, alpha=0.7)
        ax.text(len(labels) - 1, total_cells + 0.3,
                f'total={total_cells}', ha='right', va='bottom',
                fontsize=10, color='gray')

    ax.set_xticks(x)
    ax.set_xticklabels(short_labels)
    ax.set_ylabel('modulated neurons')
    ax.margins(y=0.15)
    return ax

def plot_modulated_venn(modulated_mask: dict[str, np.ndarray],
                         figsize=(6, 6),
                         ax: plt.Axes = None) -> plt.Figure:
    """Venn diagram of running-modulated neurons across stimuli.

    Of the four stimulus conditions (DG, SG, NS, Spont), the one with the
    *fewest* modulated neurons is excluded from the 3-set Venn and shown
    as a text note at the bottom instead.

    Parameters
    ----------
    modulated_mask : dict[str, np.ndarray]
        Per-stimulus boolean masks ``(n_cells,)`` for running-modulated cells.
    figsize : tuple, optional
        Figure size (only used when *ax* is not provided).
    ax : plt.Axes, optional
        Axes to draw into. Creates a new figure if None.

    Returns
    -------
    plt.Figure
    """
    # ---- raw sets for all 4 stimuli ----
    sets = {
        'DG':    set(np.where(modulated_mask['drifting_gratings'])[0]),
        'SG':    set(np.where(modulated_mask['static_gratings'])[0]),
        'NS':    set(np.where(modulated_mask['natural_scenes'])[0]),
        'Spont': set(np.where(modulated_mask['spontaneous'])[0]),
    }

    # ---- determine which stimulus to exclude (fewest modulated) ----
    sizes = {k: len(v) for k, v in sets.items()}
    excluded_key = min(sizes, key=sizes.get)
    # Venn can only show 3 sets; use the 3 largest.
    venn_keys = sorted(sets, key=lambda k: sizes[k], reverse=True)[:3]

    # alias for convenience
    A, B, C = venn_keys

    # ---- region computations ----
    sA, sB, sC = sets[A], sets[B], sets[C]
    regions = {
        f'only_{A}':       sorted(sA - sB - sC),
        f'only_{B}':       sorted(sB - sA - sC),
        f'only_{C}':       sorted(sC - sA - sB),
        f'{A}∩{B}':        sorted((sA & sB) - sC),
        f'{A}∩{C}':        sorted((sA & sC) - sB),
        f'{B}∩{C}':        sorted((sB & sC) - sA),
        f'{A}∩{B}∩{C}':   sorted(sA & sB & sC),
    }

    # ---- figure / axes ----
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    ax.set_aspect('equal')

    # ---- three-circle geometry ----
    r = 1.25
    side = 1.45
    h = side * 3**0.5 / 2

    # Assign physical triangle positions (bottom-left, bottom-right, top).
    # Prefer SG/DG on the left, NS on the right, Spont on top.
    _order = {'SG': 0, 'DG': 1, 'NS': 2, 'Spont': 3}
    sorted_vk = sorted(venn_keys, key=lambda k: _order.get(k, 9))
    bot_left_k, bot_right_k, top_k = (
        sorted_vk[0], sorted_vk[1], sorted_vk[2]
    )

    cen = {}
    label_pos = {}
    label_align = {}

    cen[bot_left_k] = np.array([-side / 2, -h / 3])
    label_pos[bot_left_k] = np.array([-side / 2 - r * 0.7, -h / 3 - r * 0.75])
    label_align[bot_left_k] = ('right', 'top')

    cen[bot_right_k] = np.array([side / 2, -h / 3])
    label_pos[bot_right_k] = np.array([side / 2 + r * 0.7, -h / 3 - r * 0.75])
    label_align[bot_right_k] = ('left', 'top')

    cen[top_k] = np.array([0, 2 * h / 3])
    label_pos[top_k] = np.array([0, 2 * h / 3 + r + 0.1])
    label_align[top_k] = ('center', 'bottom')

    color = "#7F8C8D"

    for name, pos in cen.items():
        ax.add_patch(Circle(pos, r, facecolor=color,
                            edgecolor=color, alpha=0.15, lw=2, zorder=1))
        l_pos, (ha, va) = label_pos[name], label_align[name]
        display_name = name if name == 'Spont' else name[:2].upper()
        ax.text(l_pos[0], l_pos[1], display_name,
                ha=ha, va=va, fontsize=16, fontweight='bold',
                color=color, zorder=4)

    # ---- text positions for the 7 regions (relative to circle centres) ----
    txt_offsets = {
        'top':        np.array([0,     1.50]),
        'bot_left':   np.array([-1.25, -0.65]),
        'bot_right':  np.array([1.25,  -0.65]),
        'top∩left':   np.array([-0.85,  0.45]),
        'top∩right':  np.array([0.85,   0.45]),
        'left∩right': np.array([0,     -0.85]),
        'all':        np.array([0,      0.05]),
    }

    # map region name → offset key
    def _offset_key(region_name, tl, tr, tp):
        parts = region_name.split('∩')
        # Strip 'only_' prefix so 'only_NS' → 'NS'
        parts = [p.split('only_')[-1] for p in parts]
        if len(parts) == 3:
            return 'all'
        if len(parts) == 2:
            if tp in parts:
                # intersection with top
                return 'top∩left' if tl in parts else 'top∩right'
            else:
                return 'left∩right'
        # single-set region
        p = parts[0]
        if p == tp:
            return 'top'
        if p == tl:
            return 'bot_left'
        return 'bot_right'

    for key, idxs in regions.items():
        cnt = len(idxs)
        if cnt == 0:
            continue
        ok = _offset_key(key, bot_left_k, bot_right_k, top_k)
        pos = txt_offsets[ok]

        ax.text(pos[0], pos[1], str(cnt),
                ha='center', va='center', fontsize=17, fontweight='bold', zorder=4)

        show_cells = idxs[:5]
        cell_str = ', '.join(map(str, show_cells))
        if cnt > 5:
            cell_str += ', …'
        wrapped_str = textwrap.fill(cell_str, width=16)
        ax.text(pos[0], pos[1] - 0.18, wrapped_str,
                ha='center', va='top', fontsize=7, color='#444444', zorder=4)

    # ---- excluded group note at bottom ----
    excluded_items = sorted(sets[excluded_key])
    exc_str = ', '.join(str(c) for c in excluded_items)
    ax.text(0, -2.1,
            f'{excluded_key} modulated (excluded): cell {exc_str}',
            ha='center', fontsize=9, style='italic', color='#666666', zorder=4)

    # ---- not-modulated count ----
    all_mod = set()
    for s in sets.values():
        all_mod |= s
    n_total = len(next(iter(modulated_mask.values())))
    outside = n_total - len(all_mod)
    ax.text(2.30, -1.9, f'Not modulated\n{outside}',
            ha='center', va='center', fontsize=10, fontweight='bold', color='#666666')

    ax.set_xlim(-2.6, 2.8)
    ax.set_ylim(-2.4, 2.7)
    ax.axis('off')
    
    fig.tight_layout()
    return fig


def plot_monotonicity_stacked_bar(tunings: dict[str, SpeedTuning],
                                   ax: plt.Axes = None,
                                   colors: dict[str, str]| None = None,
                                   modulated_mask: dict[str, np.ndarray] | None = None,
                                   figsize=(5, 4)) -> plt.Axes:
    """Stacked bar chart: for each stimulus, breakdown of significantly
    speed-tuned neurons by monotonicity (positive / negative / non-monotonic).

    When *modulated_mask* is provided, a wider light-grey background bar shows
    the total modulated count, and the stacked bars are restricted to the
    modulated pool so the stacked total never exceeds the background.

    Parameters
    ----------
    tunings : dict[str, SpeedTuning]
        Mapping from stimulus label to SpeedTuning (must have
        ``compute_spearman()`` called).
    ax : plt.Axes, optional
    colors : dict[str, str], optional
        Category colours. Default: positive=POS_COLOR (red),
        negative=NEG_COLOR (blue), non-monotonic=NM_COLOR (grey).
    modulated_mask : dict[str, np.ndarray] | None, optional
        Per-stimulus boolean masks ``(n_cells,)`` for running-modulated cells.
        When provided, the stacked bars are restricted to modulated cells and a
        wider background bar shows the modulated pool size.

    Returns
    -------
    plt.Axes
    """
    _, ax = _ensure_ax(ax, figsize=figsize)

    if colors is None:
        colors = {'positive': POS_COLOR, 'negative': NEG_COLOR,
                  'non-monotonic': NM_COLOR}

    labels = list(tunings.keys())
    categories = ['non-monotonic', 'negative', 'positive']

    x = np.arange(len(labels))

    # ---- wider background bar: total modulated count ----
    if modulated_mask is not None:
        n_total = len(next(iter(tunings.values())).rho)
        mod_counts = np.array([
            int(modulated_mask[lbl].sum())
            if lbl in modulated_mask and modulated_mask[lbl] is not None
            else n_total
            for lbl in labels
        ])
        ax.bar(x, mod_counts, 0.6, bottom=0, color='lightgray', alpha=0.35,
               edgecolor='gray', linewidth=0.6, label='modulated', zorder=0)
        for i in range(len(labels)):
            ax.text(i, mod_counts[i] + 0.5, str(int(mod_counts[i])),
                    ha='center', va='bottom', fontsize=9, color='gray',
                    fontweight='bold')
        ax.margins(y=0.1)

    # ---- stacked bar: tuned neurons restricted to modulated pool ----
    counts = {}
    for lbl in labels:
        t = tunings[lbl]
        assert t.monotonic_mask is not None, "call compute_spearman() first"
        assert t.significant_mask is not None, "call significance_test() first"

        mm = modulated_mask.get(lbl) if modulated_mask else None
        if mm is not None:
            n_tuning = len(t.monotonic_mask['positive'])
            if len(mm) != n_tuning:
                counts[lbl] = {
                    c: t.monotonic_mask[c].sum() for c in categories
                }
            else:
                counts[lbl] = {
                    c: (t.monotonic_mask[c] & mm).sum() for c in categories
                }
        else:
            sig = t.significant_mask
            if sig is None or sig.sum() == 0:
                counts[lbl] = {c: 0.0 for c in categories}
            else:
                counts[lbl] = {
                    c: t.monotonic_mask[c][sig].sum() for c in categories
                }

    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))

    for cat in categories:
        vals = [counts[l][cat] for l in labels]
        ax.bar(x, vals, 0.55, bottom=bottom, label=cat,
               color=colors[cat], edgecolor='white', linewidth=0.5)
        bottom += vals

    # add count labels in the middle of each segment
    for i, lbl in enumerate(labels):
        y_offset = 0.0
        for cat in categories:
            v = counts[lbl][cat]
            if v > 0:
                y_mid = y_offset + v / 2
                ax.text(i, y_mid, f'{int(v)}', ha='center', va='center',
                        fontsize=8, color='white', fontweight='bold')
            y_offset += v

    ax.set_xticks(x)
    ax.set_xticklabels([stim_to_short(l) for l in labels])
    ax.set_ylabel('# neurons')
    ax.legend(fontsize=9)
    return ax

def plot_rho_pairwise_scatter(tuning_a: SpeedTuning, tuning_b: SpeedTuning,
                               label_a: str, label_b: str,
                               ax: plt.Axes = None) -> plt.Axes:
    """Scatter plot comparing Spearman *rho* across two stimulus conditions.

    Points are coloured by joint significance and monotonicity:
    significant in both with same direction / different direction,
    significant in only one condition, or neither.

    Parameters
    ----------
    tuning_a, tuning_b : SpeedTuning
        Must have ``compute_spearman()`` called.
    label_a, label_b : str
        Axis labels.
    ax : plt.Axes, optional

    Returns
    -------
    plt.Axes
    """
    assert tuning_a.rho is not None, "call compute_spearman() first"
    assert tuning_b.rho is not None, "call compute_spearman() first"
    assert tuning_a.significant_mask is not None, "call significance_test() first"
    assert tuning_b.significant_mask is not None, "call significance_test() first"

    _, ax = _ensure_ax(ax, figsize=(5, 5))

    rho_a, rho_b = tuning_a.rho, tuning_b.rho
    sig_a, sig_b = tuning_a.significant_mask, tuning_b.significant_mask

    both = sig_a & sig_b
    only_a = sig_a & ~sig_b
    only_b = sig_b & ~sig_a
    neither = ~sig_a & ~sig_b

    ax.scatter(rho_a[neither], rho_b[neither],
               c='gray', s=10, label='neither tuned', alpha=0.7)
    ax.scatter(rho_a[only_a], rho_b[only_a],
               c=POS_COLOR, s=15, label=f'only {label_a}', alpha=0.7)
    ax.scatter(rho_a[only_b], rho_b[only_b],
               c=NEG_COLOR, s=15, label=f'only {label_b}', alpha=0.7)
    ax.scatter(rho_a[both], rho_b[both],
            c='#2ECC71', s=20, label='both tuned', alpha=0.8)

    # diagonal reference line
    lim = max(np.nanmax(np.abs(rho_a)), np.nanmax(np.abs(rho_b)))
    ax.plot([-lim, lim], [-lim, lim], 'k--', lw=0.6, alpha=0.3)
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    ax.set_xlabel(f'$\\rho$ ({label_a})')
    ax.set_ylabel(f'$\\rho$ ({label_b})')
    ax.set_title('Spearman $\\rho$ to running speed of cells')
    ax.legend(fontsize=7, loc='lower right')
    ax.set_aspect('equal')
    return ax

def plot_monotonicity_grid(tunings: dict[str, SpeedTuning],
                            responsive: dict[str, np.ndarray] | None = None,
                            speed_tuned: dict[str, np.ndarray] | None = None,
                            figsize=(6, 10),
                            legend_headers: dict[str, str] | None = None,
                            modulated_mask: dict[str, np.ndarray] | None = None) -> plt.Figure:
    """Grey background for responsive cells, coloured ρ text (by monotonicity
    category) with alpha = |ρ| for speed-tuned cells.

    Parameters
    ----------
    tunings : dict[str, SpeedTuning]
        SpeedTuning results per stimulus label.
    responsive : dict[str, np.ndarray] | None
        Per-stimulus boolean masks ``(n_cells,)`` for stimulus-driven responsive
        cells (grey background).
    speed_tuned : dict[str, np.ndarray] | None
        Per-stimulus boolean masks ``(n_cells,)`` for speed-tuned cells (diagonal
        hatch overlay).
    figsize : tuple
        Figure size.
    legend_headers : dict[str, str] | None
        Insert section header text before a legend entry. Keys are existing legend
        labels (e.g. ``'responsive'``), values are the header text to place above
        them (e.g. ``{'responsive': 'from metadata'}``).
    modulated_mask : dict[str, np.ndarray] | None
        Per-stimulus boolean masks ``(n_cells,)`` for running-modulated cells
        (e.g. from :attr:`BinaryModulation.tuned_mask`). Modulated-tuned cells
        are marked with a bold border.
    """
    labels = list(tunings.keys())
    # assert all tunings have computed results
    for t in tunings.values():
        assert t.rho is not None, "call compute_spearman() first"
        assert t.significant_mask is not None, "call significance_test() first"
        assert t.monotonic_mask is not None, "call compute_spearman() first"
        assert t.anova_p_values is not None, "call significance_test() first"

    DARK_GRAY = (0.3, 0.3, 0.3)

    J = len(labels)
    I = len(next(iter(tunings.values())).rho)

    COLS = {'positive': POS_RGB, 'negative': NEG_RGB, 'non-monotonic': NM_RGB}
    LG = (0.93, 0.93, 0.93)

    resp_mask = np.zeros((I, J), dtype=bool)
    if responsive is not None:
        for j, lbl in enumerate(labels):
            if lbl in responsive:
                resp_mask[:, j] = responsive[lbl]
    n_resp = resp_mask.sum(axis=1)

    mean_abs_rho = np.mean([np.abs(tunings[lbl].rho) for lbl in labels], axis=0)
    ns = ~np.any([tunings[lbl].significant_mask for lbl in labels], axis=0)
    mean_abs_rho[ns] = 0

    # per-cell global category: use the stimulus with the highest |ρ|
    global_cat = np.zeros(I, dtype=int)  # 0=not tuned, 1=non-mono, 2=negative, 3=positive
    for ci in range(I):
        best_rho = 0
        for j, lbl in enumerate(labels):
            t = tunings[lbl]
            if t.significant_mask[ci]:
                arho = np.abs(t.rho[ci])
                if arho > best_rho:
                    best_rho = arho
                    if t.monotonic_mask['positive'][ci]:
                        global_cat[ci] = 3
                    elif t.monotonic_mask['negative'][ci]:
                        global_cat[ci] = 2
                    else:
                        global_cat[ci] = 1

    order = np.lexsort((-n_resp, -mean_abs_rho, -global_cat))
    resp_mask = resp_mask[order]

    # --- speed-tuned mask (same ordering) ---
    st_mask = np.zeros((I, J), dtype=bool)
    if speed_tuned is not None:
        for j, lbl in enumerate(labels):
            if lbl in speed_tuned:
                st_mask[:, j] = speed_tuned[lbl]
    st_mask = st_mask[order]

    # --- modulated mask (same ordering) ---
    mod_mask = np.zeros((I, J), dtype=bool)
    if modulated_mask is not None:
        for j, lbl in enumerate(labels):
            if lbl in modulated_mask:
                mod_mask[:, j] = modulated_mask[lbl]
    mod_mask = mod_mask[order]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=figsize)

    # grey background for responsive cells
    img = np.ones((I, J, 3))
    img[resp_mask] = LG
    ax.imshow(img, aspect='auto', interpolation='nearest')

    # diagonal hatching for speed-tuned cells (under the text)
    if speed_tuned is not None:
        for i in range(I):
            for j in range(J):
                if st_mask[i, j]:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           fill=False, hatch='///', linewidth=0,
                                           color='black', alpha=0.3))

    # coloured ρ text with alpha = |ρ|
    for i in range(I):
        ci = order[i]
        for j, lbl in enumerate(labels):
            t = tunings[lbl]
            if t.significant_mask[ci]:
                if t.monotonic_mask['positive'][ci]:
                    a = np.clip(np.abs(t.rho[ci]) * 8, 0, 1)
                    col = tuple((1 - a) * c + a * COLS['positive'][k]
                                for k, c in enumerate(DARK_GRAY))
                elif t.monotonic_mask['negative'][ci]:
                    a = np.clip(np.abs(t.rho[ci]) * 8, 0, 1)
                    col = tuple((1 - a) * c + a * COLS['negative'][k]
                                for k, c in enumerate(DARK_GRAY))
                else:
                    col = COLS['non-monotonic']
                txt = f'{t.rho[ci]:.3f}'.replace('0.', '.', 1)
                p = t.anova_p_values[ci]
                ax.text(j, i, txt, ha='center', va='center',
                        fontsize=8, color=col, fontweight='bold')
                stars = _pvalue_stars(p)
                if stars:
                    ax.text(j + 0.2, i, stars, ha='left', va='center',
                            fontsize=9, color='black', fontweight='bold', alpha=0.7)

    # thick border for modulated cells
    if modulated_mask is not None:
        for i in range(I):
            for j in range(J):
                if mod_mask[i, j]:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           fill=False, edgecolor='black',
                                           linewidth=2.5))

    ax.set(xticks=range(J), xticklabels=[stim_to_short(l) for l in labels],
           yticks=range(I), yticklabels=order,
           ylabel='cell #', title='Speed-tuning by monotonicity')
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    # legend
    legend_handles = [
        Patch(facecolor=COLS['positive']),
        Patch(facecolor=COLS['negative']),
        Patch(facecolor=COLS['non-monotonic']),
        Patch(facecolor=LG),
    ]
    legend_labels = ['positive', 'negative', 'non-monotonic', 'responsive']
    if speed_tuned is not None:
        legend_handles.append(Patch(facecolor='none', hatch='///', linewidth=0,
                                    edgecolor='black'))
        legend_labels.append('speed-tuned')
    if modulated_mask is not None:
        legend_handles.append(Patch(facecolor='none', edgecolor='black',
                                    linewidth=2.5))
        legend_labels.append('modulated')

    # significance stars legend
    blank = Line2D([], [], color='none', marker='none', linestyle='')
    legend_handles.extend([blank, blank, blank])
    legend_labels.extend(['p\u2264.01 *',  'p\u2264.001 **', 'p\u2264.0001 ***'])

    # insert section headers before specified entries
    if legend_headers:
        padded_handles = []
        padded_labels = []
        for h, l in zip(legend_handles, legend_labels):
            if l in legend_headers:
                # blank spacer line
                padded_handles.append(
                    Line2D([], [], color='none', marker='none', linestyle=''))
                padded_labels.append('')
                # section header text
                padded_handles.append(
                    Line2D([], [], color='none', marker='none', linestyle=''))
                padded_labels.append(legend_headers[l])
            padded_handles.append(h)
            padded_labels.append(l)
        legend_handles, legend_labels = padded_handles, padded_labels

    ax.legend(legend_handles, legend_labels,
              loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9, frameon=False)
    fig.tight_layout()
    return fig


def plot_modulated_tuned_grid(
    tunings: dict[str, SpeedTuning],
    modulated_mask: dict[str, np.ndarray],
    figsize=(6, 10),
) -> plt.Figure:
    """Grid map combining BinaryModulation and SpeedTuning results.

    Each row is a cell, each column a stimulus. The grid shows:
    - Light blue background for running-modulated cells (from BinaryModulation).
    - Coloured ρ text for tuned cells with monotonicity category (from SpeedTuning).
    - p-value significance stars.

    Cells are sorted by total tuned count across stimuli.

    Parameters
    ----------
    tunings : dict[str, SpeedTuning]
        SpeedTuning results per stimulus label (must have ``run()`` called).
    modulated_mask : dict[str, np.ndarray]
        Per-stimulus boolean masks ``(n_cells,)`` for running-modulated cells
        (e.g. from :attr:`BinaryModulation.tuned_mask`).
    figsize : tuple
        Figure size.

    Returns
    -------
    plt.Figure
    """
    labels = list(tunings.keys())
    for t in tunings.values():
        assert t.rho is not None, "call compute_spearman() first"
        assert t.significant_mask is not None, "call significance_test() first"
        assert t.monotonic_mask is not None, "call compute_spearman() first"
        assert t.anova_p_values is not None, "call significance_test() first"

    MOD_BG = (0.9, 0.9, 0.9)    # light gray for modulated

    COLS = {'positive': POS_RGB, 'negative': NEG_RGB, 'non-monotonic': NM_RGB}

    J = len(labels)
    # Always use full 47 cells from modulated_mask (not per-stimulus rho length)
    I = len(next(iter(modulated_mask.values())))

    # Expand per-stimulus SpeedTuning results to full (I,) arrays
    expanded = {}
    for lbl in labels:
        t = tunings[lbl]
        nm = t.neuron_mask
        if nm is not None:
            # Subsetted by neuron_mask — expand back to full cell count
            rho = np.full(I, np.nan)
            rho[nm] = t.rho
            sig = np.zeros(I, dtype=bool)
            sig[nm] = t.significant_mask
            lev = np.full(I, np.nan)
            lev[nm] = t.anova_p_values
            mono = {}
            for k in ('positive', 'negative', 'non-monotonic'):
                m = np.zeros(I, dtype=bool)
                m[nm] = t.monotonic_mask[k]
                mono[k] = m
        else:
            # All cells used (e.g., spontaneous with neuron_mask=None)
            rho = t.rho.copy()
            sig = t.significant_mask.copy()
            lev = t.anova_p_values.copy()
            mono = {k: v.copy() for k, v in t.monotonic_mask.items()}
        expanded[lbl] = {'rho': rho, 'sig': sig, 'lev': lev, 'mono': mono}

    # Build modulated background mask (already full cell count)
    mod_mask = np.zeros((I, J), dtype=bool)
    for j, lbl in enumerate(labels):
        if lbl in modulated_mask:
            mod_mask[:, j] = np.asarray(modulated_mask[lbl], dtype=bool)

    # NaN-out SpeedTuning data for non-modulated cells per stimulus,
    # so they never display even if mod_mask/ordering edge cases arise
    for j, lbl in enumerate(labels):
        col_mask = mod_mask[:, j]
        ex = expanded[lbl]
        ex['rho'][~col_mask] = np.nan
        ex['sig'][~col_mask] = False
        for k in ('positive', 'negative', 'non-monotonic'):
            ex['mono'][k][~col_mask] = False

    # Sort by mean |ρ| (desc), then by modulated (desc)
    rho_stack = np.abs(np.array([expanded[lbl]['rho'] for lbl in labels]))
    with np.errstate(invalid='ignore'):
        mean_abs_rho = np.nanmean(rho_stack, axis=0)
    ns = ~np.any([expanded[lbl]['sig'] for lbl in labels], axis=0)
    mean_abs_rho[ns] = 0

    modulated_any = np.zeros(I, dtype=bool)
    for j, lbl in enumerate(labels):
        if lbl in modulated_mask:
            modulated_any |= np.asarray(modulated_mask[lbl], dtype=bool)

    order = np.lexsort((-modulated_any.astype(int), -mean_abs_rho))

    # Only plot modulated neurons (modulated in at least one stimulus)
    order = order[modulated_any[order]]
    I_mod = len(order)

    mod_mask = mod_mask[order]

    fig, ax = plt.subplots(figsize=figsize)

    # Background: light blue for modulated, white otherwise
    img = np.ones((I_mod, J, 3))
    img[mod_mask] = MOD_BG
    ax.imshow(img, aspect='auto', interpolation='nearest')

    # Text: coloured ρ / placeholder '-' for modulated-but-not-tuned cells
    for i in range(I_mod):
        ci = order[i]
        for j, lbl in enumerate(labels):
            if not mod_mask[i, j]:
                continue  # not modulated in this stimulus — leave blank
            ex = expanded[lbl]
            if ex['sig'][ci]:
                if ex['mono']['positive'][ci]:
                    col = COLS['positive']
                elif ex['mono']['negative'][ci]:
                    col = COLS['negative']
                else:
                    col = COLS['non-monotonic']
                txt = f"{ex['rho'][ci]:.3f}".replace('0.', '.', 1)
                ax.text(j, i, txt, ha='center', va='center',
                        fontsize=8, color=col, fontweight='bold')
                # p-value stars
                p = ex['lev'][ci]
                stars = _pvalue_stars(p)
                if stars:
                    ax.text(j + 0.25, i, stars, ha='left', va='center',
                            fontsize=8, color='black', fontweight='bold', alpha=0.6)
            else:
                # Modulated but not speed-tuned — show placeholder
                ax.text(j, i, '-', ha='center', va='center',
                        fontsize=10, color='gray')

    ax.set(xticks=range(J), xticklabels=[stim_to_short(l) for l in labels],
           yticks=range(I_mod), yticklabels=order,
           ylabel='cell #')
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    # Legend
    blank = Line2D([], [], color='none', marker='none', linestyle='')
    spearman_header = Line2D([], [], color='none', marker='none', linestyle='')
    legend_handles = [
        Patch(facecolor=MOD_BG, label='modulated (BinaryMod.)'),
        spearman_header,
        Patch(facecolor=COLS['positive'], label='positive'),
        Patch(facecolor=COLS['negative'], label='negative'),
        Patch(facecolor=COLS['non-monotonic'], label='non-monotonic'),
    ]
    # significance stars
    legend_handles.extend([blank, blank, blank])
    legend_labels = [
        'modulated (BinaryMod.)',
        'Spearman $\\rho$',
        'positive', 'negative', 'non-monotonic',
        'p\u2264.01 *',  'p\u2264.001 **', 'p\u2264.0001 ***',
    ]
    ax.legend(legend_handles, legend_labels,
              loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9, frameon=False)
    fig.tight_layout()
    return fig


# ==============================================================================
# Analysis 2 — Binary Conditions (Running / Still)
# ==============================================================================


class BinaryModulation:
    """Compare neural responses under **running** vs. **still** conditions.

    Classifies each trial as **running** or **still** based on running speed
    criteria, computes the Modulation Index (MI), and fits a simple
    gain model relating running and still responses:

    .. math::
        R_{\\text{run}} = a \\cdot R_{\\text{still}} + b

    , where :math:`a` and :math:`b` denotes the "multiplicative" and "additive"
    components of running-speed modulation.

    Inspired by (Christensen & Pillow, 2022)

    Parameters
    ----------
    trial_data : TrialData
        Extracted per-trial responses and running speed.
    """

    def __init__(self, trial_data: TrialData):
        self._td = trial_data

        # filled by classify_trials()
        self.run_mask = None          # np.ndarray, shape (n_trials,)
        self.still_mask = None        # np.ndarray, shape (n_trials,)
        self.ignored_mask = None      # np.ndarray, shape (n_trials,)

        # filled by compute_mi()
        self.r_run = None             # np.ndarray, shape (n_cells,)
        self.r_still = None           # np.ndarray, shape (n_cells,)
        self.delta_r = None  # np.ndarray, shape (n_cells,); r_run - r_still
        self.mi = None                # np.ndarray, shape (n_cells,); sign-safe MI

        # filled by fit_gain_model()
        self.gain_a = None            # np.ndarray, shape (n_cells,)
        self.gain_b = None            # np.ndarray, shape (n_cells,)
        self.gain_r2 = None           # np.ndarray, shape (n_cells,)
        self.gain_valid = None        # np.ndarray of bool, shape (n_cells,)

        self.condition_still = None   # list length n_cells; each item shape (n_valid_conditions,)
        self.condition_run = None     # list length n_cells; each item shape (n_valid_conditions,)
        self.n_gain_conditions = None # np.ndarray, shape (n_cells,)

        # filled by compute_modulated_neurons()
        #
        # A paired t-test per cell on condition-level (R_run, R_still) pairs
        # (built by fit_gain_model, see condition_run / condition_still).
        # Note the distinction from the aggregate quantities above: delta_r
        # is a single per-cell number (overall mean running response minus
        # overall mean still response), so a t-test cannot be run on it
        # directly — the paired per-condition differences used here come
        # from condition_run - condition_still instead.
        self.stat = None            # np.ndarray, shape (n_cells,); t-statistic (positive = running > still) or mannwhitneyu statistic
        self.pval = None            # np.ndarray, shape (n_cells,); two-sided p-value
        self.modulated_mask = None    # np.ndarray of bool, shape (n_cells,); pval < threshold

    def classify_trials(
        self,
        run_threshold: float = 3.0,
        still_threshold: float = 0.5,
    ):
        """Label trials as **running**, **still**, or **ignored**.

        Criteria:

        - **running**: mean speed > ``run_threshold`` and all speed values
          exceed ``still_threshold``.
        - **still**: mean speed < ``still_threshold`` and no speed value
          exceeds ``run_threshold``.
        - All other trials are **ignored** (discarded).

        Stores
        ------
        run_mask : np.ndarray of bool, shape ``(n_trials,)``
        still_mask : np.ndarray of bool, shape ``(n_trials,)``
        ignored_mask : np.ndarray of bool, shape ``(n_trials,)``
        """
        speed = np.asarray(self._td.running_speed)

        if speed.ndim != 2:
            raise ValueError(
                f"Expected running_speed with shape (n_trials, duration), got {speed.shape}"
            )

        mean_speed = np.nanmean(speed, axis=1)
        min_speed = np.nanmin(speed, axis=1)
        max_speed = np.nanmax(speed, axis=1)

        self.run_mask = (mean_speed > run_threshold) & (min_speed > still_threshold)
        self.still_mask = (mean_speed < still_threshold) & (max_speed < run_threshold)
        self.ignored_mask = ~(self.run_mask | self.still_mask)

        return self.run_mask, self.still_mask, self.ignored_mask

    def compute_mi(self, epsilon: float = 1e-12):
        """Compute the sign-safe Modulation Index per cell.

        The formal metric for this analysis is the **sign-safe** MI, which
        keeps the signed numerator but uses an absolute-sum denominator:

        .. math::

            MI = \\frac{R_{\\text{run}} - R_{\\text{still}}}
                       {|R_{\\text{run}}| + |R_{\\text{still}}| + \\epsilon}

        Because ΔF/F responses are signed, the naive denominator
        :math:`R_{\\text{run}} + R_{\\text{still}}` can be near zero or
        negative, which makes the naive index unbounded and can flip its
        sign relative to :math:`R_{\\text{run}} - R_{\\text{still}}`. The
        absolute-sum denominator is always positive, so :attr:`mi` is bounded
        in ``[-1, 1]`` and ``sign(mi) == sign(delta_r)`` always holds. The
        raw / denominator-thresholded ("robust") variants are diagnostic
        only and live in ``mi_audit_utils`` — this class exposes a single MI.

        Parameters
        ----------
        epsilon : float, optional
            Small positive constant added to the denominator to avoid
            division by zero when both responses are exactly zero, by
            default ``1e-12``.

        Stores
        ------
        r_run : np.ndarray, shape ``(n_cells,)``
            Mean response over running trials.
        r_still : np.ndarray, shape ``(n_cells,)``
            Mean response over still trials.
        delta_r : np.ndarray, shape ``(n_cells,)``
            ``r_run - r_still`` (denominator-free sensitivity measure). This
            is one aggregate number per cell; it is *not* a paired set and
            cannot be fed to a per-cell t-test (see
            :meth:`compute_modulated_neurons`, which uses per-condition pairs).
        mi : np.ndarray, shape ``(n_cells,)``
            Sign-safe modulation index for each cell, bounded in ``[-1, 1]``.
        """
        if self.run_mask is None or self.still_mask is None:
            raise ValueError("Please run classify_trials() first.")

        response = np.asarray(self._td.responses)
        if response.ndim != 3:
            raise ValueError(
                f"Expected response with shape (n_cells, n_trial, duration), got {response.shape}"
            )

        # Average calcium response over the response window.
        response_mean = np.nanmean(response, axis=2)  # (n_cells, n_trials)

        n_cells = response_mean.shape[0]

        if self.run_mask.sum() == 0 or self.still_mask.sum() == 0:
            self.mi = np.full(n_cells, np.nan)
            self.r_run = np.full(n_cells, np.nan)
            self.r_still = np.full(n_cells, np.nan)
            self.delta_r = np.full(n_cells, np.nan)
            return

        # Average response across running / still trials for each cell.
        # Upcast to float64 before the ratio: ΔF/F is stored as float32, and
        # computing the denominator in float32 would perturb near-±1 indices
        # at the ~1e-8 level, which is enough to flip Spearman ranks in the
        # metadata validation. float64 keeps the sign-safe MI stable.
        self.r_run = np.nanmean(response_mean[:, self.run_mask], axis=1).astype(float)      # (n_cells,)
        self.r_still = np.nanmean(response_mean[:, self.still_mask], axis=1).astype(float)  # (n_cells,)
        self.delta_r = self.r_run - self.r_still

        denom = np.abs(self.r_run) + np.abs(self.r_still) + epsilon

        self.mi = np.full(n_cells, np.nan, dtype=float)
        finite = np.isfinite(self.r_run) & np.isfinite(self.r_still)
        self.mi[finite] = self.delta_r[finite] / denom[finite]

        return

    def get_condition_labels(self):
        """Build one visual-stimulus condition label for each trial.

        Returns
        -------
        labels : np.ndarray or None
            One label per trial. For spontaneous activity, returns None.
        """
        return _get_condition_labels(self._td)
        
    def collect_condition_pairs(self, min_trials_per_state: int = 2):
        """Collect condition-level mean responses for running and still trials.

        For each cell and each stimulus condition, computes the mean ΔF/F
        response separately for running and still trials. Only conditions
        with at least ``min_trials_per_state`` trials in both states are
        retained.

        This method decouples the condition-level data collection from the
        gain-model fitting step, so that ``condition_still`` and
        ``condition_run`` can be updated independently (e.g. before running
        :meth:`compute_modulated_neurons`).

        Parameters
        ----------
        min_trials_per_state : int, optional
            Minimum number of running and (separately) still trials a
            stimulus condition must have to be included, by default 2.

        Stores
        ------
        condition_still : list of np.ndarray
            Per-cell list; each element is an array of mean still responses,
            one per valid condition.
        condition_run : list of np.ndarray
            Per-cell list; each element is an array of mean running responses,
            one per valid condition.
        n_gain_conditions : np.ndarray, shape ``(n_cells,)``
            Number of valid conditions per cell.
        """
        assert self.run_mask is not None and self.still_mask is not None, "classify_trials() have to be ran first"

        response = np.asarray(self._td.responses)
        if response.ndim != 3:
            raise ValueError(
                f"Expected response with shape (n_cells, n_trial, duration), got {response.shape}"
            )
        response_mean = np.nanmean(response, axis=2)  # (n_cells, n_trials)

        n_cells, n_trials = response_mean.shape
        labels = self.get_condition_labels()

        self.n_gain_conditions = np.zeros(n_cells, dtype=int)
        self.condition_still = [
            np.array([], dtype=float) for _ in range(n_cells)
        ]
        self.condition_run = [
            np.array([], dtype=float) for _ in range(n_cells)
        ]

        if labels is None:
            # No visual-stimulus condition structure (e.g. spontaneous):
            # condition_still/condition_run stay as empty arrays.
            return

        unique_conditions = pd.unique(labels)
        for cell_index in range(n_cells):
            still_values = []
            run_values = []
            for condition in unique_conditions:
                mask_condition = np.array(
                    [label == condition for label in labels],
                    dtype=bool,
                )
                mask_c_still = mask_condition & self.still_mask
                mask_c_run = mask_condition & self.run_mask
                if mask_c_still.sum() < min_trials_per_state:
                    continue
                if mask_c_run.sum() < min_trials_per_state:
                    continue
                mean_still = np.nanmean(response_mean[cell_index, mask_c_still])
                mean_run = np.nanmean(response_mean[cell_index, mask_c_run])

                if np.isfinite(mean_still) and np.isfinite(mean_run):
                    still_values.append(mean_still)
                    run_values.append(mean_run)

            self.condition_run[cell_index] = np.asarray(run_values, dtype=float)
            self.condition_still[cell_index] = np.asarray(still_values, dtype=float)
            self.n_gain_conditions[cell_index] = len(run_values)

    def fit_gain_model(self, min_conditions: int = 3):
        """Fit linear gain model: :math:`R_{\\text{run}} = a \\cdot R_{\\text{still}} + b`.

        Parameters
        ----------
        min_conditions : int, optional
            Minimum number of valid condition pairs a cell must have before
            a line is fit at all, by default 3. A 2-point fit is always a
            perfect but meaningless line, so this is deliberately larger
            than the 2-point minimum ``np.polyfit`` would accept.

        Stores
        ------
        gain_a : np.ndarray, shape ``(n_cells,)``
            Multiplicative coefficient.
        gain_b : np.ndarray, shape ``(n_cells,)``
            Additive offset.
        gain_valid : np.ndarray of bool, shape ``(n_cells,)``
            True only where the fit used at least ``min_conditions`` points
            with non-degenerate ``still`` values, and produced finite slope,
            intercept, and R². Always fully initialized (all False for
            stimuli with no condition structure, e.g. spontaneous), never
            left as ``None``.
        """
        assert self._td.stimulus != 'spontaneous', "Spontaneous cannt fit gain model"
        assert self.condition_still is not None and self.condition_run is not None and self.n_gain_conditions is not None, "collect_condition_pairs() have to be ran first"

        n_cells = len(self.condition_still)
        self.gain_a = np.full(n_cells, np.nan)
        self.gain_b = np.full(n_cells, np.nan)
        self.gain_r2 = np.full(n_cells, np.nan)
        self.gain_valid = np.zeros(n_cells, dtype=bool)

        if n_cells == 0 or self.n_gain_conditions.max() == 0:
            # No condition structure (e.g. spontaneous) or no valid pairs
            # for any cell: gain_model is undefined, gain_valid stays all False.
            return 

        for cell_index in range(n_cells):
            still_values = self.condition_still[cell_index]
            run_values = self.condition_run[cell_index]

            if len(run_values) < min_conditions:
                continue

            # A (near-)constant set of still-condition values makes the
            # still -> run line unidentifiable (rank-deficient fit).
            if np.ptp(still_values) <= 1e-12:
                continue

            slope, intercept = np.polyfit(still_values, run_values, deg=1)

            # check the quality of the fit
            predicted_run = slope * still_values + intercept
            ss_res = np.sum((run_values - predicted_run) ** 2)
            ss_tot = np.sum((run_values - np.mean(run_values)) ** 2)
            if ss_tot > 1e-12:
                self.gain_r2[cell_index] = 1 - ss_res / ss_tot

            self.gain_a[cell_index] = slope
            self.gain_b[cell_index] = intercept

            self.gain_valid[cell_index] = (
                np.isfinite(slope)
                and np.isfinite(intercept)
                and np.isfinite(self.gain_r2[cell_index])
            )

    def compute_modulated_neurons(self, threshold: float = 0.05, min_conditions: int = 3):
        """Per-cell t-test comparing running vs. still responses.

        Behaviour depends on whether the stimulus has a condition structure:

        * **Stimuli with conditions** (e.g. ``drifting_gratings``): a **paired**
          t-test on condition-level (R_run, R_still) pairs. This controls for
          stimulus-condition identity — if running and still trials happen to
          cover different orientations or spatial frequencies, the pairing
          cancels that out, whereas a pooled trial-level t-test would confound
          stimulus tuning with speed tuning.

        * **Spontaneous** (no conditions): a standard **two-sample** t-test
          directly on the trial-level running vs. still responses.

        For stimuli with conditions, :meth:`collect_condition_pairs` (or
        :meth:`fit_gain_model`) must have been called first. Cells with fewer
        than ``min_conditions`` valid pairs (conditions) or trials (spontaneous)
        are left as NaN / excluded from ``modulated_mask``.

        Parameters
        ----------
        threshold : float, optional
            p-value threshold for ``modulated_mask``, by default 0.05.
        min_conditions : int, optional
            For stimuli with conditions: minimum number of valid condition
            pairs. For spontaneous: minimum number of trials per group
            (running and still). By default 3.

        Stores
        ------
        stat : np.ndarray, shape ``(n_cells,)``
            t-statistic (positive = running > still).
        pval : np.ndarray, shape ``(n_cells,)``
            Two-sided p-value for the test.
        modulated_mask : np.ndarray of bool, shape ``(n_cells,)``
            Cells with ``pval < threshold`` and enough data.
        """
        assert self.condition_still is not None and self.condition_run is not None, (
            "collect_condition_pairs() or fit_gain_model() must be "
            "called before compute_modulated_neurons()"
        )

        n_cells = len(self.condition_still)
        stat = np.full(n_cells, np.nan)
        pval = np.full(n_cells, np.nan)

        # Detect spontaneous (no condition structure) by checking the first cell.
        has_conditions = (
            n_cells > 0
            and self.condition_still[0] is not None
            and len(self.condition_still[0]) > 0
        )

        if not has_conditions:
            # --- Spontaneous: Wilcoxon rank-sum test on trial-level responses ---
            response = np.asarray(self._td.responses)          # (n_cells, n_trials, duration)
            if response.ndim == 3:
                response_mean = np.nanmean(response, axis=2)   # (n_cells, n_trials)
            else:
                response_mean = response

            if self.run_mask is None or self.still_mask is None:
                raise ValueError("Please run classify_trials() first.")

            for cell_index in range(n_cells):
                run_vals = response_mean[cell_index, self.run_mask]
                still_vals = response_mean[cell_index, self.still_mask]

                run_vals = run_vals[np.isfinite(run_vals)]
                still_vals = still_vals[np.isfinite(still_vals)]

                if len(run_vals) < min_conditions or len(still_vals) < min_conditions:
                    continue

                st, p = mannwhitneyu(run_vals, still_vals, alternative="two-sided")
                stat[cell_index] = st
                pval[cell_index] = p

        else:
            # --- Stimuli with conditions: paired t-test on condition means ---
            for cell_index in range(n_cells):
                still_c = self.condition_still[cell_index]
                run_c = self.condition_run[cell_index]

                if still_c is None or run_c is None or len(still_c) < min_conditions:
                    continue

                diffs = np.asarray(run_c, dtype=float) - np.asarray(still_c, dtype=float)
                diffs = diffs[np.isfinite(diffs)]

                if len(diffs) < min_conditions:
                    continue
                st, p = ttest_1samp(diffs, 0.0)
                stat[cell_index] = st
                pval[cell_index] = p

        self.stat = stat
        self.pval = pval
        self.modulated_mask = pval < threshold


    # ------------- plotting & print -------------

    def print_modulated_cells(self):
        """Print the indices and p-values of cells with ``modulated_mask`` True."""
        assert self.modulated_mask is not None, "call compute_modulated_neurons() first"
        assert self.pval is not None, "call compute_modulated_neurons() first"
        print(f"====== {self._td.stimulus}: n={self.modulated_mask.sum()} =====\n {np.where(self.modulated_mask)[0]}")


    def plot_population_run_still(self, ax=None) -> plt.Figure:
        """Scatter each cell's mean running vs. still response."""
        if self.mi is None:
            self.compute_mi()

        fig, ax = _ensure_ax(ax, figsize=(5, 5))

        x = np.asarray(self.r_still, dtype=float)
        y = np.asarray(self.r_run, dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[valid], y[valid], alpha=0.75, label="Cells")
        if valid.sum() > 0:
            _add_identity_line(ax, x[valid], y[valid], label="Run = still")
        ax.set_xlabel("Mean response during still trials")
        ax.set_ylabel("Mean response during running trials")
        ax.set_title(f"{self._td.stimulus}: population response")
        ax.legend(frameon=False)
        return fig

    def plot_cell_gain_fit(self, cell: int, ax=None) -> plt.Figure:
        """Scatter condition-level running vs. still for one cell, with gain fit."""
        if self.gain_a is None and self._td.stimulus != "spontaneous":
            self.fit_gain_model()

        fig, ax = _ensure_ax(ax, figsize=(5, 5))

        x = np.asarray(self.condition_still[cell], dtype=float)
        y = np.asarray(self.condition_run[cell], dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[valid], y[valid], alpha=0.75, label="Stimulus conditions")

        if self.gain_a is not None:
            slope, intercept = self.gain_a[cell], self.gain_b[cell]
            if valid.sum() >= 2 and np.isfinite(slope):
                x_line = np.linspace(np.min(x[valid]), np.max(x[valid]), 100)
                ax.plot(x_line, slope * x_line + intercept, color="black",
                        linewidth=2, label=f"Fit: y={slope:.2f}x{intercept:+.3f}")

        if valid.sum() > 0:
            _add_identity_line(ax, x[valid], y[valid], label="Run = still")
        ax.set_xlabel("Mean response during still trials")
        ax.set_ylabel("Mean response during running trials")
        ax.set_title(f"{self._td.stimulus}: cell {cell} (n={valid.sum()} conditions)")
        ax.legend(frameon=False)
        return fig

    def plot_mi_distribution(self, ax=None) -> plt.Figure:
        """Histogram of the sign-safe Modulation Index across the population.

        Uses the bounded ``[-1, 1]`` sign-safe MI stored in :attr:`mi` and
        marks the population median and the no-modulation line at zero.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
        """
        if self.mi is None:
            self.compute_mi()

        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 4))
        else:
            fig = ax.figure

        valid_mi = self.mi[np.isfinite(self.mi)]
        median_mi = np.nanmedian(valid_mi)

        ax.hist(valid_mi, bins=20, range=(-1, 1), alpha=0.8, color="C0")
        ax.axvline(median_mi, linestyle="--", color="black", label=f"median = {median_mi:+.3f}")
        ax.axvline(0, linestyle=":", color="gray", label="no modulation")

        ax.set_xlim(-1, 1)
        ax.set_xlabel("Sign-safe MI")
        ax.set_ylabel("Number of cells")
        ax.set_title(f"{stim_to_short(self._td.stimulus)}\nn = {int(np.isfinite(self.mi).sum())} cells")
        ax.legend(frameon=False, fontsize=8)

        return fig

    def to_dataframe(self) -> pd.DataFrame:
        """Return per-cell results as a tidy DataFrame.

        One row per cell with the classified-response quantities and the
        fitted gain parameters. ``compute_mi`` (and, for the gain columns,
        ``fit_gain_model``) must have been run first.

        Returns
        -------
        pandas.DataFrame
            Columns: stimulus, cell, r_run, r_still, delta_r, mi, and (when
            available) gain_a, gain_b, gain_r2, gain_valid, n_gain_conditions.
        """
        if self.mi is None:
            raise ValueError("Please run compute_mi() first.")

        n_cells = len(self.mi)
        df = pd.DataFrame({
            "stimulus": self._td.stimulus,
            "cell": np.arange(n_cells),
            "r_run": np.asarray(self.r_run, dtype=float),
            "r_still": np.asarray(self.r_still, dtype=float),
            "delta_r": np.asarray(self.delta_r, dtype=float),
            "mi": np.asarray(self.mi, dtype=float),
        })

        if self.gain_a is not None:
            df["gain_a"] = np.asarray(self.gain_a, dtype=float)
            df["gain_b"] = np.asarray(self.gain_b, dtype=float)
            df["gain_r2"] = np.asarray(self.gain_r2, dtype=float)
            df["gain_valid"] = np.asarray(self.gain_valid, dtype=bool)
            df["n_gain_conditions"] = np.asarray(self.n_gain_conditions, dtype=int)

        return df


# ------------- batch pipeline & reporting helpers -------------


def run_binary_modulation_analysis(
    data,
    response_windows,
    run_threshold: float = 3.0,
    still_threshold: float = 0.5,
    min_trials_per_state: int = 1,
    min_gain_conditions: int = 3,
) -> dict:
    """Run the full binary running/still pipeline for each stimulus.

    For every stimulus in ``response_windows``, extracts trial-aligned
    responses via :func:`extract_trials` (stimuli are kept separate — no
    trials are pooled across stimulus classes), builds a
    :class:`BinaryModulation` analysis, classifies trials into
    running/still/ignored, computes the modulation index, and fits the
    condition-level gain model.

    Parameters
    ----------
    data : dict
        Data dictionary returned by :func:`load_data`.
    response_windows : dict[str, tuple]
        Mapping stimulus -> ``(offset, duration)`` passed to
        :func:`extract_trials`.
    run_threshold, still_threshold : float
        Passed to :meth:`BinaryModulation.classify_trials`.
    min_trials_per_state : int
        Minimum running/still trials per condition; passed to
        :meth:`BinaryModulation.fit_gain_model`.
    min_gain_conditions : int
        Minimum number of valid condition pairs required before a gain line
        is fit at all; passed to :meth:`BinaryModulation.fit_gain_model` as
        ``min_conditions``.

    Returns
    -------
    dict
        Mapping stimulus -> fitted :class:`BinaryModulation` instance.
    """
    results = {}
    for stimulus, response_window in response_windows.items():
        trial_data = extract_trials(
            data=data,
            stimulus=stimulus,
            response_window=response_window,
        )

        analysis = BinaryModulation(trial_data)
        analysis.classify_trials(
            run_threshold=run_threshold,
            still_threshold=still_threshold,
        )
        analysis.collect_condition_pairs(
            min_trials_per_state=min_trials_per_state,
        )

        analysis.compute_modulated_neurons()
        analysis.compute_mi()

        # Gain model requires stimulus-condition structure; skip for
        # spontaneous (which has none).
        if stimulus != "spontaneous":
            analysis.fit_gain_model(
                min_conditions=min_gain_conditions,
            )

        results[stimulus] = analysis

    return results


def summarize_binary_modulation_runs(results: dict) -> pd.DataFrame:
    """Compact per-stimulus sanity-check table for a batch of analyses.

    Parameters
    ----------
    results : dict
        Mapping stimulus -> :class:`BinaryModulation`, as returned by
        :func:`run_binary_modulation_analysis`.

    Returns
    -------
    pandas.DataFrame
        Columns: stimulus, n_trials, n_running, n_still, n_ignored,
        median_sign_safe_MI, valid_gain_fits.
    """
    rows = []
    for stimulus, analysis in results.items():
        rows.append({
            "stimulus": stimulus,
            "n_trials": int(len(analysis.run_mask)),
            "n_running": int(analysis.run_mask.sum()),
            "n_still": int(analysis.still_mask.sum()),
            "n_ignored": int(analysis.ignored_mask.sum()),
            "median_sign_safe_MI": float(np.nanmedian(analysis.mi)),
            "valid_gain_fits": int(np.asarray(
                analysis.gain_valid if analysis.gain_valid is not None else [],
                dtype=bool,
            ).sum()),
        })
    return pd.DataFrame(rows)


# ------------- downstream comparisons on the sign-safe MI -------------
#
# BinaryModulation.compute_mi() produces the sign-safe MI (numerator
# R_run - R_still over denominator |R_run| + |R_still| + epsilon), which is
# bounded in [-1, 1] and whose sign always matches R_run - R_still. The
# functions below consume that single formal metric (analysis.mi). The raw
# and denominator-thresholded ("robust") MI variants and all their
# diagnostics live separately in mi_audit_utils.py.


def compare_gratings_vs_natural(results: dict):
    """Paired grating-vs-natural-scenes comparison using the sign-safe MI.

    Compares :attr:`BinaryModulation.mi` (the single formal metric, see
    :meth:`BinaryModulation.compute_mi`) for matched cells. Trials are never
    pooled across stimulus classes: MI is computed separately for drifting
    gratings, static gratings, and natural scenes, and only the resulting
    per-cell values are combined. For each matched cell, grating MI is the
    mean of the drifting- and static-grating MI:

    .. math::
        MI_{\\text{grating}} = \\text{mean}(MI_{\\text{DG}}, MI_{\\text{SG}})

    ``MI_grating`` is compared against natural-scenes MI with a paired
    Wilcoxon signed-rank test, using only cells with a finite MI for DG, SG,
    and NS.

    Parameters
    ----------
    results : dict
        Must contain ``"drifting_gratings"``, ``"static_gratings"``, and
        ``"natural_scenes"`` keys mapping to :class:`BinaryModulation`.

    Returns
    -------
    result_df : pandas.DataFrame
        One-row summary of the paired comparison.
    valid : np.ndarray of bool, shape (n_cells,)
        Mask of cells used in the comparison.
    grating_values : np.ndarray
        Grating MI for the matched, valid cells.
    natural_values : np.ndarray
        Natural-scenes MI for the matched, valid cells.
    """
    mi_dg = np.asarray(results["drifting_gratings"].mi, dtype=float)
    mi_sg = np.asarray(results["static_gratings"].mi, dtype=float)
    mi_ns = np.asarray(results["natural_scenes"].mi, dtype=float)
    finite_dg = np.isfinite(mi_dg)
    finite_sg = np.isfinite(mi_sg)
    finite_ns = np.isfinite(mi_ns)

    mi_grating = np.nanmean(np.vstack([mi_dg, mi_sg]), axis=0)

    valid = (
        finite_dg & finite_sg & finite_ns
        & np.isfinite(mi_grating) & np.isfinite(mi_ns)
    )

    grating_values = mi_grating[valid]
    natural_values = mi_ns[valid]

    stat, p = wilcoxon(grating_values, natural_values)

    result_df = pd.DataFrame([{
        "comparison": "gratings_vs_natural_scenes",
        "n_cells": int(valid.sum()),
        "median_grating_mi": float(np.nanmedian(grating_values)),
        "median_natural_scene_mi": float(np.nanmedian(natural_values)),
        "median_difference_NS_minus_grating": float(np.nanmedian(natural_values - grating_values)),
        "wilcoxon_stat": float(stat),
        "p_value": float(p),
        "frac_NS_greater_than_grating": float(np.mean(natural_values > grating_values)),
    }])

    return result_df, valid, grating_values, natural_values


def summarize_gain_model(results: dict) -> pd.DataFrame:
    """Population summary of the condition-level gain model per stimulus.

    Spontaneous activity is excluded because it has no visual-stimulus
    condition structure, so the gain model is not defined for it.

    Parameters
    ----------
    results : dict
        Mapping stimulus -> :class:`BinaryModulation`.

    Returns
    -------
    pandas.DataFrame
        Columns: stimulus, n_valid_gain_fits, median_gain_a, median_gain_b,
        median_gain_r2, median_n_conditions, frac_gain_a_gt_1.
    """
    rows = []
    for stimulus, analysis in results.items():
        if stimulus == "spontaneous":
            continue

        valid = np.asarray(analysis.gain_valid, dtype=bool)

        rows.append({
            "stimulus": stimulus,
            "n_valid_gain_fits": int(valid.sum()),
            "median_gain_a": float(np.nanmedian(analysis.gain_a[valid])),
            "median_gain_b": float(np.nanmedian(analysis.gain_b[valid])),
            "median_gain_r2": float(np.nanmedian(analysis.gain_r2[valid])),
            "median_n_conditions": float(np.nanmedian(analysis.n_gain_conditions[valid])),
            "frac_gain_a_gt_1": float(np.mean(analysis.gain_a[valid] > 1)),
        })
    return pd.DataFrame(rows)


# ------------- plotting -------------


def plot_metric_comparison(
    binary_results: dict,
    stimuli=None,
    neurons_mask = None,
    metric: str = "mi",
    bins: int = 20,
    value_range=None,
    ax=None,
):
    """Compare one per-cell metric across an arbitrary set of stimuli.

    Draws a one-row grid of histograms, one panel per stimulus, for the
    chosen metric read directly off each :class:`BinaryModulation` instance.
    Optionally overlays a semi-transparent histogram for a subset of neurons
    (e.g. running-modulated cells identified by ``modulated_mask``).

    Parameters
    ----------
    binary_results : dict
        Mapping stimulus -> :class:`BinaryModulation` (``compute_mi`` run).
    stimuli : iterable of str or None, optional
        Which stimuli to show and in what order. ``None`` (default) uses all
        keys of ``binary_results``.
    neurons_mask : dict or np.ndarray or None, optional
        If a dict, maps stimulus -> boolean array of shape ``(n_cells,)``
        selecting neurons to highlight with a transparent overlay histogram.
        If a single boolean array, the same mask is applied to all stimuli.
        ``None`` (default) draws only the population histogram.
    metric : str, optional
        Attribute name to read per cell, e.g. ``"mi"`` (sign-safe MI, the
        default) or ``"delta_r"``.
    bins : int, optional
        Histogram bin count, by default 20.
    value_range : tuple or None, optional
        ``(low, high)`` histogram range. Defaults to ``(-1, 1)`` for
        ``metric="mi"`` (which is bounded) and to data range otherwise.
    ax : array-like of matplotlib.axes.Axes, optional
        Pre-made axes (one per stimulus). If ``None``, a figure is created.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray of matplotlib.axes.Axes
    """
    if stimuli is None:
        stimuli = list(binary_results)
    stimuli = list(stimuli)

    if value_range is None and metric == "mi":
        value_range = (-1, 1)

    if ax is None:
        fig, axes = plt.subplots(
            2,2, figsize=(8, 6),
            sharey=True, constrained_layout=True,
        )
        axes = axes.flatten()
    else:
        axes = np.atleast_1d(ax).flatten()
        fig = axes[0].figure

    for panel, stimulus in zip(axes, stimuli):
        analysis = binary_results[stimulus]
        values = np.asarray(getattr(analysis, metric), dtype=float)
        finite = np.isfinite(values)
        vals = values[finite]
        median = np.nanmedian(vals) if vals.size else np.nan

        panel.hist(vals, bins=bins, range=value_range, alpha=0.8, color="C0")
        panel.axvline(median, linestyle="--", color="black", label=f"median = {median:+.3f}")
        panel.axvline(0, linestyle="-", color="gray", alpha=0.3)

        # Overlay histogram for the masked subset (e.g. modulated neurons).
        if neurons_mask is not None:
            mask = (
                neurons_mask.get(stimulus)
                if isinstance(neurons_mask, dict)
                else neurons_mask
            )
            if mask is not None:
                mask = np.asarray(mask, dtype=bool)
                sub_vals = values[mask & finite]
                if len(sub_vals) > 0:
                    sub_median = np.nanmedian(sub_vals)
                    panel.hist(
                        sub_vals, bins=bins, range=value_range,
                        alpha=0.5, color="C3",
                    )
                    panel.axvline(
                        sub_median, linestyle="--", color="C3",
                        linewidth=1.5,
                        label=f"masked median = {sub_median:+.3f}",
                    )

        if value_range is not None:
            panel.set_xlim(*value_range)
        if neurons_mask:
            panel.set_title(f"{stim_to_short(stimulus)} (n={int(neurons_mask[stimulus].sum())}/{int(finite.sum())})")
        else:
            panel.set_title(f"{stim_to_short(stimulus)} (n={int(finite.sum())})")
        panel.set_xlabel(metric)
        panel.legend(frameon=False, fontsize=8)

    axes[0].set_ylabel("Number of cells")
    return fig, axes


def plot_grating_natural_paired_scatter(grating_values, natural_values, p_value=None, ax=None):
    """Paired scatter of grating vs. natural-scenes MI for matched cells.

    Parameters
    ----------
    grating_values, natural_values : array-like
        Matched-cell MI values, e.g. from :func:`compare_gratings_vs_natural`.
    p_value : float, optional
        Wilcoxon p-value to annotate in the title.
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    grating_values = np.asarray(grating_values, dtype=float)
    natural_values = np.asarray(natural_values, dtype=float)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure

    if grating_values.size == 0 or natural_values.size == 0:
        ax.text(0.5, 0.5, "No valid paired cells", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("Gratings MI: mean(DG, SG)")
        ax.set_ylabel("Natural scenes MI")
        ax.set_title("Gratings vs natural scenes\nn=0")
        return fig, ax

    ax.scatter(grating_values, natural_values, alpha=0.75)
    _add_identity_line(ax, grating_values, natural_values, label="NS = gratings")

    ax.set_xlabel("Gratings MI: mean(DG, SG)")
    ax.set_ylabel("Natural scenes MI")
    title = f"Gratings vs natural scenes\nn={len(grating_values)}"
    if p_value is not None:
        title += f", p={p_value:.3g}"
    ax.set_title(title)
    ax.legend(frameon=False)

    return fig, ax


def plot_gain_scatter(
    results: dict,
    stimuli=("drifting_gratings", "static_gratings", "natural_scenes"),
    figsize=(12, 4),
    r2_threshold: float = 0.0,
) -> plt.Figure:
    """Scatter gain_a vs gain_b for modulated neurons, colored by gain_r².

    Creates a 1×N subplot grid (one panel per stimulus). Each panel shows a
    scatter of gain_a (multiplicative coefficient, x) vs gain_b (additive
    offset, y) for neurons marked as modulated (``modulated_mask``) with
    valid gain fits (``gain_valid``). Point fill opacity reflects the gain
    model R² — more opaque = better fit.

    Parameters
    ----------
    results : dict
        Mapping stimulus -> fitted :class:`BinaryModulation` instance
        (``compute_running_ttest``, ``fit_gain_model`` must have been run).
    stimuli : tuple of str, optional
        Which stimuli to plot. Defaults to the three visually evoked stimuli
        (spontaneous has no gain model).
    figsize : tuple, optional
        Figure dimensions, by default ``(12, 4)``.
    r2_threshold : float, optional
        Minimum gain R² for a neuron to be plotted. Cells below this
        threshold are excluded. Defaults to 0.0 (show all modulated/valid).

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_stim = len(stimuli)
    fig, axes = plt.subplots(1, n_stim, figsize=figsize, constrained_layout=True)
    if n_stim == 1:
        axes = np.atleast_1d(axes)

    for ax, stimulus in zip(axes, stimuli):
        analysis = results[stimulus]
        mod = np.asarray(analysis.modulated_mask, dtype=bool)
        valid = np.asarray(analysis.gain_valid, dtype=bool)
        r2 = np.asarray(analysis.gain_r2, dtype=float)
        mask = mod & valid & (r2 >= r2_threshold)

        x = np.asarray(analysis.gain_a, dtype=float)
        y = np.asarray(analysis.gain_b, dtype=float)

        # Clip alpha to [0, 1]; negative R² is possible for poor fits.
        alpha = np.clip(r2, 0, 1)

        base_color = plt.cm.tab10(0)  # RGBA matching "C0"

        for i in np.where(mask)[0]:
            ax.scatter(
                x[i], y[i],
                facecolors=(base_color[0], base_color[1], base_color[2], alpha[i]),
                edgecolors=(0, 0, 0, 0.5),
                linewidths=0.3,
                zorder=3,
            )

        ax.axhline(0, linestyle=":", color="gray")
        # ax.axvline(0, linestyle="-", color="gray", linewidth=0.4)
        ax.axvline(1, linestyle=":", color="gray")
        n_shown = int(mask.sum())
        base_n = int((mod & valid).sum())
        title = f"{stim_to_short(stimulus)}\n{n_shown} / {base_n} modulated"
        ax.set_title(title)
        ax.set_xlabel("a (multiplicative)")
        ax.set_ylabel("b (additive)")

    return fig


def plot_metadata_validation(aligned: dict, validation_df: pd.DataFrame = None, ylabel: str = "Sign-safe MI (ours)"):
    """Scatter our MI against Allen ``run_mod_*`` metadata, per stimulus.

    Parameters
    ----------
    aligned : dict
        Mapping stimulus -> {"mi": array, "ref": array}, as returned by
        :func:`validate_mi_against_metadata`.
    validation_df : pandas.DataFrame, optional
        Table with ``stimulus``, ``spearman_rho``, ``p_value``, ``n_cells``
        columns, used to annotate each panel's title.
    ylabel : str, optional
        Y-axis label naming the MI variant passed in ``aligned``. Defaults to
        ``"Sign-safe MI (ours)"``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray of matplotlib.axes.Axes
    """
    stimuli = list(aligned)
    fig, axes = plt.subplots(1, len(stimuli), figsize=(4.5 * len(stimuli), 4), constrained_layout=True)
    axes = np.atleast_1d(axes)

    for ax, stimulus in zip(axes, stimuli):
        mi = aligned[stimulus]["mi"]
        ref = aligned[stimulus]["ref"]

        ax.scatter(ref, mi, alpha=0.75)

        title = stim_to_short(stimulus)
        if validation_df is not None:
            row = validation_df.loc[validation_df["stimulus"] == stimulus].iloc[0]
            title += f"\nrho={row['spearman_rho']:.2f}, p={row['p_value']:.2g}, n={int(row['n_cells'])}"
        ax.set_title(title)
        ax.set_xlabel("Allen run_mod (metadata)")
        ax.set_ylabel(ylabel)

    return fig, axes

def plot_modulation_grid(
    binary_results: dict,
    stimuli=("drifting_gratings", "static_gratings", "natural_scenes", "spontaneous"),
    figsize=(6, 10),
) -> plt.Figure:
    """Grid map of modulated neurons with gain parameters per stimulus.

    Each row is a cell, each column a stimulus/spontaneous. The grid shows:
    - Light blue background where ``modulated_mask`` is True.
    - ``(gain_a, gain_b)`` text for visual stimuli (spontaneous has no gain
      model and shows ``"—"`` instead).

    Cells are sorted by number of stimuli in which they are modulated
    (descending), so modulated-dominant cells appear at the top.

    Parameters
    ----------
    binary_results : dict
        Mapping stimulus -> fitted :class:`BinaryModulation` instance
        (``compute_modulated_neurons``, ``fit_gain_model`` must have been
        run for visual stimuli).
    stimuli : tuple of str, optional
        Which conditions to show and in what order.
    figsize : tuple, optional
        Figure size, by default ``(6, 10)``.

    Returns
    -------
    plt.Figure
    """
    labels = list(stimuli)
    J = len(labels)

    # Per-stimulus modulated mask and gain text.
    n_cells_list = [
        len(np.asarray(binary_results[lbl].modulated_mask))
        for lbl in labels
    ]
    I = max(n_cells_list)  # all stimuli share the same cell count

    mod_mask = np.zeros((I, J), dtype=bool)
    gain_texts = np.empty((I, J), dtype=object)
    gain_a_vals = np.full((I, J), np.nan)  # for sorting

    for j, lbl in enumerate(labels):
        analysis = binary_results[lbl]
        mod = np.asarray(analysis.modulated_mask, dtype=bool)
        mod_mask[:, j] = mod

        has_gain = analysis.gain_a is not None
        for i in range(I):
            if mod[i]:
                if has_gain:
                    a = analysis.gain_a[i]
                    b = analysis.gain_b[i]
                    gain_texts[i, j] = f"({a:.2f}, {b:.2f})"
                    gain_a_vals[i, j] = a
                else:
                    gain_texts[i, j] = "—"
            # else: leave gain_texts[i, j] as None (empty string below)

    MOD_BG = (0.82, 0.88, 0.97)  # light blue

    # Sort rows: by mean gain_a across modulated stimuli (gain_a available).
    # Non-modulated cells (no valid gain_a in any stimulus) go to the bottom.
    with np.errstate(invalid='ignore'):
        gain_a_means = np.nanmean(np.where(mod_mask, gain_a_vals, np.nan), axis=1)
    order = np.argsort(-gain_a_means)  # descending

    fig, ax = plt.subplots(figsize=figsize)

    # Background — reorder rows to match sorted order
    img = np.ones((I, J, 3))
    img[mod_mask] = MOD_BG
    img = img[order]          # match text/yticklabel order
    ax.imshow(img, aspect="auto", interpolation="nearest")

    # Text — only for modulated cells
    for i in range(I):
        ci = order[i]
        for j in range(J):
            if mod_mask[ci, j] and gain_texts[ci, j]:
                ax.text(
                    j, i, gain_texts[ci, j],
                    ha="center", va="center",
                    fontsize=7,
                )

    ax.set(
        xticks=range(J), xticklabels=[stim_to_short(l) for l in labels],
        yticks=range(I), yticklabels=order,
        ylabel="cell # (sorted)",
        title="Modulated neurons — gain (a, b)",
    )
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    legend_handles = [
        Patch(facecolor=MOD_BG, label="modulated"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=9,
        frameon=False,
    )
    fig.tight_layout()
    return fig


def plot_tuned_neurons_grid(computed_tuned: dict, metadata_tuned: dict = None, figsize=(6, 10)) -> plt.Figure:
    """Grid map comparing computed vs metadata-given tuned neurons.

    Each column is a stimulus, each row is a cell. Cells are sorted by
    descending total tuned-count (any source), then by both-tuned count.

    Parameters
    ----------
    computed_tuned : dict[str, np.ndarray]
        Per-stimulus boolean masks ``(n_cells,)`` for computed tuned neurons
        (e.g. from :attr:`BinaryModulation.tuned_mask`).
    metadata_tuned : dict[str, np.ndarray] or None, optional
        Per-stimulus boolean masks ``(n_cells,)`` for metadata-given tuned
        neurons (e.g. from ``p_run_mod_* < 0.05``). Stimuli not present in
        this dict are shown with computed-only tuning.
    figsize : tuple, optional
        Figure size.

    Returns
    -------
    plt.Figure
    """
    labels = list(computed_tuned.keys())
    I = len(next(iter(computed_tuned.values())))
    J = len(labels)

    # category colours
    BOTH_CLR = (0.65, 0.95, 0.65)       # green: both agree tuned
    COMP_CLR = (0.55, 0.75, 0.95)       # blue: computed only
    META_CLR = (0.95, 0.75, 0.55)       # orange: metadata only
    NONE_CLR = (0.97, 0.97, 0.97)       # near-white: neither

    # Build agreement matrix: 0=neither, 1=computed-only, 2=metadata-only, 3=both
    agreement = np.zeros((I, J), dtype=int)
    for j, lbl in enumerate(labels):
        comp = np.asarray(computed_tuned[lbl], dtype=bool)
        if metadata_tuned is not None and lbl in metadata_tuned:
            meta = np.asarray(metadata_tuned[lbl], dtype=bool)
            agreement[:, j] = np.where(comp & meta, 3,
                                       np.where(comp, 1,
                                                np.where(meta, 2, 0)))
        else:
            agreement[:, j] = comp.astype(int)

    # Sort rows
    n_tuned_any = (agreement > 0).sum(axis=1)
    n_both = (agreement == 3).sum(axis=1)
    order = np.lexsort((-n_tuned_any, -n_both, -np.arange(I)))

    bg_map = {0: NONE_CLR, 1: COMP_CLR, 2: META_CLR, 3: BOTH_CLR}

    # -- Plot --
    fig, ax = plt.subplots(figsize=figsize)

    img = np.zeros((I, J, 3))
    for i in range(I):
        for j in range(J):
            img[i, j] = bg_map[agreement[order[i], j]]
    ax.imshow(img, aspect='auto', interpolation='nearest')

    # Text markers: B = both, C = computed-only, M = metadata-only
    txt_map = {1: 'C', 2: 'M', 3: 'B'}
    txt_clr = {1: 'blue', 2: 'darkorange', 3: 'green'}
    for i in range(I):
        for j in range(J):
            val = agreement[order[i], j]
            if val > 0:
                ax.text(j, i, txt_map[val], ha='center', va='center',
                        fontsize=9, color=txt_clr[val], fontweight='bold')

    ax.set(xticks=range(J), xticklabels=[stim_to_short(l) for l in labels],
           yticks=range(I), yticklabels=order,
           ylabel='cell #', title='Tuned neurons: computed vs metadata')
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    # Legend
    ax.legend(
        [Patch(facecolor=BOTH_CLR), Patch(facecolor=COMP_CLR),
         Patch(facecolor=META_CLR)],
        ['Both tuned (B)', 'Computed only (C)', 'Metadata only (M)'],
        loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9, frameon=False,
    )
    fig.tight_layout()
    return fig


# ==============================================================================
# Analysis 3 — Predictive Encoding Models
# ==============================================================================


def tent_basis(x, centers):
    """Triangular ("tent") basis functions evaluated at ``x``.

    Each basis function peaks at 1 on its center and decays linearly to 0 at
    the neighbouring centers, so adjacent tents form a partition of unity
    between their centers. Ported from the V1Locomotion ``tent_basis.m``.

    Parameters
    ----------
    x : array-like, shape ``(N,)``
        Coordinates at which to evaluate the basis (e.g. per-trial time).
    centers : array-like, shape ``(M,)``
        Basis-function centers (knots), assumed evenly spaced.

    Returns
    -------
    np.ndarray, shape ``(N, M)``
        Basis matrix; row ``n`` holds the ``M`` tent values at ``x[n]``.
    """
    x = np.asarray(x, dtype=float).reshape(-1, 1)              # (N, 1)
    centers = np.asarray(centers, dtype=float).reshape(1, -1)  # (1, M)
    if centers.shape[1] == 1:
        return np.ones((x.shape[0], 1))
    d = np.diff(centers.ravel())
    if np.any(d <= 0):
        raise ValueError("centers must be strictly increasing")
    dscale = d.mean()                   # uniform knot spacing
    return np.maximum(1.0 - np.abs(x - centers) / dscale, 0.0)


class EncodingModel:
    """Fit and compare nested linear models of running-speed modulation.

    Four models are fit per neuron:

    ============  ===============================================================
    Model         Form
    ============  ===============================================================
    Null          :math:`R = f(S) + \\beta_0`
    Add-only      :math:`R = f(S) + \\beta_0 + \\beta_{\\text{add}} V`
    Mult-only     :math:`R = f(S) + \\beta_0 + \\beta_{\\text{mult}} (V \\times S)`
    Full          :math:`R = f(S) + \\beta_0 + \\beta_{\\text{add}} V + \\beta_{\\text{mult}} (V \\times S)`
    ============  ===============================================================

    Here :math:`f(S)` is the average response to stimulus :math:`S` (the tuning),
    :math:`\\beta_0` is a drifting baseline modeled with tent basis functions,
    :math:`V` is running speed, and :math:`(V \\times S)` is a stimulus-gated
    multiplicative term (implemented as :math:`\\text{ReLU}[1+β_mult V]`).

    Inspired by (Liska et al., 2024)

    Parameters
    ----------
    trial_data : TrialData
        Extracted per-trial responses, running speed, and stimulus params.
    n_basis : int, optional
        Number of tent basis functions for the drifting baseline, by default 5.
    """

    def __init__(self, trial_data: TrialData, n_basis: int = 5):
        self._td = trial_data
        self._n_basis = n_basis
        # filled by fit_all()
        self.r2_null = None           # np.ndarray, shape (n_cells,)
        self.r2_add = None            # np.ndarray, shape (n_cells,)
        self.r2_mult = None           # np.ndarray, shape (n_cells,)
        self.r2_full = None           # np.ndarray, shape (n_cells,)
        # cached design pieces, filled lazily by the _* helpers below
        self._fhat = None             # (n_cells, n_trials)
        self._drift = None            # (n_trials, n_basis)
        self._V = None                # (n_trials,)

    # ------------- design-matrix construction (C1–C4) -------------

    def _trial_response(self):
        """Per-trial scalar response: mean ΔF/F over the response window.

        Returns ``(n_cells, n_trials)``.
        """
        return self._td.responses.mean(axis=2)

    def _condition_labels(self):
        """Integer condition label per trial from ``stimulus_params``.

        The condition is the unique combination of every stimulus parameter
        (dg: orientation×TF; sg: orientation×SF×phase; ns: frame). For
        ``spontaneous`` (no params) all trials share a single condition.

        Returns ``(n_trials,)`` int array.
        """
        sp = self._td.stimulus_params
        n_trials = self._td.responses.shape[1]
        if not sp:
            return np.zeros(n_trials, dtype=int)
        cols = np.column_stack([np.asarray(v, dtype=float) for v in sp.values()])
        _, labels = np.unique(cols, axis=0, return_inverse=True)
        return labels

    def _stimulus_mean(self):
        """f̂(S): per-cell mean response of each trial's stimulus condition.

        Implements Plan.md's ``f(S)`` = "the average response to the stimulus":
        every trial is mapped to its condition's per-cell mean. Cached.

        Returns ``(n_cells, n_trials)``.

        Note
        ----
        Uses the plain condition mean over all trials. For cross-validated
        R² (C6) recompute f̂(S) from *training* trials inside each fold to
        avoid an optimistic Null R².
        """
        if self._fhat is None:
            R = self._trial_response()
            labels = self._condition_labels()
            fhat = np.empty_like(R)
            for c in np.unique(labels):
                m = labels == c
                fhat[:, m] = R[:, m].mean(axis=1, keepdims=True)
            self._fhat = fhat
        return self._fhat

    def _drift_basis(self):
        """Slow-drift design: ``n_basis`` tent functions over trial time.

        Returns ``(n_trials, n_basis)``, shared across cells. Cached.
        """
        if self._drift is None:
            t = self._td.time.mean(axis=1)                 # per-trial time (s)
            centers = np.linspace(t.min(), t.max(), self._n_basis)
            self._drift = tent_basis(t, centers)
        return self._drift

    def _running_speed(self):
        """V: per-trial mean running speed.

        Returns ``(n_trials,)``. Cached. Note ``extract_trials`` does not
        clamp speed, so V may contain small negative values (tracking noise);
        the linear model handles them fine, but clamp in preprocessing if you
        want strictly non-negative speeds.
        """
        if self._V is None:
            self._V = self._td.running_speed.mean(axis=1)
        return self._V

    def _stimulus_onehot(self, labels=None):
        """One-hot per-condition stimulus design, shape ``(n_trials, n_conditions)``.

        The fitted (ridge-penalized) weights on these columns are the neuron's
        tuning ``f(S) = A·s(t)`` (as in Liska/Yates); ridge shrinks the noisy
        per-condition estimates. Shared across cells. For ``spontaneous`` (a
        single condition) this is one column, collinear with the intercept —
        harmless under ridge.
        """
        if labels is None:
            labels = self._condition_labels()
        return (labels[:, None] == np.unique(labels)[None, :]).astype(float)

    def _build_design(self, cell, model="full", onehot=None, drive=None):
        """Assemble the design matrix for one cell and one nested model.

        Columns are ``[f(S), drift(n_basis), (V), (V·d̂(S))]``, included per
        ``model``:

        - ``"null"`` : f(S), drift
        - ``"add"``  : + V             (additive running term)
        - ``"mult"`` : + V·d̂(S)       (running gated by the stimulus drive)
        - ``"full"`` : + V + V·d̂(S)

        The stimulus tuning ``f(S)`` is the **fitted, ridge-penalized one-hot**
        design (:meth:`_stimulus_onehot`) — ridge shrinks the noisy per-condition
        estimates. The multiplicative term gates running by the per-condition
        drive ``d̂(S)`` (the per-fold mean, i.e. the OLS one-hot drive). No
        explicit intercept column: the (unpenalized) ``RidgeCV`` intercept in
        :meth:`fit_all` is the baseline, and the tent basis models drift.

        Parameters
        ----------
        onehot : np.ndarray, optional
            ``(n_trials, n_conditions)`` tuning design; defaults to
            :meth:`_stimulus_onehot`. (Shared across cells — the Null/Add designs
            do not depend on ``cell``.)
        drive : np.ndarray, optional
            ``(n_cells, n_trials)`` per-condition drive for the interaction; pass
            the training-fold mean during CV so it stays leakage-free (defaults
            to the cached all-trial :meth:`_stimulus_mean`).

        Returns ``(n_trials, n_features)``.
        """
        assert model in ("null", "add", "mult", "full"), f"unknown model: {model}"
        S = self._stimulus_onehot() if onehot is None else onehot
        drift = self._drift_basis()                        # (n_trials, n_basis)
        V = self._running_speed()                          # (n_trials,)
        cols = [S, drift]
        if model in ("add", "full"):
            cols.append(V[:, None])
        if model in ("mult", "full"):
            d = (self._stimulus_mean() if drive is None else drive)[cell]
            cols.append((V * d)[:, None])
        return np.column_stack(cols)

    def _fold_stimulus_mean(self, R, labels, train_idx):
        """f̂(S) computed from TRAIN trials only, mapped to every trial.

        Returns ``(n_cells, n_trials)``: each trial is assigned the mean (over
        *training* trials) of its stimulus condition, per cell. Conditions with
        no training trials fall back to the per-cell training grand mean. Used
        inside :meth:`fit_all` so the cross-validated Null R² is not optimistic.
        """
        train_mask = np.zeros(R.shape[1], dtype=bool)
        train_mask[train_idx] = True
        fhat = np.empty_like(R)
        grand = R[:, train_mask].mean(axis=1, keepdims=True)   # (n_cells, 1)
        for c in np.unique(labels):
            cond = labels == c
            tr = cond & train_mask
            fhat[:, cond] = R[:, tr].mean(axis=1, keepdims=True) if tr.any() else grand
        return fhat

    @staticmethod
    def _pooled_r2(y, yhat):
        """Pooled coefficient of determination per cell.

        ``y`` and ``yhat`` are ``(n_cells, n_trials)``; returns ``(n_cells,)``
        ``1 − Σ(y−ŷ)² / Σ(y−ȳ)²`` with ``ȳ`` the per-cell grand mean. Negative
        values mean the model predicts held-out data worse than the mean.
        """
        ss_res = ((y - yhat) ** 2).sum(axis=1)
        ss_tot = ((y - y.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            r2 = 1.0 - ss_res / ss_tot
        return np.where(ss_tot == 0, 0.0, r2)

    @staticmethod
    def _ridge_cv_predict(X_tr, Y_tr, X_te, alphas):
        """Ridge with a **per-target** GCV-selected λ and an unpenalized intercept.

        Features are z-scored on ``X_tr``; the intercept (target mean) is not
        penalised. ``Y_tr`` may be 1-D ``(n,)`` or 2-D ``(n, m)``; a 2-D array
        shares the design (one SVD) but each column (cell) selects its **own** λ,
        so the multi-target Null/Add fits match a per-cell fit exactly — the
        nested ΔR² comparison stays consistent across models. Returns test
        predictions of matching shape via a closed-form SVD solve (far faster
        than a per-cell ``RidgeCV`` loop for this many small fits).
        """
        mu = X_tr.mean(0); sd = X_tr.std(0); sd = np.where(sd == 0, 1.0, sd)
        Xz = (X_tr - mu) / sd
        one_d = Y_tr.ndim == 1
        Y = Y_tr[:, None] if one_d else Y_tr
        ybar = Y.mean(0)
        Yc = Y - ybar
        try:
            U, s, Vt = np.linalg.svd(Xz, full_matrices=False)
        except np.linalg.LinAlgError:                          # rare gesdd non-convergence
            import scipy.linalg
            U, s, Vt = scipy.linalg.svd(Xz, full_matrices=False, lapack_driver="gesvd")
        UtY = U.T @ Yc                                          # (r, m)
        n, m = Xz.shape[0], Y.shape[1]; s2 = s ** 2
        best_gcv = np.full(m, np.inf); best_a = np.full(m, alphas[0], dtype=float)
        for a in alphas:                                        # per-target GCV
            f = s2 / (s2 + a)
            rss = ((Yc - U @ (f[:, None] * UtY)) ** 2).sum(0)   # (m,)
            gcv = rss / (n * (1 - f.sum() / n) ** 2)
            better = gcv < best_gcv
            best_gcv[better] = gcv[better]; best_a[better] = a
        fj = s[:, None] / (s2[:, None] + best_a[None, :])       # (r, m)
        B = Vt.T @ (fj * UtY)                                   # (p, m)
        pred = ((X_te - mu) / sd) @ B + ybar                   # (n_te, m)
        return pred[:, 0] if one_d else pred

    @staticmethod
    def _cv_splits(n_trials, n_folds=5, cv="blocked", gap=5, random_state=0):
        """Build cross-validation folds.

        ``cv="blocked"`` returns contiguous test blocks over the trials in their
        natural (temporal) order, with the training set **purged** of trials
        within ``gap`` of each test block — so calcium/running autocorrelation
        cannot leak across the train/test boundary. Every trial is held out
        exactly once (full pooled-R² coverage); only training trials are
        dropped. ``cv="shuffled"`` returns the legacy random ``KFold`` (leaks for
        sub-second trial spacing; kept for reproducing the pre-correction fits).
        """
        if cv == "shuffled":
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
            return list(kf.split(np.arange(n_trials)))
        if cv != "blocked":
            raise ValueError(f"unknown cv {cv!r}; use 'blocked' or 'shuffled'")
        idx = np.arange(n_trials)
        bounds = np.linspace(0, n_trials, n_folds + 1).astype(int)
        splits = []
        for k in range(n_folds):
            lo, hi = bounds[k], bounds[k + 1]
            train = np.concatenate([idx[:max(0, lo - gap)], idx[min(n_trials, hi + gap):]])
            splits.append((train, idx[lo:hi]))
        return splits

    def fit_all(self, n_folds=5, alphas=None, random_state=0, splits=None,
                cv="blocked", gap=5):
        """Fit all four nested models per neuron with cross-validated R².

        Each neuron is fit by ridge regression (standardized features, λ chosen
        by GCV, unpenalized intercept) under ``n_folds``-fold cross-validation.
        By default the folds are **contiguous time blocks with the training set
        purged** within ``gap`` trials of each test block (``cv="blocked"``):
        calcium and running are slowly autocorrelated, so a shuffled/random
        K-fold interleaves held-out trials with their temporal neighbours and
        inflates ΔR² for densely-packed stimuli (natural scenes / static
        gratings, trials ~0.27 s apart) — a leakage that blocked CV removes.
        Pass ``cv="shuffled"`` to reproduce the legacy (leaky) random split.
        The stimulus tuning ``f(S)`` is a **fitted, ridge-penalized one-hot**
        design (:meth:`_stimulus_onehot`); ridge shrinks the noisy per-condition
        estimates. The multiplicative term gates running by the per-condition
        drive, recomputed from *training* trials each fold
        (:meth:`_fold_stimulus_mean`). The stimulus-only Null/Add designs are
        shared across cells and fit as a single multi-target ridge.

        Parameters
        ----------
        n_folds : int, optional
            Number of cross-validation folds, by default 5.
        alphas : array-like, optional
            Ridge penalties to search; defaults to ``np.logspace(-3, 3, 13)``.
        random_state : int, optional
            Seed for the ``cv="shuffled"`` fold split, by default 0 (unused for
            blocked CV, which is deterministic).
        splits : list of (train_idx, test_idx), optional
            Explicit cross-validation splits; when given, overrides ``cv``/``gap``.
            The test blocks must partition all trials (every trial held out
            exactly once) so the pooled R² has full coverage. Defaults to ``None``.
        cv : {"blocked", "shuffled"}, optional
            Fold scheme when ``splits`` is None, by default ``"blocked"``
            (contiguous, purged — leakage-free). ``"shuffled"`` is the legacy
            random ``KFold`` and leaks for sub-second trial spacing; use it only
            to reproduce the pre-correction numbers.
        gap : int, optional
            Purge radius (in trials) for ``cv="blocked"``: training trials within
            ``gap`` of a test block are dropped, removing calcium/running
            autocorrelation across the train/test boundary. By default 5.

        Stores
        ------
        r2_null, r2_add, r2_mult, r2_full : np.ndarray, shape ``(n_cells,)``
            Pooled cross-validated R² of each nested model (may be negative).

        Notes
        -----
        For ``spontaneous`` there is a single stimulus condition, so the one-hot
        tuning is a single (constant) column and the drive is constant; the
        multiplicative term ``V·d̂ ∝ V``, so ``r2_mult`` coincides with
        ``r2_add`` (ridge tolerates the collinearity).
        """
        if alphas is None:
            alphas = np.logspace(-3, 3, 13)

        R = self._trial_response()                 # (n_cells, n_trials) targets
        labels = self._condition_labels()          # (n_trials,)
        S = self._stimulus_onehot(labels)          # (n_trials, n_conditions) tuning design
        n_cells, n_trials = R.shape
        models = ("null", "add", "mult", "full")
        yhat = {m: np.empty_like(R) for m in models}

        if splits is None:
            splits = self._cv_splits(n_trials, n_folds, cv, gap, random_state)
        for train_idx, test_idx in splits:
            drive = self._fold_stimulus_mean(R, labels, train_idx)   # per-fold drive (leak-free)
            for model in models:
                if model in ("null", "add"):
                    # stimulus-only design is identical across cells: one shared solve
                    X = self._build_design(0, model, onehot=S, drive=drive)
                    yhat[model][:, test_idx] = self._ridge_cv_predict(
                        X[train_idx], R[:, train_idx].T, X[test_idx], alphas).T
                else:
                    for cell in range(n_cells):
                        X = self._build_design(cell, model, onehot=S, drive=drive)
                        yhat[model][cell, test_idx] = self._ridge_cv_predict(
                            X[train_idx], R[cell, train_idx], X[test_idx], alphas)

        for model in models:
            setattr(self, f"r2_{model}", self._pooled_r2(R, yhat[model]))
        return self

    def r2_decomposition(self):
        """Compute :math:`\\Delta R^2` for each term relative to the null model.

        Returns
        -------
        delta_add : np.ndarray, shape ``(n_cells,)``
            :math:`R^2_{\\text{add}} - R^2_{\\text{null}}`
        delta_mult : np.ndarray, shape ``(n_cells,)``
            :math:`R^2_{\\text{mult}} - R^2_{\\text{null}}`
        delta_full : np.ndarray, shape ``(n_cells,)``
            :math:`R^2_{\\text{full}} - R^2_{\\text{null}}`

        Raises
        ------
        RuntimeError
            If :meth:`fit_all` has not been run yet.
        """
        if self.r2_null is None:
            raise RuntimeError("call fit_all() before r2_decomposition()")
        delta_add = self.r2_add - self.r2_null
        delta_mult = self.r2_mult - self.r2_null
        delta_full = self.r2_full - self.r2_null
        return delta_add, delta_mult, delta_full

    # ------------- plotting -------------

    def plot_r2_decomposition(self, ax=None) -> plt.Figure:
        """Visualise the :math:`\\Delta R^2` breakdown across the population.

        Draws a violin (with the median) plus jittered per-cell points for the
        three nested terms — :math:`\\Delta R^2_{\\text{add}}`,
        :math:`\\Delta R^2_{\\text{mult}}`, :math:`\\Delta R^2_{\\text{full}}` —
        for this instance's stimulus. Pass ``ax`` to draw one stimulus per panel
        of a shared figure (e.g. a gratings-vs-natural comparison).

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axis to draw into; a new figure/axis is created if omitted.

        Returns
        -------
        matplotlib.figure.Figure

        Raises
        ------
        RuntimeError
            If :meth:`fit_all` has not been run yet.
        """
        delta_add, delta_mult, delta_full = self.r2_decomposition()
        series = [delta_add, delta_mult, delta_full]
        labels = [r"$\Delta R^2_{add}$", r"$\Delta R^2_{mult}$", r"$\Delta R^2_{full}$"]

        if ax is None:
            _, ax = plt.subplots(figsize=(4, 3.5))
        positions = np.arange(1, len(series) + 1)
        vp = ax.violinplot(series, positions=positions, showmedians=True,
                           showextrema=False)
        for body in vp["bodies"]:
            body.set_alpha(0.35)
        rng = np.random.default_rng(0)
        for pos, d in zip(positions, series):
            ax.scatter(rng.normal(pos, 0.04, size=len(d)), d,
                       s=8, alpha=0.4, color="k", zorder=3)
        ax.axhline(0, color="0.5", lw=0.8, ls="--")
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_ylabel(r"$\Delta R^2$ (cross-validated)")
        ax.set_title(self._td.stimulus)
        return ax.figure
