import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from functools import cached_property
from scipy.stats import spearmanr, wilcoxon
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

N_CELLS = 47
STIMULI = ['drifting_gratings', 'static_gratings', 'natural_scenes', 'spontaneous']
SHORT_STIM = ['dg', 'sg', 'ns', 'spont']
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


def load_data(path="data"):
    raw = dict(np.load(Path(path) / "visual_coding_data.npz", allow_pickle=True))
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

def _plot_mean_sem(ax, x, data, color, alpha=0.5, label=None,
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
            spon = s["stim_epoch_table"]
            spon = spon[spon["stimulus"] == "spontaneous"]
            epoch_start = int(spon["start"].values[0])  # There's only one spontaneous period per session
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
    n_cells = N_CELLS
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
    still_trial_mask : np.ndarray or None, optional
        Boolean mask ``(n_trials,)`` — True = still trial to exclude from analysis.
        Applied before neuron_mask and pref_trial_mask. None = keep all trials.
    """

    def __init__(self, trial_data: TrialData | dict[str, TrialData],
                 mode='equal_size', n_bins: int = 20,
                 neuron_mask=None, pref_trial_mask=None,
                 still_trial_mask=None):
        self._td = trial_data
        self.n_bins = n_bins
        self.mode = mode
        self.neuron_mask = neuron_mask  # if None, n_cells_picked = n_cells
        self.pref_trial_mask = pref_trial_mask  # if None, n_trials_pref = n_trials_total
        self.still_trial_mask = still_trial_mask  # if None, keep all trials

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
        self.levene_p_values: np.ndarray | None = None       # (n_cells_picked,)
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
        """Apply still_trial_mask, neuron_mask, and pref_trial_mask in order.

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
            Still-filtered speeds (aligned to neuron-masked responses).
        masked_responses : np.ndarray, shape ``(n_cells_picked, n_trials_pref)``
            Trial-masked responses (NaN where ``pref_trial_mask`` is False).
        """
        # apply still_trial_mask — exclude still (non-running) trials
        still_keep = ~self.still_trial_mask if self.still_trial_mask is not None else None
        if still_keep is not None:
            responses = responses[:, still_keep]
            speeds = speeds[still_keep]

        # apply neuron_mask
        if self.neuron_mask is not None:
            responses = responses[self.neuron_mask]

        # apply pref_trial_mask (NaN-masked per-neuron trial selection)
        if self.pref_trial_mask is not None:
            mask = self.pref_trial_mask
            if still_keep is not None:
                mask = mask[:, still_keep]
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

    def significance_test(self, n_shuffles: int = 1000, threshold: float = 0.05,
                           seed: int | None = None):
        """Shuffle running-speed labels and re-compute tuning curves to assess significance.

        Uses a permutation test: shuffles bin labels ``n_shuffles`` times to
        build a null distribution of tuning-curve variances, then computes the
        fraction of shuffled variances exceeding the observed variance as the
        p-value for each cell.

        Levene's t test of variance mentioned in (Christensen & Pillow, 2022).


        Parameters
        ----------
        n_shuffles : int, optional
            Number of shuffles, by default 1000.
        threshold : float, optional
            Significance threshold, by default 0.05.
        seed : int or None, optional
            Random seed for reproducible shuffles.

        Stores
        ------
        levene_p_values : np.ndarray, shape ``(n_cells_picked,)``
        significant_mask : np.ndarray of bool, shape ``(n_cells_picked,)``
        """
        assert self.mean_all_responses is not None, "call compute_tuning() first"
        assert self.bins_sub_ids is not None, "call compute_tuning() first"

        rng = np.random.default_rng(seed)

        # the real tuning — variance across bins
        vs_real = np.nanvar(self.mean_all_responses, axis=1)   # (n_cells_picked)

        # shuffled — permute bin labels, re-compute variance
        vs_shuffled = []
        for _ in range(n_shuffles):
            # only shuffle non-(-1) bin ids; -1 (excluded trials) stay fixed
            shuffled_bins_ids = self.bins_sub_ids.copy()
            valid_mask = shuffled_bins_ids != -1
            valid_ids = shuffled_bins_ids[valid_mask]
            shuffled_bins_ids[valid_mask] = rng.permutation(valid_ids)
            _, _, mean_all_res = self._binned_responses(bins_ids=shuffled_bins_ids) # (n_cells_picked, n_bins)
            vs_shuffled.append(np.nanvar(mean_all_res, axis=1))
        vs_shuffled = np.array(vs_shuffled).T   # (n_cells_picked, n_shuffles)

        p_values = np.mean(vs_shuffled >= vs_real[:, np.newaxis], axis=1)    # (n_cells_picked)
        significant_mask = p_values < threshold

        self.levene_p_values = p_values
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

        from scipy.stats import spearmanr
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

        _plot_mean_sem(ax, self.bins_centers, responses, semcolor,
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
                    _plot_mean_sem(ax, self.bins_centers, spont_responses[mask],
                                   color, alpha=0.15, linestyle='--',
                                   marker_facecolor='none')
            else:
                # non-significant cells as grey background
                if bg.any():
                    _plot_mean_sem(ax, self.bins_centers, responses[bg],
                                   'lightgray', alpha=0.5)

            # category cells in colour
            if n > 0:
                _plot_mean_sem(ax, self.bins_centers, responses[mask],
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
                _plot_mean_sem(ax4, self.bins_centers, spont_responses[nm_mask],
                               NTM_COLOR, alpha=0.15, linestyle='--',
                               marker_facecolor='none')

            if n > 0:
                _plot_mean_sem(ax4, self.bins_centers, responses[nm_mask],
                               NTM_COLOR, alpha=0.5)
            ax4.set_title(f'non-tuned mod. (n={n})')
            ax4.set_ylabel('')

        axes[0].set_ylabel(ylabel)
        fig.tight_layout(rect=(0, 0, 1, 0.94))

        # legend: evoked vs spont line styles
        from matplotlib.lines import Line2D
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
        spont_resps = spontaneous.mean_all_responses if spontaneous is not None else None

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
            spont_z = (spontaneous.mean_all_responses - mu) / sd
        else:
            spont_z = None

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
                             show_rho: bool = False, zscore=False) -> plt.Figure:
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
        if zscore:
            tunings[lbl].plot_tuning_curve_zscore(cells=cells, ax=ax, semcolor='gray')
        else:
            tunings[lbl].plot_tuning_curve(cells=cells, ax=ax, semcolor='gray')
            ax.set_ylim(bottom=0)
        title = lbl
        if show_rho:
            t = tunings[lbl]
            assert t.rho is not None, "call compute_spearman() first"
            rho_vals = t.rho[cells] if cells else t.rho
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
    return fig

def plot_monotonicity_stacked_bar(tunings: dict[str, SpeedTuning],
                                   ax: plt.Axes = None,
                                   colors: dict[str, str]| None = None,
                                   modulated_mask: dict[str, np.ndarray] | None = None,
                                   figsize=(5, 4)) -> plt.Axes:
    """Stacked bar chart: for each stimulus, breakdown of significantly tuned
    neurons by monotonicity (positive / negative / non-monotonic).

    When *modulated_mask* is provided, a light-grey ghost bar is drawn behind
    each stacked bar showing the total modulated pool, with the count labelled
    above it.

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
        When provided, a ghost bar shows the total modulated pool per stimulus.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = _ensure_ax(ax, figsize=figsize)

    if colors is None:
        colors = {'positive': POS_COLOR, 'negative': NEG_COLOR,
                  'non-monotonic': NM_COLOR}

    labels = list(tunings.keys())
    categories = ['non-monotonic', 'negative', 'positive']

    # modulated pool: ghost bar behind the stacked bars
    if modulated_mask is not None:
        mod_counts = np.array([
            int(modulated_mask[lbl].sum())
            if lbl in modulated_mask and modulated_mask[lbl] is not None
            else N_CELLS
            for lbl in labels
        ])
        x = np.arange(len(labels))
        ax.bar(x, mod_counts, 0.6, bottom=0, color='lightgray', alpha=0.35,
               edgecolor='gray', linewidth=0.6, label='modulated pool', zorder=0)
        for i, lbl in enumerate(labels):
            ax.text(i, mod_counts[i] + 0.5, str(int(mod_counts[i])),
                    ha='center', va='bottom', fontsize=9, color='gray',
                    fontweight='bold')

    # breakdown of significant cells in each category, per stimulus
    counts = {}
    for lbl in labels:
        t = tunings[lbl]
        assert t.monotonic_mask is not None, "call compute_spearman() first"
        if t.significant_mask is None or t.significant_mask.sum() == 0:
            counts[lbl] = {c: 0.0 for c in categories}
        else:
            sig = t.significant_mask
            n = sig.sum()
            counts[lbl] = {c: t.monotonic_mask[c][sig].sum()
                            for c in categories}

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
    ax.set_xticklabels(labels)
    ax.set_ylabel('# tuned neurons')
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

    if ax is None:
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
        assert t.levene_p_values is not None, "call significance_test() first"

    # colours (matching module-level constants)
    POS_RGB = _hex_to_rgb(POS_COLOR)
    NEG_RGB = _hex_to_rgb(NEG_COLOR)
    NM_RGB  = _hex_to_rgb(NM_COLOR)
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
    from matplotlib.patches import Rectangle
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
                p = t.levene_p_values[ci]
                stars = ''
                if p <= 0.0001:
                    stars = '***'
                elif p <= 0.001:
                    stars = '**'
                elif p <= 0.01:
                    stars = '*'
                # elif p < 0.05:
                #     stars = '*'
                ax.text(j, i, txt, ha='center', va='center',
                        fontsize=8, color=col, fontweight='bold')
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

    ax.set(xticks=range(J), xticklabels=labels,
           yticks=range(I), yticklabels=order,
           ylabel='cell #', title='Speed-tuning by monotonicity')
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    # legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
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
        assert t.levene_p_values is not None, "call significance_test() first"

    POS_RGB = _hex_to_rgb(POS_COLOR)
    NEG_RGB = _hex_to_rgb(NEG_COLOR)
    NM_RGB  = _hex_to_rgb(NM_COLOR)
    DARK_GRAY = (0.3, 0.3, 0.3)
    MOD_BG = (0.82, 0.88, 0.97)      # light blue for modulated
    COLS = {'positive': POS_RGB, 'negative': NEG_RGB, 'non-monotonic': NM_RGB}

    J = len(labels)
    I = N_CELLS

    # Build modulated background mask
    mod_mask = np.zeros((I, J), dtype=bool)
    for j, lbl in enumerate(labels):
        if lbl in modulated_mask:
            mod_mask[:, j] = np.asarray(modulated_mask[lbl], dtype=bool)

    # Sort by mean |ρ| (desc), then by modulated (desc)
    mean_abs_rho = np.mean([np.abs(tunings[lbl].rho) for lbl in labels], axis=0)
    ns = ~np.any([tunings[lbl].significant_mask for lbl in labels], axis=0)
    mean_abs_rho[ns] = 0

    modulated_any = np.zeros(I, dtype=bool)
    for j, lbl in enumerate(labels):
        if lbl in modulated_mask:
            modulated_any |= np.asarray(modulated_mask[lbl], dtype=bool)

    order = np.lexsort((-modulated_any.astype(int), -mean_abs_rho))

    mod_mask = mod_mask[order]

    fig, ax = plt.subplots(figsize=figsize)

    # Background: light blue for modulated, white otherwise
    img = np.ones((I, J, 3))
    img[mod_mask] = MOD_BG
    ax.imshow(img, aspect='auto', interpolation='nearest')

    # Text: coloured ρ + significance stars for tuned cells
    for i in range(I):
        ci = order[i]
        for j, lbl in enumerate(labels):
            t = tunings[lbl]
            if t.significant_mask[ci]:
                if t.monotonic_mask['positive'][ci]:
                    col = COLS['positive']
                elif t.monotonic_mask['negative'][ci]:
                    col = COLS['negative']
                else:
                    col = COLS['non-monotonic']
                txt = f'{t.rho[ci]:.3f}'.replace('0.', '.', 1)
                ax.text(j, i, txt, ha='center', va='center',
                        fontsize=8, color=col, fontweight='bold')
                # p-value stars
                p = t.levene_p_values[ci]
                stars = ''
                if p <= 0.0001:
                    stars = '***'
                elif p <= 0.001:
                    stars = '**'
                elif p <= 0.01:
                    stars = '*'
                if stars:
                    ax.text(j + 0.25, i, stars, ha='left', va='center',
                            fontsize=8, color='black', fontweight='bold', alpha=0.6)

    ax.set(xticks=range(J), xticklabels=labels,
           yticks=range(I), yticklabels=order,
           ylabel='cell #', title='Modulated & tuned neurons')
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    # Legend
    from matplotlib.patches import Patch as _Patch
    from matplotlib.lines import Line2D
    legend_handles = [
        _Patch(facecolor=MOD_BG, label='modulated (BinaryMod.)'),
        _Patch(facecolor=COLS['positive'], label='positive'),
        _Patch(facecolor=COLS['negative'], label='negative'),
        _Patch(facecolor=COLS['non-monotonic'], label='non-monotonic'),
    ]
    # significance stars
    blank = Line2D([], [], color='none', marker='none', linestyle='')
    legend_handles.extend([blank, blank, blank])
    legend_labels = [h.get_label() for h in legend_handles[:4]]
    legend_labels.extend(['p\u2264.01 *',  'p\u2264.001 **', 'p\u2264.0001 ***'])
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
        self.delta_r = None           # np.ndarray, shape (n_cells,); r_run - r_still
        self.mi = None                # np.ndarray, shape (n_cells,); sign-safe MI

        # filled by fit_gain_model()
        self.gain_a = None            # np.ndarray, shape (n_cells,)
        self.gain_b = None            # np.ndarray, shape (n_cells,)
        self.gain_r2 = None           # np.ndarray, shape (n_cells,)
        self.gain_valid = None        # np.ndarray of bool, shape (n_cells,)

        self.condition_still = None   # list length n_cells; each item shape (n_valid_conditions,)
        self.condition_run = None     # list length n_cells; each item shape (n_valid_conditions,)
        self.n_gain_conditions = None # np.ndarray, shape (n_cells,)

        # filled by compute_running_ttest()
        #
        # A paired t-test per cell on condition-level (R_run, R_still) pairs
        # (built by fit_gain_model, see condition_run / condition_still).
        # Note the distinction from the aggregate quantities above: delta_r
        # is a single per-cell number (overall mean running response minus
        # overall mean still response), so a t-test cannot be run on it
        # directly — the paired per-condition differences used here come
        # from condition_run - condition_still instead.
        self.t_stat = None            # np.ndarray, shape (n_cells,)
        self.t_pval = None            # np.ndarray, shape (n_cells,)
        self.tuned_mask = None            # np.ndarray, shape (n_cells,)

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
            :meth:`compute_running_ttest`, which uses per-condition pairs).
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
        
    def fit_gain_model(self, min_trials_per_state: int = 2, min_conditions: int = 3):
        """Fit linear gain model: :math:`R_{\\text{run}} = a \\cdot R_{\\text{still}} + b`.

        Parameters
        ----------
        min_trials_per_state : int, optional
            Minimum number of running and (separately) still trials a
            stimulus condition must have to contribute a data point to the
            fit, by default 2.
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
        if self.run_mask is None or self.still_mask is None:
            self.classify_trials()

        response = np.asarray(self._td.responses)
        if response.ndim != 3:
            raise ValueError(f"Expected response with shape (n_cells, n_trial, duration), got {response.shape}")
        response_mean = np.nanmean(response, axis=2) # (n_cells, n_trials)

        n_cells, n_trials = response_mean.shape
        labels = self.get_condition_labels()

        self.gain_a = np.full(n_cells, np.nan)
        self.gain_b = np.full(n_cells, np.nan)
        self.gain_r2 = np.full(n_cells, np.nan)
        self.gain_valid = np.zeros(n_cells, dtype=bool)
        self.n_gain_conditions = np.zeros(n_cells, dtype=int)
        self.condition_still = [
            np.array([], dtype=float) for _ in range(n_cells)
        ]
        self.condition_run = [
            np.array([], dtype=float) for _ in range(n_cells)
        ]

        if labels is None:
            # No visual-stimulus condition structure (e.g. spontaneous): the
            # gain model is undefined, and gain_valid is all False.
            return self.gain_a, self.gain_b

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

            run_values = np.asarray(run_values, dtype=float)
            still_values = np.asarray(still_values, dtype=float)

            self.condition_run[cell_index] = run_values
            self.condition_still[cell_index] = still_values
            self.n_gain_conditions[cell_index] = len(run_values)

            if len(run_values) < min_conditions:
                continue

            # A (near-)constant set of still-condition values makes the
            # still -> run line unidentifiable (rank-deficient fit).
            if np.ptp(still_values) <= 1e-12:
                continue

            slope, intercept = np.polyfit(still_values, run_values, deg=1)

            #check the quality of the fit
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

        return self.gain_a, self.gain_b

    def compute_running_ttest(self, threshold: float = 0.05, min_conditions: int = 3):
        """Paired t-test per cell on condition-level (R_run, R_still) pairs.

        For each cell, the per-condition mean responses under running and
        still (computed by :meth:`fit_gain_model`) are paired by stimulus
        condition. A one-sample t-test on the diffs ``d_c = R_run_c -
        R_still_c`` tests H0: mean(d) = 0 against H1: mean(d) != 0.

        This design controls for stimulus-condition identity: if running
        and still trials happen to cover different sets of orientations or
        spatial frequencies, the pairing cancels that out, whereas a pooled
        trial-level t-test would confound stimulus tuning with speed tuning.

        Requires :meth:`fit_gain_model` to have been called first (the
        condition-level pairs are built there). Cells with fewer than
        ``min_conditions`` valid pairs (e.g. spontaneous, which has none)
        are left as NaN / excluded from ``tuned_mask``.

        Parameters
        ----------
        threshold : float, optional
            p-value threshold for ``tuned_mask``, by default 0.05.
        min_conditions : int, optional
            Minimum number of valid condition pairs required to run the
            test, by default 3.

        Stores
        ------
        t_stat : np.ndarray, shape ``(n_cells,)``
            t-statistic (positive = running > still).
        t_pval : np.ndarray, shape ``(n_cells,)``
            Two-sided p-value for the paired test.
        tuned_mask : np.ndarray of bool, shape ``(n_cells,)``
            Cells with ``t_pval < threshold`` and enough conditions.
        """
        from scipy.stats import ttest_1samp

        if self.condition_still is None or self.condition_run is None:
            raise ValueError(
                "fit_gain_model() must be called before compute_running_ttest()"
            )

        n_cells = len(self.condition_still)
        t_stat = np.full(n_cells, np.nan)
        t_pval = np.full(n_cells, np.nan)

        for cell_index in range(n_cells):
            still_c = self.condition_still[cell_index]
            run_c = self.condition_run[cell_index]

            if still_c is None or run_c is None or len(still_c) < min_conditions:
                continue

            diffs = np.asarray(run_c, dtype=float) - np.asarray(still_c, dtype=float)
            diffs = diffs[np.isfinite(diffs)]

            if len(diffs) < min_conditions:
                continue
            stat, pval = ttest_1samp(diffs, 0.0)
            t_stat[cell_index] = stat
            t_pval[cell_index] = pval

        self.t_stat = t_stat
        self.t_pval = t_pval
        self.tuned_mask = t_pval < threshold


    # ------------- plotting & print -------------

    def print_tuned_cells(self):
        """Print the indices and p-values of cells with ``tuned_mask`` True."""
        assert self.tuned_mask is not None, "call compute_running_ttest() first"
        assert self.t_pval is not None, "call compute_running_ttest() first"
        print(f"Significantly tuned neurons: #{self.tuned_mask.sum()} \n {np.where(self.tuned_mask)[0]}")
        for idx in np.where(self.tuned_mask)[0]:
            print(f"  Cell {idx}: p = {self.t_pval[idx]:.5f}")

    def plot_scatter(self, cell: int = None, ax=None) -> plt.Figure:
        """Scatter plot of running vs. still responses with gain model fit.

        Parameters
        ----------
        cell : int or None, optional
            If given, plot a single cell. Otherwise show all cells
            in subplots or a combined layout.
        """
        if self.mi is None:
            self.compute_mi()

        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))
        else:
            fig = ax.figure

        if cell is None:
            x = np.asarray(self.r_still, dtype=float)
            y = np.asarray(self.r_run, dtype=float)

            valid = np.isfinite(x) & np.isfinite(y)

            ax.scatter(
                x[valid],
                y[valid],
                alpha=0.75,
                label="Cells",
            )

            title = f"{self._td.stimulus}: population response"

        else:
            if self.gain_a is None:
                self.fit_gain_model()

            x = np.asarray(self.condition_still[cell], dtype=float)
            y = np.asarray(self.condition_run[cell], dtype=float)

            valid = np.isfinite(x) & np.isfinite(y)

            ax.scatter(
                x[valid],
                y[valid],
                alpha=0.75,
                label="Stimulus conditions",
            )

            slope = self.gain_a[cell]
            intercept = self.gain_b[cell]

            if valid.sum() >= 2 and np.isfinite(slope):
                x_line = np.linspace(
                    np.min(x[valid]),
                    np.max(x[valid]),
                    100,
                )
                y_line = slope * x_line + intercept

                ax.plot(
                    x_line,
                    y_line,
                    color="black",
                    linewidth=2,
                    label=f"Fit: y={slope:.2f}x{intercept:+.3f}",
                )

            title = (
                f"{self._td.stimulus}: cell {cell} "
                f"(n={valid.sum()} conditions)"
            )

        if valid.sum() > 0:
            lower = min(np.min(x[valid]), np.min(y[valid]))
            upper = max(np.max(x[valid]), np.max(y[valid]))

            margin = 0.05 * max(upper - lower, 1e-3)
            lower -= margin
            upper += margin

            identity_kwargs = dict(linestyle="--", linewidth=1, label="Run = still")
            if cell is not None:
                identity_kwargs["color"] = "gray"

            ax.plot(
                [lower, upper],
                [lower, upper],
                **identity_kwargs,
            )

            ax.set_xlim(lower, upper)
            ax.set_ylim(lower, upper)

        ax.set_xlabel("Mean response during still trials")
        ax.set_ylabel("Mean response during running trials")
        ax.set_title(title)
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
        ax.set_title(f"{self._td.stimulus}\nn = {int(np.isfinite(self.mi).sum())} cells")
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
        analysis.compute_mi()
        analysis.fit_gain_model(
            min_trials_per_state=min_trials_per_state,
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
            "valid_gain_fits": int(np.asarray(analysis.gain_valid, dtype=bool).sum()),
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
    metric: str = "mi",
    bins: int = 20,
    value_range=None,
    ax=None,
):
    """Compare one per-cell metric across an arbitrary set of stimuli.

    Draws a one-row grid of histograms, one panel per stimulus, for the
    chosen metric read directly off each :class:`BinaryModulation` instance.
    This is the single generic cross-stimulus comparison helper; it replaces
    the earlier per-metric, per-stimulus-combination plotting functions and
    works for any subset of stimuli.

    Parameters
    ----------
    binary_results : dict
        Mapping stimulus -> :class:`BinaryModulation` (``compute_mi`` run).
    stimuli : iterable of str or None, optional
        Which stimuli to show and in what order. ``None`` (default) uses all
        keys of ``binary_results``.
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
            1, len(stimuli), figsize=(4.2 * len(stimuli), 3.8),
            sharey=True, constrained_layout=True,
        )
    else:
        axes = ax
        fig = np.atleast_1d(axes)[0].figure
    axes = np.atleast_1d(axes)

    for panel, stimulus in zip(axes, stimuli):
        analysis = binary_results[stimulus]
        values = np.asarray(getattr(analysis, metric), dtype=float)
        finite = np.isfinite(values)
        vals = values[finite]
        median = np.nanmedian(vals) if vals.size else np.nan

        panel.hist(vals, bins=bins, range=value_range, alpha=0.8, color="C0")
        panel.axvline(median, linestyle="--", color="black", label=f"median = {median:+.3f}")
        panel.axvline(0, linestyle=":", color="gray", label="no modulation")
        if value_range is not None:
            panel.set_xlim(*value_range)
        panel.set_title(f"{stimulus}\nn = {int(finite.sum())} cells")
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

    lower = min(np.min(grating_values), np.min(natural_values))
    upper = max(np.max(grating_values), np.max(natural_values))
    margin = 0.05 * max(upper - lower, 1e-3)
    lower -= margin
    upper += margin

    ax.plot([lower, upper], [lower, upper], linestyle="--", linewidth=1, color="gray", label="NS = gratings")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)

    ax.set_xlabel("Gratings MI: mean(DG, SG)")
    ax.set_ylabel("Natural scenes MI")
    title = f"Gratings vs natural scenes\nn={len(grating_values)}"
    if p_value is not None:
        title += f", p={p_value:.3g}"
    ax.set_title(title)
    ax.legend(frameon=False)

    return fig, ax


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

        title = stimulus
        if validation_df is not None:
            row = validation_df.loc[validation_df["stimulus"] == stimulus].iloc[0]
            title += f"\nrho={row['spearman_rho']:.2f}, p={row['p_value']:.2g}, n={int(row['n_cells'])}"
        ax.set_title(title)
        ax.set_xlabel("Allen run_mod (metadata)")
        ax.set_ylabel(ylabel)

    return fig, axes


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

    ax.set(xticks=range(J), xticklabels=labels,
           yticks=range(I), yticklabels=order,
           ylabel='cell #', title='Tuned neurons: computed vs metadata')
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    # Legend
    from matplotlib.patches import Patch as _Patch
    ax.legend(
        [_Patch(facecolor=BOTH_CLR), _Patch(facecolor=COMP_CLR),
         _Patch(facecolor=META_CLR)],
        ['Both tuned (B)', 'Computed only (C)', 'Metadata only (M)'],
        loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9, frameon=False,
    )
    fig.tight_layout()
    return fig


# ==============================================================================
# Analysis 3 — Predictive Encoding Models
# ==============================================================================


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

    def fit_all(self):
        """Fit all four models for every neuron.

        Stores
        ------
        r2_null : np.ndarray, shape ``(n_cells,)``
        r2_add : np.ndarray, shape ``(n_cells,)``
        r2_mult : np.ndarray, shape ``(n_cells,)``
        r2_full : np.ndarray, shape ``(n_cells,)``
        """
        raise NotImplementedError

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
        """
        raise NotImplementedError

    # ------------- plotting -------------

    def plot_r2_decomposition(self, ax=None) -> plt.Figure:
        """Visualise the :math:`\\Delta R^2` breakdown across the population.

        Suggested format: grouped box plot or violin plot with one group
        per model (Add / Mult / Full), possibly split by stimulus type.
        """
        raise NotImplementedError
