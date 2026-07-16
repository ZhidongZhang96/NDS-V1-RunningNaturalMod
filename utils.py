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
    "spontaneous": (0, 60),
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
        Running speed during each trial, shape ``(n_trials,)``.
    time : np.ndarray
        Real time points for each trial, in seconds, shape ``(n_trials, duration)``.
    stimulus_params : dict[str, np.ndarray]
        Trial-level stimulus parameters. Each entry has shape ``(n_trials,)``.
        Keys correspond to the stimulus table columns (excluding ``start``/``end``),
        e.g. ``"orientation"``, ``"temporal_frequency"``, ``"frame"``.
    """
    stimulus: str | None = None
    params: dict | None = None
    responses: np.ndarray | None = None
    running_speed: np.ndarray | None = None
    time: np.ndarray | None = None
    stimulus_params: dict | None = None


def extract_trials(
    data,
    stimulus: str,
    response_window: tuple = (0, 60),
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
    response_window : tuple of (int, int), defaults to ``(0, 60)``.
        ``(offset, duration)`` in time points relative to trial start:
        - ``offset``: frames to skip at the beginning of each trial.
        - ``duration``: frames to include; ``None`` means use the full trial
          length (minus offset).


    Returns
    -------
    TrialData
    """
    
    check_stim(stimulus)
    trial_data = TrialData(stimulus=stimulus)

    if not response_window:
        response_window = RESPONSE_WINDOWS[stimulus]
    offset, duration = response_window

    trial_data.params = {'offset': offset, 'duration': duration}

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
        trial_data.stimulus_params = {
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

    trial_data.responses = np.stack(all_responses, axis=1)     # (n_cells, n_trials, duration)
    trial_data.running_speed = np.stack(all_speeds, axis=0)    # (n_trials, duration)
    trial_data.time = np.stack(all_times, axis=0)               # (n_trials, duration)

    return trial_data


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
                    color_map[cond] = "#7f7f7f"
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


class SpeedTuning:
    """Running-speed binned tuning curves and significance tests.

    Bins trials by running speed, computes average response ± std per bin
    (tuning curve), then tests each neuron's selectivity via a shuffle test
    and reports Spearman correlation between response and running speed.

    Inspired by (Christensen & Pillow, 2022)

    Parameters
    ----------
    trial_data : TrialData
        Extracted per-trial responses and running speed.
    """

    def __init__(self, trial_data: TrialData):
        self._td = trial_data
        # filled by compute_tuning()
        self.bin_centers = None       # np.ndarray, shape (n_bins,)
        self.mean_responses = None    # np.ndarray, shape (n_bins, n_cells)
        self.std_responses = None     # np.ndarray, shape (n_bins, n_cells)
        # filled by significance_test()
        self.p_values = None          # np.ndarray, shape (n_cells,)
        self.significant_mask = None       # np.ndarray of bool, shape (n_cells,)
        # filled by compute_spearman()
        self.rho = None               # np.ndarray, shape (n_cells,)
        self.rho_p_values = None      # np.ndarray, shape (n_cells,)

    def compute_tuning(self, n_bins: int = 20):
        """Bin trials by running speed and compute tuning curves.

        Parameters
        ----------
        n_bins : int, optional
            Number of equal-width speed bins, by default 20.

        Stores
        ------
        bin_centers : np.ndarray, shape ``(n_bins,)``
        mean_responses : np.ndarray, shape ``(n_bins, n_cells)``
        std_responses : np.ndarray, shape ``(n_bins, n_cells)``
        """
        raise NotImplementedError

    def significance_test(self, n_shuffles: int = 1000):
        """Shuffle running-speed labels and re-compute tuning curves to assess significance.

        Uses Levene's test to compare the observed tuning-curve variance
        against the shuffled distribution.

        Parameters
        ----------
        n_shuffles : int, optional
            Number of shuffles, by default 1000.

        Stores
        ------
        p_values : np.ndarray, shape ``(n_cells,)``
        significant_mask : np.ndarray of bool, shape ``(n_cells,)``
        """
        raise NotImplementedError

    def compute_spearman(self):
        """Spearman rank correlation between response and running speed per cell.

        Stores
        ------
        rho : np.ndarray, shape ``(n_cells,)``
        rho_p_values : np.ndarray, shape ``(n_cells,)``
        """
        raise NotImplementedError

    # ------------- plotting -------------

    def plot_tuning_curve(self, cell: int = None, ax=None) -> plt.Figure:
        """Plot speed tuning curve(s).

        Parameters
        ----------
        cell : int or None, optional
            If given, plot a single cell. Otherwise plot all cells as
            an image / heatmap.

        Returns
        -------
        plt.Figure
        """
        raise NotImplementedError

    def plot_significant_neurons(self, ax=None) -> plt.Figure:
        """Highlight neurons that pass the significance test.

        Useful formats: bar chart of p-values with threshold line, or a
        scatter of significant vs. non-significant cells.
        """
        raise NotImplementedError


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
        self.mi = None                # np.ndarray, shape (n_cells,)

        # filled by compute_running_ttest()
        self.t_stat = None            # np.ndarray, shape (n_cells,)
        self.t_pval = None            # np.ndarray, shape (n_cells,)
        self.tuned_mask = None            # np.ndarray, shape (n_cells,)

        # filled by fit_gain_model()
        self.gain_a = None            # np.ndarray, shape (n_cells,)
        self.gain_b = None            # np.ndarray, shape (n_cells,)
        self.gain_r2 = None           # np.ndarray, shape (n_cells,)

        self.condition_still = None   # list length n_cells; each item shape (n_valid_conditions,)
        self.condition_run = None     # list length n_cells; each item shape (n_valid_conditions,)
        self.n_gain_conditions = None # np.ndarray, shape (n_cells,)

    def _validate_state(self):
        """Verify classify_trials() has been run."""
        if self.run_mask is None or self.still_mask is None:
            raise ValueError("Please run classify_trials() first.")

    @cached_property
    def _response_mean(self):
        """Trial-mean response per cell, computed once and cached."""
        response = np.asarray(self._td.responses)
        if response.ndim != 3:
            raise ValueError(
                f"Expected response with shape (n_cells, n_trial, duration), got {response.shape}"
            )
        return np.nanmean(response, axis=2)  # (n_cells, n_trials)

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

    def compute_mi(self):
        """Compute Modulation Index per cell.

        .. math::

            MI = \\frac{R_{\\text{run}} - R_{\\text{still}}}
                       {R_{\\text{run}} + R_{\\text{still}}}

        Stores
        ------
        mi : np.ndarray, shape ``(n_cells,)``
            Modulation index for each cell.
        """
        self._validate_state()
        response_mean = self._response_mean
        n_cells = N_CELLS

        if self.run_mask.sum() == 0 or self.still_mask.sum() == 0:
            self.mi = np.full(n_cells, np.nan)
            self.r_run = np.full(n_cells, np.nan)
            self.r_still = np.full(n_cells, np.nan)
            return

        # Average response across running / still trials for each cell.
        self.r_run = np.nanmean(response_mean[:, self.run_mask], axis=1)      # (n_cells,)
        self.r_still = np.nanmean(response_mean[:, self.still_mask], axis=1)  # (n_cells,)

        denom = self.r_run + self.r_still

        self.mi = np.full(n_cells, np.nan, dtype=float)
        valid = np.isfinite(denom) & (np.abs(denom) > 1e-12)

        self.mi[valid] = (
            self.r_run[valid] - self.r_still[valid]
        ) / denom[valid]

    def compute_running_ttest(self, threshold=0.05, min_conditions: int = 3):
        """Paired t-test per cell on condition-level (R_run, R_still) pairs.

        For each cell, the per-condition mean responses under running and
        still (computed by :meth:`fit_gain_model`) are paired by stimulus
        condition. A one-sample t-test on the diffs ``d_c = R_run_c -
        R_still_c`` tests H0: mean(d) = 0 against H1: mean(d) ≠ 0.

        This design controls for stimulus-condition identity: if running
        and still trials happen to cover different sets of orientations or
        spatial frequencies, the pairing cancels that out, whereas a pooled
        trial-level t-test would confound stimulus tuning with speed tuning.

        Requires :meth:`fit_gain_model` to have been called first (the
        condition-level pairs are extracted there). Cells whose
        ``condition_run`` / ``condition_still`` are ``None`` (e.g. spontaneous
        activity) are left as NaN.

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

        self._validate_state()
        n_cells = N_CELLS

        # condition-level pairs must be populated (fit_gain_model prerequisite)
        if self.condition_still is None or self.condition_run is None:
            raise ValueError(
                "fit_gain_model() must be called before compute_running_ttest()"
            )

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

        self.tuned_mask = t_pval < threshold
        self.t_stat = t_stat
        self.t_pval = t_pval

    def get_condition_labels(self):
        """Build one visual-stimulus condition label for each trial.

        Returns
        -------
        labels : np.ndarray or None
            One label per trial. For spontaneous activity, returns None.
        """
        
        n_trials = self._td.responses.shape[1]
        stimulus = self._td.stimulus
        params = self._td.stimulus_params

        # Spontaneous activity has no visual stimulus identity.
        if stimulus == "spontaneous" or params is None:
            return None

        # Natural scenes: each image frame is one visual condition.
        if stimulus == "natural_scenes":
            return np.asarray(params["frame"])

        # Drifting gratings: condition = orientation × temporal frequency.
        if stimulus == "drifting_gratings":
            condition_keys = ["orientation", "temporal_frequency"]

        # Static gratings: condition = orientation × spatial frequency × phase.
        elif stimulus == "static_gratings":
            condition_keys = ["orientation", "spatial_frequency", "phase"]

        else:
            raise ValueError(f"Unsupported stimulus: {stimulus}")
        
        labels = np.empty(n_trials, dtype=object)
        for trial_index in range(n_trials):
            labels[trial_index] = tuple(
                params[key][trial_index]
                for key in condition_keys
            )
        return labels
        
    def fit_gain_model(self, min_trials_per_state: int = 2):
        """Fit linear gain model: :math:`R_{\\text{run}} = a \\cdot R_{\\text{still}} + b`.

        Stores
        ------
        gain_a : np.ndarray, shape ``(n_cells,)``
            Multiplicative coefficient.
        gain_b : np.ndarray, shape ``(n_cells,)``
            Additive offset.
        """
        self._validate_state()
        response_mean = self._response_mean
        n_cells = N_CELLS
        labels = self.get_condition_labels()

        self.gain_a = np.full(n_cells, np.nan)
        self.gain_b = np.full(n_cells, np.nan)
        self.gain_r2 = np.full(n_cells, np.nan)
        self.n_gain_conditions = np.zeros(n_cells, dtype=int)
        self.condition_still = [None] * n_cells
        self.condition_run = [None] * n_cells

        if labels is None:
            return

        unique_conditions = pd.unique(labels)
        condition_masks = {
            cond: np.array([label == cond for label in labels], dtype=bool)
            for cond in unique_conditions
        }
        for cell_index in range(n_cells):
            still_values = []
            run_values = []
            for condition in unique_conditions:
                mask_condition = condition_masks[condition]
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

            if len(run_values) < min_trials_per_state:
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
    
    # ------------- plotting & print -------------
    
    def print_tuned_cells(self):
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

    def plot_mi_histogram(self, ax=None) -> plt.Figure:
        """Histogram of Modulation Index across the population.

        Optionally mark the median MI and compare to a null distribution
        (e.g. shuffle labels).
        """
        if self.mi is None:
            self.compute_mi()

        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 4))
        else:
            fig = ax.figure

        valid_mi = self.mi[np.isfinite(self.mi)]

        ax.hist(valid_mi, bins=20, alpha=0.8)
        median_mi = np.nanmedian(valid_mi)

        ax.axvline(median_mi, linestyle="--", color="black", label=f"median={median_mi:.3f}")
        ax.axvline(0, linestyle=":", color="gray")

        ax.set_xlabel("Modulation Index")
        ax.set_ylabel("Number of cells")
        ax.set_title(f"{self._td.stimulus}: MI distribution")
        ax.legend()

        return fig


# ------------- batch pipeline & reporting helpers -------------


def run_binary_modulation_analysis(
    data,
    response_windows,
    run_threshold: float = 3.0,
    still_threshold: float = 0.5,
    min_trials_per_state: int = 1,
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
        Passed to :meth:`BinaryModulation.fit_gain_model`.

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
        analysis.fit_gain_model(min_trials_per_state=min_trials_per_state)
        analysis.compute_running_ttest()

        results[stimulus] = analysis

    return results


def get_robust_mi(analysis: BinaryModulation, denom_threshold: float = 1e-3):
    """Return raw MI together with a robust-cell mask.

    A cell is "robust" if its MI is finite, its denominator
    (:math:`R_{\\text{run}} + R_{\\text{still}}`) is finite, and
    :math:`|R_{\\text{run}} + R_{\\text{still}}| > \\text{denom\\_threshold}`.
    This is a *filter*, not a normalization of MI itself.

    Parameters
    ----------
    analysis : BinaryModulation
        Analysis with :meth:`compute_mi` already run.
    denom_threshold : float, optional
        Minimum allowed ``|R_run + R_still|``, by default ``1e-3``.

    Returns
    -------
    mi : np.ndarray, shape (n_cells,)
        Raw modulation index (unchanged).
    robust : np.ndarray of bool, shape (n_cells,)
        Mask of cells passing the robustness filter.
    """
    mi = analysis.mi
    denom = analysis.r_run + analysis.r_still

    robust = (
        np.isfinite(mi)
        & np.isfinite(denom)
        & (np.abs(denom) > denom_threshold)
    )

    return mi, robust


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
        median_MI, valid_gain_fits.
    """
    rows = []
    for stimulus, analysis in results.items():
        rows.append({
            "stimulus": stimulus,
            "n_trials": int(len(analysis.run_mask)),
            "n_running": int(analysis.run_mask.sum()),
            "n_still": int(analysis.still_mask.sum()),
            "n_ignored": int(analysis.ignored_mask.sum()),
            "median_MI": float(np.nanmedian(analysis.mi)),
            "valid_gain_fits": int(np.isfinite(analysis.gain_a).sum()),
        })
    return pd.DataFrame(rows)


def summarize_mi_by_stimulus(results: dict, denom_threshold: float = 1e-3) -> pd.DataFrame:
    """Per-stimulus summary of raw vs. robust modulation index.

    Parameters
    ----------
    results : dict
        Mapping stimulus -> :class:`BinaryModulation`.
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
        delta = analysis.r_run - analysis.r_still

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


def _compare_gratings_vs_natural_impl(mi_dg, mi_sg, mi_ns, valid_dg, valid_sg, valid_ns, label_suffix=""):
    """Core logic shared by raw and signsafe grating-vs-natural comparisons."""
    mi_grating = np.nanmean(np.vstack([mi_dg, mi_sg]), axis=0)

    valid = (
        valid_dg & valid_sg & valid_ns
        & np.isfinite(mi_grating) & np.isfinite(mi_ns)
    )

    grating_values = mi_grating[valid]
    natural_values = mi_ns[valid]

    stat, p = wilcoxon(grating_values, natural_values)

    result_df = pd.DataFrame([{
        "comparison": f"gratings_vs_natural_scenes{label_suffix}",
        "n_cells": int(valid.sum()),
        "median_grating_MI": float(np.nanmedian(grating_values)),
        "median_natural_scene_MI": float(np.nanmedian(natural_values)),
        "median_difference_NS_minus_grating": float(np.nanmedian(natural_values - grating_values)),
        "wilcoxon_stat": float(stat),
        "p_value": float(p),
        "frac_NS_greater_than_grating": float(np.mean(natural_values > grating_values)),
    }])

    return result_df, valid, grating_values, natural_values


def compare_gratings_vs_natural(results: dict, denom_threshold: float = 1e-3):
    """Paired grating-vs-natural-scenes comparison of robust MI.

    Trials are never pooled across stimulus classes: MI is computed
    separately for drifting gratings, static gratings, and natural
    scenes, and only the resulting per-cell MI values are combined.
    For each matched cell, grating MI is the mean of the (robust)
    drifting- and static-grating MI.

    Parameters
    ----------
    results : dict
        Must contain ``"drifting_gratings"``, ``"static_gratings"``, and
        ``"natural_scenes"`` keys mapping to :class:`BinaryModulation`.
    denom_threshold : float, optional
        Passed to :func:`get_robust_mi`, by default ``1e-3``.

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
    mi_dg, robust_dg = get_robust_mi(results["drifting_gratings"], denom_threshold)
    mi_sg, robust_sg = get_robust_mi(results["static_gratings"], denom_threshold)
    mi_ns, robust_ns = get_robust_mi(results["natural_scenes"], denom_threshold)

    return _compare_gratings_vs_natural_impl(
        mi_dg, mi_sg, mi_ns, robust_dg, robust_sg, robust_ns,
    )





# ------------- sign-safe modulation index -------------
#
# The raw MI (BinaryModulation.compute_mi) uses R_run + R_still as its
# denominator. For signed ΔF/F responses this denominator can itself be
# negative, which can reverse the sign of MI relative to the biologically
# meaningful direction of R_run - R_still. These helpers do not alter
# compute_mi() or its output; they add a sign-safe alternative that keeps
# the signed numerator but replaces the denominator with |R_run| + |R_still|,
# which is never negative.


def compute_sign_safe_mi(analysis: BinaryModulation, epsilon: float = 1e-12):
    """Compute the sign-safe modulation index for one analysis.

    .. math::

        MI_{\\text{safe}} = \\frac{R_{\\text{run}} - R_{\\text{still}}}
            {|R_{\\text{run}}| + |R_{\\text{still}}| + \\epsilon}

    Unlike the raw MI, this denominator is always positive, so
    :math:`\\mathrm{sign}(MI_{\\text{safe}})` always matches
    :math:`\\mathrm{sign}(R_{\\text{run}} - R_{\\text{still}})`, and
    :math:`MI_{\\text{safe}}` is always bounded in ``[-1, 1]``.

    Parameters
    ----------
    analysis : BinaryModulation
        Analysis with :meth:`BinaryModulation.compute_mi` already run, so
        ``r_run`` / ``r_still`` are populated.
    epsilon : float, optional
        Small constant added to the denominator only to avoid division by
        zero when both responses are exactly zero, by default ``1e-12``.

    Returns
    -------
    mi_safe : np.ndarray, shape ``(n_cells,)``
    denominator_safe : np.ndarray, shape ``(n_cells,)``
        ``|R_run| + |R_still| + epsilon``.
    finite_mask : np.ndarray of bool, shape ``(n_cells,)``
        True where ``r_run`` and ``r_still`` are both finite.
    """
    r_run = np.asarray(analysis.r_run, dtype=float)
    r_still = np.asarray(analysis.r_still, dtype=float)

    finite_mask = np.isfinite(r_run) & np.isfinite(r_still)

    denominator_safe = np.abs(r_run) + np.abs(r_still) + epsilon

    mi_safe = np.full(r_run.shape, np.nan, dtype=float)
    mi_safe[finite_mask] = (
        r_run[finite_mask] - r_still[finite_mask]
    ) / denominator_safe[finite_mask]

    return mi_safe, denominator_safe, finite_mask


def summarize_mi_versions(results: dict, epsilon: float = 1e-12) -> pd.DataFrame:
    """Per-stimulus comparison of raw MI, sign-safe MI, and delta_R.

    Parameters
    ----------
    results : dict
        Mapping stimulus -> :class:`BinaryModulation` (already has
        :meth:`compute_mi` run).
    epsilon : float, optional
        Passed to :func:`compute_sign_safe_mi`, by default ``1e-12``.

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
        mi_raw = np.asarray(analysis.mi, dtype=float)
        delta_r = r_run - r_still

        mi_safe, _, _ = compute_sign_safe_mi(analysis, epsilon=epsilon)

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


def compare_gratings_vs_natural_signsafe(results: dict, epsilon: float = 1e-12):
    """Paired grating-vs-natural-scenes comparison using sign-safe MI.

    Mirrors :func:`compare_gratings_vs_natural`, but compares
    :math:`MI_{\\text{safe}}` (see :func:`compute_sign_safe_mi`) instead of
    the raw, denominator-unstable MI. Trials are never pooled across
    stimulus classes: MI_safe is computed separately for drifting
    gratings, static gratings, and natural scenes, and only the resulting
    per-cell values are combined. For each matched cell, grating MI_safe
    is the mean of the drifting- and static-grating MI_safe.

    Parameters
    ----------
    results : dict
        Must contain ``"drifting_gratings"``, ``"static_gratings"``, and
        ``"natural_scenes"`` keys mapping to :class:`BinaryModulation`.
    epsilon : float, optional
        Passed to :func:`compute_sign_safe_mi`, by default ``1e-12``.

    Returns
    -------
    result_df : pandas.DataFrame
        One-row summary of the paired comparison.
    valid : np.ndarray of bool, shape (n_cells,)
        Mask of cells used in the comparison.
    grating_values : np.ndarray
        Grating MI_safe for the matched, valid cells.
    natural_values : np.ndarray
        Natural-scenes MI_safe for the matched, valid cells.
    """
    mi_dg, _, finite_dg = compute_sign_safe_mi(results["drifting_gratings"], epsilon=epsilon)
    mi_sg, _, finite_sg = compute_sign_safe_mi(results["static_gratings"], epsilon=epsilon)
    mi_ns, _, finite_ns = compute_sign_safe_mi(results["natural_scenes"], epsilon=epsilon)

    result_df, valid, grating_values, natural_values = _compare_gratings_vs_natural_impl(
        mi_dg, mi_sg, mi_ns, finite_dg, finite_sg, finite_ns, label_suffix="_signsafe",
    )
    result_df.rename(columns={
        "median_grating_MI": "median_grating_MI_safe",
        "median_natural_scene_MI": "median_natural_scene_MI_safe",
    }, inplace=True)

    return result_df, valid, grating_values, natural_values


def validate_signsafe_mi_against_metadata(
    results: dict,
    metadata: pd.DataFrame,
    matched_cell_ids,
    epsilon: float = 1e-12,
):
    """Validate sign-safe MI against Allen's precomputed running-modulation metrics.

    Same alignment and per-stimulus comparison logic as
    :func:`validate_mi_against_metadata`, but compares
    :math:`MI_{\\text{safe}}` (see :func:`compute_sign_safe_mi`) instead of
    raw or denominator-filtered MI.

    Parameters
    ----------
    results : dict
        Must contain ``"drifting_gratings"``, ``"static_gratings"``, and
        ``"natural_scenes"`` keys mapping to :class:`BinaryModulation`.
    metadata : pandas.DataFrame
        Table containing ``cell_specimen_id``, ``run_mod_dg``,
        ``run_mod_sg``, ``run_mod_ns`` columns.
    matched_cell_ids : array-like
        Cell IDs defining row order (typically ``data["matched_cell_ids"]``).
    epsilon : float, optional
        Passed to :func:`compute_sign_safe_mi`, by default ``1e-12``.

    Returns
    -------
    validation_df : pandas.DataFrame
        Columns: stimulus, metadata_col, n_cells, spearman_rho, p_value,
        median_our_MI_safe, median_metadata_run_mod.
    aligned : dict
        Mapping stimulus -> {"mi": array, "ref": array} of the valid,
        aligned values used for the correlation (and for plotting).
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
        mi_safe, _, finite_mask = compute_sign_safe_mi(analysis, epsilon=epsilon)

        ref = meta[meta_col].to_numpy()
        valid = finite_mask & np.isfinite(mi_safe) & np.isfinite(ref)

        rho, pval = spearmanr(mi_safe[valid], ref[valid])

        rows.append({
            "stimulus": stimulus,
            "metadata_col": meta_col,
            "n_cells": int(valid.sum()),
            "spearman_rho": float(rho),
            "p_value": float(pval),
            "median_our_MI_safe": float(np.nanmedian(mi_safe[valid])),
            "median_metadata_run_mod": float(np.nanmedian(ref[valid])),
        })
        aligned[stimulus] = {"mi": mi_safe[valid], "ref": ref[valid]}

    return pd.DataFrame(rows), aligned


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

        valid = np.isfinite(analysis.gain_a) & np.isfinite(analysis.gain_b)

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


def plot_robust_mi_histograms(results: dict, denom_threshold: float = 1e-3):
    """Plot the robust MI histogram for each stimulus in ``results``.

    Robust cells satisfy ``|R_run + R_still| > denom_threshold``
    (see :func:`get_robust_mi`); this is a filter, not a normalization.

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
    running-trial response. This plot is descriptive, not a significance
    test on its own.

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


def plot_condition_gain_example(analysis: BinaryModulation, cell: int, ax=None):
    """Plot the condition-level gain fit for one representative cell.

    Thin wrapper around :meth:`BinaryModulation.plot_scatter`.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    return analysis.plot_scatter(cell=cell, ax=ax)


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


def plot_grating_natural_paired_distribution(grating_values, natural_values, p_value=None, ax=None):
    """Paired boxplot + per-cell connecting lines for grating vs. NS MI.

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
    ax.set_ylabel("Robust MI")
    title = "Paired MI comparison"
    if p_value is not None:
        title += f"\nWilcoxon p={p_value:.3g}"
    ax.set_title(title)

    return fig, ax


def plot_metadata_validation(aligned: dict, validation_df: pd.DataFrame = None, ylabel: str = "Robust MI (ours)"):
    """Scatter our MI against Allen ``run_mod_*`` metadata, per stimulus.

    Parameters
    ----------
    aligned : dict
        Mapping stimulus -> {"mi": array, "ref": array}, as returned by
        :func:`validate_mi_against_metadata` or
        :func:`validate_signsafe_mi_against_metadata`.
    validation_df : pandas.DataFrame, optional
        Table with ``stimulus``, ``spearman_rho``, ``p_value``, ``n_cells``
        columns, used to annotate each panel's title.
    ylabel : str, optional
        Y-axis label, so the panel can name whichever MI variant was passed
        in ``aligned``. Defaults to ``"Robust MI (ours)"`` for backwards
        compatibility.

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


def plot_tuned_neurons_grid(computed_tuned: dict[str, np.ndarray],
                            metadata_tuned: dict[str, np.ndarray] | None = None,
                            figsize=(6, 10)) -> plt.Figure:
    """Grid map comparing computed vs metadata-given tuned neurons.

    Each column is a stimulus, each row is a cell. Cells are sorted by
    descending total tuned-count (any source), then by both-tuned count.

    Parameters
    ----------
    computed_tuned : dict[str, np.ndarray]
        Per-stimulus boolean masks ``(n_cells,)`` for computed tuned neurons
        (e.g. from :attr:`BinaryModulation.tuned_mask`).
    metadata_tuned : dict[str, np.ndarray] | None
        Per-stimulus boolean masks ``(n_cells,)`` for metadata-given tuned
        neurons (e.g. from ``p_run_mod_* < 0.05``). Stimuli not present in
        this dict are shown with computed-only tuning.
    figsize : tuple
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

    # ── Plot ──
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


# ==============================================================================
# Exploratory diagnostics — negative-response traces
# ==============================================================================
#
# These helpers are exploratory only. They do not modify the MI definition,
# do not correct or exclude any cell, and PCA here is a pattern-detection
# tool, not a classifier of biological inhibition.


def get_negative_response_cells(results: dict, denom_threshold: float = 0.0) -> pd.DataFrame:
    """Flag cells whose MI denominator is negative or whose MI sign disagrees with :math:`\\Delta R`.

    For every stimulus and cell this reports the raw quantities behind MI
    so that sign-ambiguous or out-of-range cells can be inspected directly,
    without altering :meth:`BinaryModulation.compute_mi`.

    Parameters
    ----------
    results : dict
        Mapping stimulus -> :class:`BinaryModulation` (already has
        :meth:`compute_mi` run).
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
        mi = np.asarray(analysis.mi, dtype=float)

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


def extract_negative_trial_traces(
    analysis: BinaryModulation,
    cell_indices,
    include_states=("running", "still"),
) -> pd.DataFrame:
    """Return individual (un-averaged) trial response traces for selected cells.

    Parameters
    ----------
    analysis : BinaryModulation
        Analysis with :meth:`classify_trials` already run.
    cell_indices : iterable of int
        Cells to extract traces for.
    include_states : tuple of str, optional
        Which trial states to include, from ``{"running", "still"}``,
        by default both.

    Returns
    -------
    pandas.DataFrame
        One row per (cell, trial) pair, columns: cell, trial, state, trace
        (1-D array over the response window, not averaged), condition
        (stimulus condition label, or ``None`` for spontaneous activity).
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
        Output of :func:`extract_negative_trial_traces` (must have a
        ``"trace"`` column of equal-length 1-D arrays).
    n_components : int, optional
        Number of principal components to keep, by default 3.
    center_each_trace : bool, optional
        If True, subtract each trace's own mean before PCA. Off by default,
        because that would remove sustained negative offsets.
    scale_features : bool, optional
        If True (default), standardize each time-point column (feature)
        to zero mean / unit variance across trials before PCA.

    Returns
    -------
    scores : np.ndarray, shape (n_kept, n_components)
        PCA scores for the retained (finite) traces.
    components : np.ndarray, shape (n_components, n_timepoints)
        Principal-component temporal loadings.
    explained_variance_ratio : np.ndarray, shape (n_components,)
    metadata : pandas.DataFrame
        Rows of ``traces`` corresponding to ``scores`` (non-finite traces
        dropped), with the same index alignment as ``scores``.
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


# ------------- plotting -------------


def plot_negative_trace_examples(traces: pd.DataFrame, cell: int, n_examples: int = 6, ax=None):
    """Plot example running/still trial traces plus mean +/- std band, for one cell.

    Parameters
    ----------
    traces : pandas.DataFrame
        Output of :func:`extract_negative_trial_traces`.
    cell : int
        Cell to plot.
    n_examples : int, optional
        Number of individual raw traces to overlay per state, by default 6.
    ax : matplotlib.axes.Axes, optional

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

    Parameters
    ----------
    traces : pandas.DataFrame
        Output of :func:`extract_negative_trial_traces` (or the ``metadata``
        returned by :func:`run_negative_trace_pca`, if rows were dropped).
    pca_scores : np.ndarray, optional
        PC scores aligned row-for-row with ``traces``; if given, traces are
        additionally sorted by PC1 within each (cell, state) group.
    ax : matplotlib.axes.Axes, optional

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

    # Separators + group labels for each (cell, state) block. Groups are
    # contiguous because sorted_df is sorted by (cell, state[, PC1]).
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

    Parameters
    ----------
    scores : np.ndarray, shape (n_traces, n_components)
        From :func:`run_negative_trace_pca`.
    metadata : pandas.DataFrame
        Aligned metadata from :func:`run_negative_trace_pca`, with ``cell``
        and ``state`` columns.
    pc_x, pc_y : int, optional
        Zero-based component indices to plot, by default (0, 1) i.e. PC1/PC2.
    ax : matplotlib.axes.Axes, optional

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

    Parameters
    ----------
    components : np.ndarray, shape (n_components, n_timepoints)
        From :func:`run_negative_trace_pca`.
    explained_variance_ratio : np.ndarray, shape (n_components,)
        From :func:`run_negative_trace_pca`.
    n_show : int, optional
        Number of leading components to plot, by default 3.
    ax : matplotlib.axes.Axes, optional

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
