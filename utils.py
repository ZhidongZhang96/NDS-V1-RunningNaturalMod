import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from scipy.stats import spearmanr, wilcoxon

STIMULI = ['drifting_gratings', 'static_gratings', 'natural_scenes', 'spontaneous']
def check_stim(stimulus:str):
    assert stimulus in STIMULI, f"You must choose one of the stimulus type: {STIMULI}"

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
        if stimulus == "drifting_gratings":
            response_window = (10, 60)
        elif stimulus == "spontaneous":
            response_window = (0, 60)   # pseudo-trials
        else:
            response_window = (5, 7)    # static_gratings / natural_scenes
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

        # filled by fit_gain_model()
        self.gain_a = None            # np.ndarray, shape (n_cells,)
        self.gain_b = None            # np.ndarray, shape (n_cells,)
        self.gain_r2 = None           # np.ndarray, shape (n_cells,)

        self.condition_still = None   # list length n_cells; each item shape (n_valid_conditions,)
        self.condition_run = None     # list length n_cells; each item shape (n_valid_conditions,)
        self.n_gain_conditions = None # np.ndarray, shape (n_cells,)

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
            return self.mi

        # Average response across running / still trials for each cell.
        self.r_run = np.nanmean(response_mean[:, self.run_mask], axis=1)      # (n_cells,)
        self.r_still = np.nanmean(response_mean[:, self.still_mask], axis=1)  # (n_cells,)

        denom = self.r_run + self.r_still

        self.mi = np.full(n_cells, np.nan, dtype=float)
        valid = np.isfinite(denom) & (np.abs(denom) > 1e-12)

        self.mi[valid] = (
            self.r_run[valid] - self.r_still[valid]
        ) / denom[valid]

        return self.mi

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
        self.n_gain_conditions = np.zeros(n_cells, dtype=int)
        self.condition_still = [
            np.array([], dtype=float) for _ in range(n_cells)
        ] 
        self.condition_run = [
            np.array([], dtype=float) for _ in range(n_cells)
        ]

        if labels is None:
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

        return self.gain_a, self.gain_b
    
    # ------------- plotting -------------
    
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


# ==============================================================================
# Analysis 2 — batch pipeline & reporting helpers
# ==============================================================================


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


def compare_gratings_vs_natural(results: dict, denom_threshold: float = 1e-3):
    """Paired grating-vs-natural-scenes comparison of robust MI.

    Trials are never pooled across stimulus classes: MI is computed
    separately for drifting gratings, static gratings, and natural
    scenes, and only the resulting per-cell MI values are combined.
    For each matched cell, grating MI is the mean of the (robust)
    drifting- and static-grating MI:

    .. math::
        MI_{\\text{grating}} = \\text{mean}(MI_{\\text{DG}}, MI_{\\text{SG}})

    ``MI_grating`` is then compared against natural-scenes MI with a
    paired Wilcoxon signed-rank test, using only cells with a valid
    robust MI for DG, SG, and NS.

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
    """Validate our MI against Allen's precomputed running-modulation metrics.

    Aligns ``metadata`` to ``matched_cell_ids`` (by ``cell_specimen_id``)
    and compares, per stimulus, our MI with the corresponding Allen
    ``run_mod_*`` column using Spearman correlation.

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
    denom_threshold : float, optional
        Passed to :func:`get_robust_mi`, by default ``1e-3``.
    robust : bool, optional
        If True (default), use robust MI (see :func:`get_robust_mi`).
        If False, use raw MI with no denominator-based filtering.

    Returns
    -------
    validation_df : pandas.DataFrame
        Columns: stimulus, metadata_col, n_cells, spearman_rho, p_value,
        median_our_MI, median_metadata_run_mod.
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

        if robust:
            mi, robust_mask = get_robust_mi(analysis, denom_threshold)
        else:
            mi = analysis.mi
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
        results[stimulus].plot_mi_histogram(ax=ax)

    return fig, axes


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


def plot_metadata_validation(aligned: dict, validation_df: pd.DataFrame = None):
    """Scatter our MI against Allen ``run_mod_*`` metadata, per stimulus.

    Parameters
    ----------
    aligned : dict
        Mapping stimulus -> {"mi": array, "ref": array}, as returned by
        :func:`validate_mi_against_metadata`.
    validation_df : pandas.DataFrame, optional
        Table with ``stimulus``, ``spearman_rho``, ``p_value``, ``n_cells``
        columns, used to annotate each panel's title.

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
        ax.set_ylabel("Robust MI (ours)")

    return fig, axes


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
