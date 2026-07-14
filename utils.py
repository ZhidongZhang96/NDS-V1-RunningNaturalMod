import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass

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

            ax.plot(
                [lower, upper],
                [lower, upper],
                linestyle="--",
                linewidth=1,
                label="Run = still",
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
