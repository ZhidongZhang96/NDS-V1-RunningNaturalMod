import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass

N_CELLS = 47
STIMULI = ['drifting_gratings', 'static_gratings', 'natural_scenes', 'spontaneous']
SHORT_STIM = ['dg', 'sg', 'ns', 'spont']

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

    if not response_window:
        if stimulus == "drifting_gratings":
            response_window = (10, 60)
        elif stimulus == "spontaneous":
            response_window = (0, 60)   # pseudo-trials
        else:
            response_window = (5, 7)    # static_gratings / natural_scenes
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
POS_COLOR = '#E74C3C'                        # positive  (red)
NEG_COLOR = '#3498DB'                        # negative  (blue)
NM_COLOR  = "#6A8B6D"                        # non-monotonic (teal)
OTHER_COLOR = '#7f7f7f'                      # "Others" in condition plots


def _hex_to_rgb(h: str) -> tuple[float, float, float]:
    """Convert hex colour '#RRGGBB' to (R, G, B) in [0, 1]."""
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))

POS_RGB = _hex_to_rgb(POS_COLOR)
NEG_RGB = _hex_to_rgb(NEG_COLOR)
NM_RGB  = _hex_to_rgb(NM_COLOR)
DARK_GRAY = (0.3, 0.3, 0.3)  # for blending weak-tuned cells in grid plots


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
    label : str, optional
        Human-readable label for this analysis scenario, used in plot titles.
        If empty, auto-generated from dict keys.
    mode : str, optional
        The way to bin the data, 'equal_size' or 'equal_counts'. By default 'equal_size'
    """

    def __init__(self, trial_data: TrialData | dict[str, TrialData], label: str = "", mode='equal_size', n_bins: int = 20):
        self._td = trial_data
        self.label = label  # or ", ".join(k for k in self._td if k)
        self.n_bins = n_bins
        self.mode = mode   

        # by compute_tuning()
        self.responses: np.ndarray | None = None             # shape (n_cells, n_trials_total)
        self.speeds: np.ndarray | None = None                # shape (n_trials_total,)

        self.bins_edges: np.ndarray | None = None            # shape (n_bins+1,)
        self.bins_centers: np.ndarray | None = None          # shape (n_bins,)
        self.bins_ids: np.ndarray | None = None              # shape (n_trials_total,) starts from `1`
        self.bins_sub_ids: np.ndarray | None = None          # shape (n_trials_total,) with ignored ones `-1`

        self.mean_all_responses: np.ndarray | None = None    # shape (n_cells, n_bins)
        self.mean_responses: np.ndarray | None = None        # shape (n_bins,)
        self.std_responses: np.ndarray | None = None         # shape (n_bins,)

        # by significance_test()
        self.levene_p_values: np.ndarray | None = None       # shape (n_cells,)
        self.significant_mask: np.ndarray | None = None      # bool, shape (n_cells,)

        # by compute_spearman()
        self.rho: np.ndarray | None = None                   # shape (n_cells,)
        self.rho_p_values: np.ndarray | None = None          # shape (n_cells,)
        self.monotonic_mask : dict[str, np.ndarray] | None = None
        

    # ------------- helpers -------------

    def _pooled(self):
        """Pool trials across all stored TrialData.

        Averages the response and running-speed time windows to produce
        one value per trial, then concatenates trials from all stimuli.

        Returns
        -------
        responses : np.ndarray, shape ``(n_cells, n_trials_total)``
            Mean ΔF/F per trial (averaged over the response window).
        speed : np.ndarray, shape ``(n_trials_total,)``
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
            Bin assignments for each trial, shape ``(n_trials,)``.
            Uses ``self.bins_ids`` if None.
        mode : str, optional
            The way to bin the data
        Returns
        -------
        bins_edges : np.ndarray, shape ``(n_bins+1,)``
        bins_ids : np.ndarray, shape ``(n_trials,)``
        mean_all_responses : np.ndarray, shape ``(n_cells, n_bins)``
        """
        assert self.speeds is not None, "call compute_tuning() first"
        assert self.responses is not None, "call compute_tuning() first"

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

        # compute the mean and std of responses
        mean_all_responses = []
        for b in range(1, self.n_bins+1):
            res_bin = self.responses[:, bins_ids==b]   # (n_cells, n_trial in the bin)
            mean_all_responses.append(res_bin.mean(axis=1))
        mean_all_responses = np.array(mean_all_responses).T # (n_cells, n_bins)

        return bins_edges, bins_ids, mean_all_responses

    def _subsample(self, max_per_bin=None, seed=42):
        """Subsample the low-speed bins, to overcome distribution unbalance for further statistical test"""
        assert self.bins_ids is not None
        bins_ids = self.bins_ids.copy()    # (n_trials_total,)

        ids, bin_counts = np.unique(bins_ids, return_counts=True)
        order = bin_counts.argsort()[::-1]   # descending sorting
        if max_per_bin is None:
            max_per_bin = bin_counts[order[1]]   # the 2nd largest num

        for id in ids[order]:
            if bin_counts[id-1] > max_per_bin: 
                trials = np.where(bins_ids==id)[0]
                # subsample
                rng = np.random.default_rng(seed) 
                selected = rng.choice(trials, size=max_per_bin, replace=False)
                # set the non-selected trials' bin_ids to be `-1`
                unselected = np.setdiff1d(trials, selected)
                bins_ids[unselected] = -1

        self.bins_sub_ids = bins_ids


    # ------------- core computation -------------
    
    def run(self):
        """All in one function to compute tuning, test significance and monoticity"""
        self.compute_tuning()
        self.significance_test()
        self.compute_spearman()

    def compute_tuning(self):
        """Bin trials by running speed and compute tuning curves.

        Parameters
        ----------
        n_bins : int, optional
            Number of equal-width speed bins, by default 20.
        """
        self.responses, self.speeds = self._pooled()
        self.bins_edges, self.bins_ids, self.mean_all_responses = \
            self._binned_responses()

        # compute the mean and std across cells
        self.mean_responses = self.mean_all_responses.mean(axis=0)
        self.std_responses = self.mean_all_responses.std(axis=0)

        # subsample, for significant test
        self.bins_sub_ids = self.bins_ids
        self._subsample() 

    def significance_test(self, n_shuffles: int = 1000, threshold: float = 0.05):
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

        Stores
        ------
        levene_p_values : np.ndarray, shape ``(n_cells,)``
        significant_mask : np.ndarray of bool, shape ``(n_cells,)``
        """
        assert self.mean_all_responses is not None, "call compute_tuning() first"
        assert self.bins_sub_ids is not None, "call compute_tuning() first"

        # the real tuning — variance across bins
        vs_real = self.mean_all_responses.var(axis=1)   # (n_cells)

        # shuffled — permute bin labels, re-compute variance
        vs_shuffled = []
        for _ in range(n_shuffles):
            shuffled_bins_ids = np.random.permutation(self.bins_sub_ids)
            _, _, mean_all_res = self._binned_responses(bins_ids=shuffled_bins_ids) # (n_cells, n_bins)
            vs_shuffled.append(mean_all_res.var(axis=1))
        vs_shuffled = np.array(vs_shuffled).T   # (n_cells, n_shuffles)

        p_values = np.mean(vs_shuffled >= vs_real[:, np.newaxis], axis=1)    # (n_cells)
        significant_mask = p_values < threshold

        self.levene_p_values = p_values
        self.significant_mask = significant_mask

    def compute_spearman(self, rho_threshold=0, p_threshold = 0.05):
        """Spearman rank correlation between response and running speed per cell, to test monotonicity of tuning. 
        
        Note that only those neurons significantly tuned tested by :func:`significance_test` will be tested.

        Stores
        ------
        rho : np.ndarray, shape ``(n_cells,)``
        rho_p_values : np.ndarray, shape ``(n_cells,)``
        monotonic_mask : dict[str, np.array], with elements shape ``(n_cells,)``
        """

        assert self.responses is not None, "call compute_tuning() first"
        assert self.bins_sub_ids is not None, "call compute_tuning() first"
        assert self.significant_mask is not None, "call significance_test() first"

        from scipy.stats import spearmanr
        seq_speed_ids = self.bins_sub_ids   # (n_trials_total)
        seq_responses = self.responses  # (n_cells, n_trials_total)

        combined_mat = np.vstack([seq_speed_ids, seq_responses])    # (1+n_cells, n_trials_total)
        res = spearmanr(combined_mat, axis=1)

        rho = res.statistic[0, 1:]          # (n_cells,)
        rho_p_values = res.pvalue[0, 1:]    # (n_cells,)

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
        assert self.bins_centers is not None, "call compute_tuning() first"

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        res_cells = self.mean_all_responses[cells] if cells else self.mean_all_responses
        means = res_cells.mean(axis=0)
        sem = res_cells.std(axis=0) / np.sqrt(res_cells.shape[0])

        ax.fill_between(self.bins_centers, means - sem, means + sem,
                        color=semcolor, alpha=0.5, edgecolor='none', label=label)
        ax.plot(self.bins_centers, means, color=semcolor,
                marker='o', markersize=3,
                markerfacecolor='black', markeredgecolor='black')
        ax.set_xlabel('running speed (cm/s)')
        ax.set_ylabel('average $\\Delta$F/F')
        return ax

    def _plot_tuning_by_monotonicity(self, responses, ylabel, axes, figsize, cells=None):
        """Shared core: plot 3-panel monotonicity figure from a response matrix.

        Parameters
        ----------
        responses : np.ndarray, shape (n_cells, n_bins)
            Response values to plot (raw mean or z-scored).
        ylabel : str
            Y-axis label for the leftmost subplot.
        axes : array-like of 3 Axes or None
        figsize : tuple
        cells : list[int] or None, optional
            Subset of cells to plot. None = all cells.

        Returns
        -------
        plt.Figure
        """
        assert self.bins_centers is not None, "call compute_tuning() first"
        assert self.significant_mask is not None, "call significance_test() first"
        assert self.monotonic_mask is not None and self.rho is not None, "call compute_spearman() first"

        if cells is not None:
            cells = list(cells) if not isinstance(cells, int) else [cells]
            responses = responses[cells]
            bg = ~self.significant_mask[cells]
            rho = self.rho[cells]
            masks = {k: self.monotonic_mask[k][cells] for k in ('positive', 'negative', 'non-monotonic')}
        else:
            bg = ~self.significant_mask
            rho = self.rho
            masks = self.monotonic_mask

        cats = [
            ('positive',      POS_COLOR),
            ('negative',      NEG_COLOR),
            ('non-monotonic', NM_COLOR),
        ]

        if axes is None:
            fig, axes = plt.subplots(1, 3, figsize=figsize)
        else:
            fig = axes.flat[0].figure if hasattr(axes, 'flat') else axes[0].figure

        for ax, (key, color) in zip(axes, cats):
            mask = masks[key]
            n = mask.sum()
            label = 'non-mono' if key == 'non-monotonic' else key

            # non-significant cells as grey background
            if bg.any():
                r = responses[bg]
                m = r.mean(axis=0)
                s = r.std(axis=0) / np.sqrt(r.shape[0])
                ax.fill_between(self.bins_centers, m - s, m + s,
                                color='lightgray', alpha=0.5, edgecolor='none')
                ax.plot(self.bins_centers, m, color='lightgray',
                        marker='o', markersize=3,
                        markerfacecolor='black', markeredgecolor='black')

            # category cells in colour
            if n > 0:
                r = responses[mask]
                m = r.mean(axis=0)
                s = r.std(axis=0) / np.sqrt(r.shape[0])
                ax.fill_between(self.bins_centers, m - s, m + s,
                                color=color, alpha=0.5, edgecolor='none')
                ax.plot(self.bins_centers, m, color=color,
                        marker='o', markersize=3,
                        markerfacecolor='black', markeredgecolor='black')

            rho_mean = rho[mask].mean() if n > 0 else float('nan')
            ax.set_title(f'{label} (# {n}, $\\bar{{\\rho}}$={rho_mean:.3f})')
            ax.set_ylabel('')

        axes[0].set_ylabel(ylabel)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        return fig

    def plot_tuning_by_monotonicity(self, axes=None, figsize=(10, 3.5), cells=None) -> plt.Figure:
        """Subplots: tuning curves for positive / negative / non-monotonic cells separately."""
        assert self.mean_all_responses is not None, "call compute_tuning() first"
        return self._plot_tuning_by_monotonicity(
            self.mean_all_responses, 'average $\\Delta$F/F', axes, figsize, cells)

    def plot_tuning_by_monotonicity_zscore(self, axes=None, figsize=(10, 3.5), cells=None) -> plt.Figure:
        """Subplots with per-cell z-scored tuning curves for each monotonicity category."""
        assert self.mean_all_responses is not None, "call compute_tuning() first"

        mu = self.mean_all_responses.mean(axis=1, keepdims=True)
        sd = self.mean_all_responses.std(axis=1, keepdims=True)
        sd = np.where(sd == 0, 1.0, sd)  # avoid division by zero for flat cells
        responses_z = (self.mean_all_responses - mu) / sd

        return self._plot_tuning_by_monotonicity(
            responses_z, 'z-scored $\\Delta$F/F', axes, figsize, cells)


def plot_tuning_curves_grid(tunings: dict[str, SpeedTuning],
                             labels: list[str] | None = None,
                             cells: int | list[int] | None = None,
                             figsize=(8, 6),
                             show_rho: bool = False) -> plt.Figure:
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
        tunings[lbl].plot_tuning_curve(cells=cells, ax=ax, semcolor='gray')
        title = lbl
        if show_rho:
            t = tunings[lbl]
            assert t.rho is not None, "call compute_spearman() first"
            rho_vals = t.rho[cells] if cells else t.rho
            title += f'  ($\\bar{{\\rho}}$={rho_vals.mean():.3f})'
        ax.set_title(title)
        ax.set_ylim(bottom=0)

    # x label only on bottom row
    for ax in axes[1, :]:
        ax.set_xlabel('running speed (cm/s)')
    for ax in axes[0, :]:
        ax.set_xlabel('')
    # y label only on left column
    for ax in axes[:, 0]:
        ax.set_ylabel('average $\\Delta$F/F')
    for ax in axes[:, 1]:
        ax.set_ylabel('')

    # hide unused subplots
    for i in range(len(labels), 4):
        axes[i].set_visible(False)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig

def plot_monotonicity_stacked_bar(tunings: dict[str, SpeedTuning],
                                   ax: plt.Axes = None,
                                   colors: dict[str, str]| None = None, 
                                   figsize=(5, 4)) -> plt.Axes:
    """Stacked bar chart: for each stimulus, breakdown of significantly tuned
    neurons by monotonicity (positive / negative / non-monotonic).

    Parameters
    ----------
    tunings : dict[str, SpeedTuning]
        Mapping from stimulus label to SpeedTuning (must have
        ``compute_spearman()`` called).
    ax : plt.Axes, optional
    colors : dict[str, str], optional
        Category colours. Default: positive=POS_COLOR (red),
        negative=NEG_COLOR (blue), non-monotonic=NM_COLOR (grey).

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if colors is None:
        colors = {'positive': POS_COLOR, 'negative': NEG_COLOR,
                  'non-monotonic': NM_COLOR}

    labels = list(tunings.keys())
    categories = ['non-monotonic', 'negative', 'positive']

    # fraction of significant cells in each category, per stimulus
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
                ax.text(i, y_mid, f'# {str(int(v))}', ha='center', va='center',
                        fontsize=8, color='white', fontweight='bold')
            y_offset += v

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('# significant tuned neurons')
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
        _, ax = plt.subplots(figsize=(5, 5))

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
                            figsize=(6, 10)) -> plt.Figure:
    """Grey background for responsive cells, coloured ρ text (by monotonicity
    category) with alpha = |ρ| for speed-tuned cells."""
    labels = list(tunings.keys())
    # assert all tunings have computed results
    for t in tunings.values():
        assert t.rho is not None, "call compute_spearman() first"
        assert t.significant_mask is not None, "call significance_test() first"
        assert t.monotonic_mask is not None, "call compute_spearman() first"

    J = len(labels)
    I = len(next(iter(tunings.values())).rho)

    # colours (matching module-level constants)
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

    # --- Plot ---
    fig, ax = plt.subplots(figsize=figsize)

    # grey background for responsive cells
    img = np.ones((I, J, 3))
    img[resp_mask] = LG
    ax.imshow(img, aspect='auto', interpolation='nearest')

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
                ax.text(j, i, txt, ha='center', va='center',
                        fontsize=8, color=col, fontweight='bold')

    ax.set(xticks=range(J), xticklabels=labels,
           yticks=range(I), yticklabels=order,
           ylabel='cell #', title='Speed-tuning by monotonicity')
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    from matplotlib.patches import Patch
    ax.legend(
        [Patch(facecolor=COLS['positive']), Patch(facecolor=COLS['negative']),
         Patch(facecolor=COLS['non-monotonic']), Patch(facecolor=LG)],
        ['positive', 'negative', 'non-monotonic', 'responsive'],
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
        self.run_mask = None          # np.ndarray of bool, shape (n_trials,)
        self.still_mask = None        # np.ndarray of bool, shape (n_trials,)
        self.ignored_mask = None      # np.ndarray of bool, shape (n_trials,)
        # filled by compute_mi()
        self.mi = None                # np.ndarray, shape (n_cells,)
        # filled by fit_gain_model()
        self.gain_a = None            # np.ndarray, shape (n_cells,)
        self.gain_b = None            # np.ndarray, shape (n_cells,)

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
        raise NotImplementedError

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
        raise NotImplementedError

    def fit_gain_model(self):
        """Fit linear gain model: :math:`R_{\\text{run}} = a \\cdot R_{\\text{still}} + b`.

        Stores
        ------
        gain_a : np.ndarray, shape ``(n_cells,)``
            Multiplicative coefficient.
        gain_b : np.ndarray, shape ``(n_cells,)``
            Additive offset.
        """
        raise NotImplementedError
    
    # ------------- plotting -------------
    
    def plot_scatter(self, cell: int = None, ax=None) -> plt.Figure:
        """Scatter plot of running vs. still responses with gain model fit.

        Parameters
        ----------
        cell : int or None, optional
            If given, plot a single cell. Otherwise show all cells
            in subplots or a combined layout.
        """
        raise NotImplementedError

    def plot_mi_histogram(self, ax=None) -> plt.Figure:
        """Histogram of Modulation Index across the population.

        Optionally mark the median MI and compare to a null distribution
        (e.g. shuffle labels).
        """
        raise NotImplementedError


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
