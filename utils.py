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
    """

    def __init__(self, trial_data: TrialData | dict[str, TrialData], label: str = "", n_bins: int = 20):
        self._td = trial_data
        self.label = label  # or ", ".join(k for k in self._td if k)

        self.n_bins = n_bins

        # by compute_tuning()
        self.responses: np.ndarray | None = None             # shape (n_cells, n_trials_total)
        self.speeds: np.ndarray | None = None                # shape (n_trials_total,)
        self.bins_edges: np.ndarray | None = None            # shape (n_bins+1,)
        self.bins_centers: np.ndarray | None = None            # shape (n_bins,)
        self.bins_masking: np.ndarray | None = None          # shape (n_trials_total,)
        self.mean_all_responses: np.ndarray | None = None    # shape (n_cells, n_bins)
        self.mean_responses: np.ndarray | None = None        # shape (n_bins,)
        self.std_responses: np.ndarray | None = None         # shape (n_bins,)

        # by significance_test()
        self.levene_p_values: np.ndarray | None = None       # shape (n_cells,)
        self.significant_mask: np.ndarray | None = None      # bool, shape (n_cells,)

        # by compute_spearman()
        self.rho: np.ndarray | None = None                   # shape (n_cells,)
        self.rho_p_values: np.ndarray | None = None          # shape (n_cells,)
        self.monotonical_mask : dict[str, np.ndarray | None] = dict({
            'increasing': None, 
            'decreasing': None, 
            'non-monotonically': None
            }) 
        

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


    def _binned_responses(self, bins_masking=None):
        """Bin trials by ``bins_masking`` and average responses per bin.

        Parameters
        ----------
        bins_masking : np.ndarray or None
            Bin assignments for each trial, shape ``(n_trials,)``.
            Uses ``self.bins_masking`` if None.

        Returns
        -------
        bins_edges : np.ndarray, shape ``(n_bins+1,)``
        bins_masking : np.ndarray, shape ``(n_trials,)``
        mean_all_responses : np.ndarray, shape ``(n_cells, n_bins)``
        """
        assert self.speeds is not None, "call compute_tuning() first"
        assert self.responses is not None, "call compute_tuning() first"

        # bin the speed
        if self.bins_edges is None:
            bins_edges = np.linspace(self.speeds.min(), self.speeds.max()+1e-6, num=self.n_bins+1)
            self.bins_centers = (bins_edges[:-1] + bins_edges[1:]) / 2
        else:
            bins_edges = self.bins_edges
        if bins_masking is None:
            bins_masking = np.digitize(self.speeds, bins_edges) if self.bins_masking is None else self.bins_masking

        # compute the mean and std of responses
        mean_all_responses = []
        for b in range(1, self.n_bins+1):
            res_bin = self.responses[:, bins_masking==b]   # (n_cells, n_trial in the bin)
            mean_all_responses.append(res_bin.mean(axis=1))
        mean_all_responses = np.array(mean_all_responses).T # (n_cells, n_bins)

        return bins_edges, bins_masking, mean_all_responses


    # ------------- core computation -------------

    def compute_tuning(self):
        """Bin trials by running speed and compute tuning curves.

        Parameters
        ----------
        n_bins : int, optional
            Number of equal-width speed bins, by default 20.
        """
        self.responses, self.speeds = self._pooled()
        self.bins_edges, self.bins_masking, self.mean_all_responses = \
            self._binned_responses()
        # compute the mean and std across cells
        self.mean_responses = self.mean_all_responses.mean(axis=0)
        self.std_responses = self.mean_all_responses.std(axis=0)


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
        assert self.bins_masking is not None, "call compute_tuning() first"

        # the real tuning — variance across bins
        vs_real = self.mean_all_responses.var(axis=1)   # (n_cells)

        # shuffled — permute bin labels, re-compute variance
        vs_shuffled = []
        for _ in range(n_shuffles):
            shuffled_bins_masking = np.random.permutation(self.bins_masking)
            _, _, mean_all_res = self._binned_responses(bins_masking=shuffled_bins_masking) # (n_cells, n_bins)
            vs_shuffled.append(mean_all_res.var(axis=1))
        vs_shuffled = np.array(vs_shuffled).T   # (n_cells, n_shuffles)

        p_values = np.mean(vs_shuffled >= vs_real[:, np.newaxis], axis=1)    # (n_cells)
        significant_mask = p_values < threshold

        self.levene_p_values = p_values
        self.significant_mask = significant_mask


    def compute_spearman(self, threshold = 0.05):
        """Spearman rank correlation between response and running speed per cell, to test monotonicity of tuning. 
        
        Note that only those neurons significantly tuned tested by :func:`significance_test` will be tested.

        Stores
        ------
        rho : np.ndarray, shape ``(n_cells,)``
        rho_p_values : np.ndarray, shape ``(n_cells,)``
        monotonical_mask : dict[str, np.array], with elements shape ``(n_cells,)``
        """
        assert self.responses is not None, "call compute_tuning() first"
        assert self.bins_masking is not None, "call compute_tuning() first"
        assert self.significant_mask is not None, "call significance_test() first"

        from scipy.stats import spearmanr
        seq_speed = self.bins_masking   # (n_trials_total)
        seq_responses = self.responses  # (n_cells, n_trials_total)

        combined_mat = np.vstack([seq_speed, seq_responses])    # (1+n_cells, n_trials_total)
        res = spearmanr(combined_mat, axis=1)

        rho = res.statistic[0, 1:]          # (n_cells,)
        rho_p_values = res.pvalue[0, 1:]    # (n_cells,)

        # categorize monotonicity: increasing, decreasing, or non-monotonic but tuned
        increasing = (rho > 0) & (rho_p_values < threshold) & self.significant_mask
        decreasing = (rho < 0) & (rho_p_values < threshold) & self.significant_mask
        non_monotonically =  (rho_p_values > threshold) & self.significant_mask

        self.rho = rho
        self.rho_p_values = rho_p_values
        self.monotonical_mask = {
            'increasing': increasing,
            'decreasing': decreasing,
            'non-monotonically': non_monotonically
        }


    # ------------- plotting -------------

    def plot_tuning_curve(self, cells: list[int] | None = None, figsize=(5,3), semcolor = 'pink', ax=None) -> plt.Axes:
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

        if ax == None:
            _, ax = plt.subplots(figsize=figsize)

        res_cells = self.mean_all_responses[cells] if cells else self.mean_all_responses
        means = res_cells.mean(axis=0)     # (n_bins)
        sem = res_cells.std(axis=0) / np.sqrt(res_cells.shape[0])
        
        ax.fill_between(self.bins_centers, means-sem, means+sem, color=semcolor, alpha=0.5, edgecolor='none')
        if len(res_cells) == 1:
            ax.plot(self.bins_centers, means, color='black', marker='o')
        else:
            ax.scatter(self.bins_centers, means, s=10, color='black')
        
        ax.set_xlabel('running speed (cm/s)')
        ax.set_ylabel('average $\\Delta$F/F')

        ax.set_ylim(bottom=0)
        return ax

    def plot_tuning_cells(self):
        """Plot speed tuning for each given cells in a density map"""
        raise NotImplementedError


    def plot_significant_neurons(self, ax=None) -> plt.Axes:
        """Highlight neurons that pass the significance test.

        Useful formats: bar chart of p-values with threshold line, or a
        scatter of significant vs. non-significant cells.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw into. Creates a new one if None.

        Returns
        -------
        plt.Axes
            The axes that were drawn into.
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
