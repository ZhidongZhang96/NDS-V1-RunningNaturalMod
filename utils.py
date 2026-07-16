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
    dscale = np.diff(centers.ravel()).mean()                   # uniform knot spacing
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
        return 1.0 - ss_res / ss_tot

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
        U, s, Vt = np.linalg.svd(Xz, full_matrices=False)
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

    def fit_all(self, n_folds=5, alphas=None, random_state=0):
        """Fit all four nested models per neuron with cross-validated R².

        Each neuron is fit by ridge regression (standardized features, λ chosen
        by :class:`sklearn.linear_model.RidgeCV`, unpenalized intercept) under
        ``n_folds``-fold cross-validation. The stimulus tuning ``f(S)`` is a
        **fitted, ridge-penalized one-hot** design (:meth:`_stimulus_onehot`);
        ridge shrinks the noisy per-condition estimates. The multiplicative term
        gates running by the per-condition drive, recomputed from *training*
        trials each fold (:meth:`_fold_stimulus_mean`) so the R² is leakage-free.
        The stimulus-only Null/Add designs are shared across cells and fit as a
        single multi-target ridge.

        Parameters
        ----------
        n_folds : int, optional
            Number of cross-validation folds, by default 5.
        alphas : array-like, optional
            Ridge penalties to search; defaults to ``np.logspace(-3, 3, 13)``.
        random_state : int, optional
            Seed for the fold split, by default 0.

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
        from sklearn.model_selection import KFold

        if alphas is None:
            alphas = np.logspace(-3, 3, 13)

        R = self._trial_response()                 # (n_cells, n_trials) targets
        labels = self._condition_labels()          # (n_trials,)
        S = self._stimulus_onehot(labels)          # (n_trials, n_conditions) tuning design
        n_cells, n_trials = R.shape
        models = ("null", "add", "mult", "full")
        yhat = {m: np.empty_like(R) for m in models}

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        for train_idx, test_idx in kf.split(np.arange(n_trials)):
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
