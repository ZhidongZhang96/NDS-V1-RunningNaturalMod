"""Unit tests for the EncodingModel (Analysis 3) building blocks.

Run from the repo root in the ``allensdk`` env:  ``pytest tests/``
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from utils import tent_basis, EncodingModel, TrialData


# ----------------------------- tent_basis ---------------------------------

def test_tent_basis_hand_values():
    B = tent_basis([0.0, 0.5, 1.0], [0, 1])
    assert np.allclose(B, [[1, 0], [0.5, 0.5], [0, 1]])


def test_tent_basis_partition_of_unity():
    centers = np.linspace(0, 10, 5)
    xs = np.linspace(0, 10, 51)               # within the knot span
    assert np.allclose(tent_basis(xs, centers).sum(axis=1), 1.0)


def test_tent_basis_peak_and_single_center():
    centers = np.linspace(0, 8, 4)
    assert np.allclose(np.diag(tent_basis(centers, centers)), 1.0)   # 1 at each center
    assert np.allclose(tent_basis([1, 2, 3], [5]), 1.0)             # single center -> ones


# ----------------------------- fixtures -----------------------------------

def _synthetic_td(n_cells=3, n_trials=60, duration=5, n_cond=4, seed=0):
    rng = np.random.default_rng(seed)
    responses = rng.normal(size=(n_cells, n_trials, duration))
    running_speed = np.abs(rng.normal(size=(n_trials, duration)))
    time = np.tile(np.arange(n_trials, dtype=float)[:, None], (1, duration))
    orientation = np.tile(np.arange(n_cond), n_trials // n_cond + 1)[:n_trials].astype(float)
    return TrialData(
        stimulus="drifting_gratings",
        params={"offset": 0, "duration": duration},
        responses=responses,
        running_speed=running_speed,
        time=time,
        stimulus_params={"orientation": orientation},
    )


# --------------------------- _build_design --------------------------------

def test_stimulus_onehot():
    td = _synthetic_td(n_cond=4)
    S = EncodingModel(td)._stimulus_onehot()
    assert S.shape == (td.responses.shape[1], 4)
    assert np.allclose(S.sum(1), 1.0)          # exactly one condition active per trial


@pytest.mark.parametrize("model,extra_cols", [("null", 0), ("add", 1), ("mult", 1), ("full", 2)])
def test_build_design_columns(model, extra_cols):
    td = _synthetic_td()
    em = EncodingModel(td, n_basis=3)
    n_cond = len(np.unique(em._condition_labels()))
    X = em._build_design(0, model, onehot=em._stimulus_onehot(), drive=em._stimulus_mean())
    # design = one-hot tuning (n_cond) + drift(n_basis) + running columns
    assert X.shape == (td.responses.shape[1], n_cond + 3 + extra_cols)


# ------------------------------- fit_all ----------------------------------

def test_fit_all_finite_and_shapes():
    td = _synthetic_td()
    em = EncodingModel(td, n_basis=3).fit_all(n_folds=3)
    n_cells = td.responses.shape[0]
    for m in ("null", "add", "mult", "full"):
        r = getattr(em, f"r2_{m}")
        assert r.shape == (n_cells,)
        assert np.isfinite(r).all()
    d_add, d_mult, d_full = em.r2_decomposition()
    assert d_add.shape == d_mult.shape == d_full.shape == (n_cells,)


def test_r2_decomposition_requires_fit():
    em = EncodingModel(_synthetic_td())
    with pytest.raises(RuntimeError):
        em.r2_decomposition()


# ----------------------------- cross-validation ---------------------------

def test_cv_splits_blocked_partitions_and_purges():
    n, folds, gap = 60, 5, 5
    splits = EncodingModel._cv_splits(n, folds, "blocked", gap)
    # every trial held out exactly once (full pooled-R² coverage)
    test_all = np.sort(np.concatenate([te for _, te in splits]))
    assert np.array_equal(test_all, np.arange(n))
    for train, test in splits:
        assert np.intersect1d(train, test).size == 0            # no train/test overlap
        lo, hi = test.min(), test.max()
        # training is purged within `gap` trials of the contiguous test block
        assert not np.any((train > lo - gap) & (train < hi + gap))


def test_cv_default_is_blocked():
    assert EncodingModel.fit_all.__defaults__[-2:] == ("blocked", 5)


def test_fit_all_shuffled_still_runs():
    td = _synthetic_td()
    em = EncodingModel(td, n_basis=3).fit_all(n_folds=3, cv="shuffled")
    assert np.isfinite(em.r2_full).all()


def test_cv_splits_rejects_unknown():
    with pytest.raises(ValueError):
        EncodingModel._cv_splits(30, 5, "bogus", 5)
