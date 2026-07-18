"""
Download one or more experiment containers from AllenSDK, match cells across
sessions (A/B/C) within each container, and save locally.

Single container (``--container N``):
    Saves one standard ``.npz`` compatible with ``utils.load_data()``.

Multiple containers (``--containers N1 N2 N3``):
    Saves one ``.npz`` per container (standard format).  Pool cells at the
    per-cell metric level after analysis, NOT at raw trace level — each
    container has its own running speed and different time axes, so cell-
    level pooling after per-container analysis is the correct approach.
    Use ``utils.load_containers(dir)`` and ``utils.pool_results()`` to
    merge per-cell outputs from multiple containers.

Usage:
    python download_container.py                           # auto-pick first V1 Cux2
    python download_container.py --container 511510753      # single container
    python download_container.py --containers 511507650 511509529 --out-dir ../data
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from allensdk.core.brain_observatory_cache import BrainObservatoryCache

warnings.filterwarnings("ignore")

MANIFEST = str(Path(__file__).resolve().parent.parent / "boc" / "manifest.json")
_LETTER = {"three_session_A": "A", "three_session_B": "B", "three_session_C": "C"}
STIM_BY_SESSION = {"A": ["drifting_gratings"], "B": ["static_gratings", "natural_scenes"],
                   "C": ["locally_sparse_noise"]}
DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "data"


def log(*a):
    print(*a, flush=True)


# ── helpers ──────────────────────────────────────────────────────────────

def _find_containers(boc, container_ids):
    """Return list of container dicts given IDs, or auto-pick first V1 Cux2 with A+B."""
    if container_ids is not None:
        return boc.get_experiment_containers(ids=container_ids)

    conts = [c for c in boc.get_experiment_containers(
        targeted_structures=["VISp"], cre_lines=["Cux2-CreERT2"])
        if not c.get("failed_experiment_container", False)]
    for cont in sorted(conts, key=lambda c: c["id"]):
        exps = boc.get_ophys_experiments(experiment_container_ids=[cont["id"]])
        letters = {_LETTER.get(e["session_type"]) for e in exps}
        if "A" in letters and "B" in letters:
            return [cont]
    raise RuntimeError("No VISp Cux2-CreERT2 container with A+B found")


def _exp_map(boc, container_id):
    """Return {letter: ophys_experiment_id} for sessions A/B/C."""
    exps = boc.get_ophys_experiments(experiment_container_ids=[container_id])
    return {_LETTER[e["session_type"]]: e["id"]
            for e in exps if e["session_type"] in _LETTER}


def _download_session(boc, exp_id, letter):
    """Download one session, return dict of arrays + metadata."""
    log(f"  session {letter} (exp {exp_id}) ...")
    ds = boc.get_ophys_experiment_data(exp_id)

    ts, dff = ds.get_dff_traces()
    ts, dff = np.asarray(ts), np.asarray(dff, dtype=np.float32)

    dxcm, _ = ds.get_running_speed()
    dxcm = np.asarray(dxcm)
    running = np.zeros((2, len(dxcm)), dtype=np.float32)
    running[0] = np.nan_to_num(dxcm)

    csid = np.asarray(ds.get_cell_specimen_ids())
    roi_masks = np.asarray(ds.get_roi_mask_array(), dtype=bool)

    n = min(len(ts), dff.shape[1], len(dxcm))
    ts, dff = ts[:n], dff[:, :n]

    stim_tables = {s: ds.get_stimulus_table(s) for s in STIM_BY_SESSION[letter]}

    try:
        epoch_table = ds.get_stimulus_epoch_table()
    except Exception:
        # Fallback: build epoch table from stim tables + running-speed gaps
        log("    (epoch table unavailable, building minimal fallback)")
        rows = []
        for stim_name, df in stim_tables.items():
            start = int(df["start"].iloc[0])
            end = int(df["end"].iloc[-1])
            rows.append({"stimulus": stim_name, "start": start, "end": end})
        epoch_table = pd.DataFrame(rows)

    log(f"    dff {dff.shape}, cells={len(csid)}")
    return dict(letter=letter, session_type=f"three_session_{letter}",
                t=ts, dff=dff, running_speed=running, csid=csid,
                roi_masks=roi_masks,
                stim_tables=stim_tables, epoch_table=epoch_table)


def _match_cells(sessions):
    """Align cells across a container's A/B/C sessions by cell_specimen_id."""
    common = None
    for s in sessions:
        ids = set(s["csid"].tolist())
        common = ids if common is None else common & ids
    if common is None:
        raise ValueError("no sessions to match")
    common_ids = np.array(sorted(common))
    log(f"    cells matched: {len(common_ids)}")
    for s in sessions:
        order = np.array([{c: i for i, c in enumerate(s["csid"])}[cid] for cid in common_ids])
        s["dff"] = s["dff"][order]
        s["roi_masks"] = s["roi_masks"][order]
        s["csid"] = s["csid"][order]
    return sessions, common_ids


def _load_templates(boc, by_letter):
    """Load stimulus templates from available sessions (best-effort)."""
    templates = {}
    for letter, stim_name, template_key in [
        ("C", "locally_sparse_noise", "locally_sparse_noise"),
        ("B", "natural_scenes", "natural_scenes"),
    ]:
        if letter not in by_letter:
            continue
        try:
            ds = boc.get_ophys_experiment_data(by_letter[letter]["source_exp_id"])
            templates[template_key] = ds.get_stimulus_template(stim_name)
        except Exception:
            pass  # template not available for this container
    return templates


def _save_container_npz(out_path, by_letter, matched_ids, templates):
    """Save single container in standard load_data()-compatible .npz format."""
    data = {"matched_cell_ids": matched_ids}
    for L, s in by_letter.items():
        data[f"{L}__session_type"] = s["session_type"]
        data[f"{L}__t"] = s["t"]
        data[f"{L}__dff"] = s["dff"]
        data[f"{L}__running_speed"] = s["running_speed"]
        data[f"{L}__roi_masks"] = s["roi_masks"]
        for stim, df in s["stim_tables"].items():
            data[f"{L}__stim__{stim}__values"] = df.values.astype(np.float64)
            data[f"{L}__stim__{stim}__cols"] = np.array(list(df.columns), dtype=object)
        data[f"{L}__epoch__values"] = s["epoch_table"].values
        data[f"{L}__epoch__cols"] = np.array(list(s["epoch_table"].columns), dtype=str)
    for name, arr in templates.items():
        data[f"tmpl__{name}"] = arr
    log(f"  saving {out_path.name} ...")
    np.savez(out_path, **data)


# ── main ─────────────────────────────────────────────────────────────────

def download_containers(container_ids, out_dir):
    """Download containers, save one .npz per container."""
    boc = BrainObservatoryCache(manifest_file=MANIFEST)
    container_ids = sorted(container_ids)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for cid in container_ids:
        cont = _find_containers(boc, [cid])[0]
        exp_map = _exp_map(boc, cid)
        log(f"\nContainer {cid}: {cont['targeted_structure']} "
            f"depth={cont['imaging_depth']} cre={cont['cre_line']}")

        raw = [_download_session(boc, eid, letter) for letter, eid in sorted(exp_map.items())]
        sessions, matched_ids = _match_cells(raw)
        by_letter = {s["letter"]: s for s in sessions}

        # store source exp_id on each session for template lookup
        for L in by_letter:
            by_letter[L]["source_exp_id"] = exp_map.get(L)

        templates = _load_templates(boc, by_letter)

        fname = f"container_{cid}.npz"
        _save_container_npz(out_dir / fname, by_letter, matched_ids, templates)
        log(f"  -> {len(matched_ids)} matched cells → {out_dir / fname}")

    log(f"\nDone. {len(container_ids)} container(s) saved to {out_dir}")
    log("Use utils.load_containers() to load all .npz files in this directory.")


def main():
    parser = argparse.ArgumentParser(description="Download AllenSDK container(s) → .npz")
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--container", type=int, default=None, help="Single container ID")
    g.add_argument("--containers", type=int, nargs="+", default=None,
                   help="Multiple container IDs")
    parser.add_argument("--out-dir", type=str, default=None,
                        help=f"Output directory (default: {DEFAULT_OUT_DIR})")

    args = parser.parse_args()

    if args.container:
        cids = [args.container]
    elif args.containers:
        cids = args.containers
    else:
        boc = BrainObservatoryCache(manifest_file=MANIFEST)
        cont = _find_containers(boc, None)[0]
        cids = [cont["id"]]
        log(f"Auto-selected container {cont['id']} "
            f"(VISp, {cont['cre_line']}, depth={cont['imaging_depth']})")

    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUT_DIR
    download_containers(cids, out_dir)


if __name__ == "__main__":
    main()
