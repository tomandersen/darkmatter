"""How many minutes (grid size) does a run need before the fit levels out?

Runs run_sparc_dm_brute_cgm at one parameter set for each num_minutes in MINUTES and plots fit against num_minutes.
Close values (0.1, 0.2, ...) show whether the curve is smooth or jumps around.
Finished runs are kept in results.jsonl, so a rerun picks up where it stopped. Run from anywhere:
    python sparc/grid_size_test/grid_size_test.py
"""
import contextlib
import json
import os
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SPARC_DIR = HERE.parent
REPO_DIR = SPARC_DIR.parent
sys.path.insert(0, str(SPARC_DIR))
os.chdir(REPO_DIR)  # run_sparc_dm_brute_cgm reads ./sparc/... paths

from sparc_dm_bruteCGM import run_sparc_dm_brute_cgm  # noqa: E402

n_workers = 2  # other runs share the machine
MINUTES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]

ARGS = dict(
    power_override=16.0,
    add_core_gas_beta=2.7,
    add_core_gas_central_density=0.55,
    r_core_factor_gas=2.5,  # PAPER_PARAMS["r_core_factor_gas"]
    add_real_cgm_central_density=2.43e-4,
    r_core_factor_cgm=6.0,
    cgm_beta=0.5,
    use_core_gas_file=False,
    output_graphs=False,
    output_dir_override=str(HERE / "working"),
)

RESULTS_FILE = HERE / "results.jsonl"
LOG_DIR = HERE / "logs"


def run_one(minutes):
    LOG_DIR.mkdir(exist_ok=True)
    start = time.time()
    with open(LOG_DIR / f"run_{minutes}min.log", "w") as log, contextlib.redirect_stdout(log):
        fit, power, unobserved_gas_fraction = run_sparc_dm_brute_cgm(num_minutes=minutes, **ARGS)[:3]
    return {"num_minutes": minutes, "fit": float(fit), "power": float(power),
            "unobserved_gas_fraction": float(unobserved_gas_fraction), "seconds": time.time() - start}


def load_results():
    if not RESULTS_FILE.exists():
        return {}
    rows = map(json.loads, RESULTS_FILE.read_text().splitlines())
    return {row["num_minutes"]: row for row in rows}


def plot(results):
    rows = sorted(results.values(), key=lambda row: row["num_minutes"])
    minutes = [row["num_minutes"] for row in rows]
    fits = [row["fit"] for row in rows]
    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax in (ax_lin, ax_log):
        ax.plot(minutes, fits, "o-")
        ax.set_xlabel("num_minutes")
        ax.grid(True, alpha=0.3)
    ax_log.set_xscale("log")
    ax_lin.set_ylabel("fit")
    fig.suptitle(f"Fit against num_minutes: {ARGS['power_override']} W, core beta {ARGS['add_core_gas_beta']}, "
                 f"core gas {ARGS['add_core_gas_central_density']}, CGM {ARGS['add_real_cgm_central_density']}")
    fig.tight_layout()
    fig.savefig(HERE / "fit_vs_minutes.png", dpi=150)
    plt.close(fig)


def main():
    results = load_results()
    todo = [m for m in MINUTES if m not in results]
    print(f"{len(results)} done, {len(todo)} to run: {todo}")
    # longest first, so the 15 minute run is not left running alone at the end
    jobs = (joblib.delayed(run_one)(m) for m in sorted(todo, reverse=True))
    with joblib.Parallel(n_jobs=n_workers, return_as="generator_unordered") as pool:
        for row in pool(jobs):
            results[row["num_minutes"]] = row
            with open(RESULTS_FILE, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"{row['num_minutes']:>5} min: fit {row['fit']:.5g}  ({row['seconds'] / 60:.1f} min)", flush=True)
            plot(results)
    plot(results)
    print(f"Plot: {HERE / 'fit_vs_minutes.png'}")


if __name__ == "__main__":
    main()
