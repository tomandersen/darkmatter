"""Publication numbers for the paper, in one place. Other scripts pull them from here.

PAPER_PARAMS holds what we choose, including the core gas beta and central density: picked from the free power
fit near the publication power, not a best fit. The per-galaxy core densities are fitted and written to run_dir().

Everything from a run goes to sparc/results/<run_tag>: fits and tables at the top, model runs in subfolders, and the
optimizer runs that leave only empty folders in working/. A copy of this file goes there too, as the record of the run.
Change run_tag to start a new run. Keeping it lets paper.py resume an interrupted run, but only while this file is unchanged.
"""
import json
import shutil
from pathlib import Path

from cgm_integral import DM_CUTOFF_DENSITY

SPARC_DIR = Path(__file__).resolve().parent

PAPER_PARAMS = {
    "run_tag": "2026_11_06",
    "power_w": 24.7,                         # atomic unit of force * c
    "dm_cutoff_density": DM_CUTOFF_DENSITY,  # particles per cc
    "r_core_factor_gas": 2.5,
    "cases": {
        # Set from observed CGM gas, not fitted: inside R_max the fit always prefers less CGM. See cgm_sensitivity.py
        "cgm": {"cgm_central_density": 2.43e-4, "r_core_factor_cgm": 6.0, "cgm_beta": 0.5, "core_gas_beta": 3.1, "core_gas_central_density": 0.4},
        "no_cgm": {"cgm_central_density": 0.0, "r_core_factor_cgm": 6.0, "cgm_beta": 0.5, "core_gas_beta": 3.1, "core_gas_central_density": 0.4},
    },
    "only_galaxies": None,  # None = all SPARC galaxies
    "effort": {
        "free_power_trials": 200,
        "fixed_power_trials": 80,
        "global_fit_num_minutes": 4,
        "individual_trials": 70,
        "individual_num_minutes": 6,
        "sensitivity_num_minutes": 1,
        "test_num_minutes": 10,
        "newton_num_minutes": 5,
        "table_num_minutes": 6,
    },
}


RESULTS_DIR = SPARC_DIR / "results"


def run_dir():
    """sparc/results/<run_tag>: fits, tables and the record of the parameters."""
    path = RESULTS_DIR / PAPER_PARAMS["run_tag"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def output_dir(name):
    """A model run's folder in the run folder, as a string for output_dir_override."""
    return str(run_dir() / name)


def working_dir(name):
    """For the optimizer runs, which write no graphs and leave only empty folders."""
    return str(run_dir() / "working" / name)


def start_run(fresh=False):
    """Records the parameters in the run folder. Stops if this file changed since the run started. fresh empties the folder first."""
    path = RESULTS_DIR / PAPER_PARAMS["run_tag"]
    if fresh and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).read_text()
    record = path / "paper_params.py"
    if record.exists() and record.read_text() != source:
        raise SystemExit(f"paper_params.py has changed since run {PAPER_PARAMS['run_tag']} started (see {record}).\n"
                         f"Change run_tag to start a new run, or put back the old parameters to resume this one.")
    record.write_text(source)
    # what was actually used, including changes made in code, eg by paper.py --quick
    (path / "params_used.json").write_text(json.dumps(PAPER_PARAMS, indent=2, default=str) + "\n")
    return path


def core_densities_file(case):
    return run_dir() / f"core_densities_{case}.txt"


def case_params(case):
    """The numbers for one case."""
    return dict(PAPER_PARAMS["cases"][case])


def cgm_args(case):
    """Arguments of run_sparc_dm_brute_cgm that don't depend on the core gas."""
    params = PAPER_PARAMS["cases"][case]
    return dict(
        r_core_factor_gas=PAPER_PARAMS["r_core_factor_gas"],
        add_real_cgm_central_density=params["cgm_central_density"],
        r_core_factor_cgm=params["r_core_factor_cgm"],
        cgm_beta=params["cgm_beta"],
        dm_cutoff_density=PAPER_PARAMS["dm_cutoff_density"],
        only_galaxies=PAPER_PARAMS["only_galaxies"],
    )


def model_args(case):
    """cgm_args plus the core gas beta."""
    return dict(cgm_args(case), add_core_gas_beta=case_params(case)["core_gas_beta"])
