"""Publication numbers for the paper, in one place. Other scripts pull them from here.

PAPER_PARAMS holds what we choose. What the pipeline fits (core gas beta and central density, per-galaxy
core densities) is written to fits_dir() and read back through case_params() and model_args().

Change run_tag to start a new set of fits and folders. Keeping it lets paper.py resume an interrupted run.
"""
import json
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
        "cgm": {"cgm_central_density": 2.43e-4, "r_core_factor_cgm": 6.0, "cgm_beta": 0.5},
        "no_cgm": {"cgm_central_density": 0.0, "r_core_factor_cgm": 6.0, "cgm_beta": 0.5},
    },
    "only_galaxies": None,  # None = all SPARC galaxies
    "effort": {
        "free_power_trials": 200,
        "global_fit_trials": 150,
        "global_fit_num_minutes": 4,
        "individual_trials": 70,
        "individual_num_minutes": 6,
        "sensitivity_num_minutes": 1,
        "test_num_minutes": 10,
        "newton_num_minutes": 5,
        "table_num_minutes": 6,
    },
}


def path_suffix():
    return f"_{PAPER_PARAMS['run_tag']}"


def fits_dir():
    path = SPARC_DIR / f"paper_fits{path_suffix()}"
    path.mkdir(exist_ok=True)
    return path


def core_densities_file(case):
    return fits_dir() / f"core_densities_{case}.txt"


def global_fit_file(case):
    return fits_dir() / f"global_fit_{case}.json"


def save_global_fit(case, fit):
    global_fit_file(case).write_text(json.dumps(fit, indent=2) + "\n")


def case_params(case):
    """The numbers for one case, with its global fit included once that has been run."""
    params = dict(PAPER_PARAMS["cases"][case])
    if global_fit_file(case).exists():
        params.update(json.loads(global_fit_file(case).read_text()))
    return params


def cgm_args(case):
    """Arguments of run_sparc_dm_brute_cgm that don't depend on the core gas fit."""
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
    """cgm_args plus the fitted core gas beta. The global fit for this case must have been run."""
    if not global_fit_file(case).exists():
        raise FileNotFoundError(f"No global fit at {global_fit_file(case)}: run the global fit step in paper.py first")
    return dict(cgm_args(case), add_core_gas_beta=case_params(case)["core_gas_beta"])
