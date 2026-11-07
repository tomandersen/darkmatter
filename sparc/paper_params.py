"""Publication numbers for the paper, in one place. Other scripts pull them from here.

PAPER_PARAMS holds what we choose, including the core gas beta and central density: picked from the free power
fit near the publication power, not a best fit. The per-galaxy core densities are fitted and written to fits_dir().

Change run_tag to start a new set of fits and folders. Keeping it lets paper.py resume an interrupted run.
"""
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


def path_suffix():
    return f"_{PAPER_PARAMS['run_tag']}"


def fits_dir():
    path = SPARC_DIR / f"paper_fits{path_suffix()}"
    path.mkdir(exist_ok=True)
    return path


def core_densities_file(case):
    return fits_dir() / f"core_densities_{case}.txt"


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
