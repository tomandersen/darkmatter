"""Publication numbers for the paper, in one place. Other scripts pull them from here.

PAPER_PARAMS holds what is fixed before a run. Three numbers can only be picked from a run's own results, so they live in
<run folder>/pareto_front.txt instead, and are read into PAPER_PARAMS from there (see CHOICES):
  power_w, picked from the step 1 Pareto front of chi2 against power
  core_gas_beta and core_gas_central_density, picked from the step 1b fits at that power, used for both cases
paper.py stops and asks for each one that is not set yet. The per-galaxy core densities are fitted and written to run_dir().

Everything from a run goes to sparc/results/<run_tag>: fits and tables at the top, model runs in subfolders, and the
optimizer runs that leave only empty folders in working/. A copy of this file goes there too, as the record of the run.
Change run_tag to start a new run. Keeping it lets paper.py resume an interrupted run, but only while this file is unchanged.
"""
import json
import shutil
from pathlib import Path

from cgm_integral import DM_CUTOFF_DENSITY
from sparc_data import select_galaxies

SPARC_DIR = Path(__file__).resolve().parent

PAPER_PARAMS = {
    "run_tag": "2026_11_08-2",
    "power_w": None,                         # atomic unit of force * c. From pareto_front.txt
    "dm_cutoff_density": DM_CUTOFF_DENSITY,  # particles per cc
    "r_core_factor_gas": 2.5,
    "cases": {
        # Set from observed CGM gas, not fitted: inside R_max the fit always prefers less CGM. See cgm_sensitivity.py
        # core_gas_beta and core_gas_central_density are filled in from pareto_front.txt
        "cgm": {"cgm_central_density": 2.43e-4, "r_core_factor_cgm": 6.0, "cgm_beta": 0.5, "core_gas_beta": None, "core_gas_central_density": None},
        "no_cgm": {"cgm_central_density": 0.0, "r_core_factor_cgm": 6.0, "cgm_beta": 0.5, "core_gas_beta": None, "core_gas_central_density": None},
    },
    "galaxy_cuts": {"max_q": None, "min_inclination": None},  # None = no cut. max_q 2 and min_inclination 30 keep 153 of the 175 galaxies
    "only_galaxies": None,  # a list of names here overrides galaxy_cuts
    "effort": {
        "free_power_trials": 300,
        "fixed_power_trials": 130,
        "global_fit_num_minutes": 1,
        "individual_trials": 70,
        "individual_num_minutes": 1,
        "sensitivity_num_minutes": 0.5,
        "test_num_minutes": 12,
        "newton_num_minutes": 12,
        "table_num_minutes": 12,
    },
}


RESULTS_DIR = SPARC_DIR / "results"

# The numbers picked from a run's results, in <run folder>/pareto_front.txt as "name = value" lines
CHOICES = ("power_w", "core_gas_beta", "core_gas_central_density")
CHOICES_HEADER = """# Numbers picked from this run's results. paper.py reads them, and stops to ask for any left blank.
#   power_w: from the step 1 Pareto front below. Setting it skips step 1
#   core_gas_beta, core_gas_central_density: from the step 1b fits below, used for both cases. Setting both skips step 1b
"""


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
    load_choices()
    record_params_used()
    return path


def record_params_used():
    """What was actually used, including changes made in code, eg by paper.py --quick, and the numbers from pareto_front.txt."""
    (run_dir() / "params_used.json").write_text(json.dumps(PAPER_PARAMS, indent=2, default=str) + "\n")


def choices_file():
    return RESULTS_DIR / PAPER_PARAMS["run_tag"] / "pareto_front.txt"


def read_choices():
    """The numbers set in pareto_front.txt. A blank one is None."""
    choices = dict.fromkeys(CHOICES)
    path = choices_file()
    if path.exists():
        for line in path.read_text().splitlines():
            name, _, value = line.partition("=")
            name, value = name.strip(), value.strip()
            if name in CHOICES and not line.lstrip().startswith("#"):
                choices[name] = float(value) if value else None
    return choices


def load_choices():
    """Puts the numbers from pareto_front.txt into PAPER_PARAMS, the core gas ones in every case."""
    choices = read_choices()
    PAPER_PARAMS["power_w"] = choices["power_w"]
    for params in PAPER_PARAMS["cases"].values():
        params["core_gas_beta"] = choices["core_gas_beta"]
        params["core_gas_central_density"] = choices["core_gas_central_density"]
    return choices


def _write_choices_file(choices, notes):
    lines = [f"{name} = {'' if choices[name] is None else choices[name]}" for name in CHOICES]
    choices_file().write_text(CHOICES_HEADER + "\n".join(lines) + "\n" + notes)


def _notes():
    """Everything in pareto_front.txt after the choices: the tables the choices are picked from."""
    path = choices_file()
    if not path.exists():
        return ""
    lines = path.read_text().splitlines(keepends=True)
    last = max((i for i, line in enumerate(lines) if line.partition("=")[0].strip() in CHOICES), default=-1)
    return "".join(lines[last + 1:])


def set_choice(name, value):
    """Writes one number to pareto_front.txt and into PAPER_PARAMS."""
    choices = read_choices()
    choices[name] = value
    _write_choices_file(choices, _notes())
    load_choices()
    record_params_used()


def add_choices_notes(title, table):
    """Adds a table to the end of pareto_front.txt, as comments, replacing an older one with the same title."""
    choices = read_choices()
    notes = _notes()
    marker = f"\n# ===== {title}\n"
    notes = notes.split(marker)[0]
    notes += marker + "".join(f"# {line}\n" for line in table.splitlines())
    _write_choices_file(choices, notes)


load_choices()  # so scripts that import this file see the numbers picked for the current run_tag


FIT_DATABASES = ("global_fits", "individual_cgm", "individual_no_cgm")  # the optuna studies, one sqlite file each


def fits_db_url(name):
    """sqlite URL of one of FIT_DATABASES in the run folder."""
    return f"sqlite:///{(run_dir() / f'{name}.db').resolve()}"


def core_densities_file(case):
    return run_dir() / f"core_densities_{case}.txt"


def case_params(case):
    """The numbers for one case."""
    return dict(PAPER_PARAMS["cases"][case])


def paper_galaxies():
    """The galaxy names for this run: only_galaxies if set, otherwise everything that passes galaxy_cuts."""
    return PAPER_PARAMS["only_galaxies"] or select_galaxies(**PAPER_PARAMS["galaxy_cuts"])


def cgm_args(case):
    """Arguments of run_sparc_dm_brute_cgm that don't depend on the core gas."""
    params = PAPER_PARAMS["cases"][case]
    return dict(
        r_core_factor_gas=PAPER_PARAMS["r_core_factor_gas"],
        add_real_cgm_central_density=params["cgm_central_density"],
        r_core_factor_cgm=params["r_core_factor_cgm"],
        cgm_beta=params["cgm_beta"],
        dm_cutoff_density=PAPER_PARAMS["dm_cutoff_density"],
        only_galaxies=paper_galaxies(),
    )


def model_args(case):
    """cgm_args plus the core gas beta."""
    return dict(cgm_args(case), add_core_gas_beta=case_params(case)["core_gas_beta"])
