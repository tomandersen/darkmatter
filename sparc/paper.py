"""Runs everything for the paper, in order. Expect about 12 hours for the full run.

  1. Free power fit: the chi2 against power tradeoff behind the choice of power, core gas beta and density (with CGM)
  1b. Fixed power fit: a dense look at core gas beta and density at the publication power (with CGM). Feeds nothing later
  2. CGM sensitivity: why the CGM is set from observations and not fitted
  3. Individual galaxy fits: one core densities file for each case
  4. Tests: the graphs and numbers for the paper
  5. Added gas and halo tables, the only place R_200 is evaluated

All steps resume: stop it any time, and the same command picks up where it stopped. Numbers are in paper_params.py.
Fits go to sparc/paper_fits_<run_tag>, runs to the dm_mass_* folders, the same run_tag on the end of each.

  .venv/bin/python sparc/paper.py                    full run
  .venv/bin/python sparc/paper.py --skip-free-power  without step 1
  .venv/bin/python sparc/paper.py --quick            a few minutes on 5 galaxies, to check that it all runs
"""
import argparse
import json
import os
import time
from pathlib import Path

CASES = ("cgm", "no_cgm")
QUICK_GALAXIES = ["DDO154", "NGC2403", "UGC02953", "NGC0289", "NGC7814"]
QUICK_EFFORT = dict(free_power_trials=4, fixed_power_trials=4, global_fit_num_minutes=0.3, individual_trials=4, individual_num_minutes=0.3,
                    sensitivity_num_minutes=0.1, test_num_minutes=0.3, newton_num_minutes=0.3, table_num_minutes=0.3)


def timed_step(title, function, *args):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}", flush=True)
    start = time.time()
    result = function(*args)
    print(f"--- {title} took {(time.time() - start) / 3600:.2f} hours", flush=True)
    return result


def run_tests():
    from paper_params import PAPER_PARAMS, case_params, core_densities_file, fits_dir, model_args, path_suffix
    from sparc_dm_bruteCGM import run_sparc_dm_brute_cgm

    effort = PAPER_PARAMS["effort"]
    power = PAPER_PARAMS["power_w"]
    results_file = fits_dir() / "test_results.json"
    results = json.loads(results_file.read_text()) if results_file.exists() else {}

    def run_test(name, folder, **kwargs):
        if name in results:
            print(f"Test {name} already done")
            return
        kwargs.setdefault("num_minutes", effort["test_num_minutes"])
        err_per_point, best_power, unobserved_gas_fraction = run_sparc_dm_brute_cgm(
            output_graphs=True, output_dir_override=f"./sparc/{folder}{path_suffix()}", **kwargs)
        results[name] = {"err_per_point": err_per_point, "power_w": best_power, "unobserved_gas_fraction": unobserved_gas_fraction}
        results_file.write_text(json.dumps(results, indent=2) + "\n")

    for case in CASES:
        # the core densities made with optimize_individual_galaxies.py
        run_test(f"core_file_{case}", f"dm_mass_core_file_{case}", add_core_gas_central_density=0.0,  # ignored as we are using the file
                 use_core_gas_file=True, core_file=core_densities_file(case), power_override=power, **model_args(case))
        # one core gas density for all galaxies, from paper_params.py
        run_test(f"single_gas_{case}", f"dm_mass_single_gas_{case}", add_core_gas_central_density=case_params(case)["core_gas_central_density"],
                 power_override=power, **model_args(case))

    no_added_gas = dict(model_args("no_cgm"), add_core_gas_central_density=0.0)  # the no_cgm case adds no CGM, and no core gas here
    # no gas added, and let global P vary
    run_test("no_gas", "dm_mass_no_gas", power_override=0.0, **no_added_gas)
    # a pure Newton run: set power to 1e-5, so it is effectively off
    run_test("newton", "dm_mass_Newton", power_override=0.00001, num_minutes=effort["newton_num_minutes"], **no_added_gas)

    print(f"\n{'Test':<20}{'err/pt':>10}{'P (W)':>12}{'unobserved gas':>16}")
    for name, row in results.items():
        print(f"{name:<20}{row['err_per_point']:>10.4f}{row['power_w']:>12.4g}{row['unobserved_gas_fraction']:>16.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="a few minutes on 5 galaxies, in separate *_quick folders")
    parser.add_argument("--skip-free-power", action="store_true", help="leave out the free power fit")
    args = parser.parse_args()

    os.chdir(Path(__file__).resolve().parent.parent)  # the model opens its data files relative to the repo root

    # Imported here so the worker processes, which import this file, don't load everything twice
    from paper_params import PAPER_PARAMS, case_params

    if args.quick:
        PAPER_PARAMS.update(run_tag="quick", only_galaxies=QUICK_GALAXIES)
        PAPER_PARAMS["effort"].update(QUICK_EFFORT)

    import added_gas_table
    from cgm_sensitivity import run_cgm_sensitivity
    from optimize_individual_galaxies import optimize_individual_galaxies
    from optimizer import run_fixed_power, run_free_power

    start = time.time()
    if not args.skip_free_power:
        timed_step("Step 1: free power fit, with CGM", run_free_power, "cgm")
    timed_step("Step 1b: fixed power fit, with CGM", run_fixed_power, "cgm")
    timed_step("Step 2: CGM sensitivity", run_cgm_sensitivity)
    for case in CASES:
        timed_step(f"Step 3: individual galaxy fits, {case}", optimize_individual_galaxies, case)
    timed_step("Step 4: tests", run_tests)
    for case in CASES:
        timed_step(f"Step 5: added gas table, {case}", added_gas_table.main, case)

    print(f"\nAll done in {(time.time() - start) / 3600:.2f} hours. Run tag: {PAPER_PARAMS['run_tag']}")
    for case in CASES:
        print(f"{case}: {case_params(case)}")


# The guard matters: joblib workers import this file, and without it each one would start the whole pipeline
if __name__ == "__main__":
    main()
