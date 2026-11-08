"""Compares two finished paper.py runs that used different galaxy samples, eg the 153 good galaxies against all 175.

err/pt for two different samples is not like for like, so the per-galaxy fits are also split into the galaxies both runs share
and the galaxies only the larger run has, and those are split by why the cuts would remove them.

Usage: .venv/bin/python sparc/compare_runs.py <run_tag_a> <run_tag_b>     (run tags are folders in sparc/results)
"""
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

from sparc_data import load_properties, load_rotation_curves

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CASES = ("cgm", "no_cgm")


def individual_fits(tag, case):
    """{galaxy: best chi2/pt} from the run's per-galaxy core gas fits."""
    db = sqlite3.connect(f"file:{RESULTS_DIR / tag / f'individual_{case}.db'}?mode=ro", uri=True)
    rows = db.execute("select s.study_name, min(v.value) from studies s join trials t on s.study_id = t.study_id "
                      "join trial_values v on t.trial_id = v.trial_id where t.state = 'COMPLETE' group by s.study_name").fetchall()
    db.close()
    return dict(rows)


def core_densities(tag, case):
    return {line.split()[0]: float(line.split()[1]) for line in open(RESULTS_DIR / tag / f"core_densities_{case}.txt") if line.strip()}


def per_point(fits, points, names):
    """chi2/pt over names, weighted by each galaxy's number of points."""
    return sum(fits[n] * points[n] for n in names) / sum(points[n] for n in names) if names else float("nan")


def mond_chi2_per_point(tag, test):
    for line in open(RESULTS_DIR / tag / test / "run.txt"):
        if "MOND Chi^2 Total" in line:
            return float(line.split()[-1])
    return float("nan")


def best_fixed_power_row(tag):
    """chi2/pt, beta and density of the best step 1b fit."""
    for line in open(RESULTS_DIR / tag / "fixed_power_cgm.txt"):
        if line.strip() and line.split()[0][0].isdigit():
            return [float(x) for x in line.split()[:3]]
    return [float("nan")] * 3


def compare(tag_a, tag_b):
    properties = load_properties()
    points = {name: len(curve["R"]) for name, curve in load_rotation_curves().items()}

    print(f"A = {tag_a}\nB = {tag_b}")
    tests_a, tests_b = (json.loads((RESULTS_DIR / tag / "test_results.json").read_text()) for tag in (tag_a, tag_b))
    print(f"\n{'Test':<20}{'err/pt A':>10}{'err/pt B':>10}{'P (W) A':>10}{'P (W) B':>10}{'unobs A':>9}{'unobs B':>9}")
    for name in tests_a:
        a, b = tests_a[name], tests_b[name]
        print(f"{name:<20}{a['err_per_point']:>10.4f}{b['err_per_point']:>10.4f}{a['power_w']:>10.3f}{b['power_w']:>10.3f}"
              f"{a['unobserved_gas_fraction']:>9.3f}{b['unobserved_gas_fraction']:>9.3f}")

    print(f"\n{'MOND chi2/pt':<20}{'A':>10}{'B':>10}")
    for test in ("single_gas_cgm", "core_file_cgm"):
        print(f"{test:<20}{mond_chi2_per_point(tag_a, test):>10.2f}{mond_chi2_per_point(tag_b, test):>10.2f}")

    print(f"\n{'Best step 1b fit':<20}{'chi2/pt':>10}{'beta':>8}{'density':>9}")
    for label, tag in (("A", tag_a), ("B", tag_b)):
        chi2, beta, density = best_fixed_power_row(tag)
        print(f"{label:<20}{chi2:>10.4f}{beta:>8.2f}{density:>9.3f}")

    for case in CASES:
        fits_a, fits_b = individual_fits(tag_a, case), individual_fits(tag_b, case)
        core_a, core_b = core_densities(tag_a, case), core_densities(tag_b, case)
        shared = [n for n in fits_b if n in fits_a]
        only_b = [n for n in fits_b if n not in fits_a]
        print(f"\nPer-galaxy core gas fits, {case}: A has {len(fits_a)} galaxies, B has {len(fits_b)}, {len(shared)} shared")
        print(f"  {'group':<34}{'galaxies':>9}{'points':>8}{'chi2/pt':>9}{'median':>9}{'median n_core':>15}")

        def row(label, fits, core, names):
            median_chi2 = np.median([fits[n] for n in names]) if names else float("nan")
            median_core = np.median([core[n] for n in names if n in core]) if names else float("nan")
            print(f"  {label:<34}{len(names):>9}{sum(points[n] for n in names):>8}{per_point(fits, points, names):>9.4f}{median_chi2:>9.4f}{median_core:>15.3f}")

        row("A, all", fits_a, core_a, list(fits_a))
        row("B, all", fits_b, core_b, list(fits_b))
        row("shared, in A", fits_a, core_a, shared)
        row("shared, in B", fits_b, core_b, shared)
        row("only in B", fits_b, core_b, only_b)
        row("only in B, Q = 3", fits_b, core_b, [n for n in only_b if properties[n]["Q"] == 3])
        row("only in B, inclination < 30", fits_b, core_b, [n for n in only_b if properties[n]["Inc"] < 30])

        worst = sorted(only_b, key=lambda n: -fits_b[n])[:8]
        if worst:
            print("  Worst galaxies only in B: " + ", ".join(f"{n} {fits_b[n]:.2f} (Q{properties[n]['Q']}, i={properties[n]['Inc']:.0f})" for n in worst))
        if shared:
            changes = np.array([fits_b[n] - fits_a[n] for n in shared])
            print(f"  Shared galaxies, chi2/pt B - A: median {np.median(changes):+.4f}, largest changes {changes.min():+.3f} and {changes.max():+.3f}")


if __name__ == "__main__":
    compare(sys.argv[1], sys.argv[2])
