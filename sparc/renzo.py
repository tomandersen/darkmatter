"""Renzo's rule (Sancisi's law) test by derivatives: do the bumps in a model line up with the bumps in V_obs?

The level of a curve is fitted, so comparing values mostly measures the fit. Here only the shape counts. For each galaxy with
at least MIN_POINTS points, V_obs and each model are put on one even radius grid over the outer part of the curve
(from where V_obs first reaches OUTER_FRACTION of its peak, dropping the last EDGE_DROP points), detrended by a quadratic,
and differentiated with a Savitzky-Golay filter. The Pearson r between data and model is reported for
  d1 = dV/dR   (where the curve rises and falls)
  d2 = d2V/dR2 (bumps and dips, independent of any slope)
and as a function of lag in grid steps: features that really line up give the highest r at lag 0.

Models compared, all on the same galaxies and the same grid:
  model   this paper's fit, as in the core_file_cgm test (per-galaxy core gas from core_densities_cgm.txt)
  mond    MOND with the per-galaxy Upsilon from mond_upsilon.txt
  bar     Lelli baryons (Newton)
  gas     Lelli gas alone, to show where features come from

The global and per-galaxy fit parameters are smooth and can't make wiggles, so wiggles in the model come from the gas fed in.

  .venv/bin/python sparc/renzo.py                  results/<run_tag>/renzo
  .venv/bin/python sparc/renzo.py --min-points 40
"""
import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent))
os.chdir(Path(__file__).resolve().parent.parent)  # the model opens its data files relative to the repo root

from paper import MOND_UPSILON_FILE
from paper_params import PAPER_PARAMS, core_densities_file, model_args, run_dir, working_dir
from sparc_data import GRAPH_DPI, load_properties, load_rotation_curves
from sparc_dm_bruteCGM import run_sparc_dm_brute_cgm

CASE = "cgm"
MIN_POINTS = 30  # galaxies with at least this many rotation curve points
OUTER_FRACTION = 0.8   # the region starts where V_obs first reaches this fraction of its peak
EDGE_DROP = 2          # points dropped at the outer edge, where the filter window runs off the data
MIN_REGION_POINTS = 14  # leaves 8 after the filter's half window is dropped at each end
SG_WINDOW = 7          # Savitzky-Golay window in grid steps (odd) and polynomial order
SG_ORDER = 3
MAX_LAG = 6
BUMPY = 1.0            # bump/err at or above this: the data's bumps are at least as big as its error bars
CURVES = ("model", "mond", "bar", "gas")
COLORS = {"obs": "k", "model": "r", "mond": "c", "bar": "b", "gas": "g"}  # as in the galaxy plots


def model_curves(galaxy_names, num_minutes):
    """Run the core_file_cgm model on these galaxies and return {name: {R, obs, eobs, model, mond, bar, gas}}."""
    _, power, _, galaxies = run_sparc_dm_brute_cgm(
        add_core_gas_central_density=0.0, use_core_gas_file=True, core_file=core_densities_file(CASE),
        mond_upsilon_file=MOND_UPSILON_FILE, power_override=PAPER_PARAMS["power_w"], num_minutes=num_minutes,
        only_galaxies=galaxy_names, output_graphs=False, return_galaxies=True,
        output_dir_override=working_dir("renzo_model"), **{k: v for k, v in model_args(CASE).items() if k != "only_galaxies"})
    curves = {}
    for name, g in galaxies.items():
        v_gas = g["Vgas_test"]  # the gas as used: observed plus core gas and CGM
        v_bar_sq = np.clip(g["Vdisk"]**2 + g["Vbul"]**2 + v_gas * np.abs(v_gas), 0, None)
        lelli_bar_sq = np.clip(g["Vdisk"]**2 + g["Vbul"]**2 + g["Vgas"] * np.abs(g["Vgas"]), 0, None)
        curves[name] = dict(R=g["R"], obs=g["Vobs"], eobs=g["eVobs"], model=np.sqrt(v_bar_sq + power * g["V_dm_sq_base_kms"]),
                            mond=g["V_mond_kms"], bar=np.sqrt(lelli_bar_sq), gas=g["Vgas"])
    return curves, power


def outer_grid(c):
    """Even radius grid over the outer region, or None if it is too short."""
    R, obs = c["R"], c["obs"]
    start = int(np.argmax(obs >= OUTER_FRACTION * obs.max()))
    end = len(R) - EDGE_DROP
    if end - start < MIN_REGION_POINTS:
        return None
    return np.linspace(R[start], R[end - 1], end - start)  # same number of points as the data in the region


def derivatives(R_grid, R, v):
    """Quadratic-detrended curve and its smoothed first and second derivatives, on the interior of the grid.

    The half window at each end is dropped: there the filter extrapolates a polynomial past the data, and those end
    values (steep inner rise, edge turn-ups) would dominate the correlation instead of the bumps."""
    y = np.interp(R_grid, R, v)
    y = y - np.polyval(np.polyfit(R_grid, y, 2), R_grid)
    dr = R_grid[1] - R_grid[0]
    inner = slice(SG_WINDOW // 2, len(y) - SG_WINDOW // 2)
    return dict(R=R_grid[inner], resid=y[inner], smooth=savgol_filter(y, SG_WINDOW, SG_ORDER)[inner], d1=savgol_filter(y, SG_WINDOW, SG_ORDER, deriv=1, delta=dr)[inner],
                d2=savgol_filter(y, SG_WINDOW, SG_ORDER, deriv=2, delta=dr)[inner])


def lag_correlation(a, b, max_lag=MAX_LAG):
    """r(lag) for lag = -max_lag..max_lag grid steps; positive lag shifts b outwards."""
    lags = np.arange(-max_lag, max_lag + 1)
    r = []
    for k in lags:
        x, y = (a[k:], b[:len(b) - k]) if k >= 0 else (a[:k], b[-k:])
        r.append(pearsonr(x, y)[0] if len(x) > 4 and np.std(x) > 0 and np.std(y) > 0 else np.nan)
    return lags, np.array(r)


def analyse(name, c, props, out_dir):
    R_grid = outer_grid(c)
    if R_grid is None:
        return None
    d = {key: derivatives(R_grid, c["R"], c[key]) for key in ("obs",) + CURVES}
    R_used = d["obs"]["R"]
    row = dict(name=name, n=len(R_used), r_from=R_used[0], r_to=R_used[-1], Q=int(props["Q"]), inc=props["Inc"])
    # Bumpiness of the data: spread of the smoothed, detrended V_obs against the typical error bar in the same region
    row["bump_kms"] = np.std(d["obs"]["smooth"])
    row["err_kms"] = np.median(np.interp(R_used, c["R"], c["eobs"]))
    row["bump_over_err"] = row["bump_kms"] / row["err_kms"]
    # Bumpiness of each fit against the data: 1 = bumps of the right size, > 1 overreacts, < 1 too smooth
    for key in ("model", "mond"):
        for order in ("d1", "d2"):
            row[f"amp_{key}_{order}"] = np.std(d[key][order]) / np.std(d["obs"][order])
    lag_curves = {}
    for key in CURVES:
        for order in ("d1", "d2"):
            lags, r = lag_correlation(d["obs"][order], d[key][order])
            row[f"{key}_{order}"] = r[MAX_LAG]
            row[f"{key}_{order}_peak_lag"] = int(lags[np.nanargmax(r)]) if np.any(np.isfinite(r)) else 0
            lag_curves[(key, order)] = (lags, r)
    plot_galaxy(name, c, R_grid, d, lag_curves, row, out_dir)
    return row


def plot_galaxy(name, c, R_grid, d, lag_curves, row, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    ax.errorbar(c["R"], c["obs"], yerr=c["eobs"], fmt="ko", markersize=2, capsize=1.5, elinewidth=0.6, label="observed")
    for key in ("model", "mond", "bar"):
        ax.plot(c["R"], c[key], COLORS[key], linewidth=1.5, label=key)
    ax.axvspan(row["r_from"], row["r_to"], color="0.9", zorder=0, label="region correlated")
    ax.set(xlabel="kpc", ylabel="km/s", title=f"{name}: rotation curves", ylim=(0, None))
    ax.legend(fontsize="small", frameon=False)

    for ax, order, label in ((axes[0, 1], "d1", "dV/dR (detrended)"), (axes[1, 0], "d2", "d²V/dR² (detrended)")):
        scale = np.std(d["obs"][order]) or 1.0
        ax.plot(d["obs"]["R"], d["obs"][order] / scale, "k", linewidth=2, label="observed")
        for key in ("model", "mond"):
            s = np.std(d[key][order]) or 1.0
            ax.plot(d[key]["R"], d[key][order] / s, COLORS[key], linewidth=1.5,
                    label=f"{key}  r = {row[f'{key}_{order}']:+.2f}")
        ax.axhline(0, color="0.6", linewidth=0.5)
        ax.set(xlabel="kpc", ylabel="normalised", title=label)
        ax.legend(fontsize="small", frameon=False)

    ax = axes[1, 1]
    for key in ("model", "mond", "gas"):
        for order, style in (("d1", "-"), ("d2", "--")):
            lags, r = lag_curves[(key, order)]
            ax.plot(lags, r, color=COLORS[key], linestyle=style, linewidth=1.5, marker="o", markersize=3, label=f"{key} {order}")
    ax.axvline(0, color="0.6", linewidth=0.5)
    ax.axhline(0, color="0.6", linewidth=0.5)
    ax.set(xlabel=f"lag (grid steps of {R_grid[1] - R_grid[0]:.2f} kpc)", ylabel="Pearson r with observed", title="correlation against lag",
           ylim=(-1, 1))
    ax.legend(fontsize="small", frameon=False, ncol=2)

    for ax in axes.flat:
        ax.grid(True, linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_renzo.png", dpi=GRAPH_DPI)
    plt.close(fig)


def write_table(rows, power, out_dir):
    cols = [f"{k}_{o}" for o in ("d1", "d2") for k in CURVES]
    lines = [f"# Renzo's rule by derivatives. P = {power} W. Region: V_obs >= {OUTER_FRACTION} V_max, last {EDGE_DROP} points dropped, "
             f"quadratic detrend, Savitzky-Golay window {SG_WINDOW} order {SG_ORDER}.",
             "# Pearson r between observed and each curve at zero lag; lag = lag of the highest r for model, in grid steps.",
             f"{'galaxy':<12}{'n':>4}{'R from':>8}{'R to':>7}" + "".join(f"{c:>10}" for c in cols) + f"{'lag d1':>8}{'lag d2':>8}"]
    for row in rows:
        lines.append(f"{row['name']:<12}{row['n']:>4}{row['r_from']:>8.1f}{row['r_to']:>7.1f}" + "".join(f"{row[c]:>10.2f}" for c in cols)
                     + f"{row['model_d1_peak_lag']:>8d}{row['model_d2_peak_lag']:>8d}")

    lines.append("\n# Bumpiness. bump = spread of smoothed detrended V_obs, err = median e_Vobs, both km/s over the region.")
    lines.append("# amp = spread of the fit's derivative over the data's: 1 right size, > 1 overreacts, < 1 too smooth. "
                 "Q: SPARC quality flag, 1 best, 3 known problems.")
    amp_cols = [f"amp_{k}_{o}" for o in ("d1", "d2") for k in ("model", "mond")]
    lines.append(f"{'galaxy':<12}{'Q':>3}{'Inc':>5}{'bump':>7}{'err':>7}{'bump/err':>9}" + "".join(f"{c:>14}" for c in amp_cols))
    for row in rows:
        lines.append(f"{row['name']:<12}{row['Q']:>3d}{row['inc']:>5.0f}{row['bump_kms']:>7.2f}{row['err_kms']:>7.2f}{row['bump_over_err']:>9.2f}"
                     + "".join(f"{row[c]:>14.2f}" for c in amp_cols))

    lines.append(f"\n# Summary. Bumpy = bump/err >= {BUMPY}: the data's bumps are at least as big as the error bars.")
    subsets = [("all", rows), ("Q = 1", [r for r in rows if r["Q"] == 1]), ("Q >= 2", [r for r in rows if r["Q"] >= 2]),
               ("bumpy", [r for r in rows if r["bump_over_err"] >= BUMPY]),
               ("bumpy, Q = 1", [r for r in rows if r["bump_over_err"] >= BUMPY and r["Q"] == 1]),
               ("not bumpy", [r for r in rows if r["bump_over_err"] < BUMPY])]
    for label, subset in subsets:
        lines.append(f"\n{label}: {len(subset)} galaxies")
        if not subset:
            continue
        for order in ("d1", "d2"):
            model, mond, bar = (np.array([r[f"{k}_{order}"] for r in subset]) for k in ("model", "mond", "bar"))
            amp_model, amp_mond = (np.array([r[f"amp_{k}_{order}"] for r in subset]) for k in ("model", "mond"))
            near = sum(abs(r[f"model_{order}_peak_lag"]) <= 1 for r in subset)
            lines.append(f"  {order}: median r  model {np.nanmedian(model):+.2f}  mond {np.nanmedian(mond):+.2f}  bar {np.nanmedian(bar):+.2f}"
                         f" | model > mond in {np.sum(model > mond)}/{len(subset)} | model |lag| <= 1 in {near}"
                         f" | median amp  model {np.median(amp_model):.2f}  mond {np.median(amp_mond):.2f}")
    text = "\n".join(lines) + "\n"
    (out_dir / "renzo_table.txt").write_text(text)
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-points", type=int, default=MIN_POINTS, help="use galaxies with at least this many rotation curve points")
    parser.add_argument("--minutes", type=float, default=PAPER_PARAMS["effort"]["test_num_minutes"], help="model resolution")
    args = parser.parse_args()

    names = [name for name, g in load_rotation_curves(None).items() if len(g["R"]) >= args.min_points]
    properties = load_properties()
    print(f"{len(names)} galaxies with {args.min_points} or more points")
    out_dir = run_dir() / "renzo"
    (out_dir / "galaxies").mkdir(parents=True, exist_ok=True)

    curves, power = model_curves(names, args.minutes)
    rows = []
    for name in names:
        row = analyse(name, curves[name], properties[name], out_dir / "galaxies")
        if row is None:
            print(f"{name}: outer region too short, skipped")
        else:
            rows.append(row)
    print(write_table(rows, power, out_dir))
    print(f"Plots and table in {out_dir}")


if __name__ == "__main__":
    main()
