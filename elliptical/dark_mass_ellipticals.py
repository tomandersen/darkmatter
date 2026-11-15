"""Dark mass for the seven elliptical galaxies of Humphrey et al. 2006, from their hot gas.

The dark mass density is
    rho_dark(r) = P / c^3 * n(r)^(2/3)          (P = 16.54 W, n in particles per m^3, rho_dark in kg/m^3)
where n is the gas particle density, electrons counted, n = rho_gas / (0.6 m_p), about 1.67 times rho_gas / m_p.
That is the same 0.6 m_p per particle the SPARC code uses for ionized gas (ADDED_GAS_MASS_PER_PARTICLE in
sparc/cgm_integral.py).

The gas comes from the digitized gas mass line of Fig. 6 (humphrey2006_fig6_digitized.txt, made by digitize_fig6.py):
    rho_gas(r) = dM_gas/dr / (4 pi r^2) = M_gas / (4 pi r^3) * dln M_gas / dln r
with a smoothing spline through log M_gas against log r for the slope. Inside the radius where the gas line comes in
through the bottom of the plot (0.1 dex further in, where the spline's slope is steady), M_gas is carried in as a
power law with the slope it has there.
Where gas_density_measured.txt has a measured gas density profile (so far NGC 4649, from Humphrey et al. 2008, made by
digitize_ngc4649_2008.py), that is used instead, out to its last point, and joined in log-log to the Fig. 6 gas
further out. The gas mass there is the integral of that density.
The dark mass is then integrated out from r = 0, and the new total is stars + gas + dark mass.

Writes plots/<galaxy>.png (the four lines of Fig. 6 plus the dark mass and new total), plots/all_galaxies.png (all
seven, laid out as in Fig. 6), and dark_mass_results.txt. A run with another P or cutoff adds that to the names, as in
plots_P10_nc0.0001/ and dark_mass_results_P10_nc0.0001.txt, so the default run is not overwritten.

The low density cutoff multiplies the dark mass density by exp(-(n_c/n)^2), n_c = 1e-5 per cc by default, the same as
the SPARC code (DM_CUTOFF_DENSITY in sparc/cgm_integral.py).

  ../.venv/bin/python dark_mass_ellipticals.py                  P = 16.54 W, n_c = 1e-5 per cc
  ../.venv/bin/python dark_mass_ellipticals.py --power 10       another P
  ../.venv/bin/python dark_mass_ellipticals.py --cutoff 1e-4    another n_c, per cc
  ../.venv/bin/python dark_mass_ellipticals.py --no-cutoff      n^(2/3) all the way out
  ../.venv/bin/python dark_mass_ellipticals.py --fig6-gas-only  ignore the measured gas densities
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import UnivariateSpline

HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "humphrey2006_fig6_digitized.txt"
MEASURED_FILE = HERE / "gas_density_measured.txt"

POWER_W = 16.54                # P in the dark mass formula, W
GAS_MASS_PER_PARTICLE = 0.6    # m_p per particle, fully ionized H + He with the electrons counted
DM_CUTOFF_DENSITY = 1e-5       # n_c, particles per cc, the default low density cutoff

C_SI = 2.99792458e8
M_P = 1.67262192e-27
M_SUN = 1.98847e30
KPC_M = 3.0856776e19
MSUN_PER_KPC3 = M_SUN / KPC_M ** 3   # kg/m^3 in one M_sun/kpc^3

GAS_SPLINE_RMS_DEX = 0.004     # digitizing scatter in log M_gas, sets the spline smoothing
GAS_JOIN_DEX = 0.1             # the power law inward joins the gas spline this far in from where the gas line starts
MEASURED_SLOPE_POINTS = 3      # a measured density profile is carried inward with the slope through this many points
N_FINE = 3000                  # log r grid for the integration
BELOW_PANEL_DEX = 1.0          # the plots go this far below the bottom of the Fig. 6 panel
COMPARE_KPC = (10, 30, 100)    # radii for new total / old total, besides where the X-ray data end

ORDER = ["NGC4125", "NGC720", "NGC6482", "NGC1407", "NGC4472", "NGC4649", "NGC4261"]  # Fig. 6 order


def load_digitized(path=DATA_FILE):
    """{galaxy: dict(r, total, stars, dm, gas, log_r_range, log_m_range, data_end_kpc)} from the digitized data file."""
    galaxies, ranges, data_end = {}, {}, {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if line.startswith("#   NGC") and "log R" in line:
            ranges[parts[1]] = ((float(parts[4]), float(parts[6])), (float(parts[9]), float(parts[11])))
        if line.startswith("#   NGC") and "data_end_kpc" in line:
            data_end[parts[1]] = float(parts[3])
        if line.startswith("#") or not line.strip():
            continue
        galaxies.setdefault(parts[0], []).append([float(v) for v in parts[1:]])
    out = {}
    for gal, rows in galaxies.items():
        a = np.array(rows)
        out[gal] = dict(r=a[:, 0], total=a[:, 1], stars=a[:, 2], dm=a[:, 3], gas=a[:, 4],
                        log_r_range=ranges[gal][0], log_m_range=ranges[gal][1], data_end_kpc=data_end[gal])
    return out


def gas_model(r, gas):
    """Smooth M_gas(r) and its log slope, from the digitized gas line, and the join to the power law inward.

    The spline's slope at its very end is poorly pinned down (off by up to a factor 2 in NGC1407 and NGC4261), so the
    power law joins it GAS_JOIN_DEX in from where the gas line comes in, with the spline's value and slope there.
    Returns (spline, log_r0, log_m0, alpha): the spline of log M_gas against log r, the join, M_gas there, and the
    log slope there, used for the power law inward.
    """
    good = ~np.isnan(gas)
    x, y = np.log10(r[good]), np.log10(gas[good])
    spline = UnivariateSpline(x, y, k=4, s=len(x) * GAS_SPLINE_RMS_DEX ** 2)
    x_join = x[0] + GAS_JOIN_DEX
    return spline, x_join, float(spline(x_join)), float(spline.derivative()(x_join))


def load_measured(path=MEASURED_FILE):
    """{galaxy: (R_kpc, rho_gas in M_sun/kpc^3)} from the measured gas density file, if there is one."""
    if not path.exists():
        return {}
    rows = {}
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        gal, r, rho = line.split()[:3]
        rows.setdefault(gal, []).append((float(r), float(rho) * 1e-25 * 1e3 / MSUN_PER_KPC3))  # 1e-25 g/cc to M_sun/kpc^3
    return {gal: tuple(np.array(v).T) for gal, v in rows.items()}


def dark_mass_profile(gal_data, power_w, n_cut, measured=None):
    """Gas density, dark mass density and enclosed masses on a fine log r grid, from r = 0 out to the plot edge.

    measured: (R_kpc, rho_gas in M_sun/kpc^3) of a measured gas density profile. Where given, it is used instead of the
    Fig. 6 gas line: straight in log-log between the points, a power law through the first MEASURED_SLOPE_POINTS inside
    the first, and straight in log-log from the last point out to the Fig. 6 gas, which takes over GAS_JOIN_DEX further
    out or at its own join, whichever is further.
    """
    spline, log_r0, log_m0, alpha = gas_model(gal_data["r"], gal_data["gas"])
    log_r = np.linspace(gal_data["log_r_range"][0] - 1.0, gal_data["log_r_range"][1], N_FINE)
    r = 10 ** log_r
    inner = log_r < log_r0
    log_m_gas = np.where(inner, log_m0 + alpha * (log_r - log_r0), spline(log_r))
    slope = np.where(inner, alpha, spline.derivative()(log_r))
    m_gas = 10 ** log_m_gas
    rho_gas_msun = m_gas / (4 * np.pi * r ** 3) * slope                   # M_sun/kpc^3
    guess_inside = 10 ** log_r0      # inside this the gas is a power law carried inward, not read from data
    fig6_from = 10 ** log_r0         # the Fig. 6 gas line is used from here out

    if measured is not None:
        x, y = np.log10(measured[0]), np.log10(measured[1])
        s_in = np.polyfit(x[:MEASURED_SLOPE_POINTS], y[:MEASURED_SLOPE_POINTS], 1)[0]
        log_fig6_from = max(log_r0, x[-1] + GAS_JOIN_DEX)
        y_fig6 = float(np.interp(log_fig6_from, log_r, np.log10(rho_gas_msun)))
        log_rho = np.where(log_r < x[0], y[0] + s_in * (log_r - x[0]),
                           np.interp(log_r, np.append(x, log_fig6_from), np.append(y, y_fig6)))
        use = log_r < log_fig6_from
        rho_gas_msun = np.where(use, 10 ** log_rho, rho_gas_msun)
        # The gas mass is then the integral of that density, with M(<r) = 4 pi r^3 rho / (3 + s_in) inside the grid
        m_gas = 4 * np.pi * r[0] ** 3 * rho_gas_msun[0] / (3 + s_in) + np.concatenate(
            [[0.0], cumulative_trapezoid(4 * np.pi * r ** 3 * rho_gas_msun, np.log(r))])
        alpha = 3 + s_in
        guess_inside, fig6_from = 10 ** x[0], 10 ** log_fig6_from

    rho_gas = rho_gas_msun * MSUN_PER_KPC3                                 # kg/m^3
    n = rho_gas / (GAS_MASS_PER_PARTICLE * M_P)                           # particles per m^3
    term = np.where(n > 0, np.abs(n) ** (2.0 / 3.0), 0.0)
    if n_cut > 0:
        with np.errstate(over="ignore", divide="ignore"):
            term *= np.exp(-(n_cut * 1e6 / np.where(n > 0, n, 1e-300)) ** 2)
    rho_dark = power_w / C_SI ** 3 * term                                 # kg/m^3
    rho_dark_msun = rho_dark / MSUN_PER_KPC3                              # M_sun/kpc^3

    # Inside the first point of the grid rho_gas goes as r^(alpha-3) and rho_dark as r^(2(alpha-3)/3),
    # so M_dark(<r) = 4 pi r^3 rho_dark / (3 + 2(alpha-3)/3)
    m_dark_first = 4 * np.pi * r[0] ** 3 * rho_dark_msun[0] / (3 + 2 * (alpha - 3) / 3)
    m_dark = m_dark_first + np.concatenate(
        [[0.0], cumulative_trapezoid(4 * np.pi * r ** 3 * rho_dark_msun, np.log(r))])
    return dict(r=r, m_gas=m_gas, n_cc=n / 1e6, rho_gas_mp_cc=rho_gas / M_P / 1e6, rho_dark_mp_cc=rho_dark / M_P / 1e6,
                m_dark=m_dark, r0=guess_inside, fig6_from=fig6_from, alpha=alpha, measured=measured is not None,
                spline_rms=float(np.sqrt(np.mean((spline(np.log10(gal_data["r"][~np.isnan(gal_data["gas"])]))
                                                   - np.log10(gal_data["gas"][~np.isnan(gal_data["gas"])])) ** 2))))


def on_data_grid(gal_data, prof):
    """The dark mass profile at the radii of the data file, and the new total there."""
    r = gal_data["r"]
    def at(key):  # a cutoff can take rho_dark to exactly 0 far out, which log10 makes -inf and 10** brings back to 0
        with np.errstate(divide="ignore"):
            return 10 ** np.interp(np.log10(r), np.log10(prof["r"]), np.log10(prof[key]))
    m_gas, m_dark = at("m_gas"), at("m_dark")
    stars = gal_data["stars"]
    return dict(r=r, m_gas=m_gas, m_dark=m_dark, new_total=stars + m_gas + m_dark,
                n_cc=at("n_cc"), rho_dark_mp_cc=at("rho_dark_mp_cc"))


def draw_panel(ax, gal, gal_data, prof, grid, power_w, legend=True, small=False):
    r = gal_data["r"]
    inner = prof["r"] <= prof["r0"]  # where the gas is carried inward as a power law, below the Fig. 6 panel
    shows_inner = prof["r0"] > 10 ** gal_data["log_r_range"][0]  # only label the faint lines if they are on the plot
    lw = 1.0 if small else 1.4
    gas_dashes = (0, (7, 2.5, 1.2, 2.5, 1.2, 2.5))
    ax.plot(r, gal_data["total"], color="red", lw=lw, label="total, NFW+stars (Humphrey)")
    ax.plot(r, gal_data["stars"], color="magenta", ls=(0, (1, 2)), lw=lw + 0.3, label="stars")
    ax.plot(r, gal_data["dm"], color="blue", ls=(0, (6, 3)), lw=lw, label="NFW dark matter")
    ax.plot(r, gal_data["gas"], color="darkorange", ls=gas_dashes, lw=lw, label="gas")
    if prof["measured"]:
        meas = (prof["r"] >= prof["r0"]) & (prof["r"] <= prof["fig6_from"])
        ax.plot(prof["r"][meas], prof["m_gas"][meas], color="darkorange", lw=lw + 0.4,
                label="gas, from the measured density (Humphrey et al. 2008)")
    ax.plot(prof["r"][inner], prof["m_gas"][inner], color="darkorange", ls=gas_dashes, lw=lw, alpha=0.4,
            label=None if not shows_inner else f"gas, carried inward as r^alpha (alpha = {prof['alpha']:.2f})" if not small
            else "gas, carried inward as a power law")
    ax.plot(prof["r"][~inner], prof["m_dark"][~inner], color="green", lw=lw + 0.8, label=f"dark mass, P = {power_w:g} W")
    ax.plot(prof["r"][inner], prof["m_dark"][inner], color="green", lw=lw + 0.8, alpha=0.35,
            label="dark mass, from the gas carried inward" if shows_inner else None)
    ax.plot(r, grid["new_total"], color="black", lw=lw + 0.8, label="new total: stars + gas + dark mass")
    ax.set_xscale("log")
    ax.set_yscale("log")
    (lr0, lr1), (lm0, lm1) = gal_data["log_r_range"], gal_data["log_m_range"]
    # A decade below the Fig. 6 panel, so the gas carried inward shows. The Fig. 6 lines stop at the panel's bottom
    bottom = 10 ** (lm0 - BELOW_PANEL_DEX)
    top = max(10 ** lm1, 1.3 * np.nanmax(grid["new_total"]))
    ax.set_xlim(10 ** lr0, 10 ** lr1)
    ax.set_ylim(bottom, top)
    # Past the outermost X-ray data point the Humphrey lines are their model, extrapolated to R_vir
    ax.axvspan(gal_data["data_end_kpc"], 10 ** lr1, color="0.92", zorder=0, lw=0,
               label="beyond the X-ray data" if small else f"beyond the X-ray data ({gal_data['data_end_kpc']:.0f} kpc)")
    ax.axvline(gal_data["data_end_kpc"], color="0.45", lw=0.9, ls="--", zorder=0.5)
    ax.axhline(10 ** lm0, color="gray", lw=0.6, ls=":", label="bottom and top of the Fig. 6 panel (top about M_vir)")
    if top > 10 ** lm1 * 1.01:
        ax.axhline(10 ** lm1, color="gray", lw=0.6, ls=":")
    ax.set_xlabel("R (kpc)")
    ax.set_ylabel(r"Mass(<R) (M$_\odot$)")
    ax.text(0.96, 0.05, gal, transform=ax.transAxes, ha="right", va="bottom", fontsize=11 if small else 15,
            family="serif")
    ax.tick_params(which="both", direction="in", top=True, right=True)
    if legend:
        ax.legend(loc="upper left", fontsize=6.5 if small else 8, frameon=False)


def ratio_at(res, r_kpc):
    """New total over the Humphrey total at r_kpc."""
    d, g = res["data"], res["grid"]
    good = ~np.isnan(d["total"]) & ~np.isnan(g["new_total"])
    return 10 ** np.interp(np.log10(r_kpc), np.log10(d["r"][good]),
                           np.log10(g["new_total"][good] / d["total"][good]))


def write_results(results, power_w, n_cut, results_file):
    with open(results_file, "w") as f:
        f.write("# Dark mass for the Humphrey et al. 2006 ellipticals, from dark_mass_ellipticals.py\n")
        f.write(f"# rho_dark = P/c^3 n^(2/3), P = {power_w:g} W, n = rho_gas / ({GAS_MASS_PER_PARTICLE} m_p)"
                + (f", times exp(-(n_c/n)^2), n_c = {n_cut:g} per cc" if n_cut > 0 else ", no cutoff") + "\n")
        f.write("# Gas from the digitized gas mass line of Fig. 6. Inside r_gas0, 0.1 dex in from where it comes in through\n")
        f.write("# the bottom of the plot, M_gas is a power law r^alpha with the slope there.\n#\n")
        f.write("# Summary at R_max, the right edge of the panel, which is about R_vir (Table 4). Masses in 1e12 M_sun\n")
        f.write(f"# {'galaxy':<8} {'R_max':>7} {'r_gas0':>7} {'alpha':>6} {'M_stars':>8} {'M_gas':>8} {'M_dm':>8}"
                f" {'M_total':>8} {'M_dark':>8} {'new_tot':>8} {'new/old':>8} {'dark<r0':>8} {'n(Rmax)':>9}"
                + "".join(f" {f'n/o@{r}':>8}" for r in COMPARE_KPC) + f" {'R_data':>7} {'n/o@data':>8}\n")
        for gal, res in results.items():
            d, p, g = res["data"], res["prof"], res["grid"]
            i = -1
            inner_frac = np.interp(p["r0"], p["r"], p["m_dark"]) / p["m_dark"][-1]
            f.write(f"# {gal:<8} {d['r'][i]:7.1f} {p['r0']:7.2f} {p['alpha']:6.2f} {d['stars'][i] / 1e12:8.3f}"
                    f" {g['m_gas'][i] / 1e12:8.3f} {d['dm'][i] / 1e12:8.3f} {d['total'][i] / 1e12:8.3f}"
                    f" {g['m_dark'][i] / 1e12:8.3f} {g['new_total'][i] / 1e12:8.3f}"
                    f" {g['new_total'][i] / d['total'][i]:8.3f} {inner_frac:8.3f} {p['n_cc'][-1]:9.2e}"
                    + "".join(f" {ratio_at(res, rk):8.3f}" for rk in COMPARE_KPC)
                    + f" {d['data_end_kpc']:7.1f} {ratio_at(res, d['data_end_kpc']):8.3f}\n")
        f.write("#   r_gas0, R_max in kpc. alpha: dln M_gas/dln r at r_gas0. dark<r0: fraction of M_dark(R_max) that is\n")
        f.write("#   inside r_gas0, from the gas carried inward. n(Rmax): gas particles per cc at R_max.\n")
        f.write("#   n/o@r: new total over the Humphrey total at r kpc. R_data: where the X-ray data end, the outer edge\n")
        f.write("#   of the outermost data point's radial bin, in kpc. n/o@data: new total over the Humphrey total there\n#\n")
        f.write(f"# {'galaxy':<8} {'R_kpc':>9} {'n_gas_cc':>10} {'rho_dark':>10} {'M_total':>11} {'M_stars':>11}"
                f" {'M_dm':>11} {'M_gas':>11} {'M_dark':>11} {'M_new_tot':>11}\n")
        f.write("#   n_gas_cc: gas particles per cc. rho_dark: m_p per cc. M_total, M_stars, M_dm as digitized,\n")
        f.write("#   M_gas from the smoothed (and inside r_gas0, carried inward) gas line. Masses in M_sun\n")
        for gal, res in results.items():
            d, g = res["data"], res["grid"]
            for k in range(len(d["r"])):
                vals = [d["total"][k], d["stars"][k], d["dm"][k], g["m_gas"][k], g["m_dark"][k], g["new_total"][k]]
                f.write(f"{gal:<10} {d['r'][k]:9.4g} {g['n_cc'][k]:10.3e} {g['rho_dark_mp_cc'][k]:10.3e} "
                        + " ".join(f"{v:11.4e}" for v in vals) + "\n")
    print(f"Wrote {results_file}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--power", type=float, default=POWER_W, help="P in W, default %(default)s")
    parser.add_argument("--cutoff", type=float, default=DM_CUTOFF_DENSITY, metavar="N_C",
                        help="n_c of the exp(-(n_c/n)^2) low density cutoff, per cc, default %(default)s")
    parser.add_argument("--no-cutoff", action="store_true", help="no low density cutoff")
    parser.add_argument("--fig6-gas-only", action="store_true",
                        help="ignore gas_density_measured.txt, and use only the Fig. 6 gas line")
    args = parser.parse_args()
    n_cut = 0.0 if args.no_cutoff else args.cutoff

    data = load_digitized()
    measured = {} if args.fig6_gas_only else load_measured()
    results = {}
    for gal in ORDER:
        prof = dark_mass_profile(data[gal], args.power, n_cut, measured.get(gal))
        results[gal] = dict(data=data[gal], prof=prof, grid=on_data_grid(data[gal], prof))

    suffix = (("" if args.power == POWER_W else f"_P{args.power:g}")
              + ("" if n_cut == DM_CUTOFF_DENSITY else "_nocutoff" if n_cut == 0 else f"_nc{n_cut:g}")
              + ("_fig6gas" if args.fig6_gas_only else ""))
    plot_dir = HERE / f"plots{suffix}"
    plot_dir.mkdir(exist_ok=True)
    tag = f"P = {args.power:g} W, " + (f"cutoff n_c = {n_cut:g}/cc" if n_cut > 0 else "no cutoff")
    for gal, res in results.items():
        fig, ax = plt.subplots(figsize=(7, 5.6))
        draw_panel(ax, gal, res["data"], res["prof"], res["grid"], args.power)
        ax.set_title(f"{gal}: Humphrey et al. 2006 Fig. 6 lines, plus dark mass\n{tag}", fontsize=10)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{gal}.png", dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(17, 15))
    slots = [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2], axes[2, 1]]
    for ax in axes.flat:
        ax.set_visible(False)
    for i, (ax, (gal, res)) in enumerate(zip(slots, results.items())):
        ax.set_visible(True)
        draw_panel(ax, gal, res["data"], res["prof"], res["grid"], args.power, legend=(i == 0), small=True)
    fig.suptitle(f"Mass profiles of Humphrey et al. 2006 Fig. 6, with dark mass from the hot gas ({tag})", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(plot_dir / "all_galaxies.png", dpi=110)
    plt.close(fig)
    print(f"Wrote {len(results)} plots and all_galaxies.png to {plot_dir}")

    write_results(results, args.power, n_cut, HERE / f"dark_mass_results{suffix}.txt")
    print(f"\nnew/old: new total over the Humphrey total. spline: rms of the gas spline, dex")
    print(f"{'galaxy':<8} {'r_gas0':>7} {'alpha':>6} {'spline':>7} {'M_dark(Rmax)':>13} {'dark<r_gas0':>12}"
          + "".join(f" {f'new/old@{r}':>11}" for r in COMPARE_KPC) + f" {'R_data':>7} {'new/old@data':>13}"
          + f" {'new/old@Rmax':>13}")
    for gal, res in results.items():
        p, g, d = res["prof"], res["grid"], res["data"]
        inner_frac = np.interp(p["r0"], p["r"], p["m_dark"]) / p["m_dark"][-1]
        print(f"{gal:<8} {p['r0']:7.2f} {p['alpha']:6.2f} {p['spline_rms']:7.4f} {g['m_dark'][-1]:13.3e} {inner_frac:12.3f}"
              + "".join(f" {ratio_at(res, rk):11.2f}" for rk in COMPARE_KPC)
              + f" {d['data_end_kpc']:7.0f} {ratio_at(res, d['data_end_kpc']):13.2f} {g['new_total'][-1] / d['total'][-1]:13.2f}")


if __name__ == "__main__":
    main()
