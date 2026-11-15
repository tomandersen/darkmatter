"""Digitize the four model lines of Fig. 6 of Humphrey et al. 2006 (ApJ 646, 899), "A Chandra View of Dark Matter in
Early-Type Galaxies".

Each panel shows the enclosed mass M(<R) of one galaxy, from their best-fit NFW+stars model:
  total (red solid), stars (magenta dotted), NFW dark matter (blue dashed), gas (orange dash-dot-dot).
The data points (black diamonds) are ignored.

Each line is found by its colour, column by column, and the pixel positions are turned into (R, M) with the panel's own
axis calibration, fitted to its major tick marks. The lines are then put on a common log R grid:
  - gaps in the dashed and dotted lines are filled by interpolating in log-log
  - where the red total line hides the stars or the DM (they overlap it), or the blue dashes hide the end of the red line
    in the top right corner, the hidden one is filled from total = stars + DM, which holds for these models, as the
    paper ignores the gas in the gravitating mass
  - lines are carried straight in log-log the last few pixels to the frame, which EDGE and the frame corner hide
  - where a line is off the plot (the gas line below the bottom axis, at small R) it is left as nan

Also finds where the X-ray data end in each panel: the outer edge of the outermost data point's radial bin.

Writes humphrey2006_fig6_digitized.txt, and for checking, digitized_overlay.png, the digitized lines drawn over the figure.

  ../.venv/bin/python digitize_fig6.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
FIGURE = HERE / "humphrey2006_fig6.png"
DATA_FILE = HERE / "humphrey2006_fig6_digitized.txt"
OVERLAY = HERE / "digitized_overlay.png"

# Panel frames in figure pixels: (left, top, right, bottom), the inside edges of the black box
# x_ticks: (pixel, log10 R) and y_ticks: (pixel, log10 M) for the major tick marks, read off the figure
PANELS = {
    "NGC4125": dict(frame=(275, 133, 747, 498), x_ticks=[(322.5, 0), (482.0, 1), (641.5, 2)],
                    y_ticks=[(215.0, 12), (320.0, 11), (425.0, 10)]),
    "NGC720": dict(frame=(808, 133, 1280, 498), x_ticks=[(855.5, 0), (1014.0, 1), (1173.0, 2)],
                   y_ticks=[(217.5, 12), (321.5, 11), (425.5, 10)]),
    "NGC6482": dict(frame=(1341, 133, 1813, 498), x_ticks=[(1388.5, 0), (1546.5, 1), (1705.0, 2)],
                    y_ticks=[(220.0, 12), (323.0, 11), (426.5, 10)]),
    "NGC1407": dict(frame=(275, 572, 747, 937), x_ticks=[(320.5, 0), (473.0, 1), (625.5, 2)],
                    y_ticks=[(589.5, 13), (683.5, 12), (777.5, 11), (871.5, 10)]),
    "NGC4472": dict(frame=(808, 572, 1280, 937), x_ticks=[(852.0, 0), (999.5, 1), (1147.0, 2)],
                    y_ticks=[(615.5, 13), (702.0, 12), (789.5, 11), (876.0, 10)]),
    "NGC4649": dict(frame=(1341, 572, 1813, 937), x_ticks=[(1385.0, 0), (1532.0, 1), (1679.0, 2)],
                    y_ticks=[(614.5, 13), (695.0, 12), (775.5, 11), (856.5, 10)]),
    "NGC4261": dict(frame=(808, 1010, 1265, 1376), x_ticks=[(920.5, 0), (1035.5, 1), (1150.0, 2)],
                    y_ticks=[(1067.0, 13), (1137.5, 12), (1207.5, 11), (1277.5, 10), (1348.0, 9)]),
}

# Hue ranges in degrees for each line. Pixels between these (red/magenta and blue/magenta blends) are not used
# min_sat: saturation below which a pixel is background. The magenta dots are faint, about 0.2 at their centres
LINES = {
    "total": dict(hues=[(345, 360), (0, 12)], min_sat=0.3),
    "stars": dict(hues=[(280, 335)], min_sat=0.08),
    "dm": dict(hues=[(215, 265)], min_sat=0.3),
    "gas": dict(hues=[(15, 55)], min_sat=0.15),
}
MIN_VALUE = 0.55    # darker pixels are where a black error bar or frame line crosses a line, and not used
EDGE = 3            # pixels inside the frame to skip, so the black frame and tick marks are not picked up
MAX_GAP_DEX = 0.35  # gaps in log R wider than this, between digitized points, are not interpolated across
N_GRID = 120        # points on the common log R grid
END_EXTRAP_DEX = 0.1  # how far in log R a line may be carried past its last digitized point, to reach the frame
END_FIT_DEX = 0.15    # the slope for that is fitted to the digitized points this far in log R from the end
DATA_BAND = 30        # rows either side of the red line searched for the black data point diamonds and bars
DATA_MARGIN = 12      # pixels inside the frame skipped there, so the tick marks are not taken for data
TRACK_TOL = 3.0     # pixels, how far a clump can be from the predicted row of the line being tracked
TRACK_SPAN = 25     # columns, the prediction is a straight line through the accepted points this far back


def axis_fit(ticks):
    """Linear fit of log10 value against pixel, from the major tick marks. Returns (slope, intercept)."""
    pix, val = np.array(ticks, dtype=float).T
    return np.polyfit(pix, val, 1)


def column_clumps(hsv, frame, hues, min_sat):
    """Per column inside the frame, the clumps of pixels in the hue ranges, as {column: [(row, weight), ...]}.

    A clump is a run of nearly touching rows. Its row is the saturation weighted mean, in figure pixels.
    """
    x0, y0, x1, y1 = frame
    clumps = {}
    for x in range(x0 + EDGE, x1 - EDGE + 1):
        col = hsv[y0 + EDGE:y1 - EDGE + 1, x]
        h = col[:, 0] * 360.0 / 255.0
        s = col[:, 1] / 255.0
        v = col[:, 2] / 255.0
        hit = np.zeros(len(col), bool)
        for lo, hi in hues:
            hit |= (h >= lo) & (h < hi)
        hit &= (s > min_sat) & (v > MIN_VALUE)
        rows = np.nonzero(hit)[0]
        if len(rows):
            clumps[x] = [(y0 + EDGE + np.average(c, weights=s[c]), s[c].sum())
                         for c in np.split(rows, np.nonzero(np.diff(rows) > 3)[0] + 1)]
    return clumps


def track_one_way(clumps, columns, seed_row, history=((), ())):
    """Follow one line through the columns in the order given, starting near seed_row in the first column.

    The next row is predicted from a straight line through the accepted points of the last TRACK_SPAN columns, and the
    nearest clump is taken if it is close enough. Columns with nothing close are skipped, so a crossing line or an
    overlap blend is not picked up. history: points already tracked the other way from the start, ending at the start,
    so the first prediction has a slope. Returns the new points only.
    """
    xs, ys = list(history[0]), list(history[1])
    n_history = len(xs)
    for x in columns:
        if x not in clumps:
            continue
        if not xs:
            predict, tol = seed_row, TRACK_TOL
        else:
            near = np.abs(np.array(xs) - xs[-1]) <= TRACK_SPAN
            recent_x, recent_y = np.array(xs)[near], np.array(ys)[near]
            if len(recent_x) >= 2:
                predict = np.polyval(np.polyfit(recent_x, recent_y, 1), x)
            else:
                predict = recent_y[-1]
            tol = TRACK_TOL + 0.15 * abs(x - xs[-1])
        row = min((r for r, _ in clumps[x]), key=lambda r: abs(r - predict))
        if abs(row - predict) < tol:
            xs.append(x)
            ys.append(row)
    return xs[n_history:], ys[n_history:]


def track(clumps, seed_x, seed_row):
    """Follow one line both ways from a clump on it. Returns arrays of column and row, by column."""
    columns = sorted(clumps)
    i = columns.index(seed_x)
    # Go right, then left with the slope from that, then right again with the slope from the left
    right = track_one_way(clumps, columns[i:], seed_row)
    left = track_one_way(clumps, columns[i::-1], seed_row, (right[0][::-1], right[1][::-1]))
    right = track_one_way(clumps, columns[i:], seed_row, (left[0][::-1], left[1][::-1]))
    xs, ys = left[0][::-1] + right[0][1:], left[1][::-1] + right[1][1:]
    return np.array(xs, float), np.array(ys, float)


def heaviest(clumps, keep=lambda x, row: True):
    """(column, row) of the heaviest clump that passes keep. On a line that is alone there, so a place to start tracking."""
    return max(((x, row, w) for x, cs in clumps.items() for row, w in cs if keep(x, row)), key=lambda c: c[2])[:2]


def line_pixels(hsv, frame):
    """Track each of the four lines. Returns {line: (columns, rows)}."""
    clumps = {name: column_clumps(hsv, frame, line["hues"], line["min_sat"]) for name, line in LINES.items()}
    out = {name: track(clumps[name], *heaviest(clumps[name])) for name in ("total", "dm", "gas")}
    # The magenta dots are faint, and the red total over the blue DM blends to magenta. Start the stars on a dot that is
    # well away from both
    (tx, ty), (dx, dy) = out["total"], out["dm"]
    away = lambda x, row: (row > np.interp(x, tx, ty) + 8 and
                           (x < dx[0] or x > dx[-1] or abs(row - np.interp(x, dx, dy)) > 8))
    out["stars"] = track(clumps["stars"], *heaviest(clumps["stars"], away))
    return out


def running_median(y, half_width=2):
    out = np.empty_like(y)
    for i in range(len(y)):
        out[i] = np.median(y[max(0, i - half_width):i + half_width + 1])
    return out


def on_grid(log_r, log_m, grid):
    """Interpolate the digitized points onto the grid, leaving nan outside them and across wide gaps."""
    out = np.interp(grid, log_r, log_m, left=np.nan, right=np.nan)
    gap_starts = np.nonzero(np.diff(log_r) > MAX_GAP_DEX)[0]
    for i in gap_starts:
        out[(grid > log_r[i]) & (grid < log_r[i + 1])] = np.nan
    return out


def extend_ends(g, log_r, log_m, grid, log_m_range):
    """Carry a line the last bit to the frame, where EDGE and the frame corner hide it.

    Straight in log-log, with the slope of the last END_FIT_DEX of digitized points, by at most END_EXTRAP_DEX, and
    only while it stays on the plot, so a line that leaves through the bottom axis is not carried below it.
    """
    g = g.copy()
    for end in (0, -1):
        side = np.abs(log_r - log_r[end]) <= END_FIT_DEX
        slope, icept = np.polyfit(log_r[side], log_m[side], 1)
        beyond = (grid < log_r[0]) if end == 0 else (grid > log_r[-1])
        beyond &= np.abs(grid - log_r[end]) <= END_EXTRAP_DEX
        ext = slope * grid + icept
        beyond &= (ext >= log_m_range[0]) & (ext <= log_m_range[1] + 0.05)
        g[beyond] = ext[beyond]
    return g


def data_end(hsv, frame, total_px, total_py):
    """Column of the outer edge of the outermost X-ray data point, from its black error diamond or bar.

    The data points sit on or near the red total line, so this looks for black (dark, unsaturated) pixels within
    DATA_BAND rows of it, away from the frame and its tick marks. The red line itself is dark too, but saturated.
    """
    x0, y0, x1, y1 = frame
    sat, val = hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    black = (sat < 0.3) & (val < 0.6)
    last = None
    for x in range(x0 + DATA_MARGIN, x1 - DATA_MARGIN + 1):
        row = np.interp(x, total_px, total_py)
        lo, hi = int(max(y0 + DATA_MARGIN, row - DATA_BAND)), int(min(y1 - DATA_MARGIN, row + DATA_BAND))
        if black[lo:hi + 1, x].any():
            last = x
    return last


def digitize_panel(hsv, panel):
    frame = panel["frame"]
    ax, bx = axis_fit(panel["x_ticks"])
    ay, by = axis_fit(panel["y_ticks"])
    x0, y0, x1, y1 = frame
    log_r_range = (ax * x0 + bx, ax * x1 + bx)
    log_m_range = (ay * y1 + by, ay * y0 + by)
    grid = np.linspace(log_r_range[0], log_r_range[1], N_GRID)

    raw, gridded = {}, {}
    pixels = line_pixels(hsv, frame)
    data_end_px = data_end(hsv, frame, *pixels["total"])
    for name, (px, py) in pixels.items():
        log_r = ax * px + bx
        log_m = running_median(ay * py + by)
        raw[name] = (log_r, log_m)
        gridded[name] = extend_ends(on_grid(log_r, log_m, grid), log_r, log_m, grid, log_m_range)

    # Lines hidden under the red total: total = stars + DM in these models (the paper leaves the gas out of the total),
    # so fill the hidden one from the other two. The DM hides under the total at large R, the stars at small R, and in
    # the top right corner the blue dashes can hide the end of the red line
    total, stars, dm = (10 ** gridded[k] for k in ("total", "stars", "dm"))
    total_r, dm_r, stars_r = raw["total"][0], raw["dm"][0], raw["stars"][0]
    fill_total = np.isnan(total) & ~np.isnan(stars) & ~np.isnan(dm) & (grid > total_r[-1])
    fill_dm = np.isnan(dm) & ~np.isnan(total) & ~np.isnan(stars) & (grid > dm_r[-1]) & (total > stars)
    fill_stars = np.isnan(stars) & ~np.isnan(total) & (grid < stars_r[0])
    gridded["total"][fill_total] = np.log10(stars[fill_total] + dm[fill_total])
    gridded["dm"][fill_dm] = np.log10(total[fill_dm] - stars[fill_dm])
    gridded["stars"][fill_stars] = np.log10(total[fill_stars] - np.nan_to_num(dm[fill_stars]))
    filled = {"total": fill_total, "dm": fill_dm, "stars": fill_stars}
    return dict(grid=grid, raw=raw, lines=gridded, filled=filled, log_r_range=log_r_range, log_m_range=log_m_range,
                calib=(ax, bx, ay, by), data_end_px=data_end_px, data_end_kpc=10 ** (ax * data_end_px + bx))


def write_data(results):
    with open(DATA_FILE, "w") as f:
        f.write("# Digitized model lines of Fig. 6, Humphrey et al. 2006, ApJ 646, 899, 'A Chandra View of Dark Matter in\n")
        f.write("# Early-Type Galaxies'. Best-fit NFW+stars mass models, extrapolated by the authors out to R_vir.\n")
        f.write("# Made by digitize_fig6.py from humphrey2006_fig6.png. Enclosed mass M(<R) in solar masses, R in kpc.\n")
        f.write("#   M_total: total gravitating mass, red solid line (= stars + DM, the paper leaves the gas out of it)\n")
        f.write("#   M_stars: stellar mass, magenta dotted line (Hernquist)\n")
        f.write("#   M_dm:    NFW dark matter, blue dashed line\n")
        f.write("#   M_gas:   hot gas mass, orange dash-dot-dot line\n")
        f.write("# nan: the line is off the plot there (mostly the gas, below the bottom axis at small R).\n")
        f.write("# Where one of total, stars, DM is hidden under another line, it is filled from total = stars + DM. Lines are carried\n")
        f.write(f"# straight in log-log by up to {END_EXTRAP_DEX} dex past their last digitized point, to the frame.\n")
        f.write("# Panel axis ranges (log10 R_kpc, log10 M_sun), from the fitted tick marks:\n")
        for gal, res in results.items():
            f.write(f"#   {gal:<8} log R {res['log_r_range'][0]:6.3f} to {res['log_r_range'][1]:6.3f}"
                    f"   log M {res['log_m_range'][0]:6.3f} to {res['log_m_range'][1]:6.3f}\n")
        f.write("# Where the X-ray data end: the outer edge of the radial bin of the outermost data point (its error\n")
        f.write("# diamond or bar), in kpc:\n")
        for gal, res in results.items():
            f.write(f"#   {gal:<8} data_end_kpc {res['data_end_kpc']:7.1f}\n")
        f.write(f"#\n# {'galaxy':<8} {'R_kpc':>10} {'M_total':>11} {'M_stars':>11} {'M_dm':>11} {'M_gas':>11}\n")
        for gal, res in results.items():
            for i, lr in enumerate(res["grid"]):
                vals = [10 ** res["lines"][k][i] for k in ("total", "stars", "dm", "gas")]
                f.write(f"{gal:<10} {10 ** lr:10.4g} " + " ".join(f"{v:11.4e}" for v in vals) + "\n")
    print(f"Wrote {DATA_FILE}")


def write_overlay(image, results):
    """The digitized lines drawn over the figure, each panel in its own pixel frame, to check by eye."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (gal, res) in zip([axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2], axes[2, 1]],
                              results.items()):
        x0, y0, x1, y1 = PANELS[gal]["frame"]
        a_x, b_x, a_y, b_y = res["calib"]
        ax.imshow(np.asarray(image)[y0 - 5:y1 + 6, x0 - 5:x1 + 6], extent=(x0 - 5.5, x1 + 5.5, y1 + 5.5, y0 - 5.5))
        for name, line in LINES.items():
            g = res["lines"][name]
            ax.plot((res["grid"] - b_x) / a_x, (g - b_y) / a_y, color="black", lw=0.8, ls="-")
            raw_r, raw_m = res["raw"][name]
            ax.plot((raw_r - b_x) / a_x, (raw_m - b_y) / a_y, ".", ms=1, color="cyan")
            if name in res["filled"]:
                f = res["filled"][name]
                ax.plot(((res["grid"] - b_x) / a_x)[f], ((g - b_y) / a_y)[f], "o", ms=2.5, mfc="none", mec="green")
        ax.axvline(res["data_end_px"], color="green", lw=0.8)
        ax.set_title(gal + "   cyan: tracked pixels, black: on the grid, green: filled from total = stars + DM,"
                     f" green line: data end, {res['data_end_kpc']:.0f} kpc", fontsize=7)
        ax.axis("on")
    fig.tight_layout()
    fig.savefig(OVERLAY, dpi=110)
    print(f"Wrote {OVERLAY}")


def main():
    image = Image.open(FIGURE).convert("RGB")
    hsv = np.asarray(image.convert("HSV")).astype(float)
    results = {gal: digitize_panel(hsv, panel) for gal, panel in PANELS.items()}
    for gal, res in results.items():
        t, s, d = (10 ** res["lines"][k] for k in ("total", "stars", "dm"))
        both = ~np.isnan(t) & ~np.isnan(s) & ~np.isnan(d) & ~np.any([f for f in res["filled"].values()], axis=0)
        resid = np.log10((s + d)[both] / t[both])
        print(f"{gal:<8} log(stars+DM) - log(total) where all three were digitized: median {np.median(resid):+.3f},"
              f" max |.| {np.max(np.abs(resid)):.3f} dex")
    write_data(results)
    write_overlay(image, results)


if __name__ == "__main__":
    main()
