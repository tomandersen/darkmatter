"""Digitize the measured hot gas density of NGC 4649, Fig. 5 of Humphrey et al. 2008, ApJ 683, 161, "Weighing the
Quiescent Central Black Hole in an Elliptical Galaxy with X-ray Emitting Gas".

Same Chandra galaxy and distance (15.6 Mpc) as the 2006 paper, with deeper data reaching in to 0.15 kpc. The 15 data
points are the triangles at the centres of the error diamonds. Each is found by eye to a few pixels (APPROX), then
placed exactly with a local fit of a triangle template cut from the figure. The black and blue step lines (their
models) are ignored.

Writes gas_density_measured.txt, which dark_mass_ellipticals.py uses for the gas density where it has data, and
digitized_ngc4649_2008_check.png, the found points marked on the figure.

  ../.venv/bin/python digitize_ngc4649_2008.py
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
FIGURE = HERE / "humphrey2008_ngc4649_gas_density.png"
OUT = HERE / "gas_density_measured.txt"
CHECK = HERE / "digitized_ngc4649_2008_check.png"

# Major tick marks, (pixel, log10 value): R = 1 and 10 kpc, rho = 1 and 0.1 (1e-25 g/cc)
X_TICKS = [(364.5, 0.0), (620.5, 1.0)]
Y_TICKS = [(228.5, 0.0), (396.5, -1.0)]
# One clean triangle, its box in figure pixels (x0, y0, x1, y1) and its centre in that box
TEMPLATE_BOX = (239, 132, 250, 140)
TEMPLATE_CENTRE = (5.0, 3.5)
SEARCH = 4  # pixels either way around each APPROX point
APPROX = [(154.5, 97.5), (244, 136), (292, 154), (329, 173), (358, 195), (386, 224), (417, 261), (451, 293),
          (485.5, 324), (516, 346), (544, 377), (571, 392), (599, 431), (634, 435), (680, 506)]


def calib(ticks):
    pix, val = np.array(ticks, float).T
    return np.polyfit(pix, val, 1)


def main():
    image = Image.open(FIGURE).convert("RGB")
    dark = (255 - np.asarray(image.convert("L")).astype(float)) / 255
    x0, y0, x1, y1 = TEMPLATE_BOX
    tpl = dark[y0:y1, x0:x1]
    tpl = tpl - tpl.mean()
    h, w = tpl.shape
    cx, cy = TEMPLATE_CENTRE
    ax, bx = calib(X_TICKS)
    ay, by = calib(Y_TICKS)

    points = []
    for px, py in APPROX:
        best = max(((tpl * dark[y:y + h, x:x + w]).sum(), x + cx, y + cy)
                   for y in range(int(round(py - cy)) - SEARCH, int(round(py - cy)) + SEARCH + 1)
                   for x in range(int(round(px - cx)) - SEARCH, int(round(px - cx)) + SEARCH + 1))
        points.append(best[1:])

    with open(OUT, "w") as f:
        f.write("# Measured hot gas density, from published deprojected profiles. rho_gas in 1e-25 g/cc, R in kpc.\n")
        f.write("# dark_mass_ellipticals.py uses these instead of the Fig. 6 gas line where they have data.\n")
        f.write("# NGC4649: Humphrey et al. 2008, ApJ 683, 161, Fig. 5 data points (triangles), by digitize_ngc4649_2008.py.\n")
        f.write("#   Distance 15.6 Mpc, the same as Humphrey et al. 2006.\n")
        f.write(f"# {'galaxy':<8} {'R_kpc':>8} {'rho_gas':>9}\n")
        for x, y in points:
            f.write(f"{'NGC4649':<10} {10 ** (ax * x + bx):8.4f} {10 ** (ay * y + by):9.4f}\n")
    print(f"Wrote {OUT}")

    big = image.resize((image.width * 3, image.height * 3), Image.NEAREST)
    draw = ImageDraw.Draw(big)
    for x, y in points:
        draw.ellipse([3 * x - 3, 3 * y - 3, 3 * x + 3, 3 * y + 3], fill=(255, 0, 0))
    big.save(CHECK)
    print(f"Wrote {CHECK}")


if __name__ == "__main__":
    main()
