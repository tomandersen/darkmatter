"""Reads the SPARC tables (Lelli 2016) and picks galaxies. Everything that needs the galaxy data gets it from here.

Quality cuts belong in select_galaxies, which gives a list of names to pass to the model as only_galaxies. The model runs exactly
the galaxies it is given, so a loop over single galaxies and a global fit can't disagree about which ones are in.
"""
from pathlib import Path

import numpy as np

SPARC_DIR = Path(__file__).resolve().parent
PROPERTIES_FILE = SPARC_DIR / "SPARC_Lelli2016c.mrt.txt"
KINEMATICS_FILE = SPARC_DIR / "MassModels_Lelli2016_SigmaGas.mrt"  # the Lelli mass models plus a Sigma_gas column, from calc_gas_checkit.py

LELLI_UPSILON_DISK = 0.5
LELLI_UPSILON_BULGE = 0.7
LELLI_HE_FACTOR = 1.33
G_KPC = 4.30091e-6  # kpc (km/s)^2 / Msun

GRAPH_DPI = 300  # every saved graph


def load_properties():
    """Returns {galaxy: {'D' Mpc, 'Inc' deg, 'L36' 10^9 Lsun, 'Rd' kpc, 'MHI' 10^9 Msun, 'Vflat' km/s, 'Q' quality flag}}, in file order."""
    properties = {}
    with open(PROPERTIES_FILE) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 18 and parts[1].lstrip('-').isdigit():
                properties[parts[0]] = {'D': float(parts[2]), 'Inc': float(parts[5]), 'L36': float(parts[7]), 'Rd': float(parts[11]),
                                        'MHI': float(parts[13]), 'Vflat': float(parts[15]), 'Q': int(parts[17])}
    return properties


def load_rotation_curves(only_galaxies=None):
    """Returns {galaxy: {'R', 'Vobs', 'eVobs', 'Vgas', 'Vdisk', 'Vbul', 'Sigma_gas': arrays}}. Vdisk and Vbul are at Upsilon = 1, eVobs <= 0 is set to 1.

    only_galaxies: list of names to return, in that order. Default: all, in file order. A name not in the file is an error.
    """
    columns = ('R', 'Vobs', 'eVobs', 'Vgas', 'Vdisk', 'Vbul')
    curves = {}
    with open(KINEMATICS_FILE) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 11 or line.startswith('---') or line.startswith('Byte'):
                continue
            try:
                values = [float(p) for p in parts[2:8]] + [float(parts[10])]
            except ValueError:
                continue
            values[2] = values[2] if values[2] > 0 else 1.0
            g = curves.setdefault(parts[0], {k: [] for k in columns + ('Sigma_gas',)})
            for k, v in zip(columns + ('Sigma_gas',), values):
                g[k].append(v)

    if only_galaxies is not None:
        unknown = [name for name in only_galaxies if name not in curves]
        if unknown:
            raise ValueError(f"Not in {KINEMATICS_FILE.name}: {unknown}")
        curves = {name: curves[name] for name in only_galaxies}
    return {name: {k: np.array(v) for k, v in g.items()} for name, g in curves.items()}


def lelli_baryonic_mass(properties, curve):
    """Total baryonic mass in 10^9 Msun: 0.5 L_disk + 0.7 L_bulge + 1.33 HI. Takes one galaxy's load_properties entry and its curve, with Vbul still at Upsilon = 1."""
    # The bulge is spherical in Lelli and Vbul is for M/L = 1, so V^2 R / G at the last radius is its luminosity
    bulge_luminosity = min(float(curve['Vbul'][-1]) ** 2 * float(curve['R'][-1]) / G_KPC / 1e9, properties['L36'])
    disk_luminosity = properties['L36'] - bulge_luminosity
    return LELLI_UPSILON_DISK * disk_luminosity + LELLI_UPSILON_BULGE * bulge_luminosity + LELLI_HE_FACTOR * properties['MHI']


def select_galaxies(max_q=None, min_inclination=None):
    """Names of the galaxies that pass the cuts, in file order. None means no cut. Q is 1 (high), 2 (medium) or 3 (low)."""
    return [name for name, p in load_properties().items()
            if (max_q is None or p['Q'] <= max_q) and (min_inclination is None or p['Inc'] >= min_inclination)]
