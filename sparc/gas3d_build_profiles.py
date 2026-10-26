# Build measured gas surface density profiles (HI + H2) for the 3D gas test.
#
# Reads THINGS HI moment-0 maps and HERACLES CO(2-1) moment-0 maps (downloaded into data_3dgas/raw),
# deprojects them with the THINGS tilted-ring geometry (de Blok et al. 2008, Table 2) and writes
# face-on ring-averaged profiles to sparc/gas3d/<GAL>_profiles.csv in the same units as the
# Sigma_gas column of MassModels_Lelli2016_SigmaGas.mrt (hydrogen particles / cm^2, no helium).
#
# Run from the repo root, isolated (the FITS files are downloaded data):
#   .venv/bin/python -I sparc/gas3d_build_profiles.py
import os
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW_DIR = "./data_3dgas/raw"
OUT_DIR = "./sparc/gas3d"

# CO -> H2. Bolatto, Wolfire & Leroy 2013 (ARA&A 51, 207): alpha_CO(1-0) = 4.35 Msun/pc^2/(K km/s) including helium,
# uncertain by ~0.3 dex (factor 2) and known to be lower in some galaxy centres (Sandstrom et al. 2013 find
# ~5-10x lower in the centre of NGC 6946). CO(2-1)/CO(1-0) line ratio R21 = 0.7 (den Brok et al. 2021 find 0.65 +/- 0.2).
ALPHA_CO_10 = 4.35
R21 = 0.7
HE_FACTOR_H2 = 1.36  # alpha_CO includes helium; strip it so we count hydrogen only, like the SPARC Sigma_gas column

MSUN_PC2_TO_H_CM2 = 1.98847e33 / (3.0857e18**2) / 1.6726e-24   # 1 Msun/pc^2 of pure hydrogen = 1.249e20 H/cm^2

# Galaxy geometry. Centre, inclination and PA from de Blok et al. 2008 (AJ 136, 2648) Table 2 (THINGS);
# distance from SPARC (Lelli et al. 2016) so radii line up with the SPARC rotation curve.
# beam (arcsec) for the THINGS natural-weighted maps from the AIPS CLEAN BMAJ/BMIN header history.
GALAXIES = {
    "NGC2403": dict(ra="07h36m51.1s", dec="+65d36m02.9s", inc=62.9, pa=123.7, D_mpc=3.16,
                    hi="NGC_2403_NA_MOM0_THINGS.FITS", co="ngc2403_heracles_mom0.fits.gz", r_max_kpc=22.0),
    "NGC6946": dict(ra="20h34m52.2s", dec="+60d09m14.4s", inc=32.6, pa=242.7, D_mpc=5.52,
                    hi="NGC_6946_NA_MOM0_THINGS.FITS", co="ngc6946_heracles_mom0.fits.gz", r_max_kpc=22.0),
}

RING_WIDTH_KPC = 0.25
N_SECTORS = 36          # azimuthal coverage is measured as the fraction of 10 degree sectors that have pixels
MIN_COVERAGE = 0.6      # rings with less azimuthal coverage than this are flagged (the model then falls back to SPARC Sigma_gas)


def beam_from_history(header):
    bmaj = bmin = None
    for card in header.cards:
        s = str(card)
        if "CLEAN BMAJ=" in s:
            parts = s.split()
            bmaj = float(parts[parts.index("BMAJ=") + 1]) * 3600.0
            bmin = float(parts[parts.index("BMIN=") + 1]) * 3600.0
    return bmaj, bmin


def galaxy_plane_coords(wcs2d, shape, cfg):
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    c0 = SkyCoord(cfg["ra"], cfg["dec"], frame="fk5")
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    ra, dec = wcs2d.all_pix2world(xx, yy, 0)
    # small-angle offsets in arcsec, x positive to the east, y positive to the north
    dx = (ra - c0.ra.deg) * np.cos(np.deg2rad(c0.dec.deg)) * 3600.0
    dy = (dec - c0.dec.deg) * 3600.0
    pa = np.deg2rad(cfg["pa"])
    x_maj = dx * np.sin(pa) + dy * np.cos(pa)
    y_min = (-dx * np.cos(pa) + dy * np.sin(pa)) / np.cos(np.deg2rad(cfg["inc"]))
    arcsec_to_kpc = cfg["D_mpc"] * 1e3 * np.pi / (180.0 * 3600.0)
    R_kpc = np.hypot(x_maj, y_min) * arcsec_to_kpc
    phi = np.arctan2(y_min, x_maj)
    return R_kpc, phi, ra, dec


def ring_profile(R_kpc, phi, values, r_edges):
    """Ring mean of values (NaN = no coverage), azimuthal coverage, and clumping <v^(2/3)>/<v>^(2/3)."""
    good = np.isfinite(values)
    mean = np.zeros(len(r_edges) - 1)
    cov = np.zeros(len(r_edges) - 1)
    clump = np.ones(len(r_edges) - 1)
    sector = ((phi + np.pi) / (2 * np.pi) * N_SECTORS).astype(int) % N_SECTORS
    for k in range(len(r_edges) - 1):
        in_ring = (R_kpc >= r_edges[k]) & (R_kpc < r_edges[k + 1])
        sel = in_ring & good
        if np.sum(sel) == 0:
            continue
        cov[k] = len(np.unique(sector[sel])) / N_SECTORS
        v = values[sel]
        mean[k] = np.mean(v)
        vp = np.clip(v, 0, None)
        if np.mean(vp) > 0:
            clump[k] = np.mean(vp ** (2.0 / 3.0)) / np.mean(vp) ** (2.0 / 3.0)
    return mean, cov, clump


def build(gal, cfg):
    cos_i = np.cos(np.deg2rad(cfg["inc"]))

    # ---- HI (THINGS) ----
    hdu = fits.open(os.path.join(RAW_DIR, cfg["hi"]))[0]
    hi = np.squeeze(hdu.data).astype(float)
    wcs_hi = WCS(hdu.header).celestial
    bmaj, bmin = beam_from_history(hdu.header)
    # Jy/beam * m/s -> K km/s -> N_HI (sky), then * cos(i) for face-on
    S_kms = hi / 1000.0
    T_kms = 6.07e5 * S_kms / (bmaj * bmin)  # Rayleigh-Jeans at 21cm, beam in arcsec
    N_HI_face = 1.823e18 * T_kms * cos_i
    R_hi, phi_hi, ra_hi, dec_hi = galaxy_plane_coords(wcs_hi, hi.shape, cfg)

    # ---- CO (HERACLES) -> H2 ----
    hdu_co = fits.open(os.path.join(RAW_DIR, cfg["co"]))[0]
    co = np.squeeze(hdu_co.data).astype(float)
    wcs_co = WCS(hdu_co.header).celestial
    Sigma_H2_msun = (ALPHA_CO_10 / R21) * co * cos_i / HE_FACTOR_H2  # hydrogen only, face-on
    N_H2_face = Sigma_H2_msun * MSUN_PC2_TO_H_CM2                    # H nuclei / cm^2 (2 per molecule)
    R_co, phi_co, _, _ = galaxy_plane_coords(wcs_co, co.shape, cfg)

    r_edges = np.arange(0.0, cfg["r_max_kpc"] + RING_WIDTH_KPC, RING_WIDTH_KPC)
    r_mid = 0.5 * (r_edges[1:] + r_edges[:-1])
    hi_mean, hi_cov, hi_clump = ring_profile(R_hi, phi_hi, N_HI_face, r_edges)
    h2_mean, h2_cov, h2_clump = ring_profile(R_co, phi_co, N_H2_face, r_edges)

    # combined HI + H2 clumping at the HI pixel scale (CO sampled at the nearest CO pixel)
    px, py = wcs_co.all_world2pix(ra_hi, dec_hi, 0)
    px = np.round(px).astype(int)
    py = np.round(py).astype(int)
    inside = (px >= 0) & (px < co.shape[1]) & (py >= 0) & (py < co.shape[0])
    h2_on_hi = np.zeros_like(N_HI_face)
    h2_on_hi[inside] = np.nan_to_num(N_H2_face[py[inside], px[inside]], nan=0.0)
    _, _, tot_clump = ring_profile(R_hi, phi_hi, N_HI_face + h2_on_hi, r_edges)

    h2_mean = np.where(h2_cov >= MIN_COVERAGE, h2_mean, 0.0)  # CO outside the HERACLES field taken as zero

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{gal}_profiles.csv")
    header = ("R_kpc,Sigma_HI_cm2,cov_HI,Sigma_H2_cm2,cov_H2,clump_HI,clump_H2,clump_HI_H2\n"
              f"# {gal}: face-on ring means, hydrogen particles/cm^2 (no helium). H2 counted as H nuclei. "
              f"alpha_CO(1-0)={ALPHA_CO_10} (incl He), R21={R21}, inc={cfg['inc']}, PA={cfg['pa']}, D={cfg['D_mpc']} Mpc, "
              f"THINGS beam {bmaj:.2f}x{bmin:.2f} arcsec")
    np.savetxt(out, np.column_stack([r_mid, hi_mean, hi_cov, h2_mean, h2_cov, hi_clump, h2_clump, tot_clump]),
               delimiter=",", header=header, comments="", fmt="%.5g")
    print(f"{gal}: wrote {out}  (beam {bmaj:.2f}x{bmin:.2f}\")")
    return r_mid, hi_mean, hi_cov, h2_mean


def read_sparc_sigma(gal):
    R, S = [], []
    with open("./sparc/MassModels_Lelli2016_SigmaGas.mrt") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 11 and parts[0] == gal:
                R.append(float(parts[2]))
                S.append(float(parts[10]))
    return np.array(R), np.array(S)


def read_leroy(gal):
    name = gal[:3] + " " + gal[3:].lstrip("0")
    R, HI, H2 = [], [], []
    with open(os.path.join(RAW_DIR, "leroy2008_table7.dat")) as f:
        for line in f:
            if line[0:8].strip() == name:
                R.append(float(line[9:13]))
                HI.append(float(line[19:24]))
                h2 = line[29:35].strip()
                H2.append(float(h2) if h2 else np.nan)
    return np.array(R), np.array(HI), np.array(H2)


if __name__ == "__main__":
    for gal, cfg in GALAXIES.items():
        r, hi, hi_cov, h2 = build(gal, cfg)
        R_s, S_s = read_sparc_sigma(gal)
        R_l, HI_l, H2_l = read_leroy(gal)
        to_msun = 1.36 / MSUN_PC2_TO_H_CM2  # plot in Msun/pc^2 including helium, like Leroy et al. 2008
        plt.figure(figsize=(8, 5))
        ok = hi_cov >= MIN_COVERAGE
        plt.plot(r[ok], hi[ok] * to_msun, 'g-', label='HI THINGS (this work)')
        plt.plot(r[~ok], hi[~ok] * to_msun, 'g:', label='HI THINGS, partial coverage')
        plt.plot(r, h2 * to_msun, 'r-', label=f'H2 HERACLES, alpha_CO={ALPHA_CO_10}, R21={R21}')
        plt.plot(R_s, S_s * to_msun, 'k--', label='SPARC Sigma_gas (from Vgas)')
        plt.plot(R_l, HI_l, 'g.', markersize=3, label='HI Leroy+2008')
        plt.plot(R_l, H2_l, 'r.', markersize=3, label='H2 Leroy+2008 (alpha=4.4, R21=0.8)')
        plt.yscale('log')
        plt.ylim(0.05, 1000)
        plt.xlabel('Radius (kpc)')
        plt.ylabel('Face-on surface density incl. He (Msun/pc$^2$)')
        plt.title(f'{gal} measured gas profiles')
        plt.grid(True, linewidth=0.5, alpha=0.7)
        plt.legend(fontsize='small')
        plt.savefig(os.path.join(OUT_DIR, f"{gal}_profiles.png"), bbox_inches='tight', dpi=150)
        plt.close()
