import sys
import numpy as np
from scipy.optimize import least_squares
from scipy.integrate import quad
from scipy.special import ellipk, ellipe
import warnings

# Suppress all integration warnings to keep terminal output clean
warnings.filterwarnings("ignore")

# Convert millions of solar masses per kpc^2 to protons per cm^2
MILLIONS_SOLAR_MASS_PER_KPC_2_TO_PROTONS_PER_CM_2 = 1.249e20

# Regularization parameter to penalize "wiggles" in the surface density.
# Increase this value if the density profile is still too jagged.
# Decrease this value if the model is failing to follow real trends in V_gas.
SMOOTHING_WEIGHT = 50.0

def integrand_log(a, r):
    """The K(m) term contains a logarithmic singularity. quad handles this using points=[r]."""
    m = (4.0 * a * r) / (r + a)**2
    # Cap m slightly below 1.0 to completely prevent 'inf' crashes from ellipk
    if m > 0.9999999999: 
        m = 0.9999999999 
    return a * ellipk(m) / (r + a)

def integrand_cauchy_numerator(a, r):
    """
    The E(m) term contains a 1/(r-a) singularity. 
    Scipy's weight='cauchy' integrates exactly f(a)/(a - r).
    """
    m = (4.0 * a * r) / (r + a)**2
    if m > 0.9999999999: 
        m = 0.9999999999
    return -a * ellipe(m)

def integrand_normal_E(a, r):
    """Standard evaluation for rings that do NOT overlap the observation radius."""
    m = (4.0 * a * r) / (r + a)**2
    if m > 0.9999999999: 
        m = 0.9999999999
    
    denom = r - a
    # Safety catch for floating point division by zero
    if abs(denom) < 1e-12: 
        return 0.0
        
    return a * ellipe(m) / denom

def build_gravity_matrix(R_bins):
    """Builds the coefficient matrix relating Sigma to V^2 for a specific galaxy."""
    N = len(R_bins)
    G_eff = 4.301 # G adjusted for R(kpc), V(km/s), Sigma(M_sun/pc^2)
    
    # Create bin edges assuming R_bins are the centers of the annuli
    edges = np.zeros(N + 1)
    edges[0] = 0.0
    for i in range(1, N):
        edges[i] = (R_bins[i-1] + R_bins[i]) / 2.0
    
    if N > 1:
        edges[N] = R_bins[-1] + (R_bins[-1] - R_bins[-2]) / 2.0
    else:
        edges[N] = R_bins[0] * 2.0

    M_matrix = np.zeros((N, N))
    
    # Populate the gravitational influence matrix using Cauchy Principal Values
    for i, r in enumerate(R_bins):
        for j in range(N):
            a_in = edges[j]
            a_out = edges[j+1]
            
            if a_in <= r <= a_out:
                # Observation radius lands INSIDE this bin: requires Cauchy Principal Value
                int_log, _ = quad(integrand_log, a_in, a_out, args=(r,), points=[r], limit=200)
                int_cpv, _ = quad(integrand_cauchy_numerator, a_in, a_out, args=(r,), weight='cauchy', wvar=r, limit=200)
                M_matrix[i, j] = 2.0 * G_eff * (int_log + int_cpv)
            else:
                # Observation radius is OUTSIDE this bin: standard integration
                int_log, _ = quad(integrand_log, a_in, a_out, args=(r,), limit=200)
                int_cpv, _ = quad(integrand_normal_E, a_in, a_out, args=(r,), limit=200)
                M_matrix[i, j] = 2.0 * G_eff * (int_log + int_cpv)
                
    # Sanitize the matrix to guarantee no stray NaNs hit the solver
    M_matrix = np.nan_to_num(M_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return M_matrix

def fit_galaxy_surface_density(R_obs, V_gas_obs, smooth_weight=SMOOTHING_WEIGHT):
    """Iteratively solves for Sigma_gas (M_sun/pc^2) given R and V_gas with Tikhonov regularization."""
    Gamma_obs = V_gas_obs * np.abs(V_gas_obs) # Preserves the outward pull sign
    M_matrix = build_gravity_matrix(R_obs)
    
    def residuals(Sigma_array):
        # Base physical error: Model V^2 vs Observed V^2
        Gamma_model = M_matrix @ Sigma_array
        base_residuals = Gamma_model - Gamma_obs
        
        # Regularization: Penalize the second derivative (curvature) to force a smooth curve
        if len(Sigma_array) > 2:
            curvature_penalty = smooth_weight * np.diff(Sigma_array, n=2)
            # Append the penalty array to the residuals array
            return np.concatenate((base_residuals, curvature_penalty))
        else:
            return base_residuals

    # Initial flat guess of 2.0 M_sun/pc^2
    Sigma_guess = np.ones_like(R_obs) * 2.0 
    
    # Enforce non-negative density bounds
    result = least_squares(residuals, x0=Sigma_guess, bounds=(0, np.inf))
    return result.x

def main(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    galaxies = {}
    
    for idx, line in enumerate(lines):
        if len(line) > 70 and line[12:18].strip().replace('.', '', 1).isdigit():
            gal_id = line[0:11].strip()
            R = float(line[19:25])
            Vgas = float(line[39:45])
            
            if gal_id not in galaxies:
                galaxies[gal_id] = {'R': [], 'Vgas': [], 'indices': []}
                
            galaxies[gal_id]['R'].append(R)
            galaxies[gal_id]['Vgas'].append(Vgas)
            galaxies[gal_id]['indices'].append(idx)

    new_columns = {idx: "" for idx in range(len(lines))}

    print(f"Processing {len(galaxies)} galaxies with Smoothness Weight = {SMOOTHING_WEIGHT}...")
    
    for gal_id, data in galaxies.items():
        R_arr = np.array(data['R'])
        Vgas_arr = np.array(data['Vgas'])
        
        Sigma_fit_Msun_pc2 = fit_galaxy_surface_density(R_arr, Vgas_arr)
        
        # --- Convergence Check ---
        M_matrix = build_gravity_matrix(R_arr)
        Gamma_test = M_matrix @ Sigma_fit_Msun_pc2
        # Re-derive velocity, maintaining the sign
        Vgas_test = np.sign(Gamma_test) * np.sqrt(np.abs(Gamma_test))
        
        errors = np.abs(Vgas_test - Vgas_arr)
        max_error = np.max(errors)
        
        if max_error > 0.5:
            print(f"WARNING: Convergence/Smoothing trade-off for {gal_id} | Max error: {max_error:.2f} km/s")
        # -------------------------
        
        # Convert to particles/cm^2 (Hydrogen column density N_HI)
        N_HI_particles_cm2 = (Sigma_fit_Msun_pc2 / 1.33) * MILLIONS_SOLAR_MASS_PER_KPC_2_TO_PROTONS_PER_CM_2
        
        for i, idx in enumerate(data['indices']):
            val = N_HI_particles_cm2[i]
            new_columns[idx] = f"{val:10.3e}"
            
        print(f"Solved: {gal_id} ({len(R_arr)} radii)")

    with open(output_file, 'w') as f:
        for idx, line in enumerate(lines):
            clean_line = line.rstrip('\n')
            
            if clean_line.startswith("   69- 76 F8.2   solLum/pc2   SBbul"):
                f.write(clean_line + "\n")
                f.write("   78- 87 E10.3  part/cm2 Sigma_gas Calculated HI particle density \n")
            elif clean_line.startswith("   61- 67 F7.2   solLum/pc2   SBdisk"):
                f.write(clean_line + "      Sigma_gas\n")
            elif idx in new_columns and new_columns[idx] != "":
                f.write(f"{clean_line} {new_columns[idx]}\n")
            else:
                f.write(clean_line + "\n")

    print(f"\nSuccess! Modified data saved to {output_file}")

if __name__ == "__main__":
    INPUT_FILENAME = "./sparc/MassModels_Lelli2016c.mrt"
    OUTPUT_FILENAME = "./sparc/MassModels_Lelli2016_SigmaGas.mrt"
    main(INPUT_FILENAME, OUTPUT_FILENAME)