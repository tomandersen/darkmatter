from cgm_integral import evaluate_dm_mass_integral
import numpy as np
import os
import time
from datetime import datetime
from scipy.optimize import minimize_scalar
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings
import gc 


def run_sparc_dm_brute_cgm(add_core_gas_central_density, add_core_gas_beta, r_core_factor_gas, add_real_cgm_central_density, r_core_factor_cgm, cgm_beta, num_minutes=10, output_graphs=True, power_override = 9.0, min_mass=0, max_mass=1e99,
                           supplied_gas_density=None, only_galaxies=None, output_dir_override=None, return_galaxies=False, z_stretch_kpc=None):
    # supplied_gas_density: optional dict {galaxy_name: fn(R_m, Z_m, Theta_m) -> gas particle density [m^-3]}
    #   For those galaxies the 3D gas grid comes from fn instead of the axisymmetric Sigma_gas column (core gas / CGM are still added on top).
    #   fn gets the 3D meshgrid arrays (indexing='ij', shape R x Z x Theta) in metres / radians.
    # only_galaxies: optional list of galaxy names to run (default: all SPARC galaxies)
    # output_dir_override: write logs/plots here instead of ./sparc/dm_mass
    # return_galaxies: if True also return the per-galaxy dict (curves etc) as a fourth value
    # z_stretch_kpc: if set, z layers are spaced as z = A*sinh(u) with A = z_stretch_kpc (u uniform), so layers are
    #   ~A*du thick near the mid-plane and coarse far out. Needed for thin (~50 pc) molecular layers. Default: uniform z.
    # Record start time and date
    start_time = time.time()
    start_datetime = datetime.now()

    # Suppress integration warnings for highly dense inner rings
    warnings.filterwarnings("ignore")

    Upsilon_disk = 0.5   
    Upsilon_bulge = 0.6  # 0.6 seems to work a little better

    sqrt_Upsilon_disk = np.sqrt(Upsilon_disk)
    sqrt_Upsilon_bulge = np.sqrt(Upsilon_bulge)

    USE_LINEAR_ERROR = True
    USE_INTERPOLATED_FIT = True # Added boolean to switch between fits

    # control the length of time it takes to calculate everything. 
    num_points = int(num_minutes*3e5) 
    base_points = int(num_points**(1/3.0))
    NUM_BLOBS_THETA = int(base_points/1.5) | 1 # the | 1 makes the integer odd, odd is better, as it puts blobs on the central plane, etc.
    NUM_LAYERS_Z = int(1.5*base_points) | 1
    NUM_R_BINS = int(base_points) | 1 


    R_GRID_SCALE_FACTOR = 1.1 # How far to go past the last R data point. should be 1.1 or 1.2 or 1.3 
    # after R_GRID_SCALE_FACTOR, points at R_max observed should not feel any force from mass beyond R_GRID_SCALE_FACTOR
    # Need to go past R_max as Newton's Shell Theorem does not hold in a thin disk. (And no Jeans stuff is not good either)     
     
    USE_EXPONENTIAL_DISK = True  # without this on, disk is just hard cut at +/- h_z

    MIN_DENSITY_FOR_POWER = 0 # with zero you will get too much cgm on the analytic full halo integrasls... i think. 3e-5

    # to speed it up, maybe dont bother with heatmaps
    DO_HEATMAPS = output_graphs

    # as per the litrature, I think Lelli 2016 uses this
    def get_sparc_galaxy_scale_height(radius_kpc, R_d):
        """Calculates the vertical disk scale height (thickness) of a SPARC galaxy."""
        z_d = 0.196 * (R_d ** 0.633)
        return z_d

    # 1. Setup and Parse the Data
    properties_file = "./sparc/SPARC_Lelli2016c.mrt.txt"
    kinematics_file = "./sparc/MassModels_Lelli2016_SigmaGas.mrt" # i needed to use gas density at each radius, so a seperate python file did that.
    output_dir = "./sparc/dm_mass"
    if output_dir_override is not None:
        output_dir = output_dir_override
    output_galaxies = os.path.join(output_dir, "galaxies")
    output_gas_heatmaps = os.path.join(output_dir, "gas_heatmaps")
    output_dm_heatmaps = os.path.join(output_dir, "dm_heatmaps")
    if supplied_gas_density is None:
        supplied_gas_density = {}

    os.makedirs(output_galaxies, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_gas_heatmaps, exist_ok=True)
    os.makedirs(output_dm_heatmaps, exist_ok=True)

    rd_dict = {}
    HIMass_dict = {}
    TL_dict = {}
    with open(properties_file, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 12 and parts[1].lstrip('-').isdigit():
                gal_name = parts[0]
                try:
                    r_d = float(parts[11])
                    rd_dict[gal_name] = r_d
                    t_luminosity = float(parts[7])
                    TL_dict[gal_name] = t_luminosity
                    hi_mass = float(parts[13])
                    HIMass_dict[gal_name] = hi_mass
                except ValueError:
                    continue

    galaxies = {}

    with open(kinematics_file, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) == 11 and not line.startswith('---') and not line.startswith('Byte'):
                try:
                    gal_name = parts[0]
                    R = float(parts[2])
                    Vobs = float(parts[3])
                    eVobs = float(parts[4])
                    Vgas = float(parts[5])
                    Vdisk = float(parts[6])
                    Vbul = float(parts[7])
                    Sigma_gas = float(parts[10]) 
                    
                    Vdisk = Vdisk * sqrt_Upsilon_disk
                    Vbul = Vbul * sqrt_Upsilon_bulge

                    if eVobs <= 0:
                        eVobs = 1.0 
                        
                    if gal_name in rd_dict:
                        if gal_name not in galaxies:
                            galaxies[gal_name] = {'R': [], 'Vobs': [], 'eVobs': [], 'Vgas': [], 'Vdisk': [], 'Vbul': [], 'Sigma_gas': []}
                        
                        galaxies[gal_name]['R'].append(R)
                        galaxies[gal_name]['Vobs'].append(Vobs)
                        galaxies[gal_name]['eVobs'].append(eVobs)
                        galaxies[gal_name]['Vgas'].append(Vgas)
                        galaxies[gal_name]['Vdisk'].append(Vdisk)
                        galaxies[gal_name]['Vbul'].append(Vbul)
                        galaxies[gal_name]['Sigma_gas'].append(Sigma_gas)
                        
                except ValueError:
                    continue

    if only_galaxies is not None:
        galaxies = {gal: galaxies[gal] for gal in only_galaxies if gal in galaxies}

    for gal in galaxies:
        for key in galaxies[gal]:
            galaxies[gal][key] = np.array(galaxies[gal][key])

    # 2. Physics Constants
    G_SI = 6.67430e-11        
    c_SI = 2.9979e8           
    m_p = 1.67262e-27      
    pc_to_m = 3.086e16     
    kpc_to_m = pc_to_m * 1000
    a0_mond = 1.2e-10      
    M_sun = 1.98847e30     

    # 3. Pre-calculate Base Integrals (P=1 Watt) for ALL galaxies
    print(f"Pre-calculating base arrays for {len(galaxies)} galaxies via continuous 3D brute force...")

    # we may delete galaxies in this loop, filtering them out..
    for i, (gal, value) in enumerate(list(galaxies.items())):
        print(f"Processing {gal} ({i+1}/{len(galaxies)})...", end='\r')
        
        # try heavy
        total_mass = (TL_dict[gal] + HIMass_dict[gal])*1e9
        if (total_mass < min_mass or total_mass > max_mass):
            del galaxies[gal]
            continue
 

        g_data = galaxies[gal]
        R_obs_kpc = g_data['R']
        R_m_obs = R_obs_kpc * kpc_to_m
        V_gas_ms = g_data['Vgas'] * 1000.0
        V_disk_ms = g_data['Vdisk'] * 1000.0
        V_bul_ms = g_data['Vbul'] * 1000.0
        
        max_R_kpc = np.max(R_obs_kpc)
        grid_max_kpc = max_R_kpc
        if R_GRID_SCALE_FACTOR > 1.0:
            grid_max_kpc = R_GRID_SCALE_FACTOR * grid_max_kpc
        grid_max_m = grid_max_kpc*kpc_to_m

        R_grid_kpc = np.linspace(max_R_kpc / NUM_R_BINS, grid_max_kpc, NUM_R_BINS)
        R_grid_m = R_grid_kpc * kpc_to_m
        dR_m = R_grid_m[1] - R_grid_m[0]
        
        g_data_r_obs_kpc = R_obs_kpc
        g_data_sigma_gas = g_data['Sigma_gas']
        g_data_Vdisk = g_data['Vdisk']
        g_data_Vbul = g_data['Vbul']
        g_data_Vgas = g_data['Vgas']
        g_data_Vobs = g_data['Vobs'] 
        g_data_eVobs = g_data['eVobs'] 
        
        # if we are extending the grid past the max r, we add data in, with the values adjusted to fall off
        # I don't use these data for anything but making the interpolation grids up.
        if R_GRID_SCALE_FACTOR > 1.0:
            # grid_max - where the simulation stops. 
            # pick a point where the data grinds out. Not too far out!
            # halfway between R_max observed and grid_max
            last_r = g_data_r_obs_kpc[-1] + (grid_max_kpc - g_data_r_obs_kpc[-1])/2.0
            rscale = 1.0/R_GRID_SCALE_FACTOR # scale some things down a bit at the extra point
            g_data_r_obs_kpc = np.append(g_data_r_obs_kpc, last_r)
            g_data_sigma_gas = np.append(g_data_sigma_gas, rscale*g_data_sigma_gas[-1]) # gas density drops a bit with R.
            g_data_Vdisk = np.append(g_data_Vdisk, rscale*g_data_Vdisk[-1])
            g_data_Vbul = np.append(g_data_Vbul, rscale*g_data_Vbul[-1])
            g_data_Vgas = np.append(g_data_Vgas, g_data_Vgas[-1]) # gas velocity is often flat at the edge of the data. 
            g_data_Vobs = np.append(g_data_Vobs, g_data_Vobs[-1]) 
            g_data_eVobs = np.append(g_data_eVobs, g_data_eVobs[-1]) 
        
        Sigma_gas_cm2_grid = np.interp(R_grid_kpc, g_data_r_obs_kpc, g_data_sigma_gas)
        g_data['Vdisk_grid'] = np.interp(R_grid_kpc, g_data_r_obs_kpc, g_data_Vdisk)
        g_data['Vbul_grid'] = np.interp(R_grid_kpc, g_data_r_obs_kpc, g_data_Vbul)
        g_data['Vgas_grid'] = np.interp(R_grid_kpc, g_data_r_obs_kpc, g_data_Vgas)
        g_data['Vobs_grid'] = np.interp(R_grid_kpc, g_data_r_obs_kpc, g_data_Vobs) 
        g_data['eVobs_grid'] = np.interp(R_grid_kpc, g_data_r_obs_kpc, g_data_eVobs) 
        
        R_d = rd_dict[gal]
        hz_kpc = get_sparc_galaxy_scale_height(None, R_d)
        hz_m = hz_kpc * kpc_to_m
        
        valid_obs = R_m_obs > 0 
        Sigma_gas_m2_grid = Sigma_gas_cm2_grid * 100.0 * 100.0 
        
        # ---------------------------------------------------------
        # BRUTE FORCE 3D BLOB CALCULATION (Continuous Grid)
        # ---------------------------------------------------------
        
        # we make a square box of size grid_max_m*2 on a side.
        if z_stretch_kpc is None:
            Z_m = np.linspace(-grid_max_m, grid_max_m, NUM_LAYERS_Z) 
            dZ = Z_m[1] - Z_m[0] 
        else:
            a_m = z_stretch_kpc * kpc_to_m
            u_max = np.arcsinh(grid_max_m / a_m)
            Z_m = a_m * np.sinh(np.linspace(-u_max, u_max, NUM_LAYERS_Z))
            dZ = np.gradient(Z_m)[None, :, None] # layer thickness, broadcast over R and theta
        
        theta_rad = np.linspace(0, 2 * np.pi, NUM_BLOBS_THETA, endpoint=False)
        dTheta = 2 * np.pi / NUM_BLOBS_THETA

        R_grid_3d, Z_grid_3d, Theta_grid_3d = np.meshgrid(R_grid_m, Z_m, theta_rad, indexing='ij')
        
        dV = R_grid_3d * dTheta * dR_m * dZ 
        
        if gal in supplied_gas_density:
            # measured 3D gas supplied from outside (e.g. HI + H2 + ionized maps, each with its own vertical profile)
            gas_particle_density_grid = np.array(supplied_gas_density[gal](R_grid_3d, Z_grid_3d, Theta_grid_3d), dtype=float)
        else:
            gas_particle_density_1d = np.where(R_grid_m > 0, Sigma_gas_m2_grid / (2.0 * hz_m), 0.0)
            gas_particle_density_grid = gas_particle_density_1d[:, None, None] * np.ones_like(Z_grid_3d)

            if USE_EXPONENTIAL_DISK:
                gas_particle_density_grid *= np.exp(-np.abs(Z_grid_3d) / hz_m)
            else: 
                z_mask = np.abs(Z_grid_3d) <= hz_m
                gas_particle_density_grid[~z_mask] = 0.0

        X_blob = R_grid_3d * np.cos(Theta_grid_3d)
        Y_blob = R_grid_3d * np.sin(Theta_grid_3d)
        Z_blob = Z_grid_3d

        r_core_kpc = 2.0*(total_mass/1e11)**(1/3) 
        r_core_gas_m = (r_core_kpc * r_core_factor_gas) * kpc_to_m
        r_core_cgm_m = (r_core_kpc * r_core_factor_cgm) * kpc_to_m
        r_squared = X_blob**2 + Y_blob**2 + Z_blob**2

        # core gas density correction: I have determined that I need to add (9.5 - log10(TL))*0.2 to the core gas density.
        # Determined by fitting.
        #print(f"{gal} core_gas_density_correction {core_gas_density_correction(total_mass)} total mass: {total_mass/1e9}")
        if add_core_gas_central_density > 0.0:
            n_0_cgs = add_core_gas_central_density + core_gas_density_correction(total_mass)
            n_0_m3 =  n_0_cgs * (100)**3
            exponent = -add_core_gas_beta 
            gas_particle_density_grid += n_0_m3 * (1 + r_squared / r_core_gas_m**2)**exponent
        if add_real_cgm_central_density > 0.0:
            n_0_cgs = add_real_cgm_central_density  
            n_0_m3 =  n_0_cgs * (100)**3
            exponent = -3.0*cgm_beta/2.0
            gas_particle_density_grid += n_0_m3 * (1 + r_squared / r_core_cgm_m**2)**exponent

        # Zero out gas outside the bounding sphere of radius grid_max_m (keeps 3D shape intact)
        # this makes it so our output stats make more sense, we can just sum all the dark mass we made, etc.
        gas_particle_density_grid[r_squared >= grid_max_m**2] = 0.0
        

        g_data['R_grid_kpc'] = R_grid_kpc
        g_data['Z_grid_kpc'] = Z_m / kpc_to_m
        g_data['r_core_kpc'] = r_core_kpc
        
        gas_particle_mass_He_factor = 1.33  
        dm_gas = gas_particle_density_grid * m_p * gas_particle_mass_He_factor * dV

        # only want the sum of gas inside r_max
        r_max_m = max_R_kpc * kpc_to_m
        r_max_mask = r_squared < r_max_m**2
        g_data['sum_all_gas_particles'] = np.sum(gas_particle_density_grid[r_max_mask] * dV[r_max_mask])
       
        n_min_m3 = MIN_DENSITY_FOR_POWER*(100)**3 
        dm_mask = gas_particle_density_grid >= n_min_m3
        dm_stat_mask = dm_mask & r_max_mask # for display and output stats only get dm until r_max
        g_data['gas_density_slice_m3'] = np.where(dm_stat_mask[:, :, 0], gas_particle_density_grid[:, :, 0], 0.0)
        g_data['sum_dm_mass_integral'] = np.sum((gas_particle_density_grid[dm_stat_mask]**(2.0/3.0)) * dV[dm_stat_mask])
        dm_base_dm = np.zeros_like(gas_particle_density_grid)
        dm_base_dm[dm_mask] = (1/c_SI**3) * gas_particle_density_grid[dm_mask]**(2/3) * dV[dm_mask]

        V_gas_sq_grid = np.zeros_like(R_grid_m)
        V_dm_sq_base_grid = np.zeros_like(R_grid_m)
        V_gas_inDisk_sq_grid = np.zeros_like(R_grid_m)

        softening_sq = (0.001 * hz_m)**2 
        
        # for each radius on the x axis of R_grid_m, we sum up force from all the gas/dark matter in each blob
        # I don't need to add power at this point, since its linear, we just multiply it in when we optimize on P.
        for i_idx, R_test in enumerate(R_grid_m):
            dx = X_blob - R_test
            dy = Y_blob
            dz = Z_blob
            
            dist_sq = dx**2 + dy**2 + dz**2 + softening_sq
            dist = np.sqrt(dist_sq)
            
            da_x_factor = G_SI * dx / (dist_sq * dist)
            
            a_x_gas = np.sum(da_x_factor * dm_gas)
            a_x_dm = np.sum(da_x_factor * dm_base_dm)
            
            V_gas_sq_grid[i_idx] = -a_x_gas * R_test
            V_dm_sq_base_grid[i_idx] = max(0, -a_x_dm * R_test)
            a_x_gas_inDisk = np.sum(np.where((dz >= -1.6*hz_m) & (dz <= 1.6*hz_m), da_x_factor * dm_gas, 0.0))
            V_gas_inDisk_sq_grid[i_idx] = -a_x_gas_inDisk * R_test
        
        # _base_ means the dm still needs to be multiplied by the power.
        g_data['V_dm_sq_base_grid_kms'] = V_dm_sq_base_grid / 1e6
        g_data['Vgas_test_grid_kms'] = np.sign(V_gas_sq_grid) * np.sqrt(np.abs(V_gas_sq_grid)) / 1000.0
        g_data['Vgas_test_inDisk_grid_kms'] = np.sign(V_gas_inDisk_sq_grid) * np.sqrt(np.abs(V_gas_inDisk_sq_grid)) / 1000.0

        # thses two velocity arrays match the data from Lelli 2016 in format, no added extra point 
        g_data['V_dm_sq_base_kms'] = np.interp(R_obs_kpc, R_grid_kpc, V_dm_sq_base_grid / 1e6)
        g_data['Vgas_test'] = np.interp(R_obs_kpc, R_grid_kpc, g_data['Vgas_test_grid_kms'])
        
        V_bar_sq = np.clip(V_disk_ms**2 + V_bul_ms**2 + V_gas_ms * np.abs(V_gas_ms), 0, None)
        g_N = np.zeros_like(R_m_obs)
        g_N[valid_obs] = V_bar_sq[valid_obs] / R_m_obs[valid_obs]
        
        g_mond = (g_N + np.sqrt(g_N**2 + 4 * g_N * a0_mond)) / 2.0
        g_data['V_mond_kms'] = np.sqrt(g_mond * R_m_obs) / 1000.0
        g_data['V_mond_grid_kms'] = np.interp(R_grid_kpc, R_obs_kpc, g_data['V_mond_kms']) 

        del R_grid_3d, Z_grid_3d, Theta_grid_3d
        del X_blob, Y_blob, Z_blob
        del gas_particle_density_grid, dV, dm_gas, dm_base_dm
        gc.collect()

    print(f"\nIntegrations complete. Now at {len(galaxies)} galaxies. Optimizing global parameter for Linear Absolute Error...")


    # 4. Global Optimization Function
    def global_fit_err(P_test):
        total_err = 0
        for gal in galaxies:
            g = galaxies[gal]
            
            if USE_INTERPOLATED_FIT:
                V_dm_test_sq_grid = P_test * g['V_dm_sq_base_grid_kms']
                V_bar_sq_grid_kms = np.clip(g['Vdisk_grid']**2 + g['Vbul_grid']**2 + g['Vgas_test_grid_kms'] * np.abs(g['Vgas_test_grid_kms']), 0, None)
                V_tot_test_grid = np.sqrt(V_bar_sq_grid_kms + V_dm_test_sq_grid)
                n = len(V_tot_test_grid)
                # when using the interpolated fit, we use the central area only, ignore the tails...
                i0, i1 = max(1, int(0.05 * n)), max(2, int(0.95 * n))  # trim outer 5% each side
                if USE_LINEAR_ERROR:
                    the_err = np.sum(np.abs(g['Vobs_grid'][i0:i1] - V_tot_test_grid[i0:i1]))
                else:
                    the_err = np.sum(((g['Vobs_grid'][i0:i1] - V_tot_test_grid[i0:i1]) / g['eVobs_grid'][i0:i1])**2)
            else:
                V_dm_test_sq = P_test * g['V_dm_sq_base_kms']
                
                gas_to_use = 'Vgas'
                if add_core_gas_central_density > 0.0 or add_real_cgm_central_density > 0.0 or gal in supplied_gas_density:
                    gas_to_use = 'Vgas_test'

                V_bar_sq_kms = np.clip(g['Vdisk']**2 + g['Vbul']**2 + g[gas_to_use] * np.abs(g[gas_to_use]), 0, None) 
                V_tot_test = np.sqrt(V_bar_sq_kms + V_dm_test_sq)
                
                if USE_LINEAR_ERROR:
                    the_err = np.sum(np.abs(g['Vobs'] - V_tot_test))
                else:
                    the_err = np.sum(((g['Vobs'] - V_tot_test) / g['eVobs'])**2) 
                    
            total_err += the_err
      
        return total_err

    result = minimize_scalar(global_fit_err, bounds=(0.5, 100.0), method='bounded')
    print(f"Found best power: {result.x:.3f} (± {result.fun:.3e})")

    best_global_P = result.x
    min_found_err = result.fun
    if (power_override > 0.0):
        best_global_P = power_override
        min_found_err = 0

        
    dm_chi2_total = 0
    dm_linear_total = 0
    dm_linear_total_err = 0
    dm_grid_chi2_total = 0 
    dm_grid_linear_total = 0 
    mond_chi2_total = 0
    mond_linear_total = 0
    mond_linear_total_err = 0
    mond_linear_total_grid = 0;

    for gal in galaxies:
        g = galaxies[gal]

        gas_to_use = 'Vgas'
        if add_core_gas_central_density > 0.0 or add_real_cgm_central_density > 0.0 or gal in supplied_gas_density:
            gas_to_use = 'Vgas_test'

        V_dm_final_kms = np.sqrt(best_global_P * g['V_dm_sq_base_kms'])
        V_bar_sq_kms = np.clip(g['Vdisk']**2 + g['Vbul']**2 + g[gas_to_use] * np.abs(g[gas_to_use]), 0, None)
        V_tot_final_kms = np.sqrt(V_bar_sq_kms + V_dm_final_kms**2)
        
        # Original Data Points
        dm_chi2_total += np.sum(((g['Vobs'] - V_tot_final_kms) / g['eVobs'])**2)
        dm_linear_total_err += np.sum(np.abs(g['Vobs'] - V_tot_final_kms) / g['eVobs'])
        dm_linear_total += np.sum(np.abs(g['Vobs'] - V_tot_final_kms)) 

        # Interpolated Grid Points
        V_dm_final_grid_kms = np.sqrt(best_global_P * g['V_dm_sq_base_grid_kms'])
        V_bar_sq_grid_kms = np.clip(g['Vdisk_grid']**2 + g['Vbul_grid']**2 + g['Vgas_test_grid_kms'] * np.abs(g['Vgas_test_grid_kms']), 0, None) 
        V_tot_final_grid_kms = np.sqrt(V_bar_sq_grid_kms + V_dm_final_grid_kms**2)
        
        dm_grid_chi2_total += np.sum(((g['Vobs_grid'] - V_tot_final_grid_kms) / g['eVobs_grid'])**2)
        dm_grid_linear_total += np.sum(np.abs(g['Vobs_grid'] - V_tot_final_grid_kms))

        # MOND
        mond_chi2_total += np.sum(((g['Vobs'] - g['V_mond_kms']) / g['eVobs'])**2)
        mond_linear_total_err += np.sum(np.abs(g['Vobs'] - g['V_mond_kms']) / g['eVobs'])
        mond_linear_total += np.sum(np.abs(g['Vobs'] - g['V_mond_kms']))
        mond_linear_total_grid += np.sum(np.abs(g['Vobs_grid'] - g['V_mond_grid_kms'])) 


    # ---------------------------------------------------------
    # HELPER FUNCTIONS FOR STATS, LOGGING, AND SCATTER PLOTS
    # ---------------------------------------------------------

    def compute_galaxy_stats(galaxy_dict, best_P):
        stats = []
        global_stats = {}
        total_baryon_mass = 0.0 
        total_dm_mass = 0.0 
        dm_ratio_by_galaxy = 0.0 
        tot_dm_gas_ratio_to_rmax = 0.0
        unobserved_gas_fraction = 0.0   

        for name, g in galaxy_dict.items():
            tl = TL_dict.get(name, 0.0) * 1e9          
            hi = HIMass_dict.get(name, 0.0) * 1e9      
            m_b = tl + hi                             
            g['gas_plus_tl'] = m_b

            tot_gas_mass_kg = g['sum_all_gas_particles'] * m_p * 1.33
            tot_gas_mass_msun = tot_gas_mass_kg / M_sun
            
            hi_gas_ratio = hi / tot_gas_mass_msun if tot_gas_mass_msun > 0 else 0.0
            gas_plus_tl = tot_gas_mass_msun + tl
            
            tot_dm_mass_kg = best_P * (1.0 / c_SI**3) * g['sum_dm_mass_integral']
            tot_dm_mass_msun = tot_dm_mass_kg / M_sun
            
            dm_mb_ratio = tot_dm_mass_msun / m_b if m_b > 0 else 0.0
            dm_gas_ratio = tot_dm_mass_msun / tot_gas_mass_msun if tot_gas_mass_msun > 0 else 0.0
            dm_allbaryons_ratio = tot_dm_mass_msun / (tot_gas_mass_msun + tl) if (tot_gas_mass_msun + tl) > 0 else 0.0

            total_baryon_mass += tot_gas_mass_msun + tl 
            total_dm_mass += tot_dm_mass_msun 
            dm_ratio_by_galaxy += dm_allbaryons_ratio
            tot_dm_gas_ratio_to_rmax += tot_dm_mass_msun/hi
            unobserved_gas_fraction += (tot_gas_mass_msun-hi)/hi
            calc_halo_properties(name, g, best_P)

            stats.append({
                'name': name,
                'M_b': m_b,
                'TL': tl/m_b,
                'HIMass': hi/m_b,
                'R_max_Gas_Mass': tot_gas_mass_msun,
                'HIMass_Total_Gas_Ratio': hi_gas_ratio,
                'Gas_Plus_TL': gas_plus_tl,
                'R_max_DM_Mass': tot_dm_mass_msun,
                'DM_Mb_Ratio': dm_mb_ratio,
                'DM_Gas_Ratio': dm_gas_ratio,
                'DM_AllBaryons_Ratio': dm_allbaryons_ratio,
                'r_core_kpc': g['r_core_kpc'],
                'totR200_gas': g['totR200_gas'],
                'totR200_dm': g['totR200_dm'],
                'totRMax_gas': g['totRMax_gas'],
                'totRMax_dm': g['totRMax_dm'],
                'r_200': g['r_200'],
                'r_max': g['r_max']
            })

        global_stats = {
            'Global_R_max_Baryon_Mass': total_baryon_mass,
            'Global_R_max_DM_Mass': total_dm_mass,
            'DM_AllBaryons_Ratio': dm_ratio_by_galaxy / len(galaxy_dict),
            'Overall_R_max_DM_Ratio': total_dm_mass/total_baryon_mass,
            'Overall_DM_HILelliGas_Ratio_to_RMax': tot_dm_gas_ratio_to_rmax / len(galaxy_dict),
            'Unobserved_gas_fraction': unobserved_gas_fraction / len(galaxy_dict),
        }
        return stats, global_stats


    def calc_halo_properties(name, g, best_P):
        # I want to get the analytic numbers for the total halo mass we calculate here:
        # there are two integrals, done until r_200
        v_last_observed = g['Vobs'][-1]
        hubble_const = 71.0
        to_kpc_200 = 100.0
        r_200 = to_kpc_200 *  v_last_observed/ hubble_const
        g['r_200'] = r_200
        r_200_m = r_200 * kpc_to_m
        r_core_m = g['r_core_kpc'] * kpc_to_m
        r_core_gas_m = r_core_m * r_core_factor_gas
        r_core_cgm_m = r_core_m * r_core_factor_cgm
        # ok, now integrate the functions

        n_0_1_m3 = 0
        exponent1 = -2.0
        n_0_2_m3 = 0
        exponent2 = -2.0
        min_gas_for_dm = MIN_DENSITY_FOR_POWER*(100)**3


        n_0_m3 = 0
        total_mass = (TL_dict[name] + HIMass_dict[name])*1e9
        if add_core_gas_central_density > 0.0:
            n_0_1_cgs = add_core_gas_central_density + core_gas_density_correction(total_mass) 
            n_0_1_m3 =  n_0_1_cgs * (100)**3
            exponent1 = -add_core_gas_beta 
        
        if add_real_cgm_central_density > 0.0:
            n_0_2_cgs = add_real_cgm_central_density  
            n_0_2_m3 =  n_0_2_cgs * (100)**3
            exponent2 = -3.0*cgm_beta/2.0
        

        totR200_gas, err_gas = evaluate_dm_mass_integral(0, r_200_m, True, best_P, c_SI, r_core_gas_m, n_0_1_m3, exponent1, r_core_cgm_m, n_0_2_m3, exponent2, min_gas_for_dm)
        totR200_dm, err_dm = evaluate_dm_mass_integral(0, r_200_m, False, best_P, c_SI, r_core_gas_m, n_0_1_m3, exponent1, r_core_cgm_m, n_0_2_m3, exponent2, min_gas_for_dm)
      
        # do the same integrals out to R_max
        R2 = g['R'][-1] * kpc_to_m
        totRMax_gas, err_gas = evaluate_dm_mass_integral(0, R2, True, best_P, c_SI, r_core_gas_m, n_0_1_m3, exponent1, r_core_cgm_m, n_0_2_m3, exponent2, min_gas_for_dm)
        totRMax_dm, err_dm = evaluate_dm_mass_integral(0, R2, False, best_P, c_SI, r_core_gas_m, n_0_1_m3, exponent1, r_core_cgm_m, n_0_2_m3, exponent2, min_gas_for_dm)

        g['totR200_gas'] = totR200_gas*m_p/M_sun
        g['totR200_dm'] = totR200_dm*m_p/M_sun
        g['totRMax_gas'] = totRMax_gas*m_p/M_sun
        g['totRMax_dm'] = totRMax_dm*m_p/M_sun
        g['r_200'] = r_200
        g['r_max'] = R2/kpc_to_m



    def write_run_log(output_dir, best_global_P, dm_chi2_total, dm_linear_total_err, dm_linear_total, dm_grid_chi2_total, dm_grid_linear_total, mond_chi2_total, mond_linear_total_err, mond_linear_total, mond_linear_total_grid, global_stats, galaxy_stats):
        log_filename = os.path.join(output_dir, "run.txt")
        duration_seconds = time.time() - start_time
        
        with open(log_filename, "w") as log_file:
            log_file.write("========================================================================================================\n")
            log_file.write(f"Run Date/Time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"Run Duration: {duration_seconds:.2f} seconds ({duration_seconds / 60:.2f} minutes)\n\n")
            
            log_file.write("INPUT CONSTANTS:\n")
            log_file.write(f"  NUM_BLOBS_THETA = {NUM_BLOBS_THETA}\n")
            log_file.write(f"  NUM_LAYERS_Z = {NUM_LAYERS_Z}\n")
            log_file.write(f"  NUM_R_BINS = {NUM_R_BINS}\n")
            log_file.write(f"  Total 3D grid points per galaxy = {NUM_R_BINS * NUM_BLOBS_THETA * NUM_LAYERS_Z/1e6:.2f} million\n")
            log_file.write(f"  R_GRID_SCALE_FACTOR = {R_GRID_SCALE_FACTOR}\n")
            log_file.write(f"  USE_EXPONENTIAL_DISK = {USE_EXPONENTIAL_DISK}\n")
            log_file.write(f"  add_core_gas_central_density = {add_core_gas_central_density}\n")
            log_file.write(f"  add_core_gas_beta = {add_core_gas_beta}\n")
            log_file.write(f"  add_real_cgm_central_density = {add_real_cgm_central_density}\n")
            log_file.write(f"  cgm_beta = {cgm_beta}\n")
            log_file.write(f"  MIN_DENSITY_FOR_POWER = {MIN_DENSITY_FOR_POWER}\n")
            log_file.write(f"  Upsilon_disk = {Upsilon_disk}\n")
            log_file.write(f"  Upsilon_bulge = {Upsilon_bulge}\n\n")


            log_file.write("--- GLOBAL OPTIMIZATION RESULTS ---\n")
            log_file.write(f"  USE_LINEAR_ERROR = {USE_LINEAR_ERROR}\n")
            log_file.write(f"  USE_INTERPOLATED_FIT = {USE_INTERPOLATED_FIT}\n")
            log_file.write(f"  Best Global P: {best_global_P:.4f} W\n")
            
            number_of_points = np.sum([len(galaxies[gal]['Vobs']) for gal in galaxies]) 
            number_of_grid_points = np.sum([len(galaxies[gal]['Vobs_grid']) for gal in galaxies]) 
            
            log_file.write(f"  DM Chi^2 Total (Obs): {dm_chi2_total:.2f}\n")
            log_file.write(f"  DM Linear Total Error (Weighted Obs): {dm_linear_total_err:.2f}\n")
            log_file.write(f"  DM Linear Total Error (Unweighted Obs): {dm_linear_total:.2f} km/s, per point: {dm_linear_total/number_of_points:.2f} km/s\n\n")
            
            log_file.write(f"  DM Chi^2 Total (Interpolated Grid): {dm_grid_chi2_total:.2f}\n")
            log_file.write(f"  DM Linear Total Error (Interpolated Grid): {dm_grid_linear_total:.2f} km/s, per point: {dm_grid_linear_total/number_of_grid_points:.2f} km/s\n\n")
            
            log_file.write("--- MOND (Simple) PREDICTION ERRORS ---\n")
            log_file.write(f"  MOND Chi^2 Total: {mond_chi2_total:.2f}\n")
            log_file.write(f"  MOND Linear Total Error (Weighted): {mond_linear_total_err:.2f}\n")
            log_file.write(f"  MOND Linear Total Error (Unweighted): {mond_linear_total:.2f} km/s, per point: {mond_linear_total/number_of_points:.2f} km/s\n\n")
            log_file.write(f"  MOND Linear Total Error via grid (Unweighted): {mond_linear_total_grid:.2f} km/s, per point: {mond_linear_total_grid/number_of_grid_points:.2f} km/s\n\n")
            
            log_file.write("--- GLOBAL STATISTICAL BREAKDOWN ---\n")
            log_file.write(f"  Sum Baryon Mass to R_max: {global_stats['Global_R_max_Baryon_Mass']:.2e} M_sun\n")
            log_file.write(f"  Sum DM to R_max: {global_stats['Global_R_max_DM_Mass']:.2e} M_sun\n")
            log_file.write(f"  Average (Sum DM / Sum Baryons): {global_stats['DM_AllBaryons_Ratio']:.2f}\n")
            log_file.write(f"  Overall DM Ratio: {global_stats['Overall_R_max_DM_Ratio']:.2f}\n\n")
            log_file.write(f"  DM to Lelli HI Gas Average to R_max: {global_stats['Overall_DM_HILelliGas_Ratio_to_RMax']:.2f}\n\n")
            log_file.write(f"  Unobserved gas fraction average: {global_stats['Unobserved_gas_fraction']:.2f}\n\n")
            

            log_file.write("--- GALAXY STATISTICAL BREAKDOWN ---\n")
            log_file.write("Column Legend:\n")
            log_file.write("  name: Galaxy identifier\n")
            log_file.write("  M_b: Total baryon mass [M_sun] (TL + HIMass, assuming Upsilon = 1 - Lelli 2016 data)\n")
            log_file.write("  TL: Stellar luminosity-derived mass [fraction of M_b]\n")
            log_file.write("  HIMass: Ne utral Hydrogen mass from literature [fraction of M_b]\n")
            log_file.write("  R_max Gas Mass: Integrated gas disk mass from 3D density grid * 1.33 [fraction of M_b]\n")
            log_file.write("  R_max:Total Gas Mass + TL mass: Combined integrated gas and stellar mass [fraction of M_b]\n")
            log_file.write("  R_max: DM Mass: Integrated Dark Mass via theory formula [fraction of M_b]\n")
            log_file.write("  R_max: DM Mass/M_b: Ratio of Dark Mass to Total Baryons\n")
            log_file.write("  R_max: DM Mass/Total Gas: Ratio of Dark Mass to Integrated Gas Mass\n")
            log_file.write("  R_max: DM Mass/Total Baryons: Ratio of Dark Mass to all baryons\n")
            log_file.write("  analytic:  halo gas added to the baryons in DM calculation, to R_200 [M_sun]\n")
            log_file.write("  analytic:  halo dm mass added (no truncation, to R_200 [M_sun])\n")
            log_file.write("  analytic:  halo gas mass added (truncated at R_max [M_sun])\n")
            log_file.write("  analytic:  halo dm mass added (truncated at R_max [M_sun])\n\n")
            
            
            header = f"{'name':<14} {'M_b(sol)':<11} {'TL(f)':<9} {'HIMass(f)':<9} {'RMaxGas(f)':<9} {'DM/M_b':<9} {'DM/RMaxGs':<9} {'DM/RMaxB':<9} {'r_core_kpc':<9} {'gas(r_200)':<11} {'dm(r_200)':<11} {'gas(R_max)':<11} {'dm(R_max)':<11} {'r_200':<9} {'r_max':<9}\n"
            log_file.write(header)
            log_file.write("-" * len(header) + "\n")
            
            for row in galaxy_stats:
                log_file.write(f"{row['name']:<14} {row['M_b']:<11.2e} {row['TL']:<9.3f} {row['HIMass']:<9.3f} "
                               f"{row['R_max_Gas_Mass']/row['M_b']:<9.3f}  "
                               f"{row['DM_Mb_Ratio']:<9.3f} {row['DM_Gas_Ratio']:<9.3f} {row['DM_AllBaryons_Ratio']:<9.3f} {row['r_core_kpc']:<9.3f} "
                               f"{row['totR200_gas']:<11.2e} {row['totR200_dm']:<11.2e} {row['totRMax_gas']:<11.2e} {row['totRMax_dm']:<11.2e} {row['r_200']:<9.0f} {row['r_max']:<9.1f}\n")
            
            log_file.write("========================================================================================================\n\n")

        dest_filename = os.path.join(output_dir, "run_log.txt")
        with open(dest_filename, "a", encoding="utf-8") as dest, open(log_filename, "r", encoding="utf-8") as src:
            dest.write(src.read())


    def generate_mass_scatter_plots(output_dir, stats):
        m_b = np.array([s['M_b'] for s in stats])
        gas_plus_tl = np.array([s['Gas_Plus_TL'] for s in stats])
        tot_gas = np.array([s['R_max_Gas_Mass'] for s in stats])
        dm_mass = np.array([s['R_max_DM_Mass'] for s in stats])
        
        plt.figure(figsize=(8, 6))
        valid = (m_b > 0) & (dm_mass > 0)
        plt.scatter(m_b[valid], dm_mass[valid], alpha=0.6, color='blue', edgecolors='k', linewidth=0.5)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('R_max Baryon Mass $M_b$ [$M_{\\odot}$]', fontsize=11)
        plt.ylabel('R_max Dark Mass [$M_{\\odot}$]', fontsize=11)
        plt.title('Baryon Mass $M_b$ vs Dark Mass (to R_max)', fontsize=12)
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.savefig(os.path.join(output_dir, "mb_vs_R_max_DM_Mass.png"), bbox_inches='tight', dpi=200)
        plt.close()
        
        plt.figure(figsize=(8, 6))
        valid = (gas_plus_tl > 0) & (dm_mass > 0)
        plt.scatter(gas_plus_tl[valid], dm_mass[valid], alpha=0.6, color='darkgreen', edgecolors='k', linewidth=0.5)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('R_max Gas Mass + Stellar Mass (TL) [$M_{\\odot}$]', fontsize=11)
        plt.ylabel('R_max Dark Mass [$M_{\\odot}$]', fontsize=11)
        plt.title('Gas + Stellar Mass vs Total Dark Mass (to R_max)', fontsize=12)
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.savefig(os.path.join(output_dir, "gas_plus_tl_vs_R_max_DM_Mass.png"), bbox_inches='tight', dpi=200)
        plt.close()
        
        plt.figure(figsize=(8, 6))
        valid = (tot_gas > 0) & (dm_mass > 0)
        plt.scatter(tot_gas[valid], dm_mass[valid], alpha=0.6, color='purple', edgecolors='k', linewidth=0.5)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('R_max Integrated Gas Mass [$M_{\\odot}$]', fontsize=11)
        plt.ylabel('R_max Dark Mass [$M_{\\odot}$]', fontsize=11)
        plt.title('Gas Mass vs Total Dark Mass (to R_max)', fontsize=12)
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.savefig(os.path.join(output_dir, "gas_mass_vs_R_max_DM_Mass.png"), bbox_inches='tight', dpi=200)
        plt.close()


    def generate_density_heatmaps(galaxy_dict, best_P):
        print("\nGenerating 2D Gas and DM Heatmaps...")
        for gal in galaxy_dict:
            g = galaxy_dict[gal]
            R_kpc = g['R_grid_kpc']
            Z_kpc = g['Z_grid_kpc']
            gas_density_m3 = g['gas_density_slice_m3']
            
            gas_density_cm3 = gas_density_m3 / 1e6
            n_min_m3 = MIN_DENSITY_FOR_POWER*(100)**3 
            dm_mask = gas_density_m3 >= n_min_m3
            dm_density_kg_m3 = np.where(dm_mask, best_P * (1/c_SI**3) * (gas_density_m3**(2/3)), 0.0)
            dm_density_cm3 = dm_density_kg_m3 / (m_p * 1e6)
            
            R_mesh, Z_mesh = np.meshgrid(R_kpc, Z_kpc, indexing='ij')

            plt.figure(figsize=(6, 10))
            vmin = max(gas_density_cm3.max() * 1e-4, 1e-6)
            pcm = plt.pcolormesh(R_mesh, Z_mesh, gas_density_cm3, 
                                 shading='auto', cmap='viridis', 
                                 norm=mcolors.LogNorm(vmin=vmin, vmax=gas_density_cm3.max()))
            plt.colorbar(pcm, label='Gas Density (particles / cm$^3$)')
            plt.xlabel('Radius [x-axis] (kpc)')
            plt.ylabel('Height [z-axis] (kpc)')
            plt.title(f'{gal} 2D Gas Density (y=0 slice)')
            plt.savefig(os.path.join(output_gas_heatmaps, f"{gal}_gas_heatmap.png"), bbox_inches='tight', dpi=200)
            plt.close()

            plt.figure(figsize=(6, 10))
            vmin_dm = max(dm_density_cm3.max() * 1e-4, 1e-6)
            pcm = plt.pcolormesh(R_mesh, Z_mesh, dm_density_cm3, 
                                 shading='auto', cmap='plasma', 
                                 norm=mcolors.LogNorm(vmin=vmin_dm, vmax=dm_density_cm3.max()))
            plt.colorbar(pcm, label='DM Density Equivalent (protons / cm$^3$)')
            plt.xlabel('Radius [x-axis] (kpc)')
            plt.ylabel('Height [z-axis] (kpc)')
            plt.title(f'{gal} 2D Dark Mass Density (y=0 slice)\n$P={best_P:.2f}W$')
            plt.savefig(os.path.join(output_dm_heatmaps, f"{gal}_dm_heatmap.png"), bbox_inches='tight', dpi=200)
            plt.close()

    if DO_HEATMAPS:
        generate_density_heatmaps(galaxies, best_global_P)

    # ---------------------------------------------------------
    # RUN STATS COMPUTATION, LOGGING, AND SCATTER PLOTS
    # ---------------------------------------------------------
    print("\nComputing galaxy mass statistics, appending log file, and plotting mass comparisons...")

    galaxy_stats, global_stats = compute_galaxy_stats(galaxies, best_global_P)

    write_run_log(
        output_dir=output_dir,
        best_global_P=best_global_P,
        dm_chi2_total=dm_chi2_total,
        dm_linear_total_err=dm_linear_total_err,
        dm_linear_total=dm_linear_total,
        dm_grid_chi2_total=dm_grid_chi2_total,
        dm_grid_linear_total=dm_grid_linear_total,
        mond_chi2_total=mond_chi2_total,
        mond_linear_total_err=mond_linear_total_err,
        mond_linear_total=mond_linear_total,
        mond_linear_total_grid=mond_linear_total_grid,
        global_stats=global_stats,
        galaxy_stats=galaxy_stats
    )

    if output_graphs:
        # 5. Generate Output Plots
        print("\nGenerating individual galaxy plots and master summary...")
        all_V_obs = []
        all_V_pred = []

        for gal in galaxies:
            g = galaxies[gal]
            
            gas_to_use = 'Vgas'
            if add_core_gas_central_density > 0.0 or add_real_cgm_central_density > 0.0 or gal in supplied_gas_density:
                gas_to_use = 'Vgas_test'

            V_dm_final_obs_kms = np.sqrt(best_global_P * g['V_dm_sq_base_kms'])
            V_bar_sq_obs_kms = np.clip(g['Vdisk']**2 + g['Vbul']**2 + g[gas_to_use] * np.abs(g[gas_to_use]), 0, None)
            V_tot_final_pred_kms = np.sqrt(V_bar_sq_obs_kms + V_dm_final_obs_kms**2)
            all_V_obs.extend(g['Vobs'])
            all_V_pred.extend(V_tot_final_pred_kms)
            
            V_dm_final_grid_kms = np.sqrt(best_global_P * g['V_dm_sq_base_grid_kms'])
            V_bar_sq_grid_kms_tom = np.clip(g['Vdisk_grid']**2 + g['Vbul_grid']**2 + g['Vgas_test_grid_kms'] * np.abs(g['Vgas_test_grid_kms']), 0, None) 
            V_tot_final_grid_kms = np.sqrt(V_bar_sq_grid_kms_tom + V_dm_final_grid_kms**2)
            
            plt.figure(figsize=(9, 6))

            plt.plot(g['R_grid_kpc'], V_tot_final_grid_kms, 'r-', linewidth=2, label='baryons + dm')
            plt.plot(g['R_grid_kpc'], V_dm_final_grid_kms, 'm--', label=f'dark mass (dm)', linewidth=0.8)
            plt.plot(g['R_grid_kpc'], g['Vgas_test_grid_kms'], 'g', label="gas (as used)", linewidth=0.5)
            plt.plot(g['R_grid_kpc'], g['Vgas_test_inDisk_grid_kms'], 'cyan', label="gas (inDisk)", linestyle='dotted', linewidth=0.9)

            plt.plot(g['R'], g['Vgas'], 'g', label='gas (Lelli)', linestyle='dotted', linewidth=0.7)
            plt.plot(g['R'], g['Vdisk'], 'violet', label='disk (Lelli)', linestyle='dashdot', linewidth=0.7)
            if np.any(g['Vbul'] > 0):
                plt.plot(g['R'], g['Vbul'], 'red', label='bulge (Lelli)', linestyle='dashed', linewidth=0.7)

            lelli_all_baryon_v = np.sqrt(np.clip(g['Vdisk']**2 + g['Vbul']**2 + g['Vgas'] * np.abs(g['Vgas']), 0, None))
            plt.plot(g['R'], lelli_all_baryon_v, 'blue', label='baryons (Lelli)', linewidth=0.7)
            
            max_r = max(g['R'])
            plt.xlim(0.0, max_r + max_r*0.02) 
            plt.plot(g['R'], g['V_mond_kms'], 'c-.', linewidth=1, label='MOND')
        
            plt.errorbar(g['R'], g['Vobs'], yerr=g['eVobs'], fmt='ko', label='observed', capsize=1.5, capthick=1.0, markersize=2, elinewidth=0.6)
            
            plt.xlabel('Radius (kpc)')
            plt.ylabel('Velocity (km/s)')
            plt.title(f'{gal} Kinematics (Universal P = {best_global_P:.2f} W)')
            leg = plt.legend(fontsize="small")
            leg.get_frame().set_linewidth(0.0) 

            plt.grid(True, linewidth=0.5, alpha=0.7)
            
            plt.savefig(os.path.join(output_galaxies, f"{gal}_dm_mass.png"), bbox_inches='tight', dpi=300)
            plt.close()

        plt.figure(figsize=(9, 9))
        plt.scatter(all_V_pred,all_V_obs, alpha=0.3, edgecolors='none', c='blue')
        max_val = max(max(all_V_obs), max(all_V_pred))
        plt.plot([0, max_val], [0, max_val], 'k--', linewidth=2, label='Newton')

        plt.xlabel('Predicted Velocity (km/s)', fontsize=12)
        plt.ylabel('Observed Velocity (km/s)', fontsize=12)
        plt.title(f'Global Fit: Dark Mass Theory on SPARC Dataset\n(Total Points = {len(all_V_obs)}, Universal P = {best_global_P:.2f} W)', fontsize=14)
        plt.legend(fontsize="small")
        plt.grid(True)
        plt.axis('equal')
        plt.xlim(0, max_val + 20)
        plt.ylim(0, max_val + 20)

        plt.savefig(os.path.join(output_dir, "global_fit_vs_obs.png"), bbox_inches='tight', dpi=300)
        plt.close()

        print("\nGenerating global acceleration scatter plot...")

        all_g_obs = []
        all_g_pred = []

        for gal in galaxies:
            g = galaxies[gal]
            R_m = g['R'] * kpc_to_m
            valid = R_m > 0 
            
            gas_to_use = 'Vgas'
            if add_core_gas_central_density > 0.0 or add_real_cgm_central_density > 0.0 or gal in supplied_gas_density:
                gas_to_use = 'Vgas_test'

            V_dm_final_sq = best_global_P * g['V_dm_sq_base_kms']
            V_bar_sq_kms = np.clip(g['Vdisk']**2 + g['Vbul']**2 + np.sign(g[gas_to_use])*(g[gas_to_use]**2), 0, None)
            
            V_tot_final_sq_ms = (V_bar_sq_kms + V_dm_final_sq) * (1000.0**2)
            V_obs_sq_ms = (g['Vobs'] * 1000.0)**2
            
            g_obs = V_obs_sq_ms[valid] / R_m[valid]
            g_pred = V_tot_final_sq_ms[valid] / R_m[valid]
            
            all_g_obs.extend(g_obs)
            all_g_pred.extend(g_pred)

        g_pred_arr = np.asarray(all_g_pred)
        g_obs_arr = np.asarray(all_g_obs)
        valid_pts = (g_pred_arr > 0) & (g_obs_arr > 0)
        x_pts = g_pred_arr[valid_pts]
        y_pts = g_obs_arr[valid_pts]

        plt.figure(figsize=(9, 9))
        try:
            log_x = np.log10(x_pts)
            log_y = np.log10(y_pts)
            density = gaussian_kde(np.vstack([log_x, log_y]))(np.vstack([log_x, log_y]))
            idx = density.argsort()
            x_sorted, y_sorted, d_sorted = x_pts[idx], y_pts[idx], density[idx]

            sc = plt.scatter(x_sorted, y_sorted, c=d_sorted, cmap='viridis', s=12, alpha=0.7, edgecolors='none')
            cbar = plt.colorbar(sc, fraction=0.046, pad=0.04)
            cbar.set_label('Point Density (log-space KDE)', fontsize=11)
        except Exception:
            plt.scatter(x_pts, y_pts, alpha=0.3, edgecolors='none', c='blue')

        plt.xscale('log')
        plt.yscale('log')

        # min_val = 10**np.floor(np.log10(min(min(y_pts), min(x_pts))))
        # max_val = 10**np.ceil(np.log10(max(max(y_pts), max(x_pts))))
        min_val = min(min(y_pts), min(x_pts))
        max_val = max(max(y_pts), max(x_pts))
        min_val = min_val / 1.1
        max_val = max_val * 1.1
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.2, label='Newton')

        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)
        plt.axis('square')

        plt.ylabel('Observed Acceleration $g_{obs}$ (m/s$^2$)', fontsize=12)
        plt.xlabel('Predicted Acceleration $g_{pred}$ (m/s$^2$)', fontsize=12)
        plt.title(f'Global Fit: Dark Mass vs Observed Acceleration\n(Total Points = {len(all_g_obs)}, Universal P = {best_global_P:.2f} W)', fontsize=14)
        plt.legend(fontsize="small", loc="upper left")
        plt.grid(True, which="both", ls="--", alpha=0.5)

        plt.savefig(os.path.join(output_dir, "global_predicted_accel_vs_obs.png"), bbox_inches='tight', dpi=200)
        plt.close()



        generate_mass_scatter_plots(output_dir, galaxy_stats)

    print(f"All operations completed successfully. Files and plots saved to '{output_dir}'.")


    # return the correct per point stat, depending on the settings 
    returned_fit = 0
    if USE_INTERPOLATED_FIT:
        number_of_grid_points = np.sum([len(galaxies[gal]['Vobs_grid']) for gal in galaxies])
        if USE_LINEAR_ERROR:
            returned_fit = dm_grid_linear_total / number_of_grid_points
        else:
            returned_fit = dm_grid_chi2_total / number_of_grid_points
    else:
        number_of_points = np.sum([len(galaxies[gal]['Vobs']) for gal in galaxies])
        if USE_LINEAR_ERROR:
            returned_fit = dm_linear_total / number_of_points
        else:
            returned_fit = dm_chi2_total / number_of_points

    if return_galaxies:
        return returned_fit, best_global_P, global_stats['Unobserved_gas_fraction'], galaxies
    return returned_fit, best_global_P, global_stats['Unobserved_gas_fraction']

# core gas density correction: I have determined that I need to add (9.5 - log10(TL))*0.10 to the core gas density.
# Determined by fitting.
def core_gas_density_correction(total_mass):
    core_gas_density_correction = (9.5 - np.log10(total_mass)) * 0.09
    return core_gas_density_correction



if __name__ == "__main__":
    # You can adjust these default values as needed when running from the terminal
    err_per_point, best_power, unobserved_gas_fraction = run_sparc_dm_brute_cgm(
        add_core_gas_central_density=0.6, #0.60, #0.60
        add_core_gas_beta=1.93, 
        r_core_factor_gas=2.5, 
        add_real_cgm_central_density=0.00,  # 0.026
        r_core_factor_cgm=0.20, 
        cgm_beta = 0.50,
        num_minutes=3, 
        output_graphs=True,
        power_override = 0.0,
        min_mass = 0,
        max_mass = 1e99
    )
    print(f"Done.")
    print(f"Err per point: {err_per_point}")
    print(f"Best power: {best_power}\n")
    print(f"Unobserved gas fraction: {unobserved_gas_fraction}\n\n\n")