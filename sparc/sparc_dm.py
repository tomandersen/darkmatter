from PIL import ImagePalette
import numpy as np
from matplotlib import mathtext
import os
import matplotlib.pyplot as plt

G = 4.3009e-6             # Gravitational constant: kpc * (km/s)^2 / M_sun
c = 299792458.0           # Speed of light: m/s
MSUN_TO_KG = 1.989e30     # Solar masses to kilograms
MSUN_KPC3_TO_KG_M3 = 6.77e-29 # Mass density conversion
METRE_PER_KPC = 3.086e19
G_SI = 6.67430e-11        # Gravitational constant: m^3 kg^-1 s^-2
PROTON_MASS_KG = 1.67e-27   # Mass of a proton in kilograms

Power = 30.0 #Guess Watts per particle. You heard it here first, people!
DENSITY_THRESHOLD=0.03

CONVERT_ACCELERATION =  3.2408e-14 #Given v in km/sec, R in kpc, g = CONVERT_ACCELERATION*v^2/R
REJECT_LOW_Q_AND_LOW_INCL = False # does not make much of a difference
ADD_EXTRA_GALAXIES = False # i added a galaxy for fun. (Malin1) 

#reads the SPARC asci i data as from the web site. 
def read_ascii(data_file):
    galaxies = {}

    with open(data_file, 'r') as f:
        lines = f.readlines()
        
    # Data starts after line 25
    for line in lines[25:]:
        parts = line.strip().split()
        if not parts:
            continue
            
        name = parts[0]
        # Columns: distance R Vobs e_Vobs Vgas Vdisk Vbul SBdisk SBbul
        R = float(parts[2])
        Vobs = float(parts[3])
        e_Vobs = float(parts[4])
        Vgas = float(parts[5])
        Vdisk = float(parts[6])
        Vbul = float(parts[7])
        
        if name not in galaxies:
            galaxies[name] = {'R': [], 'Vobs': [], 'e_Vobs': [], 'Vgas': [], 'Vdisk': [], 'Vbul': []}
            
        galaxies[name]['R'].append(R)
        galaxies[name]['Vobs'].append(Vobs)
        galaxies[name]['e_Vobs'].append(e_Vobs)
        galaxies[name]['Vgas'].append(Vgas)
        galaxies[name]['Vdisk'].append(Vdisk)
        galaxies[name]['Vbul'].append(Vbul)
        
    print(f"Loaded {len(galaxies)} galaxies. Generating plots...")
    return galaxies

# Read galaxy properties from the separate structural data file
#    1- 11 A11    ---           Galaxy  Galaxy Name
#   12- 13 I2     ---           T       Hubble Type (1)
#   14- 19 F6.2   Mpc           D       Distance
#   20- 24 F5.2   Mpc         e_D       Mean error on D
#   25- 26 I2     ---         f_D       Distance Method (2)
#   27- 30 F4.1   deg           Inc     Inclination
#   31- 34 F4.1   deg         e_Inc     Mean error on Inc
#   35- 41 F7.3   10+9solLum    L[3.6]  Total Luminosity at [3.6]
#   42- 48 F7.3   10+9solLum  e_L[3.6]  Mean error on L[3.6]
#   49- 53 F5.2   kpc           Reff    Effective Radius at [3.6]
#   54- 61 F8.2   solLum/pc2    SBeff   Effective Surface Brightness at [3.6]
#   62- 66 F5.2   kpc           Rdisk   Disk Scale Length at [3.6]
#   67- 74 F8.2   solLum/pc2    SBdisk  Disk Central Surface Brightness at [3.6]
#   75- 81 F7.3   10+9solMass   MHI     Total HI mass
#   82- 86 F5.2   kpc           RHI     HI radius at 1 Msun/pc2
#   87- 91 F5.1   km/s          Vflat   Asymptotically Flat Rotation Velocity
#   92- 96 F5.1   km/s        e_Vflat   Mean error on Vflat
#   97- 99 I3     ---           Q       Quality Flag (3)
#  100-113 A14    ---           Ref.    References for HI and Ha data (4)

def read_galaxy_properties(file_path, galaxies):
    properties = {}
    # The data in the file starts around line 99
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Data starts after line 98
    for line in lines[98:]:
        parts = line.strip().split()
        if not parts:
            continue
            
        name = parts[0]
        # Columns: see above comment, we only need Rdisk (12th column, so 11 index)
        Rdisk = float(parts[11])
        incl = float(parts[5])
        quality = float(parts[17])
        #print(name, incl, "q = ", quality)
        galaxies[name]['R_d'] = Rdisk
        galaxies[name]['inclination'] = incl
        galaxies[name]['Quality'] = quality

    # cut down the galaxies like the 2016 paperSPARC: MASS MODELS FOR 175 DISK GALAXIES WITH SPITZER PHOTOMETRY AND ACCURATE ROTATION CURVES.
    # for 176 galaxies with quality flag Q >= 2 (146 LTGs and 30 ETGs).
    # and remove spirals with inclination <= 30 degrees.
    if REJECT_LOW_Q_AND_LOW_INCL:
        galaxies_rejected = []    
        for name, data in list(galaxies.items()):
            if data['inclination'] >= 30 and data['Quality'] < 3:
                pass #keep galaxy
            else: 
                galaxies_rejected.append(name)
                del galaxies[name]   # Remove the galaxy from the original dictionary  
    
    # print(f"Rejected {len(galaxies_rejected)}, named: {galaxies_rejected}")
    return galaxies

def add_extra_galaxies(galaxies):
    #Malin1 from the paper Exploring the stellar streams and satellites around the giant low  surface brightness galaxy Malin 1
    malin = {}
    malin["R"] =    [2,   5,    10,       15,    20,    30,    40,    50,   60,   70,   80,   90,       100.0]
    malin["Vobs"] = [310, 280,   265,     250,   230,   220, 212,   209,    204,  201,  198,   194,   191.0] 
    malin["e_Vobs"]=[5,    5,     5,       5,     5,     5,   5,     5,      5,    5,    5,     5,      5.0]     
    malin["Vgas"] = [3,    7,     8,       9,     10,    15,   16,    19,     22,   24,   26,    28,    32.0]     
    malin["Vdisk"] = [125,  158,   150,     142,  119,  101,   80,    74,     68,   61,   57,    55,    53.0]   
    malin["Vbul"] = [275,  208,    150,     140,  112,  94,   75,    69,     64,   59,   55,    52,    50.0]     
    

# futz with gas
    # new_gas = []
    # for vg in malin["Vgas"]:
    #     new_gas.append(2.0*vg)
    # malin["Vgas"] = new_gas 

    # no cheating.. but is there better data? malin["Vgas"] = [3,    7,     12,       20,     30,    26,   24,    23,     22,   24,   26,    28,    32.0]     
  
 
    malin['R_d'] = 20
    malin['inclination'] = 40.0
    malin['Quality'] = 1
    galaxies['Malin1'] = malin

    return galaxies


def get_sparc_galaxy_scale_height(radius_kpc, R_d):
    """
    Calculates the vertical disk scale height (thickness) of a SPARC galaxy.
    
    Parameters:
    radius_kpc (float): The target radius in kpc where you want the thickness.
                        (Note: Standard models assume uniform thickness, so 
                        the value remains constant regardless of the radius).
    R_d (float): The radial disk scale length of the specific galaxy in kpc 
                 (retrieved from the SPARC structural data tables).
                 
    Returns:
    float: The vertical disk scale height (z_d) in kpc.
    """
    # Empirically derived scale-height relation used for SPARC mass modeling
    z_d = 0.196 * (R_d ** 0.633)
    
    return z_d


# model of dark matter - depends on gas density and galaxy thickness at r.
# takes in arrays of R, vGas, disk scale factor, returns 
def dm_model(pw, rs, Vgas, R_d, name, densities, dm_densities, densities_r):
    r_prev = 0
    v_prev = 0
    Vdm = []
    # I need the density of the gas in particles per cm^3 for each radius.
    # To do this, work out the enclosed mass of gas at each R, then divide by the volume
    # of a cylinder of radius R and height 2*z_d. 
    # The mass could turn out to be negative, in which case I take a low number for density.
    
    # I wonder if I should get the enclosed gas masses, as an array, then clean it up   
    # (take out negative values, and also smooth it.  smooth with a boxcar of width 3).?.
    mass_encl_prev_kg = 0
    r_m_prev = 0
    for r, Vg in zip(rs, Vgas):
        v_m_per_sec = Vg * 1000.0
        if v_m_per_sec < 0:
            v_m_per_sec = 0 # not the best strategy here. See UGC07323 for example which has neg gas contributions AFTER a few positive ones.
        r_m = r * METRE_PER_KPC
        z_d = get_sparc_galaxy_scale_height(r, R_d)
        thickness_m = 2 * z_d * METRE_PER_KPC
        mass_encl_kg = v_m_per_sec**2 * r_m / G_SI
        
        
        z_extent_kpc = get_sparc_galaxy_scale_height(r, R_d)
        thickness_m = 2 * z_extent_kpc * METRE_PER_KPC

        shell_volume_m3 = thickness_m*(np.pi*r_m**2 - np.pi*r_m_prev**2)
        shell_volume_cm3 = shell_volume_m3 * 1e6
        
        net_mass_shell_kg = mass_encl_kg - mass_encl_prev_kg
        num_baryons_shell = net_mass_shell_kg / PROTON_MASS_KG
        baryon_density_n_cm3 = num_baryons_shell / shell_volume_cm3
        if baryon_density_n_cm3 < DENSITY_THRESHOLD: 
            #print(f"{name}: Low or negative Baryon density {baryon_density_n_cm3} cm^-3 at r {r} kpc, R_d {R_d} kpc, setting to {DENSITY_THRESHOLD}")
            baryon_density_n_cm3 = DENSITY_THRESHOLD
        num_baryons_shell = shell_volume_cm3 * baryon_density_n_cm3 #recalc incase of underflow
        net_mass_shell_kg = num_baryons_shell*PROTON_MASS_KG # redo in case we need it

        densities.append(baryon_density_n_cm3)
        densities_r.append(r)
        dm_density_baryonspercc = 0.0 # initialize

        if baryon_density_n_cm3 > 0:
            #average distance between gas particles
            d_avg_gas_m = np.cbrt(1.0 / baryon_density_n_cm3)/100 # cm to m
            dm_per_particle = (pw*d_avg_gas_m/c)*(1/c**2) #Power in watts times dist/c is energy, then 1/c^2 is mass in kg
            dm_inShell_kg = num_baryons_shell*dm_per_particle
            dm_density_baryonspercc = dm_per_particle/PROTON_MASS_KG*baryon_density_n_cm3


            #r_av = (r_m + r_m_prev)/2
            V_dm_km_per_sec = np.sqrt(G_SI*dm_inShell_kg / (r_m + 1e-10))/1000 # divide by 1000 to get km/sec
            
            #only (my weirdo) dark matter contribution here.
            Vdm.append(V_dm_km_per_sec)
        else:
            Vdm.append(0.0)
        
        dm_densities.append(dm_density_baryonspercc)
        
        mass_encl_prev = mass_encl_kg
        r_m_prev = r_m

    return Vdm

def mond_model(rs, Vgas, Vdisk, Vbul, name):
    a0 = 3700 # MOND acceleration parameter: (km/s)^2 / kpc
    VMOND = []
    root2 = np.sqrt(2.0)
    # see the image MOND_velocity_calc - bottom equation...
    for r, Vg, Vd, Vb in zip(rs, Vgas, Vdisk, Vbul):
        V_N_sq = Vg**2 + Vd**2 + Vb**2
        root_factor = np.sqrt(1.0 + (2.0*a0*r/V_N_sq)**2)
        V_mond = np.sqrt(V_N_sq/root2 * np.sqrt((1 + root_factor)))
        VMOND.append(V_mond)

    return VMOND

def run_model(galaxies, pw, densities, dm_densities, densities_r):
    fit_params = {}
    fit_params['total_linear_error'] = 0  
    fit_params['total_chi_sq_error'] = 0
    fit_params['total_linear_error_MOND'] = 0  
    fit_params['total_chi_sq_error_MOND'] = 0
    for name, data in galaxies.items():
        R = data['R']
        Vobs = data['Vobs']
        e_Vobs = data['e_Vobs']
        Vgas = data['Vgas']
        Vdisk = data['Vdisk']
        Vbul = data['Vbul']
        R_d = data['R_d']

        # Calculate sums and DM proxy
        V_dm = dm_model(pw, R, Vgas, R_d, name, densities, dm_densities, densities_r)
        galaxies[name]['V_dm'] = V_dm

        # the formula to add velocities is sqrt(g^2 + d^2 + b^2)
        Vtot_baryons = [np.sqrt(g**2 + d**2 + b**2) for g, d, b in zip(Vgas, Vdisk, Vbul)]
        galaxies[name]['Vtot_baryons'] = Vtot_baryons

        Vtot_all = [np.sqrt(g**2 + d**2 + b**2 + dm**2) for g, d, b, dm in zip(Vgas, Vdisk, Vbul, V_dm)]
        galaxies[name]['Vtot_all'] = Vtot_all

        V_MOND = mond_model(R, Vgas, Vdisk, Vbul, name)
        galaxies[name]['V_MOND'] = V_MOND

        #record errors - ignore the error on the V_obs?? not sure 
        for v_tot_all, v_obs, v_err, v_MOND in zip(Vtot_all, Vobs, e_Vobs, V_MOND):
            fit_params['total_linear_error'] += np.abs(v_tot_all - v_obs)     #/v_err)
            fit_params['total_chi_sq_error'] += (v_tot_all - v_obs)**2      #/v_err**2)
            fit_params['total_linear_error_MOND'] += np.abs(v_MOND - v_obs)     #/v_err)
            fit_params['total_chi_sq_error_MOND'] += (v_MOND - v_obs)**2      #/v_err**2)
                
    return galaxies, fit_params

def bestFit(galaxies, initialPower):
    densities = []
    dm_densities = []
    densities_r = []


    numSteps = 1000
    step = initialPower/numSteps*5
    pw = step
    print(f'bestFit starting power: {pw}')

    lowest_linear_err = 1e99
    lowest_chi_sq_err = 1e99
    lowest_linear_err_MOND = 1e99
    lowest_chi_sq_err_MOND = 1e99
    lowest_linear_err_power_MOND = 0
    lowest_chi_sq_err_power_MOND = 0
    for count in range(numSteps):
        galaxies, fit_params = run_model(galaxies, pw, densities, dm_densities, densities_r)
        linear_err = fit_params['total_linear_error']
        chi_sq_err = fit_params['total_chi_sq_error']
        linear_err_MOND = fit_params['total_linear_error_MOND']
        chi_sq_err_MOND = fit_params['total_chi_sq_error_MOND']
        if linear_err < lowest_linear_err:
            lowest_linear_err = linear_err
            lowest_linear_err_power = pw
        if chi_sq_err < lowest_chi_sq_err:
            lowest_chi_sq_err = chi_sq_err
            lowest_chi_sq_err_power = pw
        if linear_err_MOND < lowest_linear_err_MOND:
            lowest_linear_err_MOND = linear_err_MOND
            lowest_linear_err_power_MOND = pw
        if chi_sq_err_MOND < lowest_chi_sq_err_MOND:
            lowest_chi_sq_err_MOND = chi_sq_err_MOND
            lowest_chi_sq_err_power_MOND = pw
        pw += step
        # if count % 40 == 0:
        #     print(f'step: {count}, power: {pw}, linear error: {linear_err}, chi_sq error: {chi_sq_err}')

    print(f'Best fit Power for linear error: {lowest_linear_err_power}, fit: {lowest_linear_err}')
    print(f'Best fit Power for chi_sq error: {lowest_chi_sq_err_power}, fit: {lowest_chi_sq_err}')
    print(f'Best fit chi_sq MOND: {lowest_chi_sq_err_MOND}')
    print(f'Best fit linear MOND: {lowest_linear_err_MOND}')
    return lowest_linear_err_power


def main():
     
    # We will read the file and skip the first 25 lines (headers)
    galaxies = read_ascii('./sparc/MassModels_Lelli2016c.mrt')
    
    galaxies = read_galaxy_properties('./sparc/SPARC_Lelli2016c.mrt.txt', galaxies)


    # keep track of all gas densities calculated, at each R
    densities = []
    dm_densities = []
    densities_r = []

 
    # A note on velocities from stars. We measure luminousity, but want mass. So we use a constant 
    # Upsilon of, say 1/2 for the mass to luminousity ratio for stars. Since velocity is a sqrt affair
    # then a constant Upsilon just scales things by a factor of sqrt(1/2).
    # See the 2016 paper "SPARC: MASS MODELS FOR 175 DISK GALAXIES WITH SPITZER PHOTOMETRY AND ACCURATE ROTATION CURVES."
    Upsilon_disk = 0.5
    Upsilon_bulge = 0.7 
    
    # the data in the file is listed with Upsilon = 1 (stellar mass to luminousity ratio)
    # but we want to plot the mass, so we need to multiply by Upsilon. We can do this by 
    # multiplying the velocity by sqrt(Upsilon).
    sqrt_Upsilon_disk = np.sqrt(Upsilon_disk)
    sqrt_Upsilon_bulge = np.sqrt(Upsilon_bulge)
    for name, data in galaxies.items():
        Vdisk = [v * sqrt_Upsilon_disk for v in data['Vdisk']]
        Vbul = [v * sqrt_Upsilon_bulge for v in data['Vbul']]
        galaxies[name]['Vdisk'] = Vdisk
        galaxies[name]['Vbul'] = Vbul

    #extra galaxies have Upsilon already in
    if ADD_EXTRA_GALAXIES:
        galaxies = add_extra_galaxies(galaxies)


    # determine best fits
    best_power = bestFit(galaxies, Power)
    best_power = float(int(best_power))
    print(f"Best fit Power: {best_power}") 
    galaxies, fit_params = run_model(galaxies, best_power, densities, dm_densities, densities_r)

    file_name_part = f"power-{int(best_power)}W-{DENSITY_THRESHOLD}-cc"
    out_dir = f'sparc/{file_name_part}/curves/'
    os.makedirs(out_dir, exist_ok=True)

    observed_vs_pred = []

    for name, data in galaxies.items():
        # add to stats of V_obs vs V_tot_all 
        for v_obs, v_err, v_tot_all, r_kpc in zip(data['Vobs'], data['e_Vobs'], data['Vtot_all'], data['R']):
            g_obs = CONVERT_ACCELERATION*v_obs**2/r_kpc
            g_predicted = CONVERT_ACCELERATION*v_tot_all**2/r_kpc
            observed_vs_pred.append([g_predicted, g_obs]) 
        
        plt.figure(figsize=(8, 6))
        
        # Plot Vobs with error bars (black dots)
        plt.errorbar(data['R'], data['Vobs'], yerr=data['e_Vobs'], fmt='k.', label='V_observed', capsize=3, elinewidth=0.5)
         
        # Plot components
        plt.plot(data['R'], data['Vgas'], 'g:', label='Gas', linewidth=1)
        plt.plot(data['R'], data['Vdisk'], 'r--', label='Disk', linewidth=1)
        plt.plot(data['R'], data['Vbul'], color='orange', linestyle='--', label='Bulge', linewidth=1)
        

        # do 
        # Plot total (blue dashed to distinguish from Vobs)
        plt.plot(data['R'], data['Vtot_baryons'], color='blue', linestyle='--', label='gas+disk+bulge', linewidth=1.)
        
        # Plot dm
        plt.plot(data['R'], data['V_dm'], color='cyan', linestyle=':', label='DM Model', linewidth=1)
        
        # the formula to add velocities is sqrt(g^2 + d^2 + b^2) 
        plt.plot(data['R'], data['Vtot_all'], color='purple', linestyle='-', label='gas+disk+bulge+dm', linewidth=2)

        # plot MOND
        plt.plot(data['R'], data['V_MOND'], color='grey', linestyle='-', label='MOND', linewidth=2)

        plt.title(f"{name} Rotation Curve")
        plt.xlabel('Radius (kpc)')
        plt.ylabel('Velocity (km/s)')
        
        # Set axis limits
        max_R = max(data['R'])
        # Find maximum velocity to set y-limit appropriately
        all_v = data['Vobs'] + data['Vgas'] + data['Vdisk'] + data['Vbul'] + data['Vtot_baryons'] + data['V_dm'] + data['Vtot_all']
        max_V = max([v for v in all_v])
        
        plt.xlim(0, max_R + 5)
        plt.ylim(0, max_V * 1.1)  # add 10% padding on top
        
        leg = plt.legend(fontsize="small")
        # Set the border outline thickness to 0
        leg.get_frame().set_linewidth(0.0)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        out_path = os.path.join(out_dir, f"{name}-{file_name_part}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()
        
    print(f"Finished generating {len(galaxies)} plots in {out_dir}")

    # now do global stat plots
    #------------------------------


    # 1. Plot the densities histogram
    # 2. Define logarithmically spaced bins
    # This creates 50 bins between 10^0 (1) and 10^4 (10000)
    bins = np.logspace(np.log10(min(densities) - DENSITY_THRESHOLD/10), np.log10(max(densities)), 50)
 
    # 3. Plot the histogram with the custom bins
    plt.hist([densities, dm_densities], bins=bins, color=['blue', 'red'], label=['Gas', 'Model DM'])
    plt.xscale('log')

    # 3. Add labels and title
    plt.title('SPARC Gas/DM Densities, all 175 Galaxies')
    plt.xlabel("Density, n/cm^3")
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(f'sparc/{file_name_part}/gas_densities-{file_name_part}.png', dpi=300)

    # Ok now make an X-Y scatter plot of the DM velocities and radiuses 
    # plot one dot for each densities, densities_r pair, make the densities on the y scale, make the y scale logarithimic
    # make  the x scale linear 
    plt.figure(figsize=(8, 6))
    plt.scatter(densities_r, densities, color='blue', s=4, alpha=0.10, label='Gas Density')
    plt.scatter(densities_r, dm_densities, color='red', s=4, alpha=0.10, label='DM Density')
    plt.xscale('linear')
    plt.yscale('log')
    plt.title('SPARC Gas & DM Model Densities, all 175 Galaxies')   
    plt.xlabel('Radius (kpc)')
    plt.ylabel('Density (n/cm^3)')
    plt.legend()
    plt.savefig(f'sparc/{file_name_part}/densities_scatter_combined-{file_name_part}.png', dpi=300)

    # Ok now make a graph showing oberved vs predicted as a scatter plot, log/log scale.
    # Graph 4 
    observed_vs_pred_np = np.array(observed_vs_pred)
    plt.figure(figsize=(8, 6))
    plt.scatter(observed_vs_pred_np[:, 0], observed_vs_pred_np[:, 1], color='blue', s=4, alpha=0.10, label='DM Model')
    plt.xscale('log')
    plt.yscale('log')
    plt.plot([1e-12, 1e-8], [1e-12, 1e-8], color="red", linewidth=0.6, label='Newton')

    #FIT LOG 
    log_x = np.log10(observed_vs_pred_np[:, 0])
    log_y = np.log10(observed_vs_pred_np[:, 1])
    slope, intercept = np.polyfit(log_x, log_y, 1)

    # Convert the fit back to the original space to extract the power-law parameters
    # log(y) = slope * log(x) + intercept  =>  y = (10^intercept) * x^slope
    amplitude = 10**intercept
    power = slope
    print(f"Fitted Equation: y = {amplitude:.3f} * x^{power:.3f}")

    # 3. Create a logarithmically spaced X-array
    # np.logspace takes the base-10 exponents as arguments (-12 and -8)
    x_line = np.logspace(-12, -8, 200)
    y_line = amplitude * x_line ** power
    # 4. Plotting with a log-scale X-axis
    plt.plot(x_line, y_line, color="blue", label="Polyfit Line")

    plt.title(f'SPARC DM Model Acceleration vs Observed, vs Newton')
    plt.xlabel('Acceleration (m/s^2)  Predicted')
    plt.ylabel('Acceleration (m/s^2)  Observed')
    plt.legend()
    plt.savefig(f'sparc/{file_name_part}/acceleration_scatter_combined-{file_name_part}.png', dpi=300)


    # Graph 5. Plot the the same Graph 4 above only with binned data  bins
    plt.figure(figsize=(8, 6))
    plt.xscale('log')
    plt.yscale('log')
    #plot the points as a background image
    plt.scatter(observed_vs_pred_np[:, 0], observed_vs_pred_np[:, 1], color='blue', s=3, alpha=0.10, label='DM Model')
    plt.plot([1e-12, 1e-8], [1e-12, 1e-8], color="red", linewidth=0.5, label='Newton')
    
    bins = np.logspace(-12, -8, 17)  # 16 bins, log-spaced, 4 per decade from 1e-12 to 1e-8
    # my data is scattered x, y pairs, observed_vs_pred_np[:, 0], observed_vs_pred_np[:, 1]
    # I think I need to manually create a new data series based on the bins: For each bin, look through all the data that 
    # matches the bin, for the mean and standard deviation of that data, and then plot that.  
    
    #after getting the binned data I will plot a scattergram of it, NOT a histogram. 
    bin_centers = np.sqrt(bins[:-1] * bins[1:])  # geometric mean of each bin edge pair

    x_data = observed_vs_pred_np[:, 0]  # predicted acceleration
    y_data = observed_vs_pred_np[:, 1]  # observed acceleration
    bin_means = []
    bin_stds = []
    bin_valid_centers = []
    for i in range(len(bins) - 1):
        mask = (x_data >= bins[i]) & (x_data < bins[i + 1])
        y_in_bin = y_data[mask]
        # zero_count = np.count_nonzero(y_in_bin < 1e-12)

        # if zero_count > 0:
        #     print("something is fishy here in graph 5 with bin", bins[i], "to", bins[i+1], "there are", zero_count, "zeros") 

        if len(y_in_bin) > 0:
            mean_val = np.mean(np.log10(y_in_bin))  
            std_val = np.std(y_in_bin, axis=None)
            print(f" mean_val, {mean_val}, std_val, {std_val} Bin [{bins[i]:.2e}, {bins[i+1]:.2e}]")
            # Trap suspiciously large std (> 10x the mean)
            if std_val < 0.2 * mean_val:
                print(f"[DIAG] Bin [{bins[i]:.2e}, {bins[i+1]:.2e}]: n={len(y_in_bin)}, mean={mean_val:.3e}, std={std_val:.3e}")
                print(f"       min={np.min(y_in_bin):.3e}, max={np.max(y_in_bin):.3e}, median={np.median(y_in_bin):.3e}")
                # Show the top outliers
                outlier_thresh = mean_val + 5 * std_val
                outliers = y_in_bin[y_in_bin > outlier_thresh]
                if len(outliers) > 0:
                    print(f"       Outliers (>{outlier_thresh:.3e}): {outliers}")
            bin_means.append(mean_val)
            bin_stds.append(std_val)
            bin_valid_centers.append(bin_centers[i])

    #exponentiate back the bin means and stgds
    bin_means = 10 ** np.array(bin_means)
    #bin_stds = 10 ** np.array(bin_stds)

    plt.errorbar(bin_valid_centers, bin_means, yerr=bin_stds,
                 fmt='rs', markersize=5, capsize=4, elinewidth=1,
                 label='Binned mean ± std')


    # 3. Add labels and title
    plt.title('Binned LTGs - Vobs vs V_predicted')
    plt.xlabel("Predicted (m/s^2)")
    plt.ylabel('Observed (m/s^2)')
    plt.legend()
    plt.savefig(f'sparc/{file_name_part}/binnedLTGs-{file_name_part}.png', dpi=300)


if __name__ == "__main__":
    main()
