from PIL import ImagePalette
from numpy import power
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

P = 30.0 #Watts per particle. You heard it here first, people!
DENSITY_THRESHOLD=0.0002


#reads the SPARC ascii data as from the web site. 
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
        #print(name, Rdisk)
        galaxies[name]['R_d'] = Rdisk
            
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
def dm_model(rs, Vgas, R_d, name):
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
            print(f"{name}: Low or negative Baryon density {baryon_density_n_cm3} cm^-3 at r {r} kpc, R_d {R_d} kpc, setting to {DENSITY_THRESHOLD}")
            baryon_density_n_cm3 = DENSITY_THRESHOLD
        num_baryons_shell = shell_volume_cm3 * baryon_density_n_cm3 #recalc incase of underflow
        net_mass_shell_kg = num_baryons_shell*PROTON_MASS_KG # redo in case we need it

        if baryon_density_n_cm3 >= DENSITY_THRESHOLD:
            #average distance between gas particles
            d_avg_gas_m = np.cbrt(1.0 / baryon_density_n_cm3)/100 # cm to m
            dm_per_particle = (P*d_avg_gas_m/c)*(1/c**2) #Power in watts times dist/c is energy, then 1/c^2 is mass in kg
            dm_inShell_kg = num_baryons_shell*dm_per_particle

            r_av = (r_m + r_m_prev)/2
            V_dm_km_per_sec = np.sqrt(G_SI*dm_inShell_kg / (r_av + 1e-10))/1000 # divide by 1000 to get km/sec
            
            #only (my weirdo) dark matter contribution here.
            Vdm.append(V_dm_km_per_sec)
        else:
            Vdm.append(0.0)
        
        mass_encl_prev = mass_encl_kg
        r_m_prev = r_m

    return Vdm

def mond_model(rs, Vgas, Vdisk, Vbul, name):
    G_kpc = 4.302e-6 # Gravitational constant: kpc (km/s)^2 / Msun
    a0 = 3700 # MOND acceleration parameter: (km/s)^2 / kpc
    VMOND = []
    root2 = np.sqrt(2.0)
    for r, Vg, Vd, Vb in zip(rs, Vgas, Vdisk, Vbul):
        V_N_sq = Vg**2 + Vd**2 + Vb**2
        root_factor = np.sqrt(1.0 + (2.0*a0*r/V_N_sq)**2)
        V_mond = np.sqrt(V_N_sq/root2 * np.sqrt((1 + root_factor)))
        VMOND.append(V_mond)

        # # sum up all the accelerations
        # a_tot_sq = G_kpc * (Vg**2/r + Vd**2/r + Vb**2/r)
        # a_tot = np.sqrt(a_tot_sq)
        # mu = a_tot/a0
        # a_MOND = a0 * mu / np.sqrt(1 + mu**2)
        # V_mond = np.sqrt(a_MOND*r)
        # VMOND.append(V_mond)

    return VMOND

def main():
    out_dir = 'sparc/curves/initial'
    os.makedirs(out_dir, exist_ok=True)
     
    # We will read the file and skip the first 25 lines (headers)
    galaxies = read_ascii('./sparc/MassModels_Lelli2016c.mrt')
    
    galaxies = read_galaxy_properties('./sparc/SPARC_Lelli2016c.mrt.txt', galaxies)

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
        R = data['R']
        Vobs = data['Vobs']
        e_Vobs = data['e_Vobs']
        Vgas = data['Vgas']
        Vdisk = data['Vdisk']
        Vbul = data['Vbul']
        R_d = data['R_d']
        # adjust for Upsilon
        Vdisk = [v * sqrt_Upsilon_disk for v in Vdisk]
        Vbul = [v * sqrt_Upsilon_bulge for v in Vbul]

        # Calculate sums and DM proxy
        Vtot_baryons = [g + d + b for g, d, b in zip(Vgas, Vdisk, Vbul)]
        V_dm = dm_model(R, Vgas, R_d, name)
        
        plt.figure(figsize=(8, 6))
        
        # Plot Vobs with error bars (black dots)
        plt.errorbar(R, Vobs, yerr=e_Vobs, fmt='k.', label='Vobs', capsize=3, elinewidth=0.5)
         
        # Plot components
        plt.plot(R, Vgas, 'g:', label='Gas', linewidth=2)
        plt.plot(R, Vdisk, 'r--', label='Disk', linewidth=2)
        plt.plot(R, Vbul, color='orange', linestyle='--', label='Bulge', linewidth=2)
        
        # the formula to add velocities is sqrt(g^2 + d^2 + b^2)
        Vtot_baryons = [np.sqrt(g**2 + d**2 + b**2) for g, d, b in zip(Vgas, Vdisk, Vbul)]

        # do 
        # Plot total (blue dashed to distinguish from Vobs)
        plt.plot(R, Vtot_baryons, color='blue', linestyle='--', label='Total (gas+disk+bulge)', linewidth=1.5)
        
        # Plot dm
        plt.plot(R, V_dm, color='purple', linestyle='-', label='DM Model', linewidth=1.5)
        
        # the formula to add velocities is sqrt(g^2 + d^2 + b^2)
        Vtot_all     = [np.sqrt(g**2 + d**2 + b**2 + dm**2) for g, d, b, dm in zip(Vgas, Vdisk, Vbul, V_dm)]
        plt.plot(R, Vtot_all, color='cyan', linestyle='-', label='Total (gas+disk+bulge+dm)', linewidth=1.5)

        # plot MOND
        V_MOND = mond_model(R, Vgas, Vdisk, Vbul, name)
        plt.plot(R, V_MOND, color='grey', linestyle='-', label='MOND', linewidth=1.5)

        plt.title(f"{name} Rotation Curve")
        plt.xlabel('Radius (kpc)')
        plt.ylabel('Velocity (km/s)')
        
        # Set axis limits
        max_R = max(R)
        # Find maximum velocity to set y-limit appropriately
        all_v = Vobs + Vgas + Vdisk + Vbul + Vtot_baryons + V_dm + Vtot_all
        max_V = max([v for v in all_v])
        
        plt.xlim(0, max_R + 5)
        plt.ylim(0, max_V * 1.1)  # add 10% padding on top
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        out_path = os.path.join(out_dir, f"{name}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()
        
    print(f"Finished generating {len(galaxies)} plots in {out_dir}")

if __name__ == "__main__":
    main()
