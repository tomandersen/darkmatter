import numpy as np
from scipy.spatial import cKDTree

# Following the gemini app discussion "Milky Way Mock Catalogs"
# download (not included in git repo the m12i simulation)
# https://users.flatironinstitute.org/~mgrudic/fire2_public_release/core/m12i_res7100/output/snapdir_600/ 


import numpy as np
import matplotlib.pyplot as plt
import gizmo_analysis as gizmo

# ==========================================
# 1. CONFIGURATION & CONSTANTS
# ==========================================
SIM_FOLDER = "m12i_res7100" # or m10q_res30

SNAPSHOT_NUM = 600
DENSITY_THRESHOLD = 0.0000003  # cm^-3 (filtering for diffuse gas)
#DISTANCE_LIMIT = 100.0    # kpc
  
G = 4.3009e-6             # Gravitational constant: kpc * (km/s)^2 / M_sun
c = 299792458.0           # Speed of light: m/s
MSUN_TO_KG = 1.989e30     # Solar masses to kilograms
MSUN_KPC3_TO_KG_M3 = 6.77e-29 # Mass density conversion

def get_component_velocity(radii, masses):
    """Sorts particles by radius and calculates circular velocity."""
    sort_idx = np.argsort(radii)
    r_sorted = radii[sort_idx]
    m_sorted = masses[sort_idx]
    
    enclosed_mass = np.cumsum(m_sorted)
    v_circ = np.sqrt((G * enclosed_mass) / (r_sorted + 1e-6)) # Avoid div by zero
    return r_sorted, v_circ

# ==========================================
# 2. LOAD DATA & EXTRACT PROPERTIES
# ==========================================
print("Loading snapshot data...")

part = gizmo.io.Read.read_snapshots(
    'all',
    #['star', 'gas', 'dark'], 
    'index', 
    SNAPSHOT_NUM, 
    assign_hosts=True,
    simulation_directory=f'./{SIM_FOLDER}'
)
print(f"About to dump file contents keys and content descriptions")
 
for species in part:
    print(f"Species: {species}")
    for property_key in part[species]:
        print(f"  -> {property_key}: {part[species][property_key].shape}")

print(f"Dumped contents keys and content descriptions")


# Extract Radii
# Changed from 'host.distance.principal.spherical' to 'host.distance.spherical'
r_stars = part['star'].prop('host.distance.spherical')[:, 0]
r_gas_all = part['gas'].prop('host.distance.spherical')[:, 0]
r_dm_all = part['dark'].prop('host.distance.spherical')[:, 0]

# star positions
pos_stars_all = part['star'].prop('host.distance') # 3D Cartesian coords

# Extract Masses
mass_stars = part['star']['mass']
mass_gas_all = part['gas']['mass']
mass_dm_all = part['dark']['mass']

# ==========================================
# 3. FILTER AND PLOT GAS BLOBS
# ==========================================
print("Extracting and filtering gas properties...")

# Extract 3D positions (X, Y, Z in kpc) and number density (cm^-3)
pos_gas_all = part['gas'].prop('host.distance')
n_gas = part['gas'].prop('number.density')

# Create Boolean masks based on your configuration limits
# Dense gas "blobs" are typically > 1 cm^-3 (molecular/cold atomic clouds)
#mask_distance = r_gas_all < DISTANCE_LIMIT
mask_density = n_gas > DENSITY_THRESHOLD

# Combine the masks
#mask_blobs = mask_distance & mask_density
mask_blobs = mask_density

# Apply the mask to get the X and Y coordinates (Face-on view)
x_blobs = pos_gas_all[mask_blobs, 0]
y_blobs = pos_gas_all[mask_blobs, 1]

# Enhanced Statistics Printout
num_blobs = np.sum(mask_blobs)
print(f"Found {num_blobs:,} dense gas particles out of {len(r_gas_all):,}")

if num_blobs > 0:
    # Print a few statements on the stats of the filtered blobs
    print(f"  -> Minimum Density: {np.min(n_gas[mask_blobs]):.8f} cm^-3")
    print(f"  -> Maximum Density: {np.max(n_gas[mask_blobs]):.8f} cm^-3")
    print(f"  -> Mean Density:    {np.mean(n_gas[mask_blobs]):.8f} cm^-3")
    print(f"  -> Max Distance:    {np.max(r_gas_all[mask_blobs]):.8f} kpc")
    print(f"  -> Min Distance:    {np.min(r_gas_all[mask_blobs]):.8f} kpc")
    
    # Plotting the distribution
    plt.figure(figsize=(9, 8))
    
    hb = plt.hexbin(
        x_blobs, 
        y_blobs, 
        gridsize=200,      
        cmap='magma',      
        bins='log',        
        mincnt=1           
    )
    
    plt.colorbar(hb, label='Log$_{10}$(Particle Count)')
    plt.xlabel('X distance (kpc)', fontsize=12)
    plt.ylabel('Y distance (kpc)', fontsize=12)
    plt.title(f'Dense Gas Blobs Distribution (n > {DENSITY_THRESHOLD} cm$^{{-3}}$)', fontsize=14)
    
    plt.gca().set_aspect('equal', adjustable='box')
     
    plot_zoom = 1500.0 
    plt.xlim(-plot_zoom, plot_zoom)
    plt.ylim(-plot_zoom, plot_zoom)
    
    plt.tight_layout()
    plt.show()
else:
    print(f"\n[WARNING] Plot aborted: No gas particles found matching density > {DENSITY_THRESHOLD} cm^-3.")
    print(f"Max density in entire simulation gas: {np.max(n_gas):.4f} cm^-3")  
    print(f"Min density in entire simulation gas: {np.min(n_gas):.4f} cm^-3")