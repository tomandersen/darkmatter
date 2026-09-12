import os
import matplotlib.pyplot as plt

def main():
    data_file = './sparc/MassModels_Lelli2016c.mrt'
    out_dir = 'curves/initial'
    os.makedirs(out_dir, exist_ok=True)
     
    # We will read the file and skip the first 25 lines (headers)
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
    
    for name, data in galaxies.items():
        R = data['R']
        Vobs = data['Vobs']
        e_Vobs = data['e_Vobs']
        Vgas = data['Vgas']
        Vdisk = data['Vdisk']
        Vbul = data['Vbul']
        
        # Calculate sums and DM proxy
        Vtot_baryons = [g + d + b for g, d, b in zip(Vgas, Vdisk, Vbul)]
        dm = [g * 2 for g in Vgas]
        
        plt.figure(figsize=(8, 6))
        
        # Plot Vobs with error bars (BLUE solid)
        plt.errorbar(R, Vobs, yerr=e_Vobs, fmt='b-', label='Vobs', capsize=3)
        
        # Plot components
        plt.plot(R, Vgas, 'g:', label='Gas', linewidth=2)
        plt.plot(R, Vdisk, 'r--', label='Disk', linewidth=2)
        plt.plot(R, Vbul, color='orange', linestyle='--', label='Bulge', linewidth=2)
        
        # Plot total (blue dashed to distinguish from Vobs)
        plt.plot(R, Vtot_baryons, color='blue', linestyle='--', label='Total (gas+disk+bulge)', linewidth=1.5)
        
        # Plot dm
        plt.plot(R, dm, color='purple', linestyle='-', label='DM (Vgas * 2)', linewidth=1.5)
        
        plt.title(f"{name} Rotation Curve")
        plt.xlabel('Radius (kpc)')
        plt.ylabel('Velocity (km/s)')
        
        # Set axis limits
        max_R = max(R)
        # Find maximum velocity to set y-limit appropriately
        all_v = Vobs + Vgas + Vdisk + Vbul + Vtot_baryons + dm
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
