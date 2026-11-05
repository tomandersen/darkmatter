    # plan is to set thigs up so that it all runs from here making the graphs for the paper
from sparc_dm_bruteCGM import run_sparc_dm_brute_cgm

paper_core_gas_density=0.55
paper_core_gas_beta=2.25
paper_core_gas_factor=2.5

paper_cgm_density=0.025
paper_cgm_beta=0.68
paper_cgm_factor=0.20

num_minutes = 60 # per run - about 5 runs.

paper_power = 16.0 # watts

#path_suffix = "_test"

# use the core densities made with optimize_individual_galaxies.py
output_dir = "./sparc/dm_mass_core_file" + path_suffix
err_per_point, best_power, unobserved_gas_fraction = run_sparc_dm_brute_cgm(
    add_core_gas_central_density=paper_core_gas_density, # ignored as we are using the file
    add_core_gas_beta=paper_core_gas_beta, 
    r_core_factor_gas=paper_core_gas_factor,
    add_real_cgm_central_density=paper_cgm_density,  
    cgm_beta = paper_cgm_beta,
    r_core_factor_cgm=paper_cgm_factor, 
    num_minutes=num_minutes, 
    output_graphs=True,
    power_override = paper_power,
    use_core_gas_file=True,
    output_dir_override=output_dir
)

# do a run with no gas added, and let global P vary (or hold to 16?)
output_dir = "./sparc/dm_mass_no_gas" + path_suffix
err_per_point, best_power, unobserved_gas_fraction = run_sparc_dm_brute_cgm(
    add_core_gas_central_density=0.0,
    add_core_gas_beta=paper_core_gas_beta, 
    r_core_factor_gas=paper_core_gas_factor,
    add_real_cgm_central_density=0.0,  
    cgm_beta = paper_cgm_beta,
    r_core_factor_cgm=paper_cgm_factor, 
    num_minutes=num_minutes, 
    output_graphs=True,
    power_override = 0.0,
    use_core_gas_file=False,
    output_dir_override=output_dir
)


# do a run with gas added, 
output_dir = "./sparc/dm_mass_single_gas" + path_suffix
err_per_point, best_power, unobserved_gas_fraction = run_sparc_dm_brute_cgm(
    add_core_gas_central_density=paper_core_gas_density,
    add_core_gas_beta=paper_core_gas_beta, 
    r_core_factor_gas=paper_core_gas_factor,
    add_real_cgm_central_density=paper_cgm_density,  
    cgm_beta = paper_cgm_beta,
    r_core_factor_cgm=paper_cgm_factor, 
    num_minutes=num_minutes, 
    output_graphs=True,
    power_override = paper_power,
    use_core_gas_file=False,
    output_dir_override=output_dir
)

# do a pure Newton Run: do this by setting power to 1e-5, so its effectively off, 
output_dir = "./sparc/dm_mass_Newton" + path_suffix
err_per_point, best_power, unobserved_gas_fraction = run_sparc_dm_brute_cgm(
    add_core_gas_central_density=0.0, # add no core gas
    add_core_gas_beta=paper_core_gas_beta, 
    r_core_factor_gas=paper_core_gas_factor,
    add_real_cgm_central_density=0.0,   # add no cgm
    cgm_beta = paper_cgm_beta,
    r_core_factor_cgm=paper_cgm_factor, 
    num_minutes=5, #quick
    output_graphs=True,
    power_override = 0.00001, # turn off power, don't fit it.
    use_core_gas_file=False,
    output_dir_override=output_dir
)



print(f"Done.")
print(f"Err per point: {err_per_point}")
print(f"Best power: {best_power}\n")
print(f"Unobserved gas fraction: {unobserved_gas_fraction}\n\n\n")