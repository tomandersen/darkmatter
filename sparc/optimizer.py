import optuna
import time
from sparc_dm_bruteCGM import run_sparc_dm_brute_cgm

# 1. Define your objective function
def objective(trial):
    # Suggest 6 floating-point variables with predefined bounds
    add_core_gas_central_density = trial.suggest_float("add_core_gas_central_density", 0.0, 4.0)
    add_core_gas_beta = trial.suggest_float("add_core_gas_beta", 0.8, 2.2)
    r_core_factor_gas = trial.suggest_float("r_core_factor_gas", 0.1, 10) # Use log=True for learning rates/scales
    add_real_cgm_central_density = trial.suggest_float("add_real_cgm_central_density", 0,  0.1)
    r_core_factor_cgm = trial.suggest_float("r_core_factor_cgm", 0.1, 10)
    cgm_beta = trial.suggest_float("cgm_beta", 0.3, 0.8)

    num_minutes = 10
    output_graphs = False
    
    # Run your 10-minute code
    result = run_sparc_dm_brute_cgm(add_core_gas_central_density, add_core_gas_beta, r_core_factor_gas, add_real_cgm_central_density, r_core_factor_cgm, cgm_beta, num_minutes, output_graphs)
    
    return result # Optuna will try to minimize or maximize this

# 2. Use a persistent SQLite database so you can stop/resume anytime
study = optuna.create_study(
    study_name="dark_mass_opt_no_min_gas_rho", 
    storage="sqlite:///optimization.db", 
    load_if_exists=True,
    direction="minimize" # or "maximize"
)

# 3. Optimize (e.g., run for 50 trials total)
study.optimize(objective, n_trials=500)

print("Best variables:", study.best_params)
