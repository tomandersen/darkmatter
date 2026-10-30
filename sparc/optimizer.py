import optuna
import time
from sparc_dm_bruteCGM import run_sparc_dm_brute_cgm
from pathlib import Path

# 1. Define your objective function
def objective(trial):
    # Suggest 6 floating-point variables with predefined bounds
    add_core_gas_central_density = trial.suggest_float("add_core_gas_central_density", 0.5, 1.0)
    add_core_gas_beta = trial.suggest_float("add_core_gas_beta", 1.7, 3.0)
    r_core_factor_gas = 2.5# trial.suggest_float("r_core_factor_gas", 1, 5) 
    add_real_cgm_central_density = 0.00 #0.026 #trial.suggest_float("add_real_cgm_central_density", 0,  0.1)
    r_core_factor_cgm = 0.2 #trial.suggest_float("r_core_factor_cgm", 0.1, 0.5)
    cgm_beta = 0.7 #trial.suggest_float("cgm_beta", 0.5, 1.0)

    num_minutes = 4 
    output_graphs = False
    
    # Run your 10-minute code
    # power override - we set powe in all these optimiztions
    power_override = 0
    min_mass = 0
    max_mass = 1e99

    result, power, dm_to_hi_ratio = run_sparc_dm_brute_cgm(add_core_gas_central_density, add_core_gas_beta, r_core_factor_gas, add_real_cgm_central_density, r_core_factor_cgm, cgm_beta, num_minutes, output_graphs, power_override, min_mass, max_mass)
    trial.set_user_attr("dm_to_hi_ratio", dm_to_hi_ratio)

    return result, power # Optuna will try to minimize or maximize this

# 2. Use a persistent SQLite database so you can stop/resume anytime
study_name = "dark_mass_opt_" + Path(__file__).resolve().parent.parent.name + "TPE-CHI-core-gas2.5-NOCGM"
study = optuna.create_study(
    study_name=study_name, 
    storage="sqlite:///optimization.db", 
    load_if_exists=True,
    sampler=optuna.samplers.TPESampler(multivariate=True),
    directions=["minimize", "maximize"]  #  direction="minimize" # or "maximize"
)

# 3. Optimize (e.g., run for 50 trials total)
study.optimize(objective, n_trials=3300)

print("Best variables:", study.best_params)
