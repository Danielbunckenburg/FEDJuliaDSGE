import numpy as np
from scipy.optimize import minimize

def optimize_model(model, data, method="L-BFGS-B"):
    """
    Finds the posterior mode (or maximum likelihood if no priors).
    
    Args:
        model: The DSGE model instance
        data: The dataset (n_obs, T)
        method: Optimization method (default: L-BFGS-B)
        
    Returns:
        opt_res: Optimization result object
    """
    
    # 1. Identify free parameters
    free_params = []
    param_keys = []
    bounds = []
    
    for key, param in model.parameters.items():
        if not param.fixed:
            free_params.append(param.raw_value)
            param_keys.append(key)
            bounds.append(param.value_bounds)
            
    x0 = np.array(free_params)
    
    print(f"Starting optimization with {len(x0)} parameters...")
    
    # 2. Define objective function (negative log posterior)
    def objective(x):
        # Update model parameters
        for i, key in enumerate(param_keys):
            model.parameters[key].raw_value = x[i]
            
        try:
            # We want to MAXIMIZE posterior, so MINIMIZE negative posterior
            log_post = model.posterior(data)
            
            if np.isnan(log_post) or np.isinf(log_post):
                return 1e10
                
            return -log_post
        except Exception:
            return 1e10

    # 3. Optimize
    res = minimize(objective, x0, method=method, bounds=bounds, 
                   options={'disp': True, 'maxiter': 100}) # Limit iters for now
                   
    # 4. Update model with optimal parameters
    if res.success or True: # Update anyway even if maxiter reached
        for i, key in enumerate(param_keys):
            model.parameters[key].raw_value = res.x[i]
            
    return res
