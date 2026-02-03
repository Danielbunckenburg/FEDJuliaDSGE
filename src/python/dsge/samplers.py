import numpy as np
from tqdm import tqdm

class MetropolisHastings:
    """
    Random Walk Metropolis-Hastings Sampler for DSGE models.
    """
    def __init__(self, model, data, proposal_cov=None, scale=0.5):
        self.model = model
        self.data = data
        self.scale = scale
        
        # Identify free parameters
        self.param_keys = [k for k, v in model.parameters.items() if not v.fixed]
        self.n_params = len(self.param_keys)
        
        if proposal_cov is None:
            self.proposal_cov = np.eye(self.n_params) * 0.01
        else:
            self.proposal_cov = proposal_cov

    def sample(self, n_samples, x0=None, burn_in=0):
        if x0 is None:
            x_curr = np.array([self.model.parameters[k].raw_value for k in self.param_keys])
        else:
            x_curr = x0
            
        # Initial log-posterior
        log_post_curr = self._eval_posterior(x_curr)
        
        chain = np.zeros((n_samples, self.n_params))
        log_posts = np.zeros(n_samples)
        acceptances = 0
        
        print(f"Sampling {n_samples} points (Burn-in: {burn_in})...")
        
        for i in tqdm(range(n_samples + burn_in)):
            # 1. Propose new state
            # x_prop = x_curr + scale * N(0, prop_cov)
            # We use Cholesky for efficiency/stability if prop_cov is large
            noise = np.random.multivariate_normal(np.zeros(self.n_params), self.proposal_cov)
            x_prop = x_curr + self.scale * noise
            
            # Check bounds
            in_bounds = True
            for j, key in enumerate(self.param_keys):
                lb, ub = self.model.parameters[key].value_bounds
                if x_prop[j] < lb or x_prop[j] > ub:
                    in_bounds = False
                    break
            
            if not in_bounds:
                log_post_prop = -np.inf
            else:
                log_post_prop = self._eval_posterior(x_prop)
            
            # 2. Acceptance ratio
            # r = p(prop)/p(curr) -> log(r) = log_p(prop) - log_p(curr)
            log_r = log_post_prop - log_post_curr
            
            if np.log(np.random.rand()) < log_r:
                x_curr = x_prop
                log_post_curr = log_post_prop
                if i >= burn_in:
                    acceptances += 1
            
            if i >= burn_in:
                chain[i - burn_in, :] = x_curr
                log_posts[i - burn_in] = log_post_curr
                
        acceptance_rate = acceptances / n_samples
        print(f"Acceptance Rate: {acceptance_rate:.4f}")
        
        return chain, log_posts

    def _eval_posterior(self, x):
        # Temporarily set model parameters
        orig_values = []
        for i, key in enumerate(self.param_keys):
            orig_values.append(self.model.parameters[key].raw_value)
            self.model.parameters[key].raw_value = x[i]
            
        try:
            lp = self.model.posterior(self.data)
        except Exception:
            lp = -np.inf
            
        # Restore (optional if we don't care about model state after sampling)
        for i, key in enumerate(self.param_keys):
            self.model.parameters[key].raw_value = orig_values[i]
            
        return lp
