import numpy as np
from tqdm import tqdm
from .samplers import MetropolisHastings

class SMC:
    """
    Sequential Monte Carlo (SMC) Sampler for DSGE models.
    """
    def __init__(self, model, data, n_parts=1000, n_blocks=1, phi_exponent=2):
        self.model = model
        self.data = data
        self.n_parts = n_parts
        self.n_blocks = n_blocks
        self.phi_exponent = phi_exponent
        
        self.param_keys = [k for k, v in model.parameters.items() if not v.fixed]
        self.n_params = len(self.param_keys)

    def sample(self, n_stages=50):
        # 1. Initialize from prior
        particles = self._draw_from_prior(self.n_parts)
        log_prior = np.array([self._eval_prior(p) for p in particles])
        log_lh = np.array([self._eval_likelihood(p) for p in particles])
        
        # Tempering schedule
        phi = (np.arange(n_stages + 1) / n_stages)**self.phi_exponent
        
        weights = np.ones(self.n_parts) / self.n_parts
        
        print(f"Starting SMC with {self.n_parts} particles and {n_stages} stages...")
        
        for s in range(1, n_stages + 1):
            d_phi = phi[s] - phi[s-1]
            
            # 2. Correction Stage: Reweight
            # Incremental weight: exp(phi_n - phi_{n-1} * log_lh)
            inc_weights = np.exp(d_phi * log_lh)
            weights = weights * inc_weights
            
            # Normalize
            w_sum = np.sum(weights)
            if w_sum == 0:
                print("Error: Particles collapsed. Adjust tempering or prior.")
                break
                
            weights /= w_sum
            
            # ESS (Effective Sample Size)
            ess = 1.0 / np.sum(weights**2)
            
            # 3. Selection Stage: Resample if ESS too low
            if ess < self.n_parts / 2:
                particles, log_prior, log_lh = self._resample(particles, log_prior, log_lh, weights)
                weights = np.ones(self.n_parts) / self.n_parts
            
            # 4. Mutation Stage: Metropolis-Hastings steps
            # Update proposal covariance from current particle distribution
            cov = np.cov(particles, rowvar=False)
            if self.n_params == 1:
                cov = np.array([[cov]])
            
            print(f"Stage {s}/{n_stages}: ESS={ess:.1f}, Likelihood Mean={np.mean(log_lh):.2f}")
            
            # Mutate particles using MH
            particles, log_prior, log_lh = self._mutate(particles, log_prior, log_lh, phi[s], cov)
            
        return particles, weights

    def _draw_from_prior(self, n):
        # This is a bit tricky as we only have logpdf. 
        # For this demo, we'll try to use raw_value + noise or if the prior has a 'rvs' method.
        # Ideally our priors should have a sample() method.
        # Let's assume for now we use the starting values in the model.
        draws = np.zeros((n, self.n_params))
        for j, key in enumerate(self.param_keys):
            p = self.model.parameters[key]
            # If we don't have rvs, we fallback to initial value + some jitter
            draws[:, j] = p.raw_value + np.random.randn(n) * 0.01 
        return draws

    def _eval_prior(self, x):
        lp = 0.0
        for i, key in enumerate(self.param_keys):
            p = self.model.parameters[key]
            if p.prior:
                lp += p.prior.logpdf(x[i])
        return lp

    def _eval_likelihood(self, x):
        for i, key in enumerate(self.param_keys):
            self.model.parameters[key].raw_value = x[i]
        try:
            return self.model.likelihood(self.data)
        except Exception:
            return -1e10

    def _resample(self, particles, log_prior, log_lh, weights):
        idx = np.random.choice(self.n_parts, size=self.n_parts, p=weights)
        return particles[idx], log_prior[idx], log_lh[idx]

    def _mutate(self, particles, log_prior, log_lh, phi_curr, cov):
        # Move each particle slightly using MH steps
        # In a real SMC, we'd do multiple steps.
        new_particles = particles.copy()
        new_log_prior = log_prior.copy()
        new_log_lh = log_lh.copy()
        
        # Scaling for mutation
        scale = 0.5 
        
        for i in range(self.n_parts):
            # Propose
            noise = np.random.multivariate_normal(np.zeros(self.n_params), cov)
            prop = particles[i] + scale * noise
            
            lp_prop = self._eval_prior(prop)
            llh_prop = self._eval_likelihood(prop)
            
            # Tempering posterior logic:
            # log_p = log_prior + phi * log_likelihood
            log_post_curr = log_prior[i] + phi_curr * log_lh[i]
            log_post_prop = lp_prop + phi_curr * llh_prop
            
            if np.log(np.random.rand()) < (log_post_prop - log_post_curr):
                new_particles[i] = prop
                new_log_prior[i] = lp_prop
                new_log_lh[i] = llh_prop
                
        return new_particles, new_log_prior, new_log_lh
