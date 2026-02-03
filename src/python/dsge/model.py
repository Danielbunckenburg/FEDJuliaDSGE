import numpy as np
from collections import OrderedDict

class Parameter:
    def __init__(self, name, value, value_bounds=(1e-20, 1e5), 
                 fixed=False, description="", tex_label="", scaling=None, prior=None):
        self.name = name
        self.raw_value = value
        self.value_bounds = value_bounds
        self.fixed = fixed
        self.description = description
        self.tex_label = tex_label
        self.scaling = scaling
        self.prior = prior

    @property
    def value(self):
        if self.scaling:
            return self.scaling(self.raw_value)
        return self.raw_value

    @value.setter
    def value(self, val):
        self.raw_value = val

class AbstractModel:
    def __init__(self):
        self.parameters = OrderedDict()
        self.steady_state = OrderedDict()
        self.endogenous_states = OrderedDict()
        self.exogenous_shocks = OrderedDict()
        self.expected_shocks = OrderedDict()
        self.equilibrium_conditions = OrderedDict()
        self.observables = OrderedDict()
        self.observable_mappings = OrderedDict()
        
    def add_parameter(self, param):
        self.parameters[param.name] = param

    def __getitem__(self, key):
        if key in self.parameters:
            return self.parameters[key].value
        if key in self.steady_state:
            return self.steady_state[key]
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    @property
    def n_states(self):
        return len(self.endogenous_states)

    @property
    def n_shocks_exogenous(self):
        return len(self.exogenous_shocks)

    @property
    def n_shocks_expectational(self):
        return len(self.expected_shocks)

    @property
    def n_observables(self):
        return len(self.observables)

    @property
    def n_states_augmented(self):
        # Default implementation, can be overridden by models with lags
        return self.n_states

    def eqcond(self):
        raise NotImplementedError

    def measurement(self, TTT, RRR, CCC):
        raise NotImplementedError

    def solve(self, kernels=None):
        """
        Solves the model. Uses Python fallbacks if kernels=None.
        """
        gamma0, gamma1, c, psi, pi = self.eqcond()
        
        if kernels:
            # Convert to complex for gensys kernel
            g0 = gamma0.astype(np.complex128)
            g1 = gamma1.astype(np.complex128)
            c_c = c.astype(np.complex128)
            psi_c = psi.astype(np.complex128)
            pi_c = pi.astype(np.complex128)
            TTT, RRR, CCC = kernels.solve(g0, g1, c_c, psi_c, pi_c)
        else:
            from .solvers.gensys import gensys
            TTT, CCC, RRR, eu = gensys(gamma0, gamma1, c, psi, pi)
            if eu[0] != 1 or eu[1] != 1:
                # Fallback implementation might be incomplete, but we should log
                pass

        return TTT, RRR, CCC

    def likelihood(self, data):
        from .likelihood import likelihood as compute_likelihood
        return compute_likelihood(self, data)
    
    def filter(self, data, outputs=['loglh', 'states', 'variances']):
        """
        Runs the Kalman Filter and returns requested outputs.
        """
        TTT, RRR, CCC = self.solve()
        ZZ, DD, QQ, EE = self.measurement(TTT, RRR, CCC)
        s0, P0 = self.init_stationary_states(TTT, RRR, QQ)
        
        from .solvers.kalman import kalman_filter
        return kalman_filter(data, TTT, RRR, QQ, ZZ, DD, EE, s0, P0, outputs=outputs)

    def smooth(self, data):
        """
        Runs the Kalman Smoother.
        """
        TTT, RRR, CCC = self.solve()
        ZZ, DD, QQ, EE = self.measurement(TTT, RRR, CCC)
        s0, P0 = self.init_stationary_states(TTT, RRR, QQ)
        
        from .solvers.kalman import kalman_filter, kalman_smoother
        out = kalman_filter(data, TTT, RRR, QQ, ZZ, DD, EE, s0, P0, 
                            outputs=['states', 'variances', 'pred_states', 'pred_variances'])
        
        s_smooth, P_smooth, eps_smooth = kalman_smoother(TTT, RRR, QQ, out['states'], out['variances'], 
                                                        out['pred_states'], out['pred_variances'])
        return s_smooth, P_smooth, eps_smooth

    def prior(self):
        """
        Computes the log-prior of the model parameters.
        """
        log_prior = 0.0
        for param in self.parameters.values():
            if not param.fixed and param.prior:
                log_prior += param.prior.logpdf(param.raw_value)
        return log_prior

    def posterior(self, data):
        """
        Computes the log-posterior (Likelihood + Prior).
        """
        log_lh = self.likelihood(data)
        log_prior = self.prior()
        return log_lh + log_prior

    def init_stationary_states(self, TTT, RRR, QQ):
        """
        Solves the discrete Lyapunov equation for the initial variance P0.
        P = T P T' + R Q R'
        """
        from scipy.linalg import solve_discrete_lyapunov
        n_states = TTT.shape[0]
        s0 = np.zeros(n_states)
        
        # Q_mat = R @ Q @ R'
        Q_sigma = RRR @ QQ @ RRR.T
        
        try:
            P0 = solve_discrete_lyapunov(TTT, Q_sigma)
        except Exception:
            # Fallback to identity or large variance if non-stationary
            P0 = np.eye(n_states) * 10.0
            
        return s0, P0

    def init_model_indices(self):
        raise NotImplementedError("Subclasses must implement init_model_indices")

    def steadystate(self):
        raise NotImplementedError("Subclasses must implement steadystate")
