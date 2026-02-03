import numpy as np
from ..model import AbstractModel, Parameter

class AnSchorfheide(AbstractModel):
    def __init__(self, subspec="ss0"):
        super().__init__()
        self.subspec = subspec
        self.init_parameters()
        self.init_model_indices()

    def init_parameters(self):
        self.add_parameter(Parameter("tau", 1.9937, tex_label="\\tau"))
        self.add_parameter(Parameter("kappa", 0.7306, tex_label="\\kappa"))
        self.add_parameter(Parameter("psi_1", 1.1434, tex_label="\\psi_1"))
        self.add_parameter(Parameter("psi_2", 0.4536, tex_label="\\psi_2"))
        self.add_parameter(Parameter("rA", 0.0313, tex_label="rA"))
        self.add_parameter(Parameter("pi_star", 8.1508, tex_label="\\pi*"))
        self.add_parameter(Parameter("gamma_Q", 1.5, tex_label="\\gamma_Q"))
        self.add_parameter(Parameter("rho_R", 0.3847, tex_label="\\rho_R"))
        self.add_parameter(Parameter("rho_g", 0.3777, tex_label="\\rho_g"))
        self.add_parameter(Parameter("rho_z", 0.9579, tex_label="\\rho_z"))
        self.add_parameter(Parameter("sigma_R", 0.4900, tex_label="\\sigma_R"))
        self.add_parameter(Parameter("sigma_g", 1.4594, tex_label="\\sigma_g"))
        self.add_parameter(Parameter("sigma_z", 0.9247, tex_label="\\sigma_z"))
        self.add_parameter(Parameter("e_y", 0.20*0.579923, fixed=True, tex_label="e_y"))
        self.add_parameter(Parameter("e_pi", 0.20*1.470832, fixed=True, tex_label="e_\\pi"))
        self.add_parameter(Parameter("e_R", 0.20*2.237937, fixed=True, tex_label="e_R"))

    def init_model_indices(self):
        # Endogenous states
        states = ["y_t", "pi_t", "R_t", "y_t1", "g_t", "z_t", "Ey_t", "Epi_t"]
        for i, s in enumerate(states):
            self.endogenous_states[s] = i
            
        # Exogenous shocks
        shocks = ["z_sh", "g_sh", "rm_sh"]
        for i, s in enumerate(shocks):
            self.exogenous_shocks[s] = i
            
        # Expectations shocks
        expectations = ["Ey_sh", "Epi_sh"]
        for i, s in enumerate(expectations):
            self.expected_shocks[s] = i
            
        # Equilibrium conditions
        eqs = ["eq_euler", "eq_phillips", "eq_mp", "eq_y_t1", "eq_g", "eq_z", "eq_Ey", "eq_Epi"]
        for i, e in enumerate(eqs):
            self.equilibrium_conditions[e] = i

        # Observables
        obs = ["obs_gdp", "obs_cpi", "obs_nominalrate"]
        for i, o in enumerate(obs):
            self.observables[o] = i

    def eqcond(self):
        endo = self.endogenous_states
        exo  = self.exogenous_shocks
        ex   = self.expected_shocks
        eq   = self.equilibrium_conditions
        
        n = self.n_states
        gamma0 = np.zeros((n, n))
        gamma1 = np.zeros((n, n))
        c = np.zeros(n)
        psi = np.zeros((n, self.n_shocks_exogenous))
        pi = np.zeros((n, self.n_shocks_expectational))
        
        # 1. Consumption Euler Equation
        gamma0[eq["eq_euler"], endo["y_t"]] = 1.0
        gamma0[eq["eq_euler"], endo["R_t"]] = 1.0 / self["tau"]
        gamma0[eq["eq_euler"], endo["g_t"]] = -(1.0 - self["rho_g"])
        gamma0[eq["eq_euler"], endo["z_t"]] = -self["rho_z"] / self["tau"]
        gamma0[eq["eq_euler"], endo["Ey_t"]] = -1.0
        gamma0[eq["eq_euler"], endo["Epi_t"]] = -1.0 / self["tau"]
        
        # 2. NK Phillips Curve
        gamma0[eq["eq_phillips"], endo["y_t"]] = -self["kappa"]
        gamma0[eq["eq_phillips"], endo["pi_t"]] = 1.0
        gamma0[eq["eq_phillips"], endo["g_t"]] = self["kappa"]
        gamma0[eq["eq_phillips"], endo["Epi_t"]] = -1.0 / (1.0 + self["rA"] / 400.0)
        
        # 3. Monetary Policy Rule
        gamma0[eq["eq_mp"], endo["y_t"]] = -(1.0 - self["rho_R"]) * self["psi_2"]
        gamma0[eq["eq_mp"], endo["pi_t"]] = -(1.0 - self["rho_R"]) * self["psi_1"]
        gamma0[eq["eq_mp"], endo["R_t"]] = 1.0
        gamma0[eq["eq_mp"], endo["g_t"]] = (1.0 - self["rho_R"]) * self["psi_2"]
        gamma1[eq["eq_mp"], endo["R_t"]] = self["rho_R"]
        psi[eq["eq_mp"], exo["rm_sh"]] = 1.0
        
        # 4. Output lag
        gamma0[eq["eq_y_t1"], endo["y_t1"]] = 1.0
        gamma1[eq["eq_y_t1"], endo["y_t"]] = 1.0
        
        # 5. Government spending
        gamma0[eq["eq_g"], endo["g_t"]] = 1.0
        gamma1[eq["eq_g"], endo["g_t"]] = self["rho_g"]
        psi[eq["eq_g"], exo["g_sh"]] = 1.0
        
        # 6. Technology
        gamma0[eq["eq_z"], endo["z_t"]] = 1.0
        gamma1[eq["eq_z"], endo["z_t"]] = self["rho_z"]
        psi[eq["eq_z"], exo["z_sh"]] = 1.0
        
        # 7. Expected output
        gamma0[eq["eq_Ey"], endo["y_t"]] = 1.0
        gamma1[eq["eq_Ey"], endo["Ey_t"]] = 1.0
        pi[eq["eq_Ey"], ex["Ey_sh"]] = 1.0
        
        # 8. Expected inflation
        gamma0[eq["eq_Epi"], endo["pi_t"]] = 1.0
        gamma1[eq["eq_Epi"], endo["Epi_t"]] = 1.0
        pi[eq["eq_Epi"], ex["Epi_sh"]] = 1.0
        
        return gamma0, gamma1, c, psi, pi

    def measurement(self, TTT, RRR, CCC):
        endo = self.endogenous_states
        exo  = self.exogenous_shocks
        obs  = self.observables
        
        n_obs = len(self.observables)
        n_states = self.n_states
        n_shocks = self.n_shocks_exogenous
        
        zz = np.zeros((n_obs, n_states))
        dd = np.zeros(n_obs)
        ee = np.zeros((n_obs, n_obs))
        qq = np.zeros((n_shocks, n_shocks))
        
        ## Output growth
        zz[obs["obs_gdp"], endo["y_t"]]  = 1.0
        zz[obs["obs_gdp"], endo["y_t1"]] = -1.0
        zz[obs["obs_gdp"], endo["z_t"]]  = 1.0
        dd[obs["obs_gdp"]]              = self["gamma_Q"]

        ## Inflation
        zz[obs["obs_cpi"], endo["pi_t"]] = 4.0
        dd[obs["obs_cpi"]]             = self["pi_star"]

        ## Federal Funds Rate
        zz[obs["obs_nominalrate"], endo["R_t"]] = 4.0
        dd[obs["obs_nominalrate"]]             = self["pi_star"] + self["rA"] + 4.0*self["gamma_Q"]

        # Measurement error
        ee[obs["obs_gdp"], obs["obs_gdp"]]                 = self["e_y"]**2
        ee[obs["obs_cpi"], obs["obs_cpi"]]                 = self["e_pi"]**2
        ee[obs["obs_nominalrate"], obs["obs_nominalrate"]] = self["e_R"]**2

        # Variance of innovations
        qq[exo["z_sh"], exo["z_sh"]]   = self["sigma_z"]**2
        qq[exo["g_sh"], exo["g_sh"]]   = self["sigma_g"]**2
        qq[exo["rm_sh"], exo["rm_sh"]] = self["sigma_R"]**2
        
        return zz, dd, qq, ee

    def solve(self, kernels=None):
        return super().solve(kernels=kernels)

    def steadystate(self):
        # AnSchorfheide doesn't have a complex SS calculation like M1002
        pass
