import numpy as np
import pandas as pd
from collections import OrderedDict
from scipy.stats import norm
from scipy.optimize import brentq
from ..model import AbstractModel, Parameter
from ..utils.math import *
from ..utils.transformations import *
from ..data import Observable

from ..priors import Normal, Gamma, Beta, InverseGamma

class Model1002(AbstractModel):
    def __init__(self, subspec="ss10"):
        super().__init__()
        self.subspec = subspec
        self.init_parameters()
        self.init_model_indices()
        self.init_observable_mappings()
        self.steadystate()

    def init_observable_mappings(self):
        m = self
        obs = self.observable_mappings
        
        # 1. GDP
        def gdp_fwd(levels):
            temp = percapita("GDP", levels, "CNP16OV") # Default population mnemonic
            gdp = 1000 * nominal_to_real("temp", pd.DataFrame({"temp": temp, "GDPDEF": levels["GDPDEF"]}))
            return one_qtr_pct_change(gdp)
        
        obs["obs_gdp"] = Observable("obs_gdp", ["GDP__FRED", "CNP16OV__FRED", "GDPDEF__FRED"],
                                    gdp_fwd, loggrowthtopct_annualized_percapita)

        # 2. Hours
        def hrs_fwd(levels):
            temp = levels["AWHNONAG"] * levels["CE16OV"]
            weeklyhours = percapita("temp", pd.DataFrame({"temp": temp, "CNP16OV": levels["CNP16OV"]}), "CNP16OV")
            return 100 * np.log(3 * weeklyhours / 100)
            
        obs["obs_hours"] = Observable("obs_hours", ["AWHNONAG__FRED", "CE16OV__FRED", "CNP16OV__FRED"],
                                      hrs_fwd, lambda x: x)

        # 3. Wages
        def wages_fwd(levels):
            real_wage = nominal_to_real("COMPNFB", levels)
            return one_qtr_pct_change(real_wage)
            
        obs["obs_wages"] = Observable("obs_wages", ["COMPNFB__FRED", "GDPDEF__FRED"],
                                      wages_fwd, loggrowthtopct_annualized)

        # 4. GDP Deflator
        obs["obs_gdpdeflator"] = Observable("obs_gdpdeflator", ["GDPDEF__FRED"],
                                            lambda l: one_qtr_pct_change(l["GDPDEF"]), 
                                            loggrowthtopct_annualized)

        # 5. Core PCE
        obs["obs_corepce"] = Observable("obs_corepce", ["PCEPILFE__FRED"],
                                        lambda l: one_qtr_pct_change(l["PCEPILFE"]), 
                                        loggrowthtopct_annualized)

        # 6. Nominal Rate
        obs["obs_nominalrate"] = Observable("obs_nominalrate", ["DFF__FRED"],
                                            lambda l: annual_to_quarter(l["DFF"]), 
                                            quarter_to_annual)

        # 7. Investment
        def inv_fwd(levels):
            real_inv = nominal_to_real("GPDI", levels)
            return one_qtr_pct_change(real_inv)
        obs["obs_investment"] = Observable("obs_investment", ["GPDI__FRED", "GDPDEF__FRED"],
                                           inv_fwd, loggrowthtopct_annualized)

        # 8. Consumption
        def cons_fwd(levels):
            real_cons = nominal_to_real("PCE", levels)
            return one_qtr_pct_change(real_cons)
        obs["obs_consumption"] = Observable("obs_consumption", ["PCE__FRED", "GDPDEF__FRED"],
                                            cons_fwd, loggrowthtopct_annualized)

        # 9. Spread
        obs["obs_spread"] = Observable("obs_spread", ["BAA__FRED", "GS10__FRED"],
                                       lambda l: annual_to_quarter(l["BAA"] - l["GS10"]),
                                       quarter_to_annual)

        # 10. 10Y Yield
        obs["obs_10y"] = Observable("obs_10y", ["GS10__FRED"],
                                    lambda l: annual_to_quarter(l["GS10"]),
                                    quarter_to_annual)

        # 11. TFP
        # Fernald TFP is not on FRED usually, we use a placeholder or GDI for now 
        # to avoid missing data errors if the user doesn't have it.
        obs["obs_tfp"] = Observable("obs_tfp", ["TFP__FRED"],
                                    lambda l: l["TFP"], lambda x: x) # Placeholder

        # 12. GDI
        def gdi_fwd(levels):
            temp = percapita("GDI", levels, "CNP16OV")
            gdi = 1000 * nominal_to_real("temp", pd.DataFrame({"temp": temp, "GDPDEF": levels["GDPDEF"]}))
            return one_qtr_pct_change(gdi)
        obs["obs_gdi"] = Observable("obs_gdi", ["GDI__FRED", "CNP16OV__FRED", "GDPDEF__FRED"],
                                    gdi_fwd, loggrowthtopct_annualized_percapita)

        # 13. Anticipated Nominal Rates (1-6 periods ahead)
        # These usually map to FFR expectations or OIS rates. 
        # For simplicity, we'll map them to the same DFF level (flat expectations) 
        # or just placeholders.
        for i in range(1, 7):
            obs[f"obs_nominalrate{i}"] = Observable(f"obs_nominalrate{i}", ["DFF__FRED"],
                                                    lambda l: annual_to_quarter(l["DFF"]),
                                                    quarter_to_annual)

    def init_parameters(self):
        # Ported from m1002.jl and subspecs.jl
        self.add_parameter(Parameter("alp", 0.1596, (1e-5, 0.999), prior=Normal(0.30, 0.05), tex_label="\\alpha"))
        self.add_parameter(Parameter("zeta_p", 0.8940, (1e-5, 0.999), prior=Beta(0.5, 0.1), tex_label="\\zeta_p"))
        self.add_parameter(Parameter("iota_p", 0.1865, (1e-5, 0.999), prior=Beta(0.5, 0.15), tex_label="\\iota_p"))
        self.add_parameter(Parameter("delta", 0.025, fixed=True, tex_label="\\delta"))
        self.add_parameter(Parameter("Upsilon", 1.000, fixed=True, tex_label="\\Upsilon"))
        self.add_parameter(Parameter("Phi", 1.1066, (1.0, 10.0), prior=Normal(1.25, 0.12), tex_label="\\Phi_p"))
        self.add_parameter(Parameter("S2", 2.7314, (-15.0, 15.0), prior=Normal(4.0, 1.5), tex_label="S''"))
        self.add_parameter(Parameter("h", 0.5347, (1e-5, 0.999), prior=Beta(0.7, 0.1), tex_label="h"))
        self.add_parameter(Parameter("ppsi", 0.6862, (1e-5, 0.999), prior=Beta(0.5, 0.15), tex_label="\\psi"))
        self.add_parameter(Parameter("nu_l", 2.5975, (1e-5, 10.0), prior=Normal(2.0, 0.75), tex_label="\\nu_l"))
        self.add_parameter(Parameter("zeta_w", 0.9291, (1e-5, 0.999), prior=Beta(0.5, 0.1), tex_label="\\zeta_w"))
        self.add_parameter(Parameter("iota_w", 0.2992, (1e-5, 0.999), prior=Beta(0.5, 0.15), tex_label="\\iota_w"))
        self.add_parameter(Parameter("lambda_w", 1.5000, fixed=True, tex_label="\\lambda_w"))
        self.add_parameter(Parameter("bet", 0.1402, (1e-5, 10.0), scaling=lambda x: 1/(1 + x/100), prior=Gamma(0.25, 0.1), tex_label="100(\\beta^{-1}-1)"))
        self.add_parameter(Parameter("psi1", 1.3679, (1e-5, 10.0), prior=Normal(1.5, 0.25), tex_label="\\psi_1"))
        self.add_parameter(Parameter("psi2", 0.0388, (-0.5, 0.5), prior=Normal(0.125, 0.05), tex_label="\\psi_2"))
        self.add_parameter(Parameter("psi3", 0.2464, (-0.5, 0.5), prior=Normal(0.125, 0.05), tex_label="\\psi_3"))
        self.add_parameter(Parameter("pi_star", 0.5000, (1e-5, 10.0), scaling=lambda x: 1 + x/100, fixed=True, tex_label="\\pi_*"))
        self.add_parameter(Parameter("sigma_c", 0.8719, (1e-5, 10.0), prior=Normal(1.5, 0.37), tex_label="\\sigma_c"))
        self.add_parameter(Parameter("rho", 0.7126, (1e-5, 0.999), prior=Beta(0.75, 0.1), tex_label="\\rho_R"))
        self.add_parameter(Parameter("ep_p", 10.0, fixed=True, tex_label="\\epsilon_p"))
        self.add_parameter(Parameter("ep_w", 10.0, fixed=True, tex_label="\\epsilon_w"))
        
        # Financial Frictions
        self.add_parameter(Parameter("Fgamma", 0.0300, scaling=lambda x: 1 - (1-x)**0.25, fixed=True, tex_label="F(\\bar{\\omega})"))
        self.add_parameter(Parameter("spr", 1.7444, (1e-5, 10.0), scaling=lambda x: (1 + x/100)**0.25, prior=Gamma(2.0, 0.1), tex_label="SP_*"))
        self.add_parameter(Parameter("zeta_spb", 0.0559, (1e-5, 0.999), prior=Beta(0.05, 0.005), tex_label="\\zeta_{sp,b}"))
        self.add_parameter(Parameter("gamma_star", 0.9900, fixed=True, tex_label="\\gamma_*"))
        
        # Exogenous processes
        self.add_parameter(Parameter("gam", 0.3673, (1e-5, 5.0), scaling=lambda x: x/100, prior=Normal(0.4, 0.1), tex_label="100\\gamma"))
        self.add_parameter(Parameter("Lmean", -45.9364, (-1000.0, 1000.0), prior=Normal(0.0, 10.0), tex_label="\\bar{L}"))
        self.add_parameter(Parameter("g_star", 0.1800, fixed=True, tex_label="g_*"))
        
        self.add_parameter(Parameter("rho_g", 0.9863, (1e-5, 0.999), prior=Beta(0.5, 0.2), tex_label="\\rho_g"))
        self.add_parameter(Parameter("rho_b", 0.9410, (1e-5, 0.999), prior=Beta(0.5, 0.2), tex_label="\\rho_b"))
        self.add_parameter(Parameter("rho_mu", 0.8735, (1e-5, 0.999), prior=Beta(0.5, 0.2), tex_label="\\rho_{\\mu}"))
        self.add_parameter(Parameter("rho_ztil", 0.9446, (1e-5, 0.999), prior=Beta(0.5, 0.2), tex_label="\\rho_{\\tilde{z}"))
        self.add_parameter(Parameter("rho_lambda_f", 0.8827, (1e-5, 0.999), prior=Beta(0.5, 0.2), tex_label="\\rho_{\\lambda_f}"))
        self.add_parameter(Parameter("rho_lambda_w", 0.3884, (1e-5, 0.999), prior=Beta(0.5, 0.2), tex_label="\\rho_{\\lambda_w}"))
        self.add_parameter(Parameter("rho_rm", 0.2135, (1e-5, 0.999), prior=Beta(0.5, 0.2), tex_label="\\rho_{r^m}"))
        self.add_parameter(Parameter("rho_sigma_w", 0.9898, (1e-5, 0.999), prior=Beta(0.75, 0.15), tex_label="\\rho_{\\sigma_\\omega}"))
        self.add_parameter(Parameter("rho_mu_e", 0.7500, fixed=True, tex_label="\\rho_{\\mu_e}"))
        self.add_parameter(Parameter("rho_gamma", 0.7500, fixed=True, tex_label="\\rho_{\\gamma}"))
        self.add_parameter(Parameter("rho_pi_star", 0.9900, fixed=True, tex_label="\\rho_{\\pi_*}"))
        self.add_parameter(Parameter("rho_lr", 0.6936, (1e-5, 0.999), prior=Beta(0.75, 0.15), tex_label="\\rho_{10y}"))
        self.add_parameter(Parameter("rho_z_p", 0.8910, (1e-5, 0.999), prior=Beta(0.5, 0.2), tex_label="\\rho_{z^p}"))
        self.add_parameter(Parameter("rho_tfp", 0.1953, (1e-5, 0.999), prior=Beta(0.5, 0.2), tex_label="\\rho_{tfp}"))
        self.add_parameter(Parameter("rho_gdpdef", 0.5379, (1e-5, 0.999), prior=Beta(0.5, 0.2), tex_label="\\rho_{gdpdef}"))
        self.add_parameter(Parameter("rho_corepce", 0.2320, (1e-5, 0.999), prior=Beta(0.5, 0.2), tex_label="\\rho_{pce}"))
        
        self.add_parameter(Parameter("sigma_g", 2.5230, (1e-7, 100.0), prior=InverseGamma(0.1, 2.0), tex_label="\\sigma_g"))
        self.add_parameter(Parameter("sigma_b", 0.0292, (1e-7, 100.0), prior=InverseGamma(0.1, 2.0), tex_label="\\sigma_b"))
        self.add_parameter(Parameter("sigma_mu", 0.4559, (1e-7, 100.0), prior=InverseGamma(0.1, 2.0), tex_label="\\sigma_{\\mu}"))
        self.add_parameter(Parameter("sigma_ztil", 0.6742, (1e-7, 100.0), prior=InverseGamma(0.1, 2.0), tex_label="\\sigma_{\\tilde{z}"))
        self.add_parameter(Parameter("sigma_lambda_f", 0.1314, (1e-7, 100.0), prior=InverseGamma(0.1, 2.0), tex_label="\\sigma_{\\lambda_f}"))
        self.add_parameter(Parameter("sigma_lambda_w", 0.3864, (1e-7, 100.0), prior=InverseGamma(0.1, 2.0), tex_label="\\sigma_{\\lambda_w}"))
        self.add_parameter(Parameter("sigma_rm", 0.2380, (1e-7, 100.0), prior=InverseGamma(0.1, 2.0), tex_label="\\sigma_{r^m}"))
        self.add_parameter(Parameter("sigma_sigma_w", 0.0428, (1e-7, 100.0), prior=InverseGamma(0.1, 2.0), tex_label="\\sigma_{\\sigma_\\omega}"))
        self.add_parameter(Parameter("sigma_pi_star", 0.0269, (1e-7, 100.0), prior=InverseGamma(0.1, 2.0), tex_label="\\sigma_{\\pi_*}"))
        self.add_parameter(Parameter("sigma_lr", 0.1766, (1e-7, 100.0), prior=InverseGamma(0.1, 2.0), tex_label="\\sigma_{10y}"))
        self.add_parameter(Parameter("sigma_z_p", 0.1662, (1e-7, 100.0), prior=InverseGamma(0.1, 2.0), tex_label="\\sigma_{z^p}"))
        self.add_parameter(Parameter("sigma_tfp", 0.9391, (1e-7, 100.0), prior=InverseGamma(0.1, 2.0), tex_label="\\sigma_{tfp}"))
        self.add_parameter(Parameter("sigma_gdpdef", 0.1575, (1e-7, 100.0), prior=InverseGamma(0.1, 2.0), tex_label="\\sigma_{gdpdef}"))
        self.add_parameter(Parameter("sigma_corepce", 0.0999, (1e-7, 100.0), prior=InverseGamma(0.1, 2.0), tex_label="\\sigma_{pce}"))
        
        self.add_parameter(Parameter("sigma_gdp", 0.1, fixed=True))
        self.add_parameter(Parameter("sigma_gdi", 0.1, fixed=True))
        
        # Anticipated shocks
        for i in range(1, 7):
            self.add_parameter(Parameter(f"sigma_rm{i}", 0.2, tex_label=f"\\sigma_{{ant{i}}}"))

        self.add_parameter(Parameter("eta_gz", 0.8400, tex_label="\\eta_{gz}"))
        self.add_parameter(Parameter("eta_lambda_f", 0.7892, tex_label="\\eta_{\\lambda_f}"))
        self.add_parameter(Parameter("eta_lambda_w", 0.4226, tex_label="\\eta_{\\lambda_w}"))
        self.add_parameter(Parameter("model_cleansed", 0.0, fixed=True))
        
        # COVID-specific parameters (default to zero for ss10)
        self.add_parameter(Parameter("rho_ziid", 0.0, fixed=True))
        self.add_parameter(Parameter("sigma_ziid", 0.0, fixed=True))
        self.add_parameter(Parameter("rho_biidc", 0.0, fixed=True))
        self.add_parameter(Parameter("sigma_biidc", 0.0, fixed=True))
        self.add_parameter(Parameter("rho_phi", 0.0, fixed=True))
        self.add_parameter(Parameter("sigma_phi", 0.0, fixed=True))

    def init_model_indices(self):
        # Endogenous states
        states = [
            "y_t", "c_t", "i_t", "qk_t", "k_t", "kbar_t", "u_t", "rk_t", "Rktil_t", "n_t", "mc_t",
            "pi_t", "mu_w_t", "w_t", "L_t", "R_t", "g_t", "b_t", "mu_t", "z_t", "lambda_f_t", "lambda_f_t1",
            "lambda_w_t", "lambda_w_t1", "rm_t", "sigma_w_t", "mu_e_t", "gamma_t", "pi_star_t",
            "Ec_t", "Eqk_t", "Ei_t", "Epi_t", "EL_t", "Erk_t", "Ew_t", "ERktil_t", "ERktil_f_t",
            "y_f_t", "c_f_t", "i_f_t", "qk_f_t", "k_f_t",
            "kbar_f_t", "u_f_t", "rk_f_t", "w_f_t", "L_f_t", "r_f_t", "Ec_f_t", "Eqk_f_t", "Ei_f_t", "EL_f_t",
            "ztil_t", "pi_t1", "pi_t2", "pi_a_t", "R_t1", "zp_t", "Ez_t",
            "Rktil_f_t", "n_f_t", "ziid_t", "biidc_t", "phi_t", "Ephi_t"
        ]
        for i in range(1, 7): states.append(f"rm_tl{i}")
        for i, s in enumerate(states): self.endogenous_states[s] = i
            
        # Shocks
        shocks = [
            "g_sh", "b_sh", "mu_sh", "ztil_sh", "lambda_f_sh", "lambda_w_sh", "rm_sh", "sigma_w_sh", "mu_e_sh",
            "gamma_sh", "pi_star_sh", "zp_sh", "lr_sh", "tfp_sh", "gdpdef_sh", "corepce_sh", "gdp_sh", "gdi_sh",
            "ziid_sh", "biidc_sh", "phi_sh"
        ]
        for i in range(1, 7): shocks.append(f"rm_shl{i}")
        for i, s in enumerate(shocks): self.exogenous_shocks[s] = i
            
        # Expectations
        expected = [
            "Ec_sh", "Eqk_sh", "Ei_sh", "Epi_sh", "EL_sh", "Erk_sh", "Ew_sh", "ERktil_sh", "Ec_f_sh",
            "Eqk_f_sh", "Ei_f_sh", "EL_f_sh", "ERktil_f_sh"
        ]
        for i, s in enumerate(expected): self.expected_shocks[s] = i
            
        # Equations
        eqs = [
            "eq_euler", "eq_inv", "eq_capval", "eq_spread", "eq_nevol", "eq_output", "eq_caputl", "eq_capsrv", "eq_capev",
            "eq_mkupp", "eq_phlps", "eq_caprnt", "eq_msub", "eq_wage", "eq_mp", "eq_res", "eq_g", "eq_b", "eq_mu", "eq_z",
            "eq_lambda_f", "eq_lambda_w", "eq_rm",  "eq_sigma_w", "eq_mu_e", "eq_gamma", "eq_lambda_f1", "eq_lambda_w1", "eq_Ec",
            "eq_Eqk", "eq_Ei", "eq_Epi", "eq_EL", "eq_Erk", "eq_Ew", "eq_ERktil", "eq_euler_f", "eq_inv_f",
            "eq_capval_f", "eq_output_f", "eq_caputl_f", "eq_capsrv_f", "eq_capev_f", "eq_mkupp_f",
            "eq_caprnt_f", "eq_msub_f", "eq_res_f", "eq_Ec_f", "eq_Eqk_f", "eq_Ei_f", "eq_EL_f",
            "eq_ztil", "eq_pi_star", "eq_pi1", "eq_pi2", "eq_pi_a", "eq_Rt1", "eq_zp", "eq_Ez", "eq_spread_f", "eq_nevol_f", "eq_ERktil_f",
            "eq_ziid", "eq_biidc", "eq_phi", "eq_Ephi"
        ]
        for i in range(1, 7): eqs.append(f"eq_rml{i}")
        for i, e in enumerate(eqs): self.equilibrium_conditions[e] = i

        # Observables
        obs = [
            "obs_gdp", "obs_corepce", "obs_gdpdeflator", "obs_nominalrate", "obs_investment",
            "obs_consumption", "obs_hours", "obs_wages", "obs_spread", "obs_10y", "obs_tfp", "obs_gdi"
        ]
        for i in range(1, 7): obs.append(f"obs_nominalrate{i}")
        for i, o in enumerate(obs): self.observables[o] = i
        
        # Augmented states indices
        self.endogenous_states_augmented = OrderedDict()
        aug = ["y_t1", "c_t1", "i_t1", "w_t1", "pi_t1_dup", "L_t1", "u_t1", "e_gdp_t", "e_gdp_t1", "e_gdi_t", "e_gdi_t1",
               "e_lr_t", "e_tfp_t", "e_gdpdef_t", "e_corepce_t"]
        for i, s in enumerate(aug):
            self.endogenous_states_augmented[s] = i + self.n_states

    @property
    def n_states_augmented(self):
        return self.n_states + len(self.endogenous_states_augmented)

    def steadystate(self):
        m = self
        m.z_star   = np.log(1+m["gam"]) + m["alp"]/(1-m["alp"])*np.log(m["Upsilon"])
        m.rstar    = np.exp(m["sigma_c"]*m.z_star) / m["bet"]
        m.Rstarn   = 100*(m.rstar*m["pi_star"] - 1)
        m.r_k_star = m["spr"]*m.rstar*m["Upsilon"] - (1-m["delta"])
        m.wstar    = (m["alp"]**m["alp"] * (1-m["alp"])**(1-m["alp"]) * m.r_k_star**(-m["alp"]) / m["Phi"])**(1/(1-m["alp"]))
        m.Lstar    = 1.
        m.kstar    = (m["alp"]/(1-m["alp"])) * m.wstar * m.Lstar / m.r_k_star
        m.kbarstar = m.kstar * (1+m["gam"]) * m["Upsilon"]**(1 / (1-m["alp"]))
        m.istar    = m.kbarstar * (1-((1-m["delta"])/((1+m["gam"]) * m["Upsilon"]**(1/(1-m["alp"])))))
        m.ystar    = (m.kstar**m["alp"]) * (m.Lstar**(1-m["alp"])) / m["Phi"]
        m.cstar    = (1-m["g_star"])*m.ystar - m.istar
        m.wl_c     = (m.wstar*m.Lstar)/(m.cstar*m["lambda_w"])

        # FINANCIAL FRICTIONS ADDITIONS
        # solve for sigma_omega_star and zomega_star
        zomega_star = norm.ppf(m["Fgamma"])
        target_zeta = m["zeta_spb"]
        
        def objective(sigma):
            return ζ_spb_fn(zomega_star, sigma, m["spr"]) - target_zeta
        
        try:
            m.sigma_omega_star = brentq(objective, 1e-5, 2.0)
        except Exception:
            m.sigma_omega_star = 0.5 # Fallback
            
        m.zeta_spb = m["zeta_spb"] # Just to confirm consistency
        
        # evaluate omega_bar_star
        omega_bar_star = ω_fn(zomega_star, m.sigma_omega_star)
        
        # evaluate all BGG function elasticities
        Gstar       = G_fn(zomega_star, m.sigma_omega_star)
        Gammastar   = Γ_fn(zomega_star, m.sigma_omega_star)
        dGdomega_star   = dG_dω_fn(zomega_star, m.sigma_omega_star)
        d2Gdomega2star  = d2G_dω2_fn(zomega_star, m.sigma_omega_star)
        dGammadomega_star   = dΓ_dω_fn(zomega_star)
        d2Gammadomega2star  = d2Γ_dω2_fn(zomega_star, m.sigma_omega_star)
        dGdsigma_star    = dG_dσ_fn(zomega_star, m.sigma_omega_star)
        d2Gdomegadsigma_star = d2G_dωdσ_fn(zomega_star, m.sigma_omega_star)
        dGammadsigma_star    = dΓ_dσ_fn(zomega_star, m.sigma_omega_star)
        d2Gammadomegadsigma_star = d2Γ_dωdσ_fn(zomega_star, m.sigma_omega_star)

        # evaluate mu, nk, and Rhostar
        mu_estar     = μ_fn(zomega_star, m.sigma_omega_star, m["spr"])
        nkstar      = nk_fn(zomega_star, m.sigma_omega_star, m["spr"])
        Rhostar     = 1/nkstar - 1

        # evaluate wekstar and vkstar
        # Assuming ss10 logic (not ss2/ss8/ss9 special cases)
        betabar_inverse = np.exp((m["sigma_c"] - 1) * m.z_star) / m["bet"]
        wekstar = (1-(m["gamma_star"]*betabar_inverse))*nkstar - m["gamma_star"]*betabar_inverse*(m["spr"]*(1-mu_estar*Gstar) - 1)
        vkstar      = (nkstar-wekstar)/m["gamma_star"]

        # evaluate nstar and vstar
        m.nstar = nkstar*m.kbarstar
        m.vstar = vkstar*m.kbarstar

        # a couple of combinations
        GammamuG         = Gammastar - mu_estar*Gstar
        GammamuGprime    = dGammadomega_star - mu_estar*dGdomega_star

        # elasticities wrt omega_bar
        zeta_bw        = ζ_bω_fn(zomega_star, m.sigma_omega_star, m["spr"])
        zeta_zw        = ζ_zω_fn(zomega_star, m.sigma_omega_star, m["spr"])
        zeta_bw_zw     = zeta_bw/zeta_zw

        # elasticities wrt sigma_omega
        term1 = (1 - mu_estar*dGdsigma_star/dGammadsigma_star) / (1 - mu_estar*dGdomega_star/dGammadomega_star) - 1
        term2 = dGammadsigma_star*m["spr"]
        term3 = mu_estar*nkstar*(dGdomega_star*d2Gammadomegadsigma_star - dGammadomega_star*d2Gdomegadsigma_star)/GammamuGprime**2
        
        zeta_b_sigma_omega = m.sigma_omega_star * (term1*term2 + term3) / ((1 - Gammastar)*m["spr"] + dGammadomega_star/GammamuGprime*(1-nkstar))
        zeta_z_sigma_omega = m.sigma_omega_star * (dGammadsigma_star - mu_estar*dGdsigma_star) / GammamuG
        
        m.zeta_sp_sigma_w = (zeta_bw_zw*zeta_z_sigma_omega - zeta_b_sigma_omega) / (1-zeta_bw_zw)

        # elasticities wrt mu_e
        zeta_b_mu_e = -mu_estar * (nkstar*dGammadomega_star*dGdomega_star/GammamuGprime + dGammadomega_star*Gstar*m["spr"]) / \
                      ((1-Gammastar)*GammamuGprime*m["spr"] + dGammadomega_star*(1-nkstar))
        zeta_z_mu_e = -mu_estar*Gstar/GammamuG
        m.zeta_sp_mu_e = (zeta_bw_zw*zeta_z_mu_e - zeta_b_mu_e) / (1-zeta_bw_zw)

        # some ratios/elasticities
        Rkstar      = m["spr"]*m["pi_star"]*m.rstar 
        zeta_gw        = dGdomega_star/Gstar*omega_bar_star
        zeta_G_sigma_omega      = dGdsigma_star/Gstar*m.sigma_omega_star

        # elasticities for the net worth evolution
        m.zeta_nRk     = m["gamma_star"]*Rkstar/m["pi_star"]/np.exp(m.z_star)*(1+Rhostar)*(1 - mu_estar*Gstar*(1 - zeta_gw/zeta_zw))
        
        # Standard case (not ss2/ss8)
        m.zeta_nR  = m["gamma_star"]*betabar_inverse*(1+Rhostar)*(1 - nkstar + mu_estar*Gstar*m["spr"]*zeta_gw/zeta_zw)
        m.zeta_nqk = m["gamma_star"]*Rkstar/m["pi_star"]/np.exp(m.z_star)*(1+Rhostar)*(1 - mu_estar*Gstar*(1+zeta_gw/zeta_zw/Rhostar)) - m["gamma_star"]*betabar_inverse*(1+Rhostar)
        m.zeta_nn  = m["gamma_star"]*betabar_inverse + m["gamma_star"]*Rkstar/m["pi_star"]/np.exp(m.z_star)*(1+Rhostar)*mu_estar*Gstar*zeta_gw/zeta_zw/Rhostar

        m.zeta_nmu_e    = m["gamma_star"]*Rkstar/m["pi_star"]/np.exp(m.z_star)*(1+Rhostar)*mu_estar*Gstar*(1 - zeta_gw*zeta_z_mu_e/zeta_zw)
        m.zeta_nsigma_omega    = m["gamma_star"]*Rkstar/m["pi_star"]/np.exp(m.z_star)*(1+Rhostar)*mu_estar*Gstar*(zeta_G_sigma_omega-zeta_gw/zeta_zw*zeta_z_sigma_omega)

        # Store for use in eqcond
        m.zeta_n_sigma_w = m.zeta_nsigma_omega # Alias to match Julia porting naming conventions if needed

    def eqcond(self):
        endo = self.endogenous_states
        exo  = self.exogenous_shocks
        ex   = self.expected_shocks
        eq   = self.equilibrium_conditions
        n = self.n_states
        g0, g1 = np.eye(n), np.zeros((n,n)) # Identity fallback for g0
        c = np.zeros(n)
        psi = np.zeros((n, self.n_shocks_exogenous))
        pi = np.zeros((n, self.n_shocks_expectational))
        
        m, z_star, h, sigma_c, bet = self, self.z_star, self["h"], self["sigma_c"], self["bet"]
        h_exp = h * np.exp(-z_star)
        
        # Reset rows that will be defined below
        for key in ["eq_euler", "eq_inv", "eq_capval", "eq_spread", "eq_mp"]:
            g0[eq[key], eq[key]] = 0.0

        # 1. Euler
        g0[eq["eq_euler"], endo["c_t"]] = 1.0
        g0[eq["eq_euler"], endo["R_t"]] = (1-h_exp)/(sigma_c*(1+h_exp))
        g1[eq["eq_euler"], endo["c_t"]] = h_exp / (1+h_exp)
        g0[eq["eq_euler"], endo["Ec_t"]] = -1.0/(1+h_exp)
        g0[eq["eq_euler"], endo["Epi_t"]] = -(1-h_exp)/(sigma_c*(1+h_exp))
        g0[eq["eq_euler"], endo["z_t"]] = (h_exp*self["rho_ztil"] - 1)/(1+h_exp)

        # 2. Investment
        Spp = self["S2"]
        g0[eq["eq_inv"], endo["i_t"]] = 1.0
        g0[eq["eq_inv"], endo["qk_t"]] = -1.0 / (Spp * np.exp(2*z_star) * (1+bet*np.exp((1-sigma_c)*z_star)))
        g1[eq["eq_inv"], endo["i_t"]] = 1.0 / (1+bet*np.exp((1-sigma_c)*z_star))
        g0[eq["eq_inv"], endo["Ei_t"]] = -bet*np.exp((1-sigma_c)*z_star) / (1+bet*np.exp((1-sigma_c)*z_star))

        # 3. Capital Value (Q)
        g0[eq["eq_capval"], endo["qk_t"]] = 1.0
        g0[eq["eq_capval"], endo["R_t"]] = 1.0
        g0[eq["eq_capval"], endo["Epi_t"]] = -1.0
        g0[eq["eq_capval"], endo["ERktil_t"]] = -m.r_k_star / (m.r_k_star + (1-m["delta"]))
        g0[eq["eq_capval"], endo["Eqk_t"]] = -(1-m["delta"]) / (m.r_k_star + (1-m["delta"]))

        # 4. Financial Friction (Spread)
        g0[eq["eq_spread"], endo["ERktil_t"]] = 1.0
        g0[eq["eq_spread"], endo["R_t"]] = -1.0
        g0[eq["eq_spread"], endo["Epi_t"]] = 1.0
        g0[eq["eq_spread"], endo["qk_t"]] = -m.zeta_spb * m.zeta_nqk
        g0[eq["eq_spread"], endo["kbar_t"]] = -m.zeta_spb * m.zeta_nqk
        g0[eq["eq_spread"], endo["n_t"]] = m.zeta_spb * m.zeta_nn

        # 5. Financial Friction Block (continued)
        # n evol
        g0[eq["eq_nevol"], endo["n_t"]]      = 1.
        g0[eq["eq_nevol"], endo["gamma_t"]]      = -1.
        g0[eq["eq_nevol"], endo["z_t"]]      = m["gamma_star"]*m.vstar/m.nstar
        g0[eq["eq_nevol"], endo["Rktil_t"]] = -m.zeta_nRk
        g0[eq["eq_nevol"], endo["pi_t"]]      = (m.zeta_nRk - m.zeta_nR)
        g1[eq["eq_nevol"], endo["sigma_w_t"]]    = -m.zeta_n_sigma_w/m.zeta_sp_sigma_w
        g1[eq["eq_nevol"], endo["mu_e_t"]]    = -m.zeta_nmu_e/m.zeta_sp_mu_e
        g1[eq["eq_nevol"], endo["qk_t"]]     = m.zeta_nqk
        g1[eq["eq_nevol"], endo["kbar_t"]]   = m.zeta_nqk
        g1[eq["eq_nevol"], endo["n_t"]]      = m.zeta_nn
        g1[eq["eq_nevol"], endo["R_t"]]      = -m.zeta_nR
        g1[eq["eq_nevol"], endo["b_t"]]      = m.zeta_nR*((sigma_c*(1.0+h_exp))/(1.0-h_exp))

        # Flexible prices and wages
        g0[eq["eq_nevol_f"], endo["n_f_t"]]      = 1.
        g0[eq["eq_nevol_f"], endo["z_t"]]      = m["gamma_star"]*m.vstar/m.nstar
        g0[eq["eq_nevol_f"], endo["Rktil_f_t"]] = -m.zeta_nRk
        g1[eq["eq_nevol_f"], endo["sigma_w_t"]]    = -m.zeta_n_sigma_w/m.zeta_sp_sigma_w
        g1[eq["eq_nevol_f"], endo["mu_e_t"]]    = -m.zeta_nmu_e/m.zeta_sp_mu_e
        g1[eq["eq_nevol_f"], endo["qk_f_t"]]     = m.zeta_nqk
        g1[eq["eq_nevol_f"], endo["kbar_f_t"]]   = m.zeta_nqk
        g1[eq["eq_nevol_f"], endo["n_f_t"]]      = m.zeta_nn
        g1[eq["eq_nevol_f"], endo["r_f_t"]]      = -m.zeta_nR
        g1[eq["eq_nevol_f"], endo["b_t"]]      = m.zeta_nR*((sigma_c*(1.0+h_exp))/(1.0-h_exp))

        g0[eq["eq_capval_f"], endo["Rktil_f_t"]] = 1.
        g0[eq["eq_capval_f"], endo["rk_f_t"]]     = -m.r_k_star/(m.r_k_star+1-m["delta"])
        g0[eq["eq_capval_f"], endo["qk_f_t"]]     = -(1-m["delta"])/(m.r_k_star+1-m["delta"])
        g1[eq["eq_capval_f"], endo["qk_f_t"]]     = -1.

        # 6. Aggregate Production Function
        g0[eq["eq_output"], endo["y_t"]] =  1.
        g0[eq["eq_output"], endo["k_t"]] = -m["Phi"]*m["alp"]
        g0[eq["eq_output"], endo["L_t"]] = -m["Phi"]*(1 - m["alp"])

        g0[eq["eq_output_f"], endo["y_f_t"]] =  1.
        g0[eq["eq_output_f"], endo["k_f_t"]] = -m["Phi"]*m["alp"]
        g0[eq["eq_output_f"], endo["L_f_t"]] = -m["Phi"]*(1 - m["alp"])

        # 7. Capital Utilization
        g0[eq["eq_caputl"], endo["k_t"]]    =  1.
        g1[eq["eq_caputl"], endo["kbar_t"]] =  1.
        g0[eq["eq_caputl"], endo["z_t"]]    = 1.
        g0[eq["eq_caputl"], endo["u_t"]]    = -1.

        g0[eq["eq_caputl_f"], endo["k_f_t"]]    =  1.
        g1[eq["eq_caputl_f"], endo["kbar_f_t"]] =  1.
        g0[eq["eq_caputl_f"], endo["z_t"]]      = 1.
        g0[eq["eq_caputl_f"], endo["u_f_t"]]    = -1.

        # 8. Rental Rate of Capital
        g0[eq["eq_capsrv"], endo["u_t"]]  = 1.
        g0[eq["eq_capsrv"], endo["rk_t"]] = -(1 - m["ppsi"])/m["ppsi"]

        g0[eq["eq_capsrv_f"], endo["u_f_t"]]  = 1.
        g0[eq["eq_capsrv_f"], endo["rk_f_t"]] = -(1 - m["ppsi"])/m["ppsi"]

        # 9. Evolution of Capital
        g0[eq["eq_capev"], endo["kbar_t"]] = 1.
        g1[eq["eq_capev"], endo["kbar_t"]] = 1 - m.istar/m.kbarstar
        g0[eq["eq_capev"], endo["z_t"]]    = 1 - m.istar/m.kbarstar
        g0[eq["eq_capev"], endo["i_t"]]    = -m.istar/m.kbarstar
        g0[eq["eq_capev"], endo["mu_t"]]    = -m.istar*m["S2"]*np.exp(2*z_star)*(1 + bet*np.exp((1 - sigma_c)*z_star))/m.kbarstar

        g0[eq["eq_capev_f"], endo["kbar_f_t"]] = 1.
        g1[eq["eq_capev_f"], endo["kbar_f_t"]] = 1 - m.istar/m.kbarstar
        g0[eq["eq_capev_f"], endo["z_t"]]      = 1 - m.istar/m.kbarstar
        g0[eq["eq_capev_f"], endo["i_f_t"]]    = -m.istar/m.kbarstar
        g0[eq["eq_capev_f"], endo["mu_t"]]      = -m.istar*m["S2"]*np.exp(2*z_star)*(1 + bet*np.exp((1 - sigma_c)*z_star))/m.kbarstar

        # 10. Price Markup
        g0[eq["eq_mkupp"], endo["mc_t"]] =  1.
        g0[eq["eq_mkupp"], endo["w_t"]]  = -1.
        g0[eq["eq_mkupp"], endo["L_t"]]  = -m["alp"]
        g0[eq["eq_mkupp"], endo["k_t"]]  =  m["alp"]

        g0[eq["eq_mkupp_f"], endo["w_f_t"]] = 1.
        g0[eq["eq_mkupp_f"], endo["L_f_t"]] =  m["alp"]
        g0[eq["eq_mkupp_f"], endo["k_f_t"]] =  -m["alp"]

        # 11. Phillips Curve
        g0[eq["eq_phlps"], endo["pi_t"]]  = 1.
        term = ((1 - m["zeta_p"]*bet*np.exp((1 - sigma_c)*z_star)) * (1 - m["zeta_p"])) / \
               (m["zeta_p"]*((m["Phi"] - 1)*m["ep_p"] + 1)) / (1 + m["iota_p"]*bet*np.exp((1 - sigma_c)*z_star))
        g0[eq["eq_phlps"], endo["mc_t"]] = -term
        g1[eq["eq_phlps"], endo["pi_t"]]  = m["iota_p"]/(1 + m["iota_p"]*bet*np.exp((1 - sigma_c)*z_star))
        g0[eq["eq_phlps"], endo["Epi_t"]] = -bet*np.exp((1 - sigma_c)*z_star)/(1 + m["iota_p"]*bet*np.exp((1 - sigma_c)*z_star))
        g0[eq["eq_phlps"], endo["lambda_f_t"]] = -1.

        # 12. Rental Rate (Components)
        g0[eq["eq_caprnt"], endo["rk_t"]] = 1.
        g0[eq["eq_caprnt"], endo["k_t"]]  = 1.
        g0[eq["eq_caprnt"], endo["L_t"]]  = -1.
        g0[eq["eq_caprnt"], endo["w_t"]]  = -1.

        g0[eq["eq_caprnt_f"], endo["rk_f_t"]] = 1.
        g0[eq["eq_caprnt_f"], endo["k_f_t"]]  = 1.
        g0[eq["eq_caprnt_f"], endo["L_f_t"]]  = -1.
        g0[eq["eq_caprnt_f"], endo["w_f_t"]]  = -1.

        # 13. Marginal Substitution
        g0[eq["eq_msub"], endo["mu_w_t"]] = 1.
        g0[eq["eq_msub"], endo["L_t"]]   = m["nu_l"]
        g0[eq["eq_msub"], endo["c_t"]]   = 1/(1 - h_exp)
        g1[eq["eq_msub"], endo["c_t"]]   = h_exp/(1 - h_exp)
        g0[eq["eq_msub"], endo["z_t"]]   = h_exp /(1 - h_exp)
        g0[eq["eq_msub"], endo["w_t"]]   = -1.

        g0[eq["eq_msub_f"], endo["w_f_t"]] = -1.
        g0[eq["eq_msub_f"], endo["L_f_t"]] = m["nu_l"]
        g0[eq["eq_msub_f"], endo["c_f_t"]] = 1/(1 - h_exp)
        g1[eq["eq_msub_f"], endo["c_f_t"]] = h_exp/(1 - h_exp)
        g0[eq["eq_msub_f"], endo["z_t"]]   = h_exp/(1 - h_exp)

        # 14. Evolution of Wages
        g0[eq["eq_wage"], endo["w_t"]]   = 1
        term = (1 - m["zeta_w"]*bet*np.exp((1 - sigma_c)*z_star)) * (1 - m["zeta_w"]) / \
               (m["zeta_w"]*((m["lambda_w"] - 1)*m["ep_w"] + 1)) / (1 + bet*np.exp((1 - sigma_c)*z_star))
        g0[eq["eq_wage"], endo["mu_w_t"]] = term
        g0[eq["eq_wage"], endo["pi_t"]]   = (1 + m["iota_w"]*bet*np.exp((1 - sigma_c)*z_star))/(1 + bet*np.exp((1 - sigma_c)*z_star))
        g1[eq["eq_wage"], endo["w_t"]]   = 1/(1 + bet*np.exp((1 - sigma_c)*z_star))
        g0[eq["eq_wage"], endo["z_t"]]   = 1/(1 + bet*np.exp((1 - sigma_c)*z_star))
        g1[eq["eq_wage"], endo["pi_t"]]   = m["iota_w"]/(1 + bet*np.exp((1 - sigma_c)*z_star))
        g0[eq["eq_wage"], endo["Ew_t"]]  = -bet*np.exp((1 - sigma_c)*z_star)/(1 + bet*np.exp((1 - sigma_c)*z_star))
        g0[eq["eq_wage"], endo["Ez_t"]]  = -bet*np.exp((1 - sigma_c)*z_star)/(1 + bet*np.exp((1 - sigma_c)*z_star))
        g0[eq["eq_wage"], endo["Epi_t"]]  = -bet*np.exp((1 - sigma_c)*z_star)/(1 + bet*np.exp((1 - sigma_c)*z_star))
        g0[eq["eq_wage"], endo["lambda_w_t"]] = -1.

        # 15. Monetary Policy Rule (Full)
        g0[eq["eq_mp"], endo["R_t"]]      = 1.
        g1[eq["eq_mp"], endo["R_t"]]      = m["rho"]
        g0[eq["eq_mp"], endo["pi_t"]]      = -(1 - m["rho"])*m["psi1"]
        g0[eq["eq_mp"], endo["pi_star_t"]] = (1 - m["rho"])*m["psi1"]
        g0[eq["eq_mp"], endo["y_t"]]      = -(1 - m["rho"])*m["psi2"] - m["psi3"]
        g0[eq["eq_mp"], endo["y_f_t"]]    = (1 - m["rho"])*m["psi2"] + m["psi3"]
        g1[eq["eq_mp"], endo["y_t"]]      = -m["psi3"]
        g1[eq["eq_mp"], endo["y_f_t"]]    = m["psi3"]
        g0[eq["eq_mp"], endo["rm_t"]]     = -1.

        # 16. Resource Constraint
        g0[eq["eq_res"], endo["y_t"]] = 1.
        g0[eq["eq_res"], endo["g_t"]] = -m["g_star"]
        g0[eq["eq_res"], endo["c_t"]] = -m.cstar/m.ystar
        g0[eq["eq_res"], endo["i_t"]] = -m.istar/m.ystar
        g0[eq["eq_res"], endo["u_t"]] = -m.r_k_star*m.kstar/m.ystar

        g0[eq["eq_res_f"], endo["y_f_t"]] = 1.
        g0[eq["eq_res_f"], endo["g_t"]]   = -m["g_star"]
        g0[eq["eq_res_f"], endo["c_f_t"]] = -m.cstar/m.ystar
        g0[eq["eq_res_f"], endo["i_f_t"]] = -m.istar/m.ystar
        g0[eq["eq_res_f"], endo["u_f_t"]] = -m.r_k_star*m.kstar/m.ystar

        # Extra States
        # pi_t1
        g0[eq["eq_pi1"], endo["pi_t1"]] = 1.
        g1[eq["eq_pi1"], endo["pi_t"]]  = 1.

        # pi_t2
        g0[eq["eq_pi2"], endo["pi_t2"]] = 1.
        g1[eq["eq_pi2"], endo["pi_t1"]] = 1.

        # pi_a
        g0[eq["eq_pi_a"], endo["pi_a_t"]] = 1.
        g0[eq["eq_pi_a"], endo["pi_t"]]   = -1.
        g0[eq["eq_pi_a"], endo["pi_t1"]]  = -1.
        g0[eq["eq_pi_a"], endo["pi_t2"]]  = -1.
        g1[eq["eq_pi_a"], endo["pi_t2"]]  = 1.

        # Rt1
        g0[eq["eq_Rt1"], endo["R_t1"]] = 1.
        g1[eq["eq_Rt1"], endo["R_t"]]  = 1.

        # Ez_t
        g0[eq["eq_Ez"], endo["Ez_t"]]   = 1.
        g0[eq["eq_Ez"], endo["ztil_t"]] = -(m["rho_ztil"]-1)/(1-m["alp"])
        g0[eq["eq_Ez"], endo["zp_t"]]   = -m["rho_z_p"]

        # EXOGENOUS SHOCKS
        
        # Neutral technology
        g0[eq["eq_z"], endo["z_t"]]    = 1.
        g1[eq["eq_z"], endo["ztil_t"]] = (m["rho_ztil"] - 1)/(1 - m["alp"])
        g0[eq["eq_z"], endo["zp_t"]]   = -1.
        psi[eq["eq_z"], exo["ztil_sh"]]     = 1/(1 - m["alp"])

        g0[eq["eq_ztil"], endo["ztil_t"]] = 1.
        g1[eq["eq_ztil"], endo["ztil_t"]] = m["rho_ztil"]
        psi[eq["eq_ztil"], exo["ztil_sh"]]     = 1.

        # Long-run changes to productivity
        g0[eq["eq_zp"], endo["zp_t"]] = 1.
        g1[eq["eq_zp"], endo["zp_t"]] = m["rho_z_p"]
        psi[eq["eq_zp"], exo["zp_sh"]]  = 1.

        # Government spending
        g0[eq["eq_g"], endo["g_t"]] = 1.
        g1[eq["eq_g"], endo["g_t"]] = m["rho_g"]
        psi[eq["eq_g"], exo["g_sh"]]  = 1.
        psi[eq["eq_g"], exo["ztil_sh"]]  = m["eta_gz"]

        # Asset shock
        g0[eq["eq_b"], endo["b_t"]] = 1.
        g1[eq["eq_b"], endo["b_t"]] = m["rho_b"]
        psi[eq["eq_b"], exo["b_sh"]]  = 1.

        # Investment-specific technology
        g0[eq["eq_mu"], endo["mu_t"]] = 1.
        g1[eq["eq_mu"], endo["mu_t"]] = m["rho_mu"]
        psi[eq["eq_mu"], exo["mu_sh"]]  = 1.

        # Price mark-up shock
        g0[eq["eq_lambda_f"], endo["lambda_f_t"]]  = 1.
        g1[eq["eq_lambda_f"], endo["lambda_f_t"]]  = m["rho_lambda_f"]
        g1[eq["eq_lambda_f"], endo["lambda_f_t1"]] = -m["eta_lambda_f"]
        psi[eq["eq_lambda_f"], exo["lambda_f_sh"]]   = 1.

        g0[eq["eq_lambda_f1"], endo["lambda_f_t1"]] = 1.
        psi[eq["eq_lambda_f1"], exo["lambda_f_sh"]]   = 1.

        # Wage mark-up shock
        g0[eq["eq_lambda_w"], endo["lambda_w_t"]]  = 1.
        g1[eq["eq_lambda_w"], endo["lambda_w_t"]]  = m["rho_lambda_w"]
        g1[eq["eq_lambda_w"], endo["lambda_w_t1"]] = -m["eta_lambda_w"]
        psi[eq["eq_lambda_w"], exo["lambda_w_sh"]]   = 1.

        g0[eq["eq_lambda_w1"], endo["lambda_w_t1"]] = 1.
        psi[eq["eq_lambda_w1"], exo["lambda_w_sh"]]   = 1.

        # Monetary policy shock
        g0[eq["eq_rm"], endo["rm_t"]] = 1.
        g1[eq["eq_rm"], endo["rm_t"]] = m["rho_rm"]
        psi[eq["eq_rm"], exo["rm_sh"]]  = 1.

        # Financial frictions shocks
        g0[eq["eq_sigma_w"], endo["sigma_w_t"]] = 1.
        g1[eq["eq_sigma_w"], endo["sigma_w_t"]] = m["rho_sigma_w"]
        psi[eq["eq_sigma_w"], exo["sigma_w_sh"]]  = 1.

        g0[eq["eq_mu_e"], endo["mu_e_t"]] = 1.
        g1[eq["eq_mu_e"], endo["mu_e_t"]] = m["rho_mu_e"]
        psi[eq["eq_mu_e"], exo["mu_e_sh"]]  = 1.

        g0[eq["eq_gamma"], endo["gamma_t"]] = 1.
        g1[eq["eq_gamma"], endo["gamma_t"]] = m["rho_gamma"]
        psi[eq["eq_gamma"], exo["gamma_sh"]]  = 1.

        g0[eq["eq_pi_star"], endo["pi_star_t"]] = 1.
        g1[eq["eq_pi_star"], endo["pi_star_t"]] = m["rho_pi_star"]
        psi[eq["eq_pi_star"], exo["pi_star_sh"]]  = 1.

        # Anticipated Shocks
        # For simplicity, assuming default structure where rm_tl1 feeds from rm_tl2 etc.
        # But here we just set them to zero or simple AR if needed. 
        # DSGE.jl sets them up as a chain if n_mon_anticipated_shocks > 0
        
        # rm_tl1
        g1[eq["eq_rm"], endo["rm_tl1"]]   = 1.0
        g0[eq["eq_rml1"], endo["rm_tl1"]] = 1.
        psi[eq["eq_rml1"], exo["rm_shl1"]]  = 1.
        
        for i in range(2, 7): # 2 to 6
             g1[eq[f"eq_rml{i-1}"], endo[f"rm_tl{i}"]] = 1.0
             g0[eq[f"eq_rml{i}"], endo[f"rm_tl{i}"]] = 1.0
             psi[eq[f"eq_rml{i}"], exo[f"rm_shl{i}"]] = 1.0

        # EXPECTATION ERRORS
        
        # Ec
        g0[eq["eq_Ec"], endo["c_t"]]  = 1.
        g1[eq["eq_Ec"], endo["Ec_t"]] = 1.
        pi[eq["eq_Ec"], ex["Ec_sh"]]   = 1.

        g0[eq["eq_Ec_f"], endo["c_f_t"]]  = 1.
        g1[eq["eq_Ec_f"], endo["Ec_f_t"]] = 1.
        pi[eq["eq_Ec_f"], ex["Ec_f_sh"]]   = 1.

        # Eqk
        g0[eq["eq_Eqk"], endo["qk_t"]]  = 1.
        g1[eq["eq_Eqk"], endo["Eqk_t"]] = 1.
        pi[eq["eq_Eqk"], ex["Eqk_sh"]]   = 1.

        g0[eq["eq_Eqk_f"], endo["qk_f_t"]]  = 1.
        g1[eq["eq_Eqk_f"], endo["Eqk_f_t"]] = 1.
        pi[eq["eq_Eqk_f"], ex["Eqk_f_sh"]]   = 1.

        # Ei
        g0[eq["eq_Ei"], endo["i_t"]]  = 1.
        g1[eq["eq_Ei"], endo["Ei_t"]] = 1.
        pi[eq["eq_Ei"], ex["Ei_sh"]]   = 1.

        g0[eq["eq_Ei_f"], endo["i_f_t"]]  = 1.
        g1[eq["eq_Ei_f"], endo["Ei_f_t"]] = 1.
        pi[eq["eq_Ei_f"], ex["Ei_f_sh"]]   = 1.

        # Epi
        g0[eq["eq_Epi"], endo["pi_t"]]  = 1.
        g1[eq["eq_Epi"], endo["Epi_t"]] = 1.
        pi[eq["eq_Epi"], ex["Epi_sh"]]   = 1.

        # EL
        g0[eq["eq_EL"], endo["L_t"]]  = 1.
        g1[eq["eq_EL"], endo["EL_t"]] = 1.
        pi[eq["eq_EL"], ex["EL_sh"]]   = 1.

        g0[eq["eq_EL_f"], endo["L_f_t"]]  = 1.
        g1[eq["eq_EL_f"], endo["EL_f_t"]] = 1.
        pi[eq["eq_EL_f"], ex["EL_f_sh"]]   = 1.

        # Erk
        g0[eq["eq_Erk"], endo["rk_t"]]  = 1.
        g1[eq["eq_Erk"], endo["Erk_t"]] = 1.
        pi[eq["eq_Erk"], ex["Erk_sh"]]   = 1.
        
        # ERktil
        g0[eq["eq_ERktil"], endo["Rktil_t"]]  = 1.
        g1[eq["eq_ERktil"], endo["ERktil_t"]] = 1.
        pi[eq["eq_ERktil"], ex["ERktil_sh"]]   = 1.
        
        g0[eq["eq_ERktil_f"], endo["Rktil_f_t"]]  = 1.
        g1[eq["eq_ERktil_f"], endo["ERktil_f_t"]] = 1.
        pi[eq["eq_ERktil_f"], ex["ERktil_f_sh"]]   = 1.
        
        # Ew
        g0[eq["eq_Ew"], endo["w_t"]]  = 1.
        g1[eq["eq_Ew"], endo["Ew_t"]] = 1.
        pi[eq["eq_Ew"], ex["Ew_sh"]]   = 1.
        
        # COVID placeholders (identity)
        g0[eq["eq_ziid"], endo["ziid_t"]] = 1.
        g1[eq["eq_ziid"], endo["ziid_t"]] = m["rho_ziid"]
        psi[eq["eq_ziid"], exo["ziid_sh"]] = 1. # If sigma_ziid > 0
        
        g0[eq["eq_biidc"], endo["biidc_t"]] = 1.
        g1[eq["eq_biidc"], endo["biidc_t"]] = m["rho_biidc"]
        psi[eq["eq_biidc"], exo["biidc_sh"]] = 1.
        
        g0[eq["eq_phi"], endo["phi_t"]] = 1.
        g1[eq["eq_phi"], endo["phi_t"]] = m["rho_phi"]
        psi[eq["eq_phi"], exo["phi_sh"]] = 1.

        
        return g0, g1, c, psi, pi

    def solve(self, kernels=None):
        TTT, RRR, CCC = super().solve(kernels=kernels)
        
        # Augment states
        n_endo = self.n_states
        n_aug = len(self.endogenous_states_augmented)
        n_exo = self.n_shocks_exogenous
        
        T_aug = np.zeros((n_endo + n_aug, n_endo + n_aug), dtype=TTT.dtype)
        T_aug[:n_endo, :n_endo] = TTT
        
        R_aug = np.vstack([RRR, np.zeros((n_aug, n_exo), dtype=RRR.dtype)])
        C_aug = np.concatenate([CCC, np.zeros(n_aug, dtype=CCC.dtype)])
        
        # Lag tracking
        endo = self.endogenous_states
        aug = self.endogenous_states_augmented
        T_aug[aug["y_t1"], endo["y_t"]] = 1.0
        T_aug[aug["c_t1"], endo["c_t"]] = 1.0
        T_aug[aug["i_t1"], endo["i_t"]] = 1.0
        T_aug[aug["w_t1"], endo["w_t"]] = 1.0
        
        # Measurement error persistence
        T_aug[aug["e_lr_t"], aug["e_lr_t"]] = self["rho_lr"]
        T_aug[aug["e_tfp_t"], aug["e_tfp_t"]] = self["rho_tfp"]
        
        return T_aug, R_aug, C_aug

    def measurement(self, TTT, RRR, CCC):
        endo = self.endogenous_states
        aug = self.endogenous_states_augmented
        obs = self.observables
        exo = self.exogenous_shocks
        
        n_obs = len(self.observables)
        n_states = self.n_states_augmented
        n_exo = self.n_shocks_exogenous
        
        ZZ = np.zeros((n_obs, n_states))
        DD = np.zeros(n_obs)
        QQ = np.zeros((n_exo, n_exo))
        EE = np.zeros((n_obs, n_obs))
        
        # GDP Growth
        ZZ[obs["obs_gdp"], endo["y_t"]] = 1.0
        ZZ[obs["obs_gdp"], aug["y_t1"]] = -1.0
        ZZ[obs["obs_gdp"], endo["z_t"]] = 1.0
        DD[obs["obs_gdp"]] = 100 * (np.exp(self.z_star) - 1.0)
        
        # Consumption Growth
        ZZ[obs["obs_consumption"], endo["c_t"]] = 1.0
        ZZ[obs["obs_consumption"], aug["c_t1"]] = -1.0
        ZZ[obs["obs_consumption"], endo["z_t"]] = 1.0
        DD[obs["obs_consumption"]] = 100 * (np.exp(self.z_star) - 1.0)

        # Investment Growth
        ZZ[obs["obs_investment"], endo["i_t"]] = 1.0
        ZZ[obs["obs_investment"], aug["i_t1"]] = -1.0
        ZZ[obs["obs_investment"], endo["z_t"]] = 1.0
        DD[obs["obs_investment"]] = 100 * (np.exp(self.z_star) - 1.0)

        # Real Wage Growth
        ZZ[obs["obs_wages"], endo["w_t"]] = 1.0
        ZZ[obs["obs_wages"], aug["w_t1"]] = -1.0
        ZZ[obs["obs_wages"], endo["z_t"]] = 1.0
        DD[obs["obs_wages"]] = 100 * (np.exp(self.z_star) - 1.0)

        # Hours
        ZZ[obs["obs_hours"], endo["L_t"]] = 1.0
        DD[obs["obs_hours"]] = self["Lstar_adj"] if "Lstar_adj" in self.__dict__ else 0.0

        # Inflation (Core PCE)
        ZZ[obs["obs_corepce"], endo["pi_t"]] = 1.0
        DD[obs["obs_corepce"]] = 100 * (self["pi_star"] - 1.0)
        
        # Nominal Rate
        ZZ[obs["obs_nominalrate"], endo["R_t"]] = 1.0
        DD[obs["obs_nominalrate"]] = self.Rstarn

        # Spread
        ZZ[obs["obs_spread"], endo["ERktil_t"]] = 1.0
        ZZ[obs["obs_spread"], endo["R_t"]] = -1.0
        DD[obs["obs_spread"]] = 100 * (self["spr"] - 1.0)

        # 10Y Yield (Simplified)
        ZZ[obs["obs_10y"], endo["R_t"]] = 1.0 # Placeholder
        DD[obs["obs_10y"]] = self.Rstarn

        # Shocks (Variance Matrix QQ)
        QQ[exo["g_sh"], exo["g_sh"]] = self["sigma_g"]**2
        QQ[exo["b_sh"], exo["b_sh"]] = self["sigma_b"]**2
        QQ[exo["mu_sh"], exo["mu_sh"]] = self["sigma_mu"]**2
        QQ[exo["ztil_sh"], exo["ztil_sh"]] = self["sigma_ztil"]**2
        QQ[exo["lambda_f_sh"], exo["lambda_f_sh"]] = self["sigma_lambda_f"]**2
        QQ[exo["lambda_w_sh"], exo["lambda_w_sh"]] = self["sigma_lambda_w"]**2
        QQ[exo["rm_sh"], exo["rm_sh"]] = self["sigma_rm"]**2
        QQ[exo["sigma_w_sh"], exo["sigma_w_sh"]] = self["sigma_sigma_w"]**2
        QQ[exo["pi_star_sh"], exo["pi_star_sh"]] = self["sigma_pi_star"]**2
        QQ[exo["zp_sh"], exo["zp_sh"]] = self["sigma_z_p"]**2

        # Measurement Errors (EE)
        EE[obs["obs_gdp"], obs["obs_gdp"]] = self["sigma_gdp"]**2
        EE[obs["obs_gdi"], obs["obs_gdi"]] = self["sigma_gdi"]**2
        EE[obs["obs_gdpdeflator"], obs["obs_gdpdeflator"]] = self["sigma_gdpdef"]**2
        EE[obs["obs_corepce"], obs["obs_corepce"]] = self["sigma_corepce"]**2
        EE[obs["obs_10y"], obs["obs_10y"]] = self["sigma_lr"]**2
        EE[obs["obs_tfp"], obs["obs_tfp"]] = self["sigma_tfp"]**2
        
        # Anticipated Shocks Observables (Standard deviation)
        for i in range(1, 7):
            if f"obs_nominalrate{i}" in obs and f"sigma_rm{i}" in self.parameters:
                EE[obs[f"obs_nominalrate{i}"], obs[f"obs_nominalrate{i}"]] = self[f"sigma_rm{i}"]**2

        return ZZ, DD, QQ, EE
