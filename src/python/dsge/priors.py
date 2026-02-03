import numpy as np
import scipy.stats as stats

class AbstractPrior:
    def logpdf(self, x):
        raise NotImplementedError
    
    def pdf(self, x):
        return np.exp(self.logpdf(x))

class Normal(AbstractPrior):
    """
    Normal distribution prior.
    params: mu (mean), sigma (std)
    """
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
    
    def logpdf(self, x):
        return stats.norm.logpdf(x, loc=self.mu, scale=self.sigma)
    
    def __repr__(self):
        return f"Normal(mu={self.mu}, sigma={self.sigma})"

class Gamma(AbstractPrior):
    """
    Gamma distribution prior.
    params: a (shape, alpha), b (scale, theta = 1/beta)
    """
    def __init__(self, a, b):
        self.a = a
        self.b = b
    
    def logpdf(self, x):
        return stats.gamma.logpdf(x, a=self.a, scale=self.b)
    
    def __repr__(self):
        return f"Gamma(a={self.a}, b={self.b})"

class Beta(AbstractPrior):
    """
    Beta distribution prior.
    params: a (alpha), b (beta)
    """
    def __init__(self, a, b):
        self.a = a
        self.b = b
    
    def logpdf(self, x):
        return stats.beta.logpdf(x, self.a, self.b)

    def __repr__(self):
        return f"Beta(a={self.a}, b={self.b})"

class InverseGamma(AbstractPrior):
    """
    Inverse Gamma distribution prior.
    scipy.stats.invgamma uses shape 'a' and scale 'scale'.
    pdf(x) = x**(-a-1) / gamma(a) * exp(-1/x) (standardized)
    
    Common DSGE param: s (std), nu (degrees of freedom)
    This implementation wraps scipy directly for now with a/scale.
    """
    def __init__(self, a, scale):
        self.a = a
        self.scale = scale
    
    def logpdf(self, x):
        return stats.invgamma.logpdf(x, a=self.a, scale=self.scale)

    def __repr__(self):
        return f"InverseGamma(a={self.a}, scale={self.scale})"

class Uniform(AbstractPrior):
    """
    Uniform distribution prior.
    params: a (min), b (max)
    """
    def __init__(self, a, b):
        self.a = a
        self.b = b
        
    def logpdf(self, x):
        return stats.uniform.logpdf(x, loc=self.a, scale=self.b-self.a)

    def __repr__(self):
        return f"Uniform(a={self.a}, b={self.b})"
