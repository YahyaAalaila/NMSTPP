import numpy as np

from model.base import BasePointProcess
from model.kernel import DeepBasisKernel, DeepFourierKernel, ExponentialDecayingKernel, SpatioTemporalDeepBasisKernel


class ExpDecayingHawkes(BasePointProcess):
    """
    Hawkes process with exponentially decaying kernel
    """
    def __init__(self, T, S, beta, mu, data_dim, numerical_int=True, int_res=100):
        super(ExpDecayingHawkes, self).__init__(T, S, data_dim, numerical_int, int_res)
        self.kernel = ExponentialDecayingKernel(beta)
        self._mu    = mu
    
    def mu(self):
        """
        return base intensity
        """
        return self._mu


class DeepFourierPointProcess(BasePointProcess):
    """
    Point Process with Deep Fourier Kernel
    """
    def __init__(self, T, S, noise_dim, data_dim, numerical_int=True, int_res=100):
        """
        Args:
        - T:             time horizon. e.g. (0, 1)
        - S:             bounded space for marks. e.g. a two dimensional box region [(0, 1), (0, 1)]
        - noise_dim:     dimension of noise prior
        - data_dim:      dimension of input data
        - numerical_int: numerical integral flag
        - int_res:       numerical integral resolution
        """
        super(DeepFourierPointProcess, self).__init__(T, S, data_dim, numerical_int, int_res)
        # configuration
        self.noise_dim = noise_dim
        # deep fourier kernel
        self.kernel    = DeepFourierKernel(self.noise_dim, self.data_dim)
    
    def mu(self):
        """
        return base intensity
        """
        return 1.

    def forward(self, X, n_sampled_fouriers=200):
        """
        custom forward function returning conditional intensities and corresponding log-likelihood
        """
        # precompute fourier features
        self.kernel.precompute_fourier_features(n_sampled_fouriers=n_sampled_fouriers)
        # return conditional intensities and corresponding log-likelihood
        return self.log_likelihood(X)



class DeepBasisPointProcess(BasePointProcess):
    """
    Point Process with Deep NN basis
    """
    def __init__(self, 
                 T, S, mu, 
                 n_basis, basis_dim, data_dim, 
                 numerical_int=True, int_res=100, 
                 init_gain=5e-1, init_bias=1e-3, init_std=1.,
                 nn_width=5):
        """
        Args:
        - T:             time horizon. e.g. (0, 1)
        - S:             bounded space for marks. e.g. a two dimensional box region [(0, 1), (0, 1)]
        - n_basis:       number of basis functions
        - basis_dim:     dimension of basis function
        - data_dim:      dimension of input data
        - numerical_int: numerical integral flag
        - int_res:       numerical integral resolution
        - nn_width:      the width of each layer in kernel basis NN
        """
        super(DeepBasisPointProcess, self).__init__(T, S, data_dim, numerical_int, int_res)
        # configuration
        self.n_basis   = n_basis
        self.basis_dim = basis_dim
        self._mu       = mu
        # deep nn basis kernel
        self.kernel    = DeepBasisKernel(n_basis, data_dim, basis_dim, 
                                         init_gain=init_gain, init_bias=init_bias, init_std=init_std,
                                         nn_width=nn_width)
    
    def mu(self):
        """
        return base intensity
        """
        return self._mu

    def forward(self, X, n_sampled_fouriers=200):
        """
        custom forward function returning conditional intensities and corresponding log-likelihood
        """
        # return conditional intensities and corresponding log-likelihood
        return self.log_likelihood(X)



class SpatioTemporalDeepBasisPointProcess(BasePointProcess):
    """
    Spatio-temporal Point Process with Deep NN basis
    """
    def __init__(self, 
                 T, S, 
                 n_basis, basis_dim, data_dim, 
                 init_gain=5e-1, init_bias=1e-3, init_std=1.,
                 numerical_int=True, int_res=100, nn_width=5, beta=1):
        """
        See `DeepBasisPointProcess` for more details
        """
        super(SpatioTemporalDeepBasisPointProcess, self).__init__(T, S, data_dim, numerical_int, int_res)
        # configuration
        self.n_basis   = n_basis
        self.basis_dim = basis_dim
        # spatio-temporal deep nn basis kernel
        self.kernel    = SpatioTemporalDeepBasisKernel(n_basis, data_dim, basis_dim, 
                                                       init_gain=init_gain, init_bias=init_bias, init_std=init_std,
                                                       nn_width=nn_width, beta=beta)
    
    def mu(self):
        """
        return base intensity
        """
        return 1.

    def forward(self, X, n_sampled_fouriers=200):
        """
        custom forward function returning conditional intensities and corresponding log-likelihood
        """
        # return conditional intensities and corresponding log-likelihood
        return self.log_likelihood(X)
