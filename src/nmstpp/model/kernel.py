import torch

class ExponentialDecayingKernel(torch.nn.Module):
    """
    Exponential Decaying Kernel
    """
    def __init__(self, beta):
        """
        Arg:
        - beta: decaying rate
        """
        super(ExponentialDecayingKernel, self).__init__()
        self._beta = beta # torch.nn.Parameter(torch.rand([1]), requires_grad=True)
        # self._beta = torch.nn.Parameter(torch.rand([1]), requires_grad=True)
    
    def forward(self, x, y):
        """
        customized forward function returning kernel evaluation at x and y with 
        size [ batch_size, batch_size ], where
        - x: the first input with size  [ batch_size, data_dim ]
        - y: the second input with size [ batch_size, data_dim ] 
        """
        # return torch.nn.functional.softplus(self._beta * torch.exp(- self._beta * (x - y)))
        return self._beta * torch.exp(- self._beta * torch.abs(x - y))
        


class DeepNetworkBasis(torch.nn.Module):
    """
    Deep Neural Network Basis Kernel

    This class directly models the kernel-induced feature mapping by a deep 
    neural network.
    """
    def __init__(self, data_dim, basis_dim, 
                 init_gain=5e-1, init_bias=1e-3, nn_width=5):
        """
        Args:
        - data_dim:  dimension of input data point
        - basis_dim: dimension of basis function
        - nn_width:  the width of each layer in NN
        """
        super(DeepNetworkBasis, self).__init__()
        # configurations
        self.data_dim  = data_dim
        self.basis_dim = basis_dim
        # init parameters for net
        self.init_gain   = init_gain
        self.init_bias   = init_bias
        # network for basis function
        self.net = torch.nn.Sequential(
            torch.nn.Linear(data_dim, nn_width),  # [ data_dim, n_hidden_nodes ]
            torch.nn.Softplus(), 
            torch.nn.Linear(nn_width, nn_width),  # [ n_hidden_nodes, n_hidden_nodes ]
            torch.nn.Softplus(), 
            torch.nn.Linear(nn_width, nn_width),  # [ n_hidden_nodes, n_hidden_nodes ]
            torch.nn.Softplus(), 
            torch.nn.Linear(nn_width, basis_dim), # [ n_hidden_nodes, basis_dim ]
            torch.nn.Sigmoid())
        self.net.apply(self.init_weights)

    def init_weights(self, m):
        """
        initialize weight matrices in network
        """
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight, gain=self.init_gain)
            m.bias.data.fill_(self.init_bias)

    def forward(self, x):
        """
        customized forward function returning basis function evaluated at x
        with size [ batch_size, data_dim ]
        """
        return self.net(x) * 2 - 1         # [ batch_size, basis_dim ]



class DeepBasisKernel(torch.nn.Module):
    """
    Deep Basis Kernel
    """
    def __init__(self, 
                 n_basis, data_dim, basis_dim, 
                 init_gain=5e-1, init_bias=1e-3, init_std=1.,
                 nn_width=5):
        """
        Arg:
        - n_basis:   number of basis functions
        - data_dim:  dimension of input data point
        - basis_dim: dimension of basis function
        - nn_width:  the width of each layer in basis NN
        """
        super(DeepBasisKernel, self).__init__()
        # configurations
        self.n_basis   = n_basis
        self.data_dim  = data_dim
        self.basis_dim = basis_dim
        # set of basis functions and corresponding weights
        self.xbasiss   = torch.nn.ModuleList([])
        self.ybasiss   = torch.nn.ModuleList([])
        self.weights   = torch.nn.ParameterList([])
        for i in range(n_basis):
            self.xbasiss.append(DeepNetworkBasis(data_dim, basis_dim, 
                                                 init_gain=init_gain, init_bias=init_bias,
                                                 nn_width=nn_width))
            self.ybasiss.append(DeepNetworkBasis(data_dim, basis_dim, 
                                                 init_gain=init_gain, init_bias=init_bias,
                                                 nn_width=nn_width))
            self.weights.append(torch.nn.Parameter(torch.empty(1).normal_(mean=0,std=init_std), requires_grad=True))
            
    def forward(self, x, y):
        """
        customized forward function returning kernel evaluation at x and y with 
        size [ batch_size, batch_size ], where
        - x: the first input with size  [ batch_size, data_dim ]
        - y: the second input with size [ batch_size, data_dim ] 
        """
        K = []
        for weight, xbasis, ybasis in zip(self.weights, self.xbasiss, self.ybasiss):
            xbasis_func = xbasis(x)                                   # [ batch_size, basis_dim ]
            ybasis_func = ybasis(y)                                   # [ batch_size, basis_dim ]
            weight      = torch.nn.functional.softplus(weight)        # scalar
            ki          = (weight * xbasis_func * ybasis_func).sum(1) # [ batch_size ]
            K.append(ki)
        K = torch.stack(K, 1).sum(1)
        return K



class SpatioTemporalDeepBasisKernel(torch.nn.Module):
    """
    Spatio-temporal Deep Basis Kernel
    """
    def __init__(self, 
                 n_basis, data_dim, basis_dim, 
                 init_gain=5e-1, init_bias=1e-3, init_std=1.,
                 nn_width=5, beta=1):
        """
        Arg:
        - n_basis:   number of basis functions
        - data_dim:  dimension of input data point
        - basis_dim: dimension of basis function
        - nn_width:  the width of each layer in basis NN
        """
        super(SpatioTemporalDeepBasisKernel, self).__init__()
        # decoupled kernels
        self.spatialkernel  = DeepBasisKernel(n_basis, data_dim-1, basis_dim, 
                                              init_gain=init_gain, init_bias=init_bias, init_std=init_std,
                                              nn_width=nn_width)
        # self.temporalkernel = ExponentialDecayingKernel(beta)
        self.beta = beta

    def temporalkernel(self, x, y):
        return self.beta * torch.exp(- self.beta * (x - y))
      
    def forward(self, x, y):
        """
        customized forward function returning kernel evaluation at x and y with 
        size [ batch_size, batch_size ], where
        - x: the first input with size  [ batch_size, data_dim ]
        - y: the second input with size [ batch_size, data_dim ] 
        """
        xt, yt = x[:, 0].clone(), y[:, 0].clone()   # [ batch_size ]
        xs, ys = x[:, 1:].clone(), y[:, 1:].clone() # [ batch_size, data_dim - 1 ]
        tval   = self.temporalkernel(xt, yt)        # [ batch_size ]
        sval   = self.spatialkernel(xs, ys)         # [ batch_size ]
        return tval * sval                          # [ batch_size ]


class DeepJointFourierSpectrum(torch.nn.Module):
    """
    Deep Joint Fourier Spectrum

    This class models the joint Fourier spectrum p(w, u), which is approximated 
    by sampling a number of fourier features from a deep neural network.
    """
    def __init__(self, noise_dim, fourier_dim, 
                 noise_mean=0, noise_std=1e+1,
                 init_gain=1e-1, init_bias=1e-3):
        """
        Args:
        - noise_dim:   dimension of input noise
        - fourier_dim: dimension of two output fourier features
        """
        super(DeepJointFourierSpectrum, self).__init__()
        # configurations
        self.noise_dim   = noise_dim
        self.fourier_dim = fourier_dim
        # noise parameters
        self.noise_mean  = noise_mean
        self.noise_std   = noise_std
        # init parameters for net
        self.init_gain   = init_gain
        self.init_bias   = init_bias
        self.net         = torch.nn.Sequential(
            torch.nn.Linear(noise_dim, 100),       # [ n_sampled_fouriers, n_hidden_nodes ]
            torch.nn.Softplus(), 
            torch.nn.Linear(100, 100),             # [ n_hidden_nodes, n_hidden_nodes ]
            torch.nn.Softplus(), 
            torch.nn.Linear(100, 100),             # [ n_hidden_nodes, n_hidden_nodes ]
            torch.nn.Softplus(), 
            torch.nn.Linear(100, fourier_dim * 2), # [ n_hidden_nodes, 2 * fourier_dim ]
            torch.nn.Sigmoid())
        self.net.apply(self.init_weights)

    def init_weights(self, m):
        """
        initialize weight matrices in network
        """
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight, gain=self.init_gain)
            m.bias.data.fill_(self.init_bias)

    def forward(self, n_sampled_fouriers):
        """
        customized forward function returning `n_sampled_fouriers` fourier 
        features w and u.
        """
        noise    = torch.FloatTensor(n_sampled_fouriers, self.noise_dim).normal_(mean=self.noise_mean,std=self.noise_std)
        fouriers = self.net(noise) * 2 - 1                # rescale output to (-1, 1)
        ws       = fouriers[:, :self.fourier_dim].clone() # [ n_sampled_fouriers, fourier_dim ]
        us       = fouriers[:, self.fourier_dim:].clone() # [ n_sampled_fouriers, fourier_dim ]
        return ws, us



class DeepFourierKernel(torch.nn.Module):
    """
    Deep Non-Stationary Fourier Kernel k(x, y)
    """
    def __init__(self, noise_dim, data_dim):
        """
        Args:
        - noise_dim:          dimension of noise prior
        - data_dim:           dimension of input data
        """
        super(DeepFourierKernel, self).__init__()
        # configuration
        self.noise_dim   = noise_dim          # dimension of noise prior
        self.data_dim    = data_dim           # dimension of input data
        self.precomputed = False
        # fourier specturm specified by a deep neural network
        self.p  = DeepJointFourierSpectrum(self.noise_dim, self.data_dim)
        self.q  = DeepJointFourierSpectrum(self.noise_dim, self.data_dim)
        # scalar parameters
        # Note: set constant to make kernel simpler.
        self.c1 = 1e-4 # torch.nn.Parameter(torch.randn((1), requires_grad=True))
        self.c2 = 1e-4 # torch.nn.Parameter(torch.randn((1), requires_grad=True))

    def precompute_fourier_features(self, n_sampled_fouriers):
        """
        Precompute fourier features
        """
        # sampled fourier features
        self.pre_ws_p, self.pre_us_p = self.p(n_sampled_fouriers)  # 2 * [ n_sampled_fouriers, fourier_dim ]
        self.pre_ws_q, self.pre_us_q = self.q(n_sampled_fouriers)  # 2 * [ n_sampled_fouriers, fourier_dim ]
        self.precomputed = True

    def forward(self, x, y, n_sampled_fouriers=None):
        """
        return kernel evaluation at x and y with size [ batch_size, batch_size ], 
        where
        - x: the first input with size  [ batch_size, data_dim ]
        - y: the second input with size [ batch_size, data_dim ] 
        - n_sampled_fouriers: number of sampled fourier features (if None, then 
                              use precomputed fourier features)
        """
        # sampled fourier features
        if n_sampled_fouriers is not None:
            ws_p, us_p = self.p(n_sampled_fouriers)   # 2 * [ n_sampled_fouriers, fourier_dim ]
            ws_q, us_q = self.q(n_sampled_fouriers)   # 2 * [ n_sampled_fouriers, fourier_dim ]
        # use precomputed fourier features
        elif n_sampled_fouriers is None and self.precomputed is True:
            ws_p, us_p = self.pre_ws_p, self.pre_us_p # 2 * [ n_sampled_fouriers, fourier_dim ]
            ws_q, us_q = self.pre_ws_q, self.pre_us_q # 2 * [ n_sampled_fouriers, fourier_dim ]
        else:
            raise Exception("No available Fourier features.")
        # transpose
        ws_p, us_p = torch.transpose(ws_p, 0, 1), torch.transpose(us_p, 0, 1)
        ws_q, us_q = torch.transpose(ws_q, 0, 1), torch.transpose(us_q, 0, 1)
        # cosine and sine
        cos_wx_uy_p = torch.cos(torch.matmul(x, ws_p) - torch.matmul(y, us_p)) # [ batch_size, n_sampled_fouriers ]
        sin_wx_uy_q = torch.sin(torch.matmul(x, ws_q) - torch.matmul(y, us_q)) # [ batch_size, n_sampled_fouriers ]
        # integral
        eval = self.c1 * cos_wx_uy_p.mean(1) - \
               self.c2 * sin_wx_uy_q.mean(1) - \
               4 ** self.data_dim * \
               torch.prod(torch.pow(torch.sinc(x), self.data_dim)) * \
               torch.prod(torch.pow(torch.sinc(y), self.data_dim))          # [ batch_size ]

        # NOTE: currently apply softplus to prohibit negative values
        # NOTE: -0.5 to make the value smaller
        # return (torch.nn.functional.softplus(eval) - 0.5)/10 # [ batch_size ]
        return eval + 5e-5 # [ batch_size ]
