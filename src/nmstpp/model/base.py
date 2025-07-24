
from abc import abstractmethod
import itertools
import torch
import numpy as np

class BasePointProcess(torch.nn.Module):
    """
    Point Process Base Class
    """
    @abstractmethod
    def __init__(self, T, S, data_dim, numerical_int=True, int_res=100):
        """
        Args:
        - T:             time horizon. e.g. (0, 1)
        - S:             bounded space for marks. e.g. a two dimensional box region [(0, 1), (0, 1)]
        - data_dim:      dimension of input data
        - numerical_int: numerical integral flag
        - int_res:       numerical integral resolution
        """
        super(BasePointProcess, self).__init__()
        # configuration
        self.data_dim      = data_dim
        self.T             = T # time horizon. e.g. (0, 1)
        self.S             = S # bounded space for marks. e.g. a two dimensional box region [(0, 1), (0, 1)]
        self.numerical_int = numerical_int
        assert len(S) + 1 == self.data_dim, "Invalid space dimension"

        # numerical likelihood integral preparation
        if int_res is not None:
            self.int_res  = int_res
            self.tt       = torch.FloatTensor(np.linspace(self.T[0], self.T[1], int_res))  # [ in_res ]
            self.ss       = [ np.linspace(S_k[0], S_k[1], int_res) for S_k in self.S ]     # [ data_dim - 1, in_res ]
            # spatio-temporal coordinates that need to be evaluated
            self.t_coords = torch.ones((int_res ** (data_dim - 1), 1))                     # [ int_res^(data_dim - 1), 1 ]
            self.s_coords = torch.FloatTensor(np.array(list(itertools.product(*self.ss)))) # [ int_res^(data_dim - 1), data_dim - 1 ]
            # unit volumn
            self.unit_vol = np.prod([ S_k[1] - S_k[0] for S_k in self.S ] + [ self.T[1] - self.T[0] ]) / (self.int_res) ** self.data_dim

    def numerical_integral(self, X):
        """
        return conditional intensity evaluation at grid points, the numerical 
        integral can be further calculated by summing up these evaluations and 
        scaling by the unit volumn.
        """
        batch_size, seq_len, _ = X.shape
        integral = []
        for t in self.tt:
            # all possible points at time t (x_t) 
            t_coord = self.t_coords * t
            xt      = torch.cat([t_coord, self.s_coords], 1) # [ int_res^(data_dim - 1), data_dim ] 
            xt      = xt\
                .unsqueeze_(0)\
                .repeat(batch_size, 1, 1)\
                .reshape(-1, self.data_dim)                  # [ batch_size * int_res^(data_dim - 1), data_dim ]
            # history points before time t (H_t)
            mask = ((X[:, :, 0].clone() <= t) * (X[:, :, 0].clone() > 0))\
                .unsqueeze_(-1)\
                .repeat(1, 1, self.data_dim)                 # [ batch_size, seq_len, data_dim ]
            ht   = X * mask                                  # [ batch_size, seq_len, data_dim ]
            ht   = ht\
                .unsqueeze_(1)\
                .repeat(1, self.int_res ** (self.data_dim - 1), 1, 1)\
                .reshape(-1, seq_len, self.data_dim)         # [ batch_size * int_res^(data_dim - 1), seq_len, data_dim ]
            # lambda and integral 
            lams = torch.nn.functional.softplus(self.cond_lambda(xt, ht))\
                .reshape(batch_size, -1)                     # [ batch_size, int_res^(data_dim - 1) ]
            integral.append(lams)                            
        # NOTE: second dimension is time, third dimension is mark space
        integral = torch.stack(integral, 1)                  # [ batch_size, int_res, int_res^(data_dim - 1) ]
        return integral
    
    def cond_lambda(self, xi, hti):
        """
        return conditional intensity given x
        Args:
        - xi:   current i-th point       [ batch_size, data_dim ]
        - hti:  history points before ti [ batch_size, seq_len, data_dim ]
        Return:
        - lami: i-th lambda              [ batch_size ]
        """
        # if length of the history is zero
        if hti.size()[0] == 0:
            return self.mu()
        # otherwise treat zero in the time (the first) dimension as invalid points
        batch_size, seq_len, _ = hti.shape
        mask = hti[:, :, 0].clone() > 0                                          # [ batch_size, seq_len ]
        xi   = xi.unsqueeze_(1).repeat(1, seq_len, 1).reshape(-1, self.data_dim) # [ batch_size * seq_len, data_dim ]
        hti  = hti.reshape(-1, self.data_dim)                                    # [ batch_size * seq_len, data_dim ]
        K    = self.kernel(xi, hti).reshape(batch_size, seq_len)                 # [ batch_size, seq_len ]
        K    = K * mask                                                          # [ batch_size, seq_len ]
        lami = K.sum(1) + self.mu()                                              # [ batch_size ]
        return lami

    def log_likelihood(self, X, n_sampled_fouriers=200):
        """
        return log-likelihood given sequence X
        Args:
        - X:      input points sequence [ batch_size, seq_len, data_dim ]
        Return:
        - lams:   sequence of lambda    [ batch_size, seq_len ]
        - loglik: log-likelihood        [ batch_size ]
        """
        batch_size, seq_len, _ = X.shape
        lams     = [
            torch.nn.functional.softplus(self.cond_lambda(
                X[:, i, :].clone(), 
                X[:, :i, :].clone())) + 1e-5
            for i in range(seq_len) ]
        lams     = torch.stack(lams, dim=1)                                   # [ batch_size, seq_len ]
        # log-likelihood
        mask     = X[:, :, 0] > 0                                             # [ batch_size, seq_len ]
        sumlog   = torch.log(lams) * mask                                     # [ batch_size, seq_len ]
        integral = self.numerical_integral(X)                                 # [ batch_size, int_res, int_res^(data_dim - 1) ]
        if self.numerical_int:
            loglik = sumlog.sum(1) - integral.sum(-1).sum(-1) * self.unit_vol # [ batch_size ]
        else: 
            # TODO: integral in analytical form
            pass
        return lams, loglik

    @abstractmethod
    def mu(self):
        """
        return base intensity
        """
        raise NotImplementedError()

    @abstractmethod
    def forward(self, X):
        """
        custom forward function returning conditional intensities and corresponding log-likelihood
        """
        # return conditional intensities and corresponding log-likelihood
        return self.log_likelihood(X)
