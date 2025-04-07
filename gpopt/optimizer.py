import logging
from typing import Callable, Optional, Tuple

import torch
import gpytorch
import numpy as np
from gpytorch.kernels import ScaleKernel
from gpytorch.models import GP
from scipy.optimize import minimize
from gpytorch.constraints import Interval
from gpopt.utils import func_wraper, tensor_to_hashable
from functools import lru_cache
from gpopt.kernels.matern_kernel_grad import MaternKernelGrad
from gpytorch.kernels.rbf_kernel_grad import RBFKernelGrad
__all__ = ['GPOPT']

AnalyticFunctionType = Callable[[np.array], Tuple[float, np.array]]
from gpytorch.settings import max_cholesky_size

max_cholesky_size.__enter__(max_cholesky_size(1e12))

class AnalyticGradMean(gpytorch.means.Mean):
    def __init__(self, func: AnalyticFunctionType, cache_prior, batch_shape=torch.Size(), **kwargs):
        """

        :param func:
        :param batch_shape:
        :param kwargs:
        """
        super(AnalyticGradMean, self).__init__()
        self.batch_shape = batch_shape
        self.register_parameter(name="constant",
                                parameter=torch.nn.Parameter(torch.zeros(*batch_shape, 1)))  # auto shift
        self.func = self.func_wrap(func) if cache_prior else self.func_wrap_no_cache(func)

    def forward(self, input):
        batch_shape = torch.broadcast_shapes(self.batch_shape, input.shape[:-2])
        mean = self.constant.unsqueeze(-1).expand(*batch_shape, input.size(-2), input.size(-1) + 1).contiguous()
        mean[..., 1:] = 0
        f = self.func(input)
        mean[..., :] += f
        return mean

    @staticmethod
    def func_wrap(func: AnalyticFunctionType):
        @lru_cache
        def f_(x):
            return func(np.array(x))

        def f(x):
            x_np = x.detach().numpy()
            r = np.empty((len(x_np), len(x_np[0]) + 1))
            for i in range(len(x_np)):
                value, grad = f_(tensor_to_hashable(x_np[i]))
                r[i] = [value] + list(grad)

            return torch.tensor(r)
        return f

    @staticmethod
    def func_wrap_no_cache(func: AnalyticFunctionType):
        def f_(x):
            return func(np.array(x))

        def f(x):
            x_np = x.detach().numpy()
            r = np.empty((len(x_np), len(x_np[0]) + 1))
            for i in range(len(x_np)):
                value, grad = f_(tensor_to_hashable(x_np[i]))
                r[i] = [value] + list(grad)

            return torch.tensor(r)
        return f


class GPModelWithDerivatives(gpytorch.models.ExactGP):
    base_kernel = MaternKernelGrad
    # base_kernel = gpytorch.kernels.RBFKernelGrad()
    def __init__(self, train_x, train_y, likelihood, analytic_prior: Optional[AnalyticFunctionType] = None,
                 cache_prior=True):
        super(GPModelWithDerivatives, self).__init__(train_x, train_y, likelihood)
        if analytic_prior:
            self.mean_module = AnalyticGradMean(analytic_prior,cache_prior)
        else:
            self.mean_module = gpytorch.means.ConstantMeanGrad()
        self.covar_module = ScaleKernel(self.base_kernel(ard_num_dims=train_x[0].shape[0]))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class GPOPT:
    def __init__(self, f, x0, tol=1e-6, analytic_prior=None, model: Optional[type(GP)] = None,
                 cache_prior=True,
                 last_n_train:int = 0,length_scale:float=1.0):
        """
        The constructor for the GPOPT class.

        Parameters:
        :param f:  (function): The function to be optimized.
        :param x0:  (array): The initial point for the optimization.
        :param tol: (float): The tolerance for the optimization. The optimization stops when the gradient norm is less than tol.
        :param analytic_prior:  (function): An optional analytic prior for the mean of the GP.
        :param model: (GPModelWithDerivatives): An optional initial model for the GP.
        :param last_n_train: (int): The number of last n steps points to consider. 0 to use all.
                                    good for large D and large y change
        :param length_scale: (float): The length scale for the GP kernel.
        """
        self.train_x = []
        self.train_y = []
        self.model = None
        self.tol = tol
        self.f = func_wraper(f)
        self.x0 = x0
        self.x = x0
        self.last_y = np.inf
        self.last_x = x0
        self.last_n_train = last_n_train
        self.len_x = len(x0)
        self.analytic_prior = analytic_prior
        self.cache_prior=cache_prior
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(noise_constraint=Interval(1e-9, 1e-7),
                                                                           num_tasks=self.len_x + 1)  # Value + Derivative
        self.length_scale = length_scale

    @property
    def _x_tensor(self):
        sub_train = self.train_x[-self.last_n_train:] if self.last_n_train else self.train_x
        return torch.stack(sub_train).reshape(-1, self.len_x)

    @property
    def _y_tensor(self):
        sub_train = self.train_y[-self.last_n_train:] if self.last_n_train else self.train_y
        return torch.stack(sub_train).reshape(-1, self.len_x + 1)

    @property
    def _best_grad_max(self ):
        return (self._y_tensor[:,1:]**2).max(-1).values.min()**0.5
    def _train(self):

        model = GPModelWithDerivatives(self._x_tensor, self._y_tensor, self.likelihood,
                                       analytic_prior=self.analytic_prior,cache_prior=self.cache_prior)
        model.mean_module.constant = torch.nn.Parameter(
            self._y_tensor[:, 0].mean())
        if self.analytic_prior is not None:
            # shift the mean to the last point
            with torch.no_grad():
                model.mean_module.constant = torch.nn.Parameter(torch.Tensor([0.0]))
                model.mean_module.constant = torch.nn.Parameter(
                    self._y_tensor[-1, 0] - model.mean_module.forward(self._x_tensor[-1].reshape([1,-1]))[0,0])
        model.mean_module.constant.requires_grad = False
        model.covar_module.base_kernel.lengthscale =  self.length_scale * torch.ones(self.len_x)
        model.covar_module.base_kernel.raw_lengthscale.requires_grad = False
        if self.model:
            model.covar_module.outputscale = self.model.covar_module.outputscale
            model.covar_module.base_kernel.lengthscale = self.model.covar_module.base_kernel.lengthscale
            model.likelihood.noise = self.model.likelihood.noise
            model.covar_module.raw_outputscale = self.model.covar_module.raw_outputscale
        model.train()
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=0.5)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, model)
        training_iter = 10
        output = model(self._x_tensor)
        loss = -mll(output, self._y_tensor)
        for i in range(training_iter):
            optimizer.zero_grad()
            output = model(self._x_tensor)
            loss = -mll(output, self._y_tensor)
            loss.backward()
            logging.info('Iter %d/%d - Loss: %.3f   lengthscale: %s   noise: %.3f' % (
                i + 1, training_iter, loss.item(),
                str(model.covar_module.base_kernel.lengthscale.detach().numpy()),
                model.likelihood.noise.item()
            )+str(model.covar_module.raw_outputscale) )
            optimizer.step()
        self.model = model

    def eval_surrogate(self, x):
        self.model.eval()
        with torch.no_grad():
            r = self.model(torch.tensor(x).reshape(1, -1)).mean
        return r[0][0].detach().numpy(), r[0][1:].detach().numpy()
    def _find_surrogate_min(self):
        tol = np.clip(self._best_grad_max*0.1, self.tol*0.1, self.tol*10)
        x_f = minimize(self.eval_surrogate, self.x, method='BFGS', jac=True, options={'gtol':tol})
        if not x_f.success:
            self.model.covar_module.base_kernel.lengthscale = self.length_scale * 0.1
            print(x_f.message)
        return x_f.x

    def step(self):
        x = self.x
        logging.info(f"calling f @ {x}")
        r = self.f(x)
        y = r[0]
        dy = r[1:]
        self.train_x += [torch.tensor(x)]
        self.train_y += [r]
        print(f"f = {y} max_grad = {(dy ** 2).max() ** 0.5}")
        if ((dy ** 2).max() ** 0.5 < self.tol).all():
            print(f"converged @ {x}")
            return True
        if y > self.last_y:
            self.x = self.last_x.copy()
        else:
            self.last_y = y.detach()
            self.last_x = x.copy()

        self._train()
        self.x = self._find_surrogate_min()
        return False

    def optimize(self, dump=None):
        while True:
            if self.step():
                break
        return self.x

    def __reduce__(self):
        return self._load, (self.train_x, self.train_y, self.tol, self.x0)

    @staticmethod
    def _load(train_x, train_y, tol, x0):
        new_cls = GPOPT(lambda x: x, x0, tol)
        new_cls.train_x = train_x
        new_cls.train_y = train_y
        return new_cls
