import logging

import torch
import gpytorch
import numpy as np
from scipy.optimize import minimize

from gpopt.utils import func_wraper

__all__ = ['GPOPT']


class GPModelWithDerivatives(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModelWithDerivatives, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMeanGrad()
        self.base_kernel = gpytorch.kernels.RBFKernelGrad()
        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class GPOPT:
    def __init__(self, f, x0, tol=1e-6):
        self.train_x = []
        self.train_y = []
        self.model = None
        self.tol = tol
        self.f = func_wraper(f)
        self.x0 = x0
        self.x = x0
        self.last_y = np.inf
        self.last_x = x0
        self.len_x = len(x0)
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(noise_constraint=None,
            num_tasks=self.len_x + 1)  # Value + Derivative

    @property
    def _x_tensor(self):
        return torch.stack(self.train_x).reshape(-1, self.len_x)

    @property
    def _y_tensor(self):
        return torch.stack(self.train_y).reshape(-1, self.len_x + 1)

    def _train(self):

        model = GPModelWithDerivatives(self._x_tensor, self._y_tensor, self.likelihood)
        model.mean_module.constant = torch.nn.Parameter(self._y_tensor[:, 0].mean()+1.0)
        model.likelihood.noise= 1e-4
        if self.model:
            model.covar_module.outputscale = self.model.covar_module.outputscale
            model.covar_module.base_kernel.lengthscale = self.model.covar_module.base_kernel.lengthscale
        model.train()
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=0.5)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, model)
        training_iter = 50
        for i in range(training_iter):
            optimizer.zero_grad()
            output = model(self._x_tensor)
            loss = -mll(output, self._y_tensor)
            loss.backward()
            logging.info('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, training_iter, loss.item(),
                model.covar_module.base_kernel.lengthscale.item(),
                model.likelihood.noise.item()
            ))
            optimizer.step()
        self.model = model
    def _find_surrogate_min(self):
        self.model.eval()

        def surrogate(x):
            with torch.no_grad():
                r = self.model(torch.tensor(x).reshape(1, -1)).mean
            return r[0][0].detach().numpy(), r[0][1:].detach().numpy()

        x_f = minimize(surrogate, self.x, method='BFGS', jac=True, tol=self.tol*0.01)
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
            self.x = self.last_x
        else:
            self.last_y = y
            self.last_x = x

        self._train()
        self.x = self._find_surrogate_min()
        return False

    def optimize(self):
        while True:
            if self.step():
                break
        return self.x
