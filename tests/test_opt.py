import io
import pickle
from typing import Tuple

import gpytorch
import numpy as np

from gpopt.utils import func_with_grad, rosenbrock
from gpopt.optimizer import GPOPT
from unittest import TestCase
# for debugger
# import sys
# import logging
# logger = logging.getLogger()
# logger.level = logging.INFO
# stream_handler = logging.StreamHandler(sys.stdout)
# logger.addHandler(stream_handler)


class GPOptTest(TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_rosenbrock(self):
        x0 = np.random.rand(10)
        opt = GPOPT(rosenbrock, x0, tol=1e-6)
        xf = opt.optimize()
        print(len(opt.train_x))

    def test_orth_func(self):
        x0 = np.random.rand(10)
        opt = GPOPT(func_with_grad, x0, tol=1e-6)
        xf = opt.optimize()

    def test_analy_prior(self):
        x0 = np.random.rand(10)

        def example_analytic_prior_func(x) -> Tuple[float, np.ndarray]:
            """
            this is demo of use self define analytic prior function
            it should take a vector x and return a tuple of (value, grad)
            :param x: np.ndarray
            :return: value: float, grad : np.ndarray
            """
            return np.sum(x ** 2), 2 * x

        opt = GPOPT(rosenbrock, x0, tol=1e-6, analytic_prior=example_analytic_prior_func)
        xf = opt.optimize()
        print(len(opt.train_x))

    def test_self_define_kernel(self):
        """
        this is demo of use self define kernel function(rbf as exmaple)
        simply herit from gpytorch.kernels.Kernel and redefine base_kernel attribute
        :return:
        """
        from gpopt.optimizer import GPModelWithDerivatives
        class RBFGPWithDerivatives(GPModelWithDerivatives):
            base_kernel = gpytorch.kernels.RBFKernelGrad()

        opt = GPOPT(rosenbrock, np.random.rand(10), tol=1e-4, model=RBFGPWithDerivatives)
        xf = opt.optimize()
    def test_dump(self):
        x0 = np.random.rand(10)
        opt = GPOPT(func_with_grad, x0, tol=0.1)
        xf = opt.optimize()
        with io.BytesIO() as mem_file:
            mem_file.write(pickle.dumps(opt))
            mem_file.seek(0)
            data = mem_file.read()
            opt_new = pickle.loads(data)

# if __name__ == '__main__':
#     test = GPOptTest()
#     test.setUp()
#     test.test_rosenbrock()
