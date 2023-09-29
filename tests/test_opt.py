import io
import pickle
from pickle import dumps

import numpy as np

from gpopt.utils import func_with_grad, rosenbrock
from gpopt.optimizer import GPOPT
from unittest import TestCase


class GPOptTest(TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_rosenbrock(self):
        x0 = np.random.rand(10)
        opt = GPOPT(rosenbrock, x0, tol=1e-4)
        xf = opt.optimize()

    def test_orth_func(self):
        x0 = np.random.rand(10)
        opt = GPOPT(func_with_grad, x0, tol=1e-4)
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