import numpy as np

from gpopt.utils import func_with_grad
from gpopt.optimizer import GPOPT

x0 = np.random.rand(100)
opt = GPOPT(func_with_grad, x0,tol = 1e-2)
xf = opt.optimize()
