import numpy as np

from gpopt.utils import func_with_grad
from gpopt.optimizer import GPOPT

x0 = np.random.rand(10)
opt = GPOPT(func_with_grad, x0, tol=1e-5)
xf = opt.optimize()

# dump for debug
import pickle
with open('opt.pkl', 'wb') as f:
    pickle.dump(opt, f)