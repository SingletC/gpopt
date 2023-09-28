## Gradient-GP-Regression based optimizer

### install:
```bash
git clone https://github.com/SingletC/gpopt.git
cd gpopt
python -m pip install .
```
### Verifation of installation (optional):
```bash
python -m gpopt.test
````
### usage:
```python

from gpopt.optimizer import GPOPT
from gpopt.utils import func_with_grad # your function here
import numpy as np
# Sig: func_with_grad: Callable[[np.ndarray], Tuple[float, np.ndarray]]
x0 = np.random.rand(100)
opt = GPOPT(func_with_grad, x0,tol = 1e-2)
xf = opt.optimize()

```
