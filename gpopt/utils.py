import torch
def func_with_grad(x):
    x = torch.tensor(x, requires_grad=True)
    y = torch.sin(x).sum()
    y.backward()
    dx = x.grad
    return float(y.detach().numpy()), dx.detach().numpy()


def func_wraper(f):
    def f_with_grad(x):
        y0 = f(x)
        return torch.tensor([float(y0[0]), *y0[1]])

    return f_with_grad
