import torch


def func_with_grad(x):
    x = torch.tensor(x, requires_grad=True)
    y = torch.sin(x).sum()
    y.backward()
    dx = x.grad
    return float(y.detach().numpy()), dx.detach().numpy()


def rosenbrock(x):
    """
    rosebrock function for test
    :param x:
    :return:
    """
    x = torch.tensor(x, requires_grad=True)
    x_1 = torch.roll(x, -1, dims=0)[:-1]
    x_0 = x[:-1]
    y = torch.sum((x_1 - x_0 ** 2) ** 2 + (x_0 - 1) ** 2, 0)
    y.backward()
    dx = x.grad
    return float(y.detach().numpy()), dx.detach().numpy()


def func_wraper(f):
    def f_with_grad(x):
        y0 = f(x)
        return torch.tensor([float(y0[0]), *y0[1]])

    return f_with_grad


def tensor_to_hashable(array):
    """ Convert a tensor to a hashable type. This example uses a tuple. """
    return tuple(array)
