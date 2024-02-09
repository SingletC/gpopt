#!/usr/bin/env python3

import torch
from linear_operator.operators import KroneckerProductLinearOperator

from gpytorch.kernels.matern_kernel import MaternKernel


class MaternKernelGrad(MaternKernel):
    r"""
    Computes a covariance matrix of the RBF kernel that models the covariance
    between the values and partial derivatives for inputs :math:`\mathbf{x_1}`
    and :math:`\mathbf{x_2}`.

    See :class:`gpytorch.kernels.Kernel` for descriptions of the lengthscale options.

    .. note::

        This kernel does not have an `outputscale` parameter. To add a scaling parameter,
        decorate this kernel with a :class:`gpytorch.kernels.ScaleKernel`.

    :param ard_num_dims: Set this if you want a separate lengthscale for each input
        dimension. It should be `d` if x1 is a `n x d` matrix. (Default: `None`.)
    :param batch_shape: Set this if you want a separate lengthscale for each batch of input
        data. It should be :math:`B_1 \times \ldots \times B_k` if :math:`\mathbf x1` is
        a :math:`B_1 \times \ldots \times B_k \times N \times D` tensor.
    :param active_dims: Set this if you want to compute the covariance of only
        a few input dimensions. The ints corresponds to the indices of the
        dimensions. (Default: `None`.)
    :param lengthscale_prior: Set this if you want to apply a prior to the
        lengthscale parameter. (Default: `None`)
    :param lengthscale_constraint: Set this if you want to apply a constraint
        to the lengthscale parameter. (Default: `Positive`.)
    :param eps: The minimum value that the lengthscale can take (prevents
        divide by zero errors). (Default: `1e-6`.)

    :ivar torch.Tensor lengthscale: The lengthscale parameter. Size/shape of parameter depends on the
        ard_num_dims and batch_shape arguments.

    Example:
        >>> x = torch.randn(10, 5)
        >>> # Non-batch: Simple option
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernelGrad())
        >>> covar = covar_module(x)  # Output: LinearOperator of size (60 x 60), where 60 = n * (d + 1)
        >>>
        >>> batch_x = torch.randn(2, 10, 5)
        >>> # Batch: Simple option
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernelGrad())
        >>> # Batch: different lengthscale for each batch
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernelGrad(batch_shape=torch.Size([2])))
        >>> covar = covar_module(x)  # Output: LinearOperator of size (2 x 60 x 60)
    """

    def forward(self, x1, x2, diag=False, **params):
        batch_shape = x1.shape[:-2]
        n_batch_dims = len(batch_shape)
        n1, d = x1.shape[-2:]
        n2 = x2.shape[-2]

        K = torch.zeros(*batch_shape, n1 * (d + 1), n2 * (d + 1), device=x1.device, dtype=x1.dtype)

        if not diag:
            # Scale the inputs by the lengthscale (for stability)
            # Form all possible rank-1 products for the gradient and Hessian blocks
            outer = x1.view(*batch_shape, n1, 1, d) - x2.view(*batch_shape, 1, n2, d)
            outer = torch.transpose(outer, -1, -2).contiguous()

            # scaled_diff = x1_.view(*batch_shape, n1, 1, d) - x2_.view(*batch_shape, 1, n2, d)
            # scaled_diff = diff / self.lengthscale.unsqueeze(-2)
            # scaled_diff = torch.transpose(scaled_diff, -1, -2).contiguous()


            # 1) Kernel block
            radius = self.covar_dist(x1, x2, square_dist=False, **params)
            radius_scale = radius / self.lengthscale
            radius2_scale = radius**2
            root_five = 2.23606797749979
            K_11 = ((1.0 + root_five * radius_scale + 5.0 / 3.0 * radius2_scale)
                    * torch.exp(-root_five * radius_scale))
            K[..., :n1, :n2] = K_11
            # 2) First gradient block
            # TODO there is error compared to AD, possible bug
            outer1 = outer.view(*batch_shape, n1, n2 * d)
            # diff = torch.transpose(outer1, -1, -2).contiguous()
            prefactor_jac = 5. / (3. * self.lengthscale ** 3) * (self.lengthscale + root_five * radius)
            prefactor_jac = prefactor_jac.repeat([*([1] * (n_batch_dims + 1)), d])
            prefactor_jac = (prefactor_jac * outer1).view(*batch_shape, n1, n2 * d)
            # jac = p*exp(-sqrt(5)*r/l)
            jac = prefactor_jac * torch.exp(-root_five * radius/self.lengthscale).repeat([*([1] * (n_batch_dims + 1)), d])
            K[..., :n1, n2:] = jac

            # 3) Second gradient block
            # TODO there is error compared to AD, possible bug
            # the same
            outer2 = outer.transpose(-1, -3).reshape(*batch_shape, n2, n1 * d)
            outer2 = outer2.transpose(-1, -2)
            prefactor_jac2 = 5. / (3. * self.lengthscale ** 3) * (self.lengthscale + root_five * radius)
            prefactor_jac2 = prefactor_jac2.repeat([*([1] * n_batch_dims), d, 1])
            prefactor_jac2 = (prefactor_jac2 * -outer2).view(*batch_shape, n1*d, n2)
            jac2 = prefactor_jac2 * torch.exp(-root_five * radius / self.lengthscale).repeat([*([1] * n_batch_dims), d, 1])
            K[..., n1:, :n2] = jac2

            # 4) Hessian block
            # outer3 diff x diff  : nd x nd
            outer3 = outer1.repeat([*([1] * n_batch_dims), d, 1]) * outer2.repeat([*([1] * (n_batch_dims + 1)), d])
            # P = 5. * outer3
            P = 5. * outer3
            pre_factor = 5. / (3. * self.lengthscale ** 4)
            hess__block = (self.lengthscale + root_five * radius) * self.lengthscale
            # hess__block = (torch.eye(d) * (self.lengthscale + root_five * radius) * self.lengthscale)
            kp = KroneckerProductLinearOperator(torch.eye(d, d, device=x1.device, dtype=x1.dtype).repeat(*batch_shape, 1, 1),hess__block,
                                                )
            prefactor = (kp.to_dense() - P) * pre_factor
            exp_ = torch.exp(-root_five * radius / self.lengthscale)
            exp_ = KroneckerProductLinearOperator(torch.ones(d, d, device=x1.device, dtype=x1.dtype).repeat(*batch_shape, 1, 1),exp_,)
            K[..., n1:, n2:] = (prefactor * exp_).to_dense()
            # Symmetrize for stability
            if n1 == n2 and torch.eq(x1, x2).all():
                K = 0.5 * (K.transpose(-1, -2) + K)

            # Apply a perfect shuffle permutation to match the MutiTask ordering
            pi1 = torch.arange(n1 * (d + 1)).view(d + 1, n1).t().reshape((n1 * (d + 1)))
            pi2 = torch.arange(n2 * (d + 1)).view(d + 1, n2).t().reshape((n2 * (d + 1)))
            K = K[..., pi1, :][..., :, pi2]

            return K

        else:
            if not (n1 == n2 and torch.eq(x1, x2).all()):
                raise RuntimeError("diag=True only works when x1 == x2")

            kernel_diag = super(MaternKernelGrad, self).forward(x1, x2, diag=True)
            grad_diag = torch.ones(*batch_shape, n2, d, device=x1.device, dtype=x1.dtype) / self.lengthscale.pow(2)
            grad_diag = grad_diag.transpose(-1, -2).reshape(*batch_shape, n2 * d)
            k_diag = torch.cat((kernel_diag, grad_diag), dim=-1)
            pi = torch.arange(n2 * (d + 1)).view(d + 1, n2).t().reshape((n2 * (d + 1)))
            return k_diag[..., pi]

    def num_outputs_per_input(self, x1, x2):
        return x1.size(-1) + 1
