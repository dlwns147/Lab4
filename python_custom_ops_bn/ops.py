import torch
from torch import Tensor
from typing import Tuple

# Step 1: Define custom operators using torch.library API
@torch.library.custom_op("my_ops::batchnorm_forward", mutates_args=("running_mean", "running_var"))
def batchnorm_forward(
    input: Tensor,           # [N, C, H, W]
    gamma: Tensor,           # [C]
    beta: Tensor,            # [C]
    running_mean: Tensor,    # [C]
    running_var: Tensor,     # [C]
    training: bool,
    momentum: float,
    eps: float
) -> Tuple[Tensor, Tensor, Tensor]:
    """forward pass of BatchNorm for 4D input [N, C, H, W].

    Must behave like torch.nn.BatchNorm2d:
    - training=True : normalize with the batch statistics (biased variance) and update
                      running_mean / running_var in place with momentum
                      (running_var accumulates the unbiased variance).
    - training=False: normalize with running_mean / running_var and leave them unchanged.

    Returns (output, save_mean, save_invstd). save_mean [C] and save_invstd [C] are the mean
    and 1 / sqrt(var + eps) that were used to normalize; batchnorm_backward receives them.

    Note: a custom operator may only modify the inputs listed in mutates_args, and must not
    return a tensor that is one of its inputs or shares memory with one.
    """

    N, C, H, W = input.shape

    # Compute statistics over N, H, W for each channel
    dims = (0, 2, 3)

    if training:
        # Batch mean and biased variance
        batch_mean = input.mean(dim=dims)
        batch_var = input.var(dim=dims, unbiased=False)

        # Statistics used for normalization
        save_mean = batch_mean
        save_invstd = torch.rsqrt(batch_var + eps)

        # Update running statistics in-place
        with torch.no_grad():
            unbiased_var = input.var(dim=dims, unbiased=True)

            running_mean.mul_(1 - momentum).add_(batch_mean, alpha=momentum)
            running_var.mul_(1 - momentum).add_(unbiased_var, alpha=momentum)

    else:
        # Use running statistics during evaluation
        save_mean = running_mean
        save_invstd = torch.rsqrt(running_var + eps)

    # Normalize and apply affine transformation
    mean = save_mean.view(1, C, 1, 1)
    invstd = save_invstd.view(1, C, 1, 1)
    gamma_view = gamma.view(1, C, 1, 1)
    beta_view = beta.view(1, C, 1, 1)

    output = (input - mean) * invstd * gamma_view + beta_view

    return output, save_mean.clone(), save_invstd.clone()


@torch.library.custom_op("my_ops::batchnorm_backward", mutates_args=())
def batchnorm_backward(
    grad_output: Tensor,     # [N, C, H, W]
    input: Tensor,           # [N, C, H, W]
    gamma: Tensor,           # [C]
    save_mean: Tensor,       # [C]
    save_invstd: Tensor,     # [C]
    training: bool
) -> Tuple[Tensor, Tensor, Tensor]:
    """backward pass of BatchNorm for 4D input.

    Returns the gradients of the loss w.r.t. input, gamma and beta, given
    grad_output = dL/d(output) and the tensors saved by batchnorm_forward.
    - training=True : save_mean / save_invstd are batch statistics, so they depend on input.
    - training=False: save_mean / save_invstd come from the running statistics (constants).
    """

    # Reshape per-channel tensors [C] -> [1, C, 1, 1] to broadcast over [N, C, H, W]
    mean = save_mean.view(1, -1, 1, 1)
    invstd = save_invstd.view(1, -1, 1, 1)
    g = gamma.view(1, -1, 1, 1)
    reduce_dims = (0, 2, 3)

    # Normalized input used in the forward pass: x_hat = (x - mean) * invstd
    x_hat = (input - mean) * invstd

    # y = gamma * x_hat + beta
    # dL/dbeta  = sum_{n,h,w} dy
    # dL/dgamma = sum_{n,h,w} dy * x_hat
    grad_beta = grad_output.sum(dim=reduce_dims)
    grad_gamma = (grad_output * x_hat).sum(dim=reduce_dims)

    if training:
        # mean and invstd are functions of input. With M = N*H*W elements per channel:
        # dL/dx = gamma * invstd / M * (M * dy - sum(dy) - x_hat * sum(dy * x_hat))
        M = input.numel() // input.size(1)
        grad_input = (g * invstd / M) * (
            M * grad_output
            - grad_beta.view(1, -1, 1, 1)
            - x_hat * grad_gamma.view(1, -1, 1, 1)
        )
    else:
        # Running statistics are constants, so BN is an affine map per channel:
        # dL/dx = dy * gamma * invstd
        grad_input = grad_output * g * invstd

    return grad_input, grad_gamma, grad_beta


# Step 2: Connect forward and backward with autograd
# This connects our custom forward/backward operators to PyTorch's 
# autograd system, allowing gradients to flow during backpropagation
class BatchNormCustom(torch.autograd.Function):
    """
    Custom Batch Normalization for 4D inputs [N, C, H, W].
    
    Bridges custom operators with PyTorch's autograd engine.
    - forward(): calls custom forward operator and saves context
    - backward(): calls custom backward operator using saved context

    Usage:
        output = BatchNormCustom.apply(input, gamma, beta, running_mean, running_var, training, momentum, eps)
    """
    @staticmethod
    def forward(ctx, input, gamma, beta, running_mean, running_var, training, momentum, eps):
        output, save_mean, save_invstd = torch.ops.my_ops.batchnorm_forward(
            input, gamma, beta, running_mean, running_var, training, momentum, eps
        )
        ctx.save_for_backward(input, gamma, save_mean, save_invstd)
        ctx.training = training  # non-tensor state is stored directly on ctx
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, gamma, save_mean, save_invstd = ctx.saved_tensors
        grad_input, grad_gamma, grad_beta = torch.ops.my_ops.batchnorm_backward(
            grad_output, input, gamma, save_mean, save_invstd, ctx.training
        )
        # Return gradients for all forward inputs (None for non-tensor args)
        return grad_input, grad_gamma, grad_beta, None, None, None, None, None