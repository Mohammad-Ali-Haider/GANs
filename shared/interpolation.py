import torch

@torch.no_grad()
def lerp(z1: torch.Tensor, z2: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Linear interpolation between z1 and z2.
    z1, z2: same shape (1, z_dim, 1, 1) or (1, z_dim)
    returns: (steps, ...) batch of interpolated latents
    """
    alphas = torch.linspace(0, 1, steps, device=z1.device).view(-1, *([1] * (z1.ndim - 1)))
    return (1 - alphas) * z1 + alphas * z2