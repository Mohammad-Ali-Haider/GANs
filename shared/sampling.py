import torch

@torch.no_grad()
def sample_unconditional(G, z_dim: int, device: str, n: int = 64) -> torch.Tensor:
    """
    For MLP GAN / DCGAN: no labels.
    Expects G(z) where z is:
      - MLP: (N, z_dim)
      - DCGAN: (N, z_dim, 1, 1)
    We'll infer based on G input you choose.
    """
    z = torch.randn(n, z_dim, 1, 1, device=device)
    return G(z)

@torch.no_grad()
def sample_conditional(G, z_dim: int, device: str, digit: int, n: int = 64) -> torch.Tensor:
    """
    For Conditional DCGAN: G(z, y)
      z: (N, z_dim, 1, 1)
      y: (N,)
    """
    z = torch.randn(n, z_dim, 1, 1, device=device)
    y = torch.full((n,), int(digit), device=device, dtype=torch.long)
    return G(z, y)

@torch.no_grad()
def sample_all_digits(G, z_dim: int, device: str, per_class: int = 8) -> torch.Tensor:
    """
    For Conditional DCGAN: generate a batch with labels 0..9 repeated.
    Total N = 10 * per_class.
    """
    n = 10 * per_class
    z = torch.randn(n, z_dim, 1, 1, device=device)
    y = torch.arange(0, 10, device=device).repeat_interleave(per_class)
    return G(z, y)