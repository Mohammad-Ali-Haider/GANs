from pathlib import Path
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image

def show_grid(images: torch.Tensor, title: str = "", nrow: int = 8) -> None:
    """
    images: (N, C, H, W) in [-1,1] recommended
    """
    grid = make_grid(images.detach().cpu(), nrow=nrow, normalize=True, value_range=(-1, 1))
    plt.figure(figsize=(6, 6))
    plt.axis("off")
    if title:
        plt.title(title)
    plt.imshow(grid.permute(1, 2, 0))
    plt.show()

def save_grid(images: torch.Tensor, out_path: str | Path, nrow: int = 8) -> None:
    """
    Saves images as PNG. Converts [-1,1] -> [0,1] for saving.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image((images.detach().cpu() + 1) / 2, out_path, nrow=nrow)