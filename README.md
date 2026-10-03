# GANs

A collection of PyTorch and Jupyter notebook experiments exploring generative
adversarial networks, from handwritten digit generation to image translation
and synthetic faces.

## Experiments

| Directory | Focus |
| --- | --- |
| `01_mlp_gan` | Basic fully connected GAN on MNIST |
| `02_dcgan` | Convolutional GAN on MNIST |
| `03_cdcgan` | Conditional digit generation |
| `04_latent_interpolation` | Morphing between generated images |
| `05_dcgan_fashionmnist` | Clothing image generation |
| `06_wgan_mnist` | Wasserstein GAN with weight clipping |
| `07_wgan_gp_mnist` | Wasserstein GAN with gradient penalty |
| `08_pix2pix_facades` | Building layout to facade translation |
| `09_pix2pix_edge_to_image` | Shoe outlines to shoe images |
| `10_face_does_not_exist` | Synthetic face generation using CelebA |

## Project layout

- `experiments/`: training notebooks, sampling notebooks, and Pix2Pix drawing demos.
- `shared/`: helpers for paths, random seeds, weight initialization, sampling,
  interpolation, and image grids.
- `datasets/`: local training data, excluded from Git.

Start with `experiments/01_mlp_gan/mlp_gan.ipynb` and follow the numbered
experiments. Notebooks use CUDA when available and otherwise run on CPU.
They require the libraries imported in their cells; `pyproject.toml` does not
currently list the complete training environment.

Downloaded datasets, model checkpoints, and generated training sample folders
are excluded from Git. Sampling and drawing demos need locally trained
checkpoints in the corresponding experiment's `models/` directory.
