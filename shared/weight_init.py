import torch.nn as nn

def dcgan_weights_init(m: nn.Module) -> None:
    """
    DCGAN-style init:
      Conv weights ~ N(0, 0.02)
      BN weights ~ N(1, 0.02), BN bias = 0
    """
    name = m.__class__.__name__
    if "Conv" in name:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in name:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)