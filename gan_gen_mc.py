import torch.nn as nn
import torch

from gan_mc import get_default_device, save_samples, to_device
from get_models import get_generator_16, get_generator_64

if __name__ == '__main__':
    device = get_default_device()
    generator = get_generator_64(128)

    generator.load_state_dict(torch.load("G_mc_texture.pth", map_location=device))
    generator = to_device(generator, device)
    fixed_latent = torch.randn(800, 128, 1, 1, device=device)
    save_samples(9999, fixed_latent, generator, "mc_skin_generated", [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)])
