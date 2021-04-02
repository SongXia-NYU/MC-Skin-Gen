import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torchvision.utils import save_image
from tqdm import tqdm

from get_models import *


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGBA')


def train(retrain=False, start_idx=0):
    batch_size = 16
    latent_size = 128
    stats = (0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)
    train_ds = ImageFolder("mc_skin_64", transform=T.Compose([T.ToTensor(), T.Normalize(*stats)]),
                           loader=pil_loader)
    # train_ds = Subset(train_ds, torch.arange(10))
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0, pin_memory=True)
    # show_batch(train_dl, 32, stats)
    device = get_default_device()
    train_dl = DeviceDataLoader(train_dl, device)

    discriminator = get_discriminator_64()
    discriminator = to_device(discriminator, device)
    generator = get_generator_64(latent_size)
    generator = to_device(generator, device)

    if retrain:
        discriminator.load_state_dict(torch.load("D_mc_texture.pth", map_location=device))
        generator.load_state_dict(torch.load("G_mc_texture.pth", map_location=device))

    sample_dir = 'mc_skin_generated'
    os.makedirs(sample_dir, exist_ok=True)

    fixed_latent = torch.randn(64, latent_size, 1, 1, device=device)
    save_samples(0, fixed_latent, generator, sample_dir, stats)

    epochs = 600

    history = fit(epochs, discriminator, generator, train_dl, fixed_latent, batch_size, latent_size, device,
                  sample_dir, stats, start_idx=start_idx)
    torch.save(generator.state_dict(), 'G_mc_texture.pth')
    torch.save(discriminator.state_dict(), 'D_mc_texture.pth')
    torch.save(history, "history-{}.pth".format(start_idx))

    losses_g, losses_d, real_scores, fake_scores = history
    plt.plot(losses_d, '-')
    plt.plot(losses_g, '-')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Discriminator', 'Generator'])
    plt.title('Losses')
    plt.show()
    plt.plot(real_scores, '-')
    plt.plot(fake_scores, '-')
    plt.xlabel('epoch')
    plt.ylabel('score')
    plt.legend(['Real', 'Fake'])
    plt.title('Scores')
    plt.show()


def fit(epochs, discriminator, generator, train_dl, fixed_latent, batch_size, latent_size, device, sample_dir,
        stats, start_idx=1):
    torch.cuda.empty_cache()

    # Losses & scores
    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []

    # Create optimizers
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=0.00002, betas=(0.5, 0.999))
    opt_g = torch.optim.Adam(generator.parameters(), lr=0.00005, betas=(0.5, 0.999))

    for epoch in tqdm(range(epochs)):
        t_loss_d = 0.
        t_loss_g = 0.
        t_real_score = 0.
        t_fake_score = 0.
        n_total = 0
        for real_images, _ in train_dl:
            # Train discriminator
            loss_d, real_score, fake_score = train_discriminator(real_images, opt_d, discriminator, generator,
                                                                 batch_size, latent_size, device)
            # Train generator
            n_gen_train = 1 + epoch // 50
            # for i in range(5):
            loss_g = train_generator(opt_g, discriminator, generator, batch_size, latent_size, device)
            t_loss_d += loss_d * real_images.shape[0]
            t_loss_g += loss_g * real_images.shape[0]
            t_real_score += real_score * real_images.shape[0]
            t_fake_score += fake_score * real_images.shape[0]
            n_total += real_images.shape[0]

        # Record losses & scores
        losses_g.append(t_loss_g/n_total)
        losses_d.append(t_loss_d/n_total)
        real_scores.append(t_real_score/n_total)
        fake_scores.append(t_fake_score/n_total)

        if epoch % 5 == 0:
            # Log losses & scores (last batch)
            print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
                epoch + 1, epochs, t_loss_g/n_total, t_loss_d/n_total, t_real_score/n_total, t_fake_score/n_total))

            # Save generated images
            save_samples(epoch + start_idx, fixed_latent, generator, sample_dir, stats, show=False)

    return losses_g, losses_d, real_scores, fake_scores


def save_samples(index, latent_tensors, generator, sample_dir, stats, show=True):
    fake_images = generator(latent_tensors)
    fake_fname = 'generated-images-{0:0=4d}.png'.format(index)
    save_image(denorm(fake_images, stats), os.path.join(sample_dir, fake_fname), nrow=8)
    print('Saving', fake_fname)
    if show:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0))
        plt.show()


def train_discriminator(real_images, opt_d, discriminator, generator, batch_size, latent_size, device):
    # Clear discriminator gradients
    opt_d.zero_grad()

    # Pass real images through discriminator
    real_preds = discriminator(real_images)
    real_targets = torch.ones(real_images.size(0), 1, device=device)
    real_loss = F.binary_cross_entropy(real_preds, real_targets)
    real_score = torch.mean(real_preds).item()

    # Generate fake images
    latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
    fake_images = generator(latent)

    # Pass fake images through discriminator
    fake_targets = torch.zeros(fake_images.size(0), 1, device=device)
    fake_preds = discriminator(fake_images)
    fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)
    fake_score = torch.mean(fake_preds).item()

    # Update discriminator weights
    loss = real_loss + fake_loss
    loss.backward()
    opt_d.step()
    return loss.item(), real_score, fake_score


def train_generator(opt_g, discriminator, generator, batch_size, latent_size, device):
    # Clear generator gradients
    opt_g.zero_grad()

    # Generate fake images
    latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
    fake_images = generator(latent)

    # Try to fool the discriminator
    preds = discriminator(fake_images)
    targets = torch.ones(batch_size, 1, device=device)
    loss = F.binary_cross_entropy(preds, targets)

    # Update generator weights
    loss.backward()
    opt_g.step()

    return loss.item()


def show_images(images, nmax, stats):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(make_grid(denorm(images.detach()[:nmax], stats), nrow=8).permute(1, 2, 0))
    plt.show()


def denorm(img_tensors, stats):
    return img_tensors * stats[1][0] + stats[0][0]


def show_batch(dl, nmax, stats):
    for images, _ in dl:
        show_images(images, nmax, stats)
        break


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


if __name__ == '__main__':
    train(retrain=False, start_idx=0)
