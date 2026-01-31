import torch
import torchvision

from src.model import Discriminator, Generator
from src.train import train_dcgan

if __name__=="__main__":
    image_size = 64
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,),(0.5,))
        ])

    train_data = torchvision.datasets.MNIST(root="data/mnist", train=True, download=True, transform=transform)

    train_batch_size = 128
    num_workers = 2
    train_dataloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=train_batch_size, num_workers=num_workers, shuffle=True)


    lr_D = 0.0002
    lr_G = 0.0002
    num_epochs = 50
    latent_dim = 100
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    discriminator = Discriminator(init_channels=64, img_channels=1).to(device)
    fixed_noise = torch.normal(0., 1., size=(16, latent_dim, 1, 1))
    generator = Generator(init_channels=1024, latent_dim=latent_dim, img_channels=1).to(device)
    train_dcgan(D=discriminator, G=generator, lr_G=lr_G, lr_D=lr_D, latent_dim=latent_dim,
                dataloader=train_dataloader, num_epochs=num_epochs,
                device=device, fixed_noise=fixed_noise, visualize=True, print_every=100)