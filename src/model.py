from torch import nn

class GeneratorBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, act: nn.Module = nn.ReLU()):
        super(GeneratorBlock, self).__init__()
        self.conv2d_trans = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, padding=padding, stride=stride
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.act = act

class Generator(nn.Sequential):
  def __init__(self, init_channels: int = 1024, latent_dim: int = 100, img_channels: int = 3):
    super(Generator, self).__init__()
    self.append(
        GeneratorBlock(in_channels=latent_dim, out_channels=init_channels, stride=1, kernel_size=4, padding=0)
    ) # output shape: (init_channels, 4, 4)
    self.append(
        GeneratorBlock(in_channels=init_channels, out_channels=init_channels//2, stride=2, kernel_size=4, padding=1)
    ) # output shape: (init_channels//2, 8, 8)
    self.append(
        GeneratorBlock(in_channels=init_channels//2, out_channels=init_channels//4, stride=2, kernel_size=4, padding=1)
    ) # output shape: (init_channels//4, 16, 16)
    self.append(
        GeneratorBlock(in_channels=init_channels//4, out_channels=init_channels//8, stride=2, kernel_size=4, padding=1)
    ) # output shape: (init_channels//8, 32, 32)
    self.append(
        nn.ConvTranspose2d(in_channels=init_channels//8, out_channels=img_channels, stride=2, kernel_size=4, padding=1)
    ) # output shape: (init_channels//16, 64, 64)
    self.append(nn.Tanh())

class DiscriminatorBlock(nn.Sequential):
  def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, alpha: float = 0.2):
    super(DiscriminatorBlock, self).__init__()
    self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
    self.batch_norm = nn.BatchNorm2d(out_channels)
    self.leaky_relu = nn.LeakyReLU(alpha)


class Discriminator(nn.Sequential):
  def __init__(self, init_channels: int = 64, img_channels: int = 3):
    super(Discriminator, self).__init__()
    self.append(
        DiscriminatorBlock(in_channels=img_channels, out_channels=init_channels, kernel_size=4, stride=2, padding=1)
    ) # output shape: (init_channels, 32, 32)
    self.append(
        DiscriminatorBlock(in_channels=init_channels, out_channels=init_channels*2, kernel_size=4, stride=2, padding=1)
    ) # output shape: (init_channels*2, 16, 16)
    self.append(
        DiscriminatorBlock(in_channels=init_channels*2, out_channels=init_channels*4, kernel_size=4, stride=2, padding=1)
    ) # output shape: (init_channels*4, 8, 8)
    self.append(
        DiscriminatorBlock(in_channels=init_channels*4, out_channels=init_channels*8, kernel_size=4, stride=2, padding=1)
    ) # output shape: (init_channels*8, 4, 4)
    self.append(
        nn.Conv2d(in_channels=init_channels*8, out_channels=1, kernel_size=4, stride=2, padding=0)
    ) # output shape: (1, 1, 1)
