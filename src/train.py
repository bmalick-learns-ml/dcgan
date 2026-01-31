import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn

from src.utils import show_batch

def update_discriminator(x, z, D, G, criterion, trainer_D):
    batch_size = x.shape[0]
    ones = torch.ones((batch_size,), device=x.device)
    zeros = torch.zeros((batch_size,), device=x.device)

    trainer_D.zero_grad()

    real_y = D(x)
    fake_y = D(G(z))
    loss_D = (criterion(real_y, ones.reshape(real_y.shape)) +
            criterion(fake_y, zeros.reshape(fake_y.shape))) / 2
    loss_D.backward()
    trainer_D.step()
    return loss_D.item()

def update_generator(z, D, G, criterion, trainer_G):
    batch_size = z.shape[0]
    ones = torch.ones((batch_size,), device=z.device)

    trainer_G.zero_grad()

    fake_y = D(G(z))
    loss_G = criterion(fake_y, ones.reshape(fake_y.shape))
    loss_G.backward()
    trainer_G.step()
    return loss_G.item()


def train_dcgan(D, G, lr_D, lr_G, latent_dim, dataloader, num_epochs, device, fixed_noise, visualize=True, print_every=50):
    print(f"Device: {device}")
    criterion = nn.BCEWithLogitsLoss(reduction="mean")

    for w in D.parameters(): torch.nn.init.normal_(w, 0., 0.02)
    for w in G.parameters(): torch.nn.init.normal_(w, 0., 0.02)

    # trainer_D = torch.optim.SGD(D.parameters(), lr_D, momentum=0.5)
    # trainer_G = torch.optim.SGD(G.parameters(), lr_G, momentum=0.5)
    trainer_D = torch.optim.Adam(D.parameters(), lr_D, betas=(0.5, 0.999))
    trainer_G = torch.optim.Adam(G.parameters(), lr_G, betas=(0.5, 0.999))

    metrics = []

    for epoch in range(num_epochs):
        loss_D = 0.
        loss_G = 0.
        num_instances = 0
        for step_num, (x,l) in enumerate(dataloader):
            x = x.to(device)
            batch_size = x.shape[0]
            num_instances += batch_size
            z = torch.normal(0., 1., size=(batch_size, latent_dim, 1, 1), device=x.device)

            loss_D += update_discriminator(x=x, z=z, D=D, G=G, criterion=criterion, trainer_D=trainer_D)
            loss_G += update_generator(z=z, D=D, G=G, criterion=criterion, trainer_G=trainer_G)
            # if step_num % print_every==0:
            #   print(f"[Epoch {epoch}/{num_epochs}] [Step {step_num}/{len(dataloader)}] loss_D: {loss_D/num_instances:.4f}, loss_G: {loss_G/num_instances:.4f}")


        loss_D /= num_instances
        loss_G /= num_instances
        metrics.append([loss_D, loss_G])
        print(f"[Epoch {epoch}/{num_epochs}] loss_D: {loss_D:.4f}, loss_G: {loss_G:.4f}")

        if visualize:
            os.makedirs("visualizations", exist_ok=True)
            # z = torch.normal(0., 1., size=(16, latent_dim, 1, 1), device=x.device)
            fake_data = G(fixed_noise.to(device)).detach().cpu()
            show_batch((fake_data,None), save_name=f"visualizations/generated-{epoch:02d}.png")
    metrics = np.array(metrics)
    plt.plot(metrics[:, 0], label="loss_D")
    plt.plot(metrics[:, 1], label="loss_G")
    plt.legend()
    plt.show()