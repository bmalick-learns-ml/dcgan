import matplotlib.pyplot as plt
import numpy as np


def show_batch(batch, figsize=(10,8), num_images=None, save_name=None):
    images, labels = batch
    num_images = num_images if num_images is not None else len(images)
    nrows = int(np.sqrt(num_images))
    ncols = int(np.ceil(num_images / nrows))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.ravel()
    for ax in axes: ax.axis("off")
    for i in range(num_images):
        axes[i].imshow(images[i][0], cmap="gray")
    
    if save_name is not None: plt.savefig(save_name)