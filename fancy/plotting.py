import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import torch


def image_interact(arr, color_channel=None, figsize=(8, 8), colorbar=False, **imshow_kwargs):
    if isinstance(arr, torch.Tensor):
        arr = arr.cpu().numpy()
    if color_channel is not None:
        assert arr.shape[color_channel] in (3, 4)
        arr = np.moveaxis(arr, color_channel, -1)
    n_sliders = len(arr.shape) - 2 if color_channel is None else len(arr.shape) - 3
    dim_names = [f'dim {i}' for i in range(n_sliders)]

    def f(**coords):
        plt.figure(figsize=figsize)
        im = plt.imshow(arr[tuple(coords[n] for n in dim_names)], **imshow_kwargs)
        if colorbar:
            plt.colorbar(im)
        plt.show()

    widgets.interact(f, **{dim_names[i]: widgets.IntSlider(min=0, max=arr.shape[i] - 1, step=1, value=0) for i in
                           range(n_sliders)})


def interactive_3D_scatter(pts, figsize=(8, 8), **kwargs):
    vmax = np.max(np.abs(pts[:, 1:])) * 1.1
    ind = [pts[:, 0] == z for z in np.arange(np.min(pts[:, 0]), np.max(pts[:, 0])+1)]
    def f(z):
        plt.figure(figsize=figsize)
        plt.scatter(pts[ind[z], 1], pts[ind[z], 2], **kwargs)
        plt.xlim(-vmax, vmax)
        plt.ylim(-vmax, vmax)
        plt.gca().set_aspect(1)
        plt.title(f'z={z}')
        plt.show()
    widgets.interact(f, z=widgets.IntSlider(
        min=np.min(pts[:, 0]), max=np.max(pts[:, 0]), step=1, value=np.min(pts[:, 0]), ))