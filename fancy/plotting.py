import ipywidgets as widgets
import matplotlib.pyplot as plt


def image_interact(arr, figsize=(8, 8), colorbar=False, **imshow_kwargs):
    n_sliders = len(arr.shape) - 2
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