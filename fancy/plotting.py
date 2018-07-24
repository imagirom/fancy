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