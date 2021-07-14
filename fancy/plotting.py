import ipywidgets as widgets
import matplotlib.pyplot as plt
import collections
import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None


def plot_image(img, figsize=None, figheight=None, figwidth=None, title=None, colorbar=False, colorbar_distance=None,
               colorbar_width=0.05, keep_centered=False, **imshow_kwargs):
    if not colorbar:
        # no extra space for colorbar needed
        colorbar_distance = 0
        colorbar_width = 0
    # default colorbar distance to image to half of its width
    colorbar_distance = colorbar_width / 2 if colorbar_distance is None else colorbar_width
    # calculate the aspect ratio
    aspect = 1 / (img.shape[1] / img.shape[0] + colorbar_distance + colorbar_width)

    # if given, use figsize
    if figsize is not None:
        assert figheight is None
        assert figwidth is None
        figwidth, figheight = figsize

    if figwidth is None:
        # default to height 8
        figheight = 8 if figheight is None else figheight
        figwidth = figheight / aspect
    elif figheight is None:
        figheight = figwidth * aspect
    else:  # both specified
        if figheight / figwidth < aspect:
            figwidth = figheight / aspect
            # print('adjusted width')
        else:
            figheight = figwidth * aspect
            # print('adjusted height')
    fig = plt.figure(figsize=(figwidth, figheight))
    ax = plt.axes([0, 0.0, 1 - aspect * (colorbar_width + colorbar_distance), 1])  # left, bottom, width, height
    if keep_centered:
        assert 'vmin' not in imshow_kwargs or 'vmax' not in imshow_kwargs, \
            f'keep centered does not work with both vmin and vmax being specified'
        v = imshow_kwargs.get('vmax', -imshow_kwargs.get('vmin', -np.max(np.abs(img))))
        assert v >= 0, \
            f'cannot keep centered with vmax smaller or vmax smaller 0.'
        imshow_kwargs['vmin'], imshow_kwargs['vmax'] = -v, v
    im = ax.imshow(img, interpolation='nearest', **imshow_kwargs)
    ax.grid(False)
    if title is not None:
        ax.set_title(title)
    if colorbar:
        cax = plt.axes([1 - aspect * colorbar_width, 0.0, aspect * colorbar_width, 1])
        plt.colorbar(mappable=im, cax=cax)
    return fig, im


def image_interact(arr, cat_along=None, color_channel=None, slider_labels=None, **plot_image_kwargs):
    # convert to numpy
    if torch is not None and isinstance(arr, torch.Tensor):
        arr = arr.cpu().numpy()
    
    n_dims = len(arr.shape)
    # make it so all dimension indices are positive
    color_channel = n_dims + color_channel if color_channel is not None and color_channel < 0 else color_channel
    
    # concatenate along specified axes
    if cat_along is not None:
        # convert cat_along to numpy array
        if not isinstance(cat_along, collections.Iterable):
            cat_along = [cat_along]
        cat_along = [n_dims + dim if dim < 0 else dim for dim in cat_along]
        cat_along = np.array(cat_along, dtype=np.int32)

        # convert negative axis specifications to positive ones
        cat_along[cat_along < 0] += len(arr.shape)
        cat_axes = (((-1) ** np.arange(len(cat_along)))*.5 - 1.5).astype(np.int32)  # alternate concatenation dimension (-1, -2, ..)
        if color_channel == n_dims-1:
            cat_axes -= 1
        elif color_channel == n_dims-2:
            cat_axes[cat_axes == -2] -= 1
        print(cat_axes)

        for cat_axis, dim in zip(cat_axes, sorted(cat_along, reverse=True)):
            arr = np.concatenate(np.moveaxis(arr, dim, 0), axis=cat_axis)
            # if the axis belonging to the color channel moved, adjust it
            if color_channel is not None and dim < color_channel:
                color_channel -= 1

    # move color channel to correct position
    if color_channel is not None:
        assert arr.shape[color_channel] in (3, 4), \
            f'color channel needs to be either RGB or RGBA. Got shape {arr.shape} and channel {color_channel}'
        arr = np.moveaxis(arr, color_channel, -1)

    # get the number and labels and initialize the sliders
    n_sliders = len(arr.shape) - 2 if color_channel is None else len(arr.shape) - 3
    
    if slider_labels is None:
        # default slicer labels to 'index, [0 - max]'
        slider_labels = [f'{i}, [0 - {arr.shape[i]-1}]' for i in range(n_sliders)]
    sliders = {slider_labels[i]: widgets.IntSlider(min=0, max=arr.shape[i] - 1, step=1, value=0)
               for i in range(n_sliders)}

    # FIXME: if matplotlib.get_backend() == 'nbAgg':

    # plot initial image
    #fig, im = plot_image(arr[(0,) * n_sliders], **plot_image_kwargs)

    # function for interaction
    def f(**coords):
        #im.set_data(arr[tuple(coords[n] for n in slider_labels)])
        #fig.canvas.draw_idle()
        plot_image(arr[tuple(coords[n] for n in slider_labels)], **plot_image_kwargs)

    widgets.interact(f, **sliders)


def interactive_3D_scatter(pts, figsize=(8, 8), **kwargs):
    vmax = np.max(np.abs(pts[:, 1:])) * 1.1
    ind = [pts[:, 0] == z for z in np.arange(np.min(pts[:, 0]), np.max(pts[:, 0]) + 1)]

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


def affinity_hist(affinities, offsets, n_bins=100, one_per_length=True, plot_extra='', **hist_kwargs):
    # ax = plt.gca() if ax is None else ax
    offsets = np.array(offsets)

    if one_per_length:
        dim = len(offsets[0])
        assert dim <= 3
        if dim < 3:
            id_func = lambda off: np.linalg.norm(off)
        else:
            id_func = lambda off: (off[0], np.linalg.norm(off[1:]))
        values = np.empty(len(offsets), dtype=object)
        values[...] = [id_func(off) for off in offsets]
        _, ind = np.unique(values, return_index=True)
        ind = sorted(ind)
        offsets = offsets[ind]
        affinities = affinities[ind]

    for off, aff in zip(offsets, affinities):
        plt.figure()
        plt.title(f'Affinity Histogram for offset {off} \n {plot_extra}')
        plt.hist(aff.flatten(), bins=hist_kwargs.pop('bins', 100), density=hist_kwargs.pop('density', True),
                 **hist_kwargs)
        plt.show()
    print(offsets)


