import numpy as np
import torch


def clip_extreme(imgs, percentile=5, allowed_steepness=3, dim=2):
    """
    normalization robust to outliers
    :param imgs: input images with arbitrary shape; last dim dimensions are assumed to be image dimensions
    :param percentile:
    :param dim: data dimensionality (probably 1, 2 or 3)
    :return: normalized data
    """
    if isinstance(imgs, torch.Tensor):
        return_torch = True
        device = imgs.device
        dtype = imgs.dtype
        imgs = imgs.cpu().numpy()
    else:
        return_torch=False
    img_shape = imgs.shape[-dim:]
    imgs = imgs.reshape(imgs.shape[:-dim] + (-1,))
    # mins = np.min(imgs, axis=-1)
    # maxs = np.max(imgs, axis=-1)
    soft_mins = np.percentile(imgs, percentile, axis=-1)
    soft_maxs = np.percentile(imgs, 100-percentile, axis=-1)
    ranges = soft_maxs - soft_mins
    print(ranges)
    allowed_extra = ranges * percentile/100 * allowed_steepness
    imgs = imgs.clip((soft_mins - allowed_extra)[..., None],
                     (soft_maxs - allowed_extra)[..., None])
    imgs = imgs.reshape(imgs.shape[:-1] + img_shape)
    if return_torch:
        return torch.tensor(imgs, dtype=dtype).to(device)
    else:
        return imgs


if __name__ == '__main__':
    a = np.array([
        [[1, 2, 3],
         [1, 2, 3],
         [1, 2, 3],
         [1, 2, 3],
         [1, 2, 3],
         [1, 2, 3],
         [100, 20, -132412]],

        [[1, 2, 3],
         [4, 5, 6],
         [4, 5, 6],
         [4, 5, 6],
         [4, 5, 6],
         [4, 5, 6],
         [100, 20, -4]],
    ], dtype=int)

    print(clip_extreme(a, percentile=5))
