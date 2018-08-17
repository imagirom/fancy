import numpy as np
import colorsys


def hsv_to_rgb(h, s, v):
    return np.array(colorsys.hsv_to_rgb(h, s, v), dtype=np.float32)


def get_distinct_colors(n, min_sat=.5, min_val=.5):
    huePartition = 1.0 / (n + 1)
    hues = np.arange(0, n) * huePartition
    saturations = np.random.rand(n) * (1-min_sat) + min_sat
    values = np.random.rand(n) * (1-min_val) + min_val
    return np.stack([hsv_to_rgb(h, s, v) for h, s, v in zip(hues, saturations, values)], axis=0)


def colorize_segmentation(seg, ignore_label=None, ignore_color=(0, 0, 0)):
    assert isinstance(seg, np.ndarray)
    assert seg.dtype.kind in ('u', 'i')
    if ignore_label is not None:
        ignore_ind = seg == ignore_label
    seg = seg - np.min(seg)
    colors = get_distinct_colors(np.max(seg) + 1)
    np.random.shuffle(colors)
    result = colors[seg]
    if ignore_label is not None:
        result[ignore_ind] = ignore_color
    return result


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    seg = np.random.randint(0, 100, (10, 10), dtype=np.int32)
    plt.imshow(colorize_segmentation(seg))
    plt.show()
