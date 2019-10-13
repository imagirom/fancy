import numpy as np
from collections import Iterable


def _normalize_slice(sl, size=None):
    start, stop, step = sl.start, sl.stop, sl.step
    if start is None:
        assert stop is None and step is None
        start, stop, step = 0, size, 1
    elif stop is None:
        stop = size
    if step is None:
        step = 1
    assert all(x is not None for x in [start, stop, step]), 'Must give a size'
    start, stop = [border if border >= 0 else border + size
                   for border in (start, stop)]
    assert start >= 0 and stop >= 0
    return slice(start, stop, step)


def shrink_slice(sl, margin):
    if isinstance(sl, Iterable):
        return tuple(shrink_slice(s, margin) for s in sl)
    sl = _normalize_slice(sl)
    assert margin % sl.step == 0
    return slice(sl.start + margin, sl.stop - margin, sl.step)


def _is_positive(sl):
    return sl.start >= 0 and sl.stop >= 0


def slice_overlap(*slices, size=None):
    """
    Calculates the overlap between two slices (or tuples of slices for higher dimensions) in the global and local
    reference frames

    :param sl0: slice or tuple
    :param sl1: slice or tuple
    :param size: int or tuple
    :return:
        tuple
    """
    if isinstance(slices[0], Iterable):   # more than one dimension
        sizes = (None,) * len(slices[0]) if size is None else size
        assert all(len(sl) == len(slices[0]) for sl in slices[1:])
        return tuple(zip(*(slice_overlap(*(sl[i] for sl in slices), size=size) for i, size in enumerate(sizes))))

    slices = [_normalize_slice(sl, size) for sl in slices]
    assert all(_is_positive(sl) for sl in slices)
    if not all(sl.step in (1, None) for sl in slices):
        raise NotImplementedError
    global_intersection = slice(max(sl.start for sl in slices), min(sl.stop for sl in slices))
    relative_intersections = [slice(global_intersection.start - sl.start, global_intersection.stop - sl.start)
                              for sl in slices]
    return [global_intersection] + relative_intersections


def slice_hull(*slices, size=None):
    """
    Calculate tight global slice containing all slices

    :param sl0: slice or tuple
    :param sl1: slice or tuple
    :param size: int or tuple
    :return:
        tuple
    """
    if isinstance(slices[0], Iterable):   # more than one dimension
        sizes = (None,) * len(slices[0]) if size is None else size
        assert all(len(sl) == len(slices[0]) for sl in slices[1:])
        return tuple(slice_hull(*(sl[i] for sl in slices), size=size) for i, size in enumerate(sizes))

    slices = [_normalize_slice(sl, size) for sl in slices]
    assert all(_is_positive(sl) for sl in slices)
    if not all(sl.step in (1, None) for sl in slices):
        raise NotImplementedError
    global_hull = slice(min(sl.start for sl in slices), max(sl.stop for sl in slices))
    return global_hull


def minimum_global_shape(*slices):
    return tuple(sl.stop for sl in slice_hull(*slices))


def slices_to_array(slice_list):
    slice_list = np.array(slice_list)
    shape = slice_list.shape
    return np.array([[sl.start, sl.stop, sl.step] for sl in slice_list.flatten()]).reshape(*shape, 3) # changed dtype from int32


def array_to_slices(array):
    assert array.shape[-1] == 3
    shape = array.shape[:-1]
    return np.array([slice(*args) for args in array.reshape(-1, 3)]).reshape(*shape)


def slice_argsort_lexicographical(slice_list):
    starts = slices_to_array(slice_list)[:, :, 0]
    return np.lexsort(tuple(starts.transpose()[::-1]))


def slice_argsort_diagonal(slice_list):
    diag = slices_to_array(slice_list)[:, :, 0].sum(1)
    return np.argsort(diag)


def slice_argsort(slice_list, order='lex'):
    if order == 'lex':
        return slice_argsort_lexicographical(slice_list)
    elif order == 'diag':
        return slice_argsort_diagonal(slice_list)
    else:
        assert False, f'order must be in ("lex", "diag"), but got {order}'


def center_slice(big_shape, center_shape):
    to_crop = [s_old - s_new for s_old, s_new in zip(big_shape, center_shape)]
    assert all([crop % 2 == 0 for crop in to_crop]), f'{to_crop}'
    to_crop = [crop//2 for crop in to_crop]
    return tuple(slice(crop, size-crop) for size, crop in zip(big_shape, to_crop))


if __name__ == '__main__':
    a = np.array([[1, 2, 3], [1, 2, 3], [3, 4, 5], [3, 6, 5]])
    print(a)
    print(a[center_slice(a.shape, (2, 3))])
    print(slice(0).step)
    slices = [(slice(15, 20), slice(6, 12)), (slice(15, 25), slice(5, 15))]
    print('sorted lex:', slice_argsort_lexicographical(slices))
    print('sorted diag:', slice_argsort_diagonal(slices))
    print(slice_overlap(*slices))
    print(slice_hull(*slices))
