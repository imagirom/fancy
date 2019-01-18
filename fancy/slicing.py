import numpy as np

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


def _is_positive(sl):
    return sl.start >= 0 and sl.stop >= 0


def slice_overlap(sl0, sl1, size=None):
    """
    Calculates the overlap between two slices (or tuples of slices for higher dimensions) in the global and local
    reference frames

    :param sl0: slice or tuple
    :param sl1: slice or tuple
    :param size: int or tuple
    :return:
        tuple
    """
    if isinstance(sl0, tuple):   # more than one dimension
        sizes = (None,) * len(sl0) if size is None else size
        assert len(sl0) == len(sl1) == len(sizes)
        return tuple(zip(*(slice_overlap(sl0[i], sl1[i], size) for i, size in enumerate(sizes))))

    sl0, sl1 = [_normalize_slice(sl, size) for sl in (sl0, sl1)]
    assert all(_is_positive(sl) for sl in (sl0, sl1))
    if not all(sl.step in (1, None) for sl in (sl0, sl1)):
        raise NotImplementedError
    global_intersection = slice(max(sl0.start, sl1.start), min(sl0.stop, sl1.stop))
    relative_intersections = [slice(global_intersection.start - sl.start, global_intersection.stop - sl.start)
                              for sl in (sl0, sl1)]
    return [global_intersection] + relative_intersections


if __name__ == '__main__':
    print(slice(0).step)
    print(slice_overlap((slice(10, 20), slice(6, 12)),
                        (slice(15, 25), slice(3, 15))))