import numpy as np
import multiprocessing as mp
import itertools


def verbose_wrapper(func, name='function'):
    def verbose_func(*args, **kwargs):
        print(f'{name} called with args {args} and kwargs {kwargs}')
        return func(*args, **kwargs)
    return verbose_func


# pretty unnecessary, this is basically mp.Pool.apply
def execute_parallel(func, args_list=None, kwargs_list=None, n_jobs=4, verbose=True):
    assert args_list is not None or kwargs_list is not None
    #if verbose:
    #    func = verbose_wrapper(func)
    pool = mp.Pool(processes=n_jobs)
    if args_list is not None and kwargs_list is not None:
        results = [pool.apply_async(func, args=args, kwds=kwargs)
                   for args, kwargs in zip(args_list, kwargs_list)]
    elif args_list is not None:
        results = [pool.apply_async(func, args=args)
                   for args in args_list]
    elif kwargs_list is not None:
        results = [pool.apply_async(func, kwds=kwargs)
                   for kwargs in kwargs_list]
    else:
        assert False, 'either args_list or kwargs_list must not be None'
    results = [p.get() for p in results]
    return results


def grid_evaluate(func, *args, **kwargs):
    """
    Evaluates the function func on the cartesian product of the lists given for all the args and kwargs.

    :param func: callable
        Function to be evaluated.
    :param args:
        Lists of positional argument values to be iterated over.
    :param kwargs:
        Lists of keyword argument values to be iterated over.

    :return:
        dict
    """
    n_jobs = kwargs.pop('n_jobs', 4)
    return_args = kwargs.pop('return_args', False)
    shape = tuple([len(arg) for arg in args + tuple(kwargs.values())])

    args_and_kwargs = list(itertools.product(*(args + tuple(kwargs.values()))))
    n_args = len(args)
    args_list = [a[:n_args] for a in args_and_kwargs]
    kwargs_list = [dict(zip(kwargs.keys(), a[n_args:])) for a in args_and_kwargs]

    result = np.empty(len(args_and_kwargs), dtype=object)
    result[...] = execute_parallel(func, args_list, kwargs_list, n_jobs)
    result = result.reshape(shape)
    if return_args:  # wrap args in numpy array with same shape as result
        arg_array = np.empty(len(args_and_kwargs), dtype=object)
        arg_array[...] = args_and_kwargs
        arg_array = arg_array.reshape(shape)
        result = (result, arg_array)
    return result


if __name__ == '__main__':
    def prod(x, y):
        return x * y

    print(grid_evaluate(prod, np.arange(10), y=10*np.arange(5)).astype(np.int32))

    def func(x, y):
        for i in range(10000):
            z = np.exp(12)**.5


    from timeit import default_timer as timer
    start = timer()
    print(execute_parallel(func, kwargs_list=[{'x': 0, 'y': 0}] * 100, n_jobs=4))
    end = timer()
    time_parallel = end - start

    start = timer()
    print(execute_parallel(func, kwargs_list=[{'x': 0, 'y': 0}] * 100, n_jobs=1))
    end = timer()
    time_sequential = end - start

    print(f'speedup: {time_sequential / time_parallel}')
