import ast
import inspect


def extract_return(func, file=__file__):
    fname = func.__name__
    for x in ast.walk(ast.parse(open(file).read())):
        if not(isinstance(x, ast.FunctionDef)):
            continue
        if not(x.name == fname):
            continue
        for b in x.body:
            if isinstance(b, ast.Return):
                if isinstance(b.value, ast.Name):
                    return b.value.id,
                elif hasattr(b.value, 'elts'):
                    assert all(hasattr(v, 'id') for v in b.value.elts), \
                        'cannot extract return names from nested iterable'
                    return [v.id for v in b.value.elts]
                else:
                    assert False, 'could not extract return names'


ACTING_MODES = ['add', 'replace']


def act_on_dict(func, input_names=None, output_names=None, mode='add'):
    """
    Decorator used to make function take a single dict as input and alter that dict as output
    :param func:
    :param input_names:
    :param output_names:
    :return:
    """
    assert mode in ACTING_MODES, f'mode has to be one of {ACTING_MODES}'
    # use names of return variables of func if keys to save returned values is not specified
    if output_names is None:
        output_names = extract_return(func)

    # use argument names in case keys to get input values is not specified
    if input_names is None:
        args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, annotations = inspect.getfullargspec(func)
        requires = (args if defaults is None else args[:len(args) - len(defaults)]) + \
                   (kwonlyargs if kwonlydefaults is None else kwonlyargs[:len(kwonlyargs) - len(kwonlydefaults)])
        uses = args + kwonlyargs
    else:
        args = input_names
        varkw = None
        kwonlyargs = []

        requires = args
        uses = args

    # define function to act on dictionary
    def inner(dictionary):
        # check that all required arguments are present
        for arg in inner.requires:
            assert arg in dictionary, \
                f"key '{arg}' whose value is required by function '{func.__name__}' is missing"

        # apply function
        if varkw is not None:
            returns = func(**dictionary)
        else:
            returns = func(
                **{arg: dictionary[arg] for arg in args if arg in dictionary},
                **{kwonlyarg: dictionary[kwonlyarg] for kwonlyarg in kwonlyargs if kwonlyarg in dictionary})

        # add to input or construct new dict based on mode
        if mode == 'add':
            result = dictionary
        else:
            result = {}
        for name, value in zip(output_names, returns):
            result[name] = value

        return result

    # add attributes to function specifying which keys are required, used, provided
    inner.requires = requires
    inner.uses = uses
    inner.provides = output_names

    return inner


def extract_from_dict(*keys):
    """
    returns function that takes a dictionary as input and returns a list values of that dictionary at specified keys
    :param keys: keys to use
    :return: function mapping dictionaries to lists of their values at keys
    """
    def extractor(dictionary):
        return [dictionary(key) for key in keys]
    return extractor


@act_on_dict
def foo(x, v=3, *, y, t=2):
    a = 10 * x
    b = 10 * y
    return [a, b]


@act_on_dict
def bar(x, y):
    a = x
    b = y
    return a, b


if __name__=='__main__':
    d = {'x': 1, 'y': 2, 'z': 0}
    d = foo(d)
    print(d)
    #print(extract_return('foo'))
