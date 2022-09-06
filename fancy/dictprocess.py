import ast
import inspect


def extract_return(func, file=None):
    if file==None:
        file = inspect.getsourcefile(func)
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


def act_on_dict(output_names=None, input_names=None, mode='add'):
    """
    Decorator used to make function take a single dict as input and alter that dict as output
    :param func: Function to be decorated.
    :param input_names: Keys of dict whose values should be used as inputs of func.
    If None, this gets extracted from function signature
    :param output_names: Keys used to store outputs of func in. If not specified, this gets extracted from the functions
    return statements
    :param mode: If 'add': Output of func is added to input dict. If 'replace': New dictionary only with outputs of func
    is returned
    :return: Dictionary containing outputs of func, and, depending on mode, everything the input contained
    """
    def wrapper(func):
        assert mode in ACTING_MODES, f'mode has to be one of {ACTING_MODES}'
        # use names of return variables of func if keys to save returned values is not specified
        if output_names is None:
            provides = extract_return(func)
        else:
            provides = output_names

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
            if input_names is not None:
                returns = func(*(dictionary[arg] for arg in args))
            elif varkw is not None:
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
            for name, value in zip(provides, returns):
                result[name] = value

            return result

        # add attributes to function specifying which keys are required, used, provided
        inner.requires = requires
        inner.uses = uses
        inner.provides = provides

        return inner

    if callable(output_names):
        func = output_names
        output_names = None
        return wrapper(func)
    else:
        return wrapper


def extract_from_dict(*keys):
    """
    returns function that takes a dictionary as input and returns a list values of that dictionary at specified keys
    :param keys: keys to use
    :return: function mapping dictionaries to lists of their values at keys
    """
    def extractor(dictionary):
        return [dictionary(key) for key in keys]
    return extractor


if __name__=='__main__':
    @act_on_dict
    def foo(x, v=3, *, y, t=2):
        a = 10 * x
        b = 10 * y
        return [a, b]

    d = {'x': 1, 'y': 2, 'z': 0}
    d = foo(d)
    print(d)  # {'x': 1, 'y': 2, 'z': 0, 'a': 10, 'b': 20}
    #print(extract_return('foo'))
