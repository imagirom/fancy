from copy import deepcopy


def recursive_choice_inplace(d, key):
    if not isinstance(d, dict):
        return d
    for k in d:
        if k == key:
            return recursive_choice_inplace(d[k], key)
        d[k] = recursive_choice_inplace(d[k], key)
    return d


def copy_nested_dict(d: dict):
    assert isinstance(d, dict), f'Not a dict: {d}, {type(d)}'

    def dd(d):
        if isinstance(d, dict):
            return {key: dd(value) for key, value in d.items()}
        else:
            return d

    return dd(d)


def recursive_choice(d, key):
    return recursive_choice_inplace(copy_nested_dict(d), key)


def recursive_update_inplace(d1, d2):
    '''
    Update d1 with the data from d2 recursively
    :param d1: dict
    :param d2: dict
    :return: None
    '''
    for key, value in d2.items():
        if key in d1 and isinstance(d1[key], dict) and isinstance(value, dict):
            recursive_update_inplace(d1[key], value)
        else:
            d1[key] = value


def recursive_update(d1, d2):
    d1 = deepcopy(d1)
    recursive_update_inplace(d1, d2)
    return d1


if __name__ == '__main__':

    d = {
        'C': {
            'A': 2,
            'B': 7,
        },
        'D': {
            'C': {},
            'D': {
                'B': 3,
                'A': 5
            }
        }
    }

    print(recursive_choice(d, 'B'))

    d1 = {
        'A': {
            'B': 0,
            'C': 0,
        },
        'D': {
            'E': 0,
            'F': {
                'G': 0
            }
        }
    }

    d2 = {
        'A': {
            'B': 1,
            'H': 1,
        },
        'D': {
            'I': 1,
            'F': 1,
        }
    }
    print(recursive_update(d1, d2))
