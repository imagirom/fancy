from copy import deepcopy


def recursive_choice_inplace(d, key):
    if not isinstance(d, dict):
        return d
    for k in d:
        if k == key:
            return recursive_choice_inplace(d[k], key)
        d[k] = recursive_choice_inplace(d[k], key)
    return d


def recursive_choice(d, key):
    return recursive_choice_inplace(deepcopy(d), key)


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
