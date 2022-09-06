
def named_children_recursive(module, name='model', at_depth=None, max_depth=None, separator='.'):
    """
    Returns a generator that recursively traverses named submodules of module (including itself),
    yielding pairs of 'paths' (such as model.encoder.1) and corresponding modules.
    """
    memo = set()
    def print_named_recursive(name, module, depth):
        if (at_depth is None or depth == at_depth) and module not in memo:
            yield name, module
            memo.add(module)
        if max_depth is None or depth < max_depth:
            for child_name, child in module.named_children():
                if module is None:
                    continue
                yield from print_named_recursive(f'{name}{separator}{child_name}', child, depth+1)
    if at_depth is not None:
        assert max_depth is None, f"Please specify only one of 'max_depth' and 'at_depth'"
        max_depth = at_depth
    yield from print_named_recursive(name, module, 0)
