from pandas import DataFrame

from pext.funcs import EXTENSION_FUNCS


def monkey_patch_data_frame():
    for func_name, func in EXTENSION_FUNCS.items():
        if not hasattr(DataFrame, func_name):
            setattr(DataFrame, func_name, func)
        else:
            raise AttributeError('Function %s already exists in DataFrame' % func_name)

