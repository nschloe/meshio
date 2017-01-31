# -*- coding: utf-8 -*-
#
import numpy


def dtype_digits(dt):
    '''Given a NumPy float dtype, return the number of significant decimal
    digits. Useful for formatted output.
    '''
    # get precision, cf.
    # <https://docs.scipy.org/doc/numpy/user/basics.types.html>
    if dt == numpy.float16:
        mantissa = 10
    elif dt == numpy.float32:
        mantissa = 23
    elif dt == numpy.float64:
        mantissa = 52
    else:
        raise RuntimeError('Unknown float type')
    significant_digits = mantissa * numpy.log(2) / numpy.log(10)
    return int(significant_digits)
