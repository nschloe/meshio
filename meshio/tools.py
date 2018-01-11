# -*- coding: utf-8 -*-
#
import sys


def get_byteorder(array):
    if array.dtype.byteorder in ['>', '<']:
        return array.dtype.byteorder

    assert array.dtype.byteorder == '='
    return '<' if sys.byteorder == 'little' else '>'
