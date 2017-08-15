# -*- coding: utf-8 -*-
#
from __future__ import print_function

import pipdate

from .__about__ import (
    __version__,
    __author__,
    __author_email__,
    __website__
    )

# pylint: disable=wildcard-import
from .helpers import *

if pipdate.needs_checking(__name__):
    print(pipdate.check(__name__, __version__))
