# -*- coding: utf-8 -*-
#
import pipdated

from .__about__ import (
    __version__,
    __author__,
    __author_email__,
    __website__
    )

from .helpers import *

if pipdated.needs_checking(__name__):
    print(pipdated.check(__name__, __version__))
