"""Constant values required for modelling."""

import sympy as sm

from ollie.container import Container


GRAVITY = Container(symbol=sm.Symbol(r"g"), value=sm.Float(9.81))
