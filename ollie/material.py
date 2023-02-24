""""""

import dataclasses

import sympy as sm

from ollie.container import Container


@dataclasses.dataclass
class Glue:
    """"""

    specific_mass: Container = Container(
        symbol=sm.Symbol(r"\gamma_{glue}"), value=0.210
    )


@dataclasses.dataclass
class Polyurethane:
    """"""

    density: Container = Container(symbol=sm.Symbol(r"\rho_{pu}}"), value=1130)


@dataclasses.dataclass
class Maple:
    """"""

    density: Container = Container(symbol=sm.Symbol(r"\rho_{maple}"), value=705.0)


@dataclasses.dataclass
class Steel:
    """"""

    density: Container = Container(symbol=sm.Symbol(r"\rho_{steel}"), value=7700.0)
