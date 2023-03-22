"""Materials and their properties."""

import dataclasses
from abc import ABC

import sympy as sm

from ollie.container import Container


class Material(ABC):
    """Abstract base class for material dataclasses."""
    pass


@dataclasses.dataclass
class Glue(Material):
    """Dataclass container properties of glue."""

    specific_mass: Container = Container(
        symbol=sm.Symbol(r"\gamma_{glue}"), value=0.210
    )


@dataclasses.dataclass
class Polyurethane(Material):
    """Dataclass containing properties of polyurethane."""

    density: Container = Container(symbol=sm.Symbol(r"\rho_{pu}}"), value=1130)


@dataclasses.dataclass
class Maple(Material):
    """Dataclass containing properties of maple."""

    density: Container = Container(symbol=sm.Symbol(r"\rho_{maple}"), value=705.0)


@dataclasses.dataclass
class Steel(Material):
    """Dataclass containing properties of steel."""

    density: Container = Container(symbol=sm.Symbol(r"\rho_{steel}"), value=7700.0)
