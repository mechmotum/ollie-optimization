"""Mixin for all model object classes."""

from __future__ import annotations

import sympy as sm

from ollie.container import Container


def is_auxiliary_data(container: Container) -> bool:
    """Utility function to tell if a ``Container`` is auxiliary data."""
    return (
        isinstance(container.value, (sm.Expr, sm.Symbol))
        and bounds is None
        and guess is None
    )

def is_static_parameter(container: Container) -> bool:
    """Utility function to tell if a ``Container`` is a static parameter."""
    return (
        container.bounds is not None
        and container.bounds[0] < container.bounds[1]
        and container.guess is not None
    )


class ModelObject:
    """Mixin class for all model object classes to inherit from."""

    def symbols_to_values_mapping(self) -> dict[sm.Symbol, sm.Expr | float]:
        """Mappings from symbols to auxiliary data (expressions or numbers)."""
        return {
            container.symbol: container.value
            for container in self.__dict__.values()
            if is_auxiliary_data(container)
        }

    def num_static_parameters(self) -> int:
        """Number of static parameters an instance will yield."""
        return len(self.static_parameters)

    def static_parameters(self) -> tuple[sm.Symbol]:
        """The static parameters (as symbols) that an instance yields."""
        return tuple(
            container.symbol
            for container in self.__dict__.values()
            if is_static_parameter(container)
        )

    def static_parameter_bounds(self) -> tuple[tuple[float, float]]:
        """The bounds on an instance's static parameters."""
        return tuple(
            container.bounds
            for container in self.__dict__.values()
            if is_static_parameter(container)
        )

    def static_parameter_guesses(self) -> tuple[float]:
        """The guesses for an instance's static parameters."""
        return tuple(
            container.guess
            for container in self.__dict__.values()
            if is_static_parameter(container)
        )
