"""Mixin for all model object classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import sympy as sm

from ollie.container import Container
from ollie.typing import DynamicSymbol


def is_auxiliary_data(attribute: Any) -> bool:
    """Utility function to tell if an attribute is auxiliary data."""
    return (
        isinstance(attribute, Container)
        and isinstance(attribute.value, (sm.Expr, sm.Symbol))
        and bounds is None
        and guess is None
    )


def is_state(attribute: Any) -> bool:
    """Utility function to tell if an attribute is a state."""
    return (
        isinstance(attribute, Container)
        and isinstance(attribute.symbol, DynamicSymbol)
        and attribute.state_equation is not None
        and attribute.bounds is not None
        and attribute.bounds[0] < attribute.bounds[1]
        and attribute.guess is not None
    )


def is_control(attribute: Any) -> bool:
    """Utility function to tell if an attribute is a control."""
    return (
        isinstance(attribute, Container)
        and isinstance(attribute.symbol, DynamicSymbol)
        and attribute.state_equation is None
        and attribute.bounds is not None
        and attribute.bounds[0] < attribute.bounds[1]
        and attribute.guess is not None
    )


def is_static_parameter(attribute: Any) -> bool:
    """Utility function to tell if an attribute is a static parameter."""
    return (
        isinstance(attribute, Container)
        and isinstance(attribute.Symbol, sm.Symbol)
        and attribute.state_equation is None
        and attribute.bounds is not None
        and attribute.bounds[0] < attribute.bounds[1]
        and attribute.guess is not None
    )


class ModelObject:
    """Mixin class for all model object classes to inherit from."""

    @property
    def symbols_to_values_mapping(self) -> dict[sm.Symbol, sm.Expr | float]:
        """Mappings from symbols to auxiliary data (expressions or numbers)."""
        mapping = {
            container.symbol: container.value
            for container in self.__dict__.values()
            if is_auxiliary_data(container)
        }
        for attribute in self.__dict__.values():
            if isinstance(attribute, ModelObject):
                mapping = mapping | attribute.symbols_to_values_mapping
        return mapping

    @property
    def number_states(self) -> int:
        """Number of states an instance will yield."""
        return len(self.states)

    @property
    def states(self) -> tuple[DynamicSymbol]:
        """The states (as dynamic symbols) that an instance yields."""
        states = tuple(
            container.symbol
            for container in self.__dict__.values()
            if is_state(container)
        )
        for attribute in self.__dict__.values():
            if isinstance(attribute, ModelObject):
                states += attribute.states
        return states

    @property
    def state_bounds(self) -> tuple[tuple[float, float]]:
        """The bounds on an instance's states."""
        bounds = tuple(
            container.bounds
            for container in self.__dict__.values()
            if is_state(container)
        )
        for attribute in self.__dict__.values():
            if isinstance(attribute, ModelObject):
                bounds += attribute.state_bounds
        return bounds

    @property
    def state_guesses(self) -> tuple[float]:
        """The guesses for an instance's states."""
        guess = tuple(
            container.guess
            for container in self.__dict__.values()
            if is_state(container)
        )
        for attribute in self.__dict__.values():
            if isinstance(attribute, ModelObject):
                guess += attribute.state_guesses
        return guess

    @property
    def number_controls(self) -> int:
        """Number of controls an instance will yield."""
        return len(self.controls)

    @property
    def controls(self) -> tuple[DynamicSymbol]:
        """The controls (as dynamic symbols) that an instance yields."""
        controls = tuple(
            container.symbol
            for container in self.__dict__.values()
            if is_control(container)
        )
        for attribute in self.__dict__.values():
            if isinstance(attribute, ModelObject):
                controls += attribute.controls
        return controls

    @property
    def control_bounds(self) -> tuple[tuple[float, float]]:
        """The bounds on an instance's controls."""
        bounds = tuple(
            container.bounds
            for container in self.__dict__.values()
            if is_control(container)
        )
        for attribute in self.__dict__.values():
            if isinstance(attribute, ModelObject):
                bounds += attribute.control_bounds
        return bounds

    @property
    def control_guesses(self) -> tuple[float]:
        """The guesses for an instance's controls."""
        guess = tuple(
            container.guess
            for container in self.__dict__.values()
            if is_control(container)
        )
        for attribute in self.__dict__.values():
            if isinstance(attribute, ModelObject):
                guess += attribute.control_guesses
        return guess

    @property
    def number_static_parameters(self) -> int:
        """Number of static parameters an instance will yield."""
        return len(self.static_parameters)

    @property
    def static_parameters(self) -> tuple[sm.Symbol]:
        """The static parameters (as symbols) that an instance yields."""
        static_parameters = tuple(
            container.symbol
            for container in self.__dict__.values()
            if is_static_parameter(container)
        )
        for attribute in self.__dict__.values():
            if isinstance(attribute, ModelObject):
                static_parameters += attribute.static_parameters
        return static_parameters

    @property
    def static_parameter_bounds(self) -> tuple[tuple[float, float]]:
        """The bounds on an instance's static parameters."""
        bounds = tuple(
            container.bounds
            for container in self.__dict__.values()
            if is_static_parameter(container)
        )
        for attribute in self.__dict__.values():
            if isinstance(attribute, ModelObject):
                bounds += attribute.static_parameter_bounds
        return bounds

    @property
    def static_parameter_guesses(self) -> tuple[float]:
        """The guesses for an instance's static parameters."""
        guess = tuple(
            container.guess
            for container in self.__dict__.values()
            if is_static_parameter(container)
        )
        for attribute in self.__dict__.values():
            if isinstance(attribute, ModelObject):
                guess += attribute.static_parameter_guesses
        return guess

    @abstractmethod
    def __repr__(self) -> str:
        """Formatted representation of the model object."""
        pass
