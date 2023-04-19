"""Components for modelling connections between other components."""

from abc import ABC, abstractmethod

import sympy as sm
import sympy.physics.mechanics as me

from ollie.container import Container
from ollie.model import ModelObject


INF = 1e5
EPS = 1e-1


class ConnectionBase(ABC, ModelObject):
    """Abstract base class from which all connection classes inherit."""

    @abstractmethod
    def __repr__(self) -> str:
        """Formatted representation of the connection object."""
        pass


class Friction(ConnectionBase):
    """Simple friction model to connect a human and deck via the grip tape."""

    def __init__(
        self,
        name: str,
        *,
        frame: me.ReferenceFrame,
        coefficient: float = 0.8,
    ) -> None:
        """Initialize the friction instance.

        Explanation
        ===========

        There are six path constraints involving slack variables to ensure that
        the friction equations are applied correctly:

        #1: Makes sure friction is anywhere within the positive or negative
            bound of friction relative to the COF and N
        #2: Create absolute value of relative foot sliding velocity
        #3: Create absolute value of relative foot sliding velocity
        #4: When foot is sliding (i.e. absolute_... is not 0),
            mu*f_normal = slack_positive + slack_negative
        #5: When sliding negative, slack_positive = 0, together with rule #3,
            this will ensure that friction is "maximal".
        #6: When sliding positive, slack_negative = 0, together with rule #3,
            this will ensure that friction is "maximal".

        """

        # Constants
        self.coefficient = Container(
            symbol=sm.Symbol(r"mu_{grip}"),
            value=coefficient,
        )

        # Mechanics
        self.frame = frame
        self.point = me.Point(f"P_{{{name}}}")

        # Dynamics (state and control) variables
        self.acceleration = Container(
            symbol=me.dynamicsymbols(f"acc_{{{name}}}"),
            bounds=(-10.0, 10.0),
            guess=0.0,
        )
        self.velocity = Container(
            symbol=me.dynamicsymbols(f"vel_{{{name}}}"),
            state_equation=self.acceleration,
            bounds=(-50.0, 50.0),
            guess=0.0,
        )
        self.position = Container(
            symbol=me.dynamicsymbols(f"pos_{{{name}}}"),
            state_equation=self.velocity,
            bounds=(0.0, 1.0),
            guess=0.5,
        )

        # Force (control) variables
        self.normal_force = Container(
            symbol=me.dynamicsymbols(f"R_{{{name}}}"),
            bounds=(0.0, 1.0),
            guess=0.0,
        )
        self.friction_force = Container(
            symbol=me.dynamicsymbols(f"F_{{{name}}}"),
            bounds=(0.0, 1.0),
            guess=0.0,
        )

        # Slack (control) variables
        self.slack_positive = Container(
            symbol=me.dynamicsymbols(f"s_{{{name}+}}"),
            bounds=(0.0, INF),
            guess=0.0,
        )
        self.slack_negative = Container(
            symbol=me.dynamicsymbols(f"s_{{{name}-}}"),
            bounds=(-INF, 0.0),
            guess=0.0,
        )

    @property
    def path_constraints(self) -> tuple[sm.Expr]:
        """The path constraints (as expressions) that an instance requires."""
        bounded_friction_force = (
            self.coefficient * self.normal_force
            - self.slack_positive - self.slack_negative
        )
        path_constraints = (
            bounded_friction_force,
            self.absolute_velocity + self.velocity,
            self.absolute_velocity - self.velocity,
            bounded_friction_force * self.absolute_velocity,
            (self.absolute_velocity + self.velocity) * self.slack_positive,
            (self.absolute_velocity - self.velocity) * self.slack_negative,
        )
        return path_constraints

    @property
    def path_constraint_bounds(self) -> tuple[tuple[float, float]]:
        """The bounds on an instance's path constraints."""
        bounds = (
            (0, INF),
            (0, INF),
            (0, INF),
            (-EPS, EPS),
            (-EPS, EPS),
            (-EPS, EPS),
        )
        return bounds

    def __repr__(self) -> str:
        """Formatted representation of the connection object."""
        return (
            f"{self.__class__.__name__}(name={self.name}, frame={self.frame}, "
            f"coefficient={self.coefficient})"
        )
