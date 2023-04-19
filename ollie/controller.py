"""Human controllers that ride and ollie a skateboard."""

from abc import ABC

import sympy as sm
import sympy.physics.mechanics as me

from ollie.constant import GRAVITY
from ollie.container import Container
from ollie.model import ModelObject


class ControllerBase(ABC, ModelObject):
    """Abstract base class from which all controller classes inherit."""
    pass


class PointMassHuman(ControllerBase):
    """Simple human controller consisting of a floating point mass."""

    def __init__(self, *, height: float = 1.8, mass: float = 80) -> None:

        # Constants
        self.height = Container(symbol=sm.Symbol(r"h_{human}"), value=height)
        self.mass = Container(symbol=sm.Symbol(r"m_{human}"), value=mass)

        # Mechanics
        self.origin = me.Point(r"O_{human}")
        self.frame = me.ReferenceFrame(r"A_{human}")
        self.mass_center = me.Point(r"P_{human}")

        # reach_min = 0.466 / 1.80 * height
        # reach_max = 1.130 / 1.80 * height
        # # State variable
        # self.length_rear_leg = Container(
        #     symbol=me.dynamicsymbols(r"l_{rear\_leg}"),
        #     state_equation=self.lengthening_rear_leg,
        #     bounds=(reach_min, reach_max),
        #     guess=0.5 * self.height,
        # )
        # # State variable
        # self.length_front_leg = Container(
        #     symbol=me.dynamicsymbols(r"l_{front\_leg}"),
        #     bounds=(reach_min, reach_max),
        #     guess=0.5 * self.height,
        # )

        # State variables
        # force_mass_center = (
        #     self.force_rear_foot * skateboard.frame.y
        #     + self.force_front_foot * skateboard.frame.y
        #     - self.force_rear_foot * skateboard.frame.x
        #     + self.force_front_foot * skateboard.frame.x
        #     - self.mass * GRAVITY * self.frame.y
        # )
        # self.vel_mass_center_x = Container(
        #     symbol=me.dynamicsymbols(r"dx_{human}"),
        #     state_equation=(mass_center_force.dot(frame.x) / self.mass),
        #     bounds=(-0.5, 0.5),
        #     guess=0.0,
        # )
        # self.vel_mass_center_y = Container(
        #     symbol=me.dynamicsymbols(r"dy_{human}"),
        #     state_equation=(mass_center_force.dot(frame.y) / self.mass),
        #     bounds=(0, 5.0),
        #     guess=1.0,
        # )
        # self.pos_mass_center_x = Container(
        #     symbol=me.dynamicsymbols(r"x_{human}"),
        #     state_equation=self.vel_mass_center_x,
        #     bounds=(-0.5, 0.5),
        #     guess=0.0,
        # )
        # self.pos_mass_center_x = Container(
        #     symbol=me.dynamicsymbols(r"y_{human}"),
        #     state_equation=self.vel_mass_center_y,
        #     bounds=(0, 5.0),
        #     guess=1.0,
        # )
