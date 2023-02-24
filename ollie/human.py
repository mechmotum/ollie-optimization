""""""

from abc import ABC

import sympy as sm
import sympy.physics.mechanics as me

from ollie.container import Container


class HumanBase(ABC):

    def __init__(self, *, height: float = 1.8) -> None:

        # Constants
        self.height = Container(symbol=sm.Symbol(r"h_{human}"), value=height)

        # Mechanics
        self.origin = me.Point(r"O_{human}")
        self.frame = me.ReferenceFrame(r"A_{human}")

        self.mass_center = me.Point(r"P_{human}")

        self.pos_mass_center_x = Container(
            symbol=me.dynamicsymbols("x"),
            bounds=(-0.3, 1.0),
            guess=0.0
        )
        self.pos_mass_center_y = Container(
            symbol=me.dynamicsymbols("y"),
            bounds=(0.0, 1.0),
            guess=0.5,
        )

        self.feet_frame = me.ReferenceFrame(r"A_{feet}")

        self.rear_foot = me.Point(r"P_{rf}")
        self.front_foot = me.Point(r"P_{ff}")

        self.pos_rear_foot = Container(
            symbol=me.dynamicsymbols(r"q_{rf}"),
            bounds=[0.0, 1.0],
            guess=0.5,
        )
        self.pos_front_foot = Container(
            symbol=me.dynamicsymbols(r"q_{ff}"),
            bounds=[0.0, 1.0],
            guess=0.5,
        )


class ForceHuman(HumanBase):
    pass


class PointMassHuman(HumanBase):
    pass
