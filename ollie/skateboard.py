""""""

from abc import ABC, abstractmethod
from enum import Enum, auto, unique

import numpy as np
import sympy as sm
import sympy.physics.mechanics as me

from ollie.container import Container
from ollie.material import Glue, Polyurethane, Maple, Steel


class DeckBase(ABC):
    """Abstract base class from which all skateboard deck classes inherit."""

    def __init__(
        self,
        *,
        width: float = 0.21,
        ply: int = 7,
        veneer_thickness: float = 0.0016,
    ) -> None:
        """"""

        # Constants
        self.width = Container(symbol=sm.Symbol(r"w_{deck}"), value=width)
        self.ply = Container(symbol=sm.Symbol(r"ply_{deck}"), value=ply)
        self.veneer_thickness = Container(
            symbol=sm.Symbol(r"t_{veneer}"), value=veneer_thickness
        )

        # Static parameter variables
        self.wheelbase = Container(
            symbol=sm.Symbol(r"wb_{deck}"), bounds=[0.05, 1], guess=0.444
        )
        self.length = Container(
            symbol=sm.Symbol(r"l_{deck}"), bounds=[0.05, 1.5], guess=0.83
        )

        # Mechanics
        self.origin = me.Point(r"O_{deck}")
        self.frame = me.ReferenceFrame(r"A_{deck}")

        # Calculate mass properties
        # self._calculate_mass()
        # self._calculate_inertia()
        # self._calculate_mass_center()

    @abstractmethod
    def _calculate_mass(self) -> None:
        pass

    @abstractmethod
    def _calculate_inertia(self) -> None:
        pass

    @abstractmethod
    def _calculate_mass_center(self) -> None:
        pass

    def _construct_rigid_body(self) -> None:
        """"""
        self.body = me.RigidBody(
            name=r"B_{deck}",
            masscenter=self.mass_center,
            frame=self.frame,
            mass=self.mass,
            inertia=(self.inertia, ),
        )


class FlatRectangularDeck(DeckBase):
    """"""

    def _calculate_mass(self) -> None:
        """"""
        self.thickness = Container(
            symbol=sm.Symbol(r"t_{deck}"),
            value=self.ply * self.veneer_thickness,
        )
        # 50% of glue evaporates while drying only between plies glue is applied
        self.area_density = Container(
            symbol=sm.Symbol(r"\rho_A_{deck}"),
            value=(
                self.thickness * Maple.density
                + (sm.S.Half * Glue.specific_mass * (self.ply - sm.S.One))
            ),
        )
        self.mass = Container(
            symbol=sm.Symbol(r"m_{deck}"),
            value=(self.length * self.width * self.area_density),
        )

    def _calculate_inertia(self) -> None:
        """

        Steiner == Parallel axis theorem

        """
        self.inertia = Container(
            symbol=sm.Symbol(r"I_{deck}"),
            value=(
                sm.Rational(1, 12) * self.mass
                * (self.length**sm.Integer(2) + self.thickness**sm.Integer(2))
            )
        )

    def _calculate_mass_center(self) -> None:
        """"""
        self.mass_center = me.Point(r"P_{deck}")
        self.mass_center.set_pos(self.origin, sm.S.Zero)


class SegmentDeck(DeckBase):
    """"""

    def __init__(self):
        """"""
        super().__init__()

        # Static parameter variables
        self.tail_length = Container(
            symbol=sm.Symbol(r"l_{tail}"), bounds=[0.05, 0.3], guess=0.13
        )
        self.tail_inclination = Container(
            symbol=sm.Symbol(r"\phi_{tail}"),
            bounds=[np.deg2rad(0), np.deg2rad(38)],
            guess=np.deg2rad(20),
        )

        self.back_pocket = me.Point(r"P_{bp}")
        self.back_pocket.set_pos(self.origin, -0.5 * self.length * self.frame.x)

        self.tail_frame = me.ReferenceFrame(r"A_{tail}")
        self.tail_frame.orient_axis(self.frame, self.frame.z, self.tail_inclination.symbol)

    def _calculate_mass(self) -> None:
        raise NotImplementedError

    def _calculate_inertia(self) -> None:
        raise NotImplementedError

    def _calculate_mass_center(self) -> None:
        raise NotImplementedError


class Axle:
    """"""

    def __init__(self, *, diameter: float = 0.008, width: float = 0.21) -> None:
        """"""

        # Constants
        self.diameter = Container(symbol=sm.Symbol(r"d_{axle}"), value=diameter)
        self.width = Container(symbol=sm.Symbol(r"w_{axle}"), value=width)

        self.origin = me.Point(r"O_{axle}")
        self.frame = me.ReferenceFrame(r"N_{axle}")

        # self._calculate_mass()
        # self._calculate_inertia()
        # self._calculate_mass_center()

    def _calculate_mass(self) -> None:
        """"""
        self.mass = Container(
            symbol=sm.Symbol(r"m_{axle}"),
            value=(
                sm.pi
                * (sm.S.Half * self.diameter) ** sm.Integer(2)
                * self.width
                * Steel.density
            ),
        )

    def _calculate_inertia(self) -> None:
        raise NotImplementedError

    def _calculate_mass_center(self) -> None:
        raise NotImplementedError


class Wheel:
    """"""

    def __init__(self, axle: Axle, *, width: float = 0.031) -> None:
        """"""

        # Related components
        self.axle = axle

        # Constants
        self.width = Container(symbol=sm.Symbol("w_{wheel}"), value=width)

        # Static parameter variables
        self.radius = Container(
            symbol=sm.Symbol(r"r_{wheel}"), bounds=[0.5 * self.axle.diameter.value + 0.001, 0.3], guess=0.0245
        )

        # self._calculate_mass()
        # self._calculate_inertia()
        # self._calculate_mass_center()

    def _calculate_mass(self) -> None:
        self.mass = Container(
            symbol=sm.Symbol(r"m_{wheel}"),
            value=(
                sm.Rational(1, 4)
                * sm.pi
                * Polyurethane.density
                * self.width
                * (
                    (sm.Integer(2) * self.radius) ** sm.Integer(2)
                    - self.axle.diameter ** sm.Integer(2)
                )
            ),  # V=pi*h*(D^2-d^2)/4
        )

    def _calculate_inertia(self) -> None:
        raise NotImplementedError

    def _calculate_mass_center(self) -> None:
        raise NotImplementedError


class Truck:
    """"""

    def __init__(self, axle: Axle, wheels: Wheel) -> None:
        """"""

        # Related components
        self.axle = Axle
        self.wheels = Wheel

        # Static parameter variables
        self.height = Container(
            symbol=sm.Symbol(r"h_{truck}"), bounds=[0.045, 0.3], guess=0.053
        )

        # self._calculate_mass()
        # self._calculate_inertia()
        # self._calculate_mass_center()

    def _calculate_mass(self) -> None:
        """"""
        self.mass = Container(
            symbol=sm.Symbol(r"m_{truck}"),
            value=(
                (sm.Float(0.366) - self.axle.mass) * self.height / sm.Float(0.053)
            ),  # truck mass scales linear with height compared to measured truck
        )

    def _calculate_inertia(self) -> None:
        raise NotImplementedError

    def _calculate_mass_center(self) -> None:
        raise NotImplementedError


@unique
class SegmentShape(Enum):
    Semicircle = auto()
    Cuboid = auto()
    Triangle = auto()
    Cylinder = auto()


class Skateboard:
    def __init__(self, *, deck: DeckBase, trucks: Truck):
        """"""
        self.deck = deck
        self.trucks = trucks

        self.origin = me.Point(r"O_{skateboard}")

        # self._set_point_positions()
        # self._calculate_mass()
        # self._calculate_inertia()
        # self._calculate_mass_center()

    @property
    def wheels(self):
        return self.trucks.wheels

    def _set_point_positions(self):
        raise NotImplementedError

    def _calculate_mass(self):
        raise NotImplementedError

    def _calculate_inertia(self):
        raise NotImplementedError

    def _calculate_mass_center(self):
        raise NotImplementedError
