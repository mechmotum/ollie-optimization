"""Components for modelling skateboards."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto, unique

import numpy as np
import sympy as sm
import sympy.physics.mechanics as me

from ollie.container import Container
from ollie.inertia import (
    inertia_of_cuboid,
    inertia_of_cylinder,
    inertia_of_isosceles_triangular_prism,
)
from ollie.material import Glue, Polyurethane, Maple, Steel
from ollie.model import ModelObject


class DeckBase(ABC, ModelObject):
    """Abstract base class from which all skateboard deck classes inherit.

    Attributes
    ==========

    width : ``Container``
        Width of the deck in m.
    ply : ``Container``
        Number of plies in the maple plywood that the deck is constructed from.
    veneer_thickness : ``Container``
        Thickness of a single plywood veneer in m.
    thickness : ``Container``
        Thickness of the plywood deck in m.
    area_density : ``Container``
        Mass per unit area of the deck plywood in kg/m^3. Note that in its
        calculation, 50% of glue evaporates while drying only between plies
        glue is applied.
    wheelbase : ``Container``
        Distance between the truck centers in m. Can be treated as a static
        parameter variable.
    length : ``Container``
        Length of the flat portion of the deck in m. Can be treated as a static
        parameter variable.
    origin : ``sympy.physics.mechanics.Point``
        Point representing the origin of the deck.
    frame : ``sympy.physics.mechanics.ReferenceFrame``
        Body-fixed reference frame for the deck.
    body : ``sympy.physics.mechanics.RigidBody``

    """

    def __init__(
        self,
        *,
        width: float = 0.21,
        ply: int = 7,
        veneer_thickness: float = 0.0016,
        wheelbase: float | None = None,
        length: float | None = None,
    ) -> None:
        """Initialize attributes generic to all decks."""

        # Constants
        self.width = Container(symbol=sm.Symbol(r"w_{deck}"), value=width)
        self.ply = Container(symbol=sm.Symbol(r"ply_{deck}"), value=ply)
        self.veneer_thickness = Container(
            symbol=sm.Symbol(r"t_{veneer}"), value=veneer_thickness
        )

        self.thickness = Container(
            symbol=sm.Symbol(r"t_{deck}"),
            value=self.ply * self.veneer_thickness,
        )
        self.area_density = Container(
            symbol=sm.Symbol(r"\rho_A_{deck}"),
            value=(
                self.thickness * Maple.density
                + (sm.S.Half * Glue.specific_mass * (self.ply - sm.S.One))
            ),
        )

        # Static parameter variables
        if wheelbase is not None:
            self.wheelbase = Container(symbol=sm.Symbol(r"wb_{deck}"), value=wheelbase)
        else:
            self.wheelbase = Container(
                symbol=sm.Symbol(r"wb_{deck}"), bounds=(0.05, 1), guess=0.444,
            )
        if length is not None:
            self.length = Container(r"l_{deck}", value=length)
        else:
            self.length = Container(
                symbol=sm.Symbol(r"l_{deck}"), bounds=(0.05, 1.5), guess=0.57,
            )

        # Mechanics
        self.origin = me.Point(r"O_{deck}")
        self.frame = me.ReferenceFrame(r"A_{deck}")
        self.mass_center = me.Point(r"C_{deck}")

        self._calculate_mass()
        self._calculate_inertia()

    @abstractmethod
    def _calculate_mass(self) -> None:
        """Calculate and instantiate any mass-related attributes."""
        pass

    @abstractmethod
    def _calculate_inertia(self) -> None:
        """Calculate and instantiate any inertia-related attributes."""
        pass

    def _construct_rigid_body(self) -> None:
        """Create a ``sympy.physics.mechanics.RigidBody`` for the deck."""
        self.body = me.RigidBody(
            name=r"B_{deck}",
            masscenter=self.mass_center,
            frame=self.frame,
            mass=self.mass,
            inertia=(self.inertia, ),
        )


class FlatDeck(DeckBase):
    """A flat rectangular skateboard deck without a traditional nose and tail."""

    def __init__(
        self,
        *,
        width: float = 0.21,
        ply: int = 7,
        veneer_thickness: float = 0.0016,
        wheelbase: float | None = None,
        length: float | None = None,
        tail_length: float | None = None,
        tail_inclination: float | None = None,
    ) -> None:
        """Initializer the segment deck instance."""
        super().__init__(
            width=width,
            ply=ply,
            veneer_thickness=veneer_thickness,
            wheelbase=wheelbase,
            length=length,
        )

    def _calculate_mass(self):
        """Calculate and instantiate the deck's mass-related attributes."""
        self.mass = Container(
            symbol=sm.Symbol(r"m_{deck}"),
            value=self.wheelbase * self.width * self.area_density,
        )
        self.mass_center.set_pos(self.origin, 0)

    def _calculate_inertia(self):
        """Calculate and instantiate the deck's inertia-related attributes."""
        self.inertia = Container(
            symbol=sm.Symbol('I_{deck}'),
            value=inertia_of_cuboid(
                self.frame,
                self.mass,
                x_dim=self.length,
                y_dim=self.thickness,
                z_dim=self.width,
            ),
        )

    def __repr__(self) -> str:
        """Formatted representation of the flat deck."""
        return (
            f"{self.__class__.__name__}(width={self.width}), ply={self.ply}, "
            f"veneer_thickness={self.veneer_thickness}, "
            f"wheelbase={self.wheelbase}, length={self.length})"
        )


class SegmentDeck(DeckBase):
    """A segmented skateboard deck with inclined semicircular nose and tail."""

    def __init__(
        self,
        *,
        width: float = 0.21,
        ply: int = 7,
        veneer_thickness: float = 0.0016,
        wheelbase: float | None = None,
        length: float | None = None,
        tail_length: float | None = None,
        tail_inclination: float | None = None,
    ) -> None:
        """Initializer the segment deck instance."""

        # Static parameter variables
        if tail_length is not None:
            self.tail_length = Container(
                symbol=sm.Symbol(r"l_{tail}"), value=tail_length,
            )
        else:
            self.tail_length = Container(
                symbol=sm.Symbol(r"l_{tail}"), bounds=[0.05, 0.3], guess=0.13,
            )
        if tail_inclination is not None:
            self.tail_inclination = Container(
                symbol=sm.Symbol(r"\phi_{tail}"),
                value=sm.Float(np.deg2rad(tail_inclination)),
            )
        else:
            self.tail_inclination = Container(
                symbol=sm.Symbol(r"\phi_{tail}"),
                bounds=[np.deg2rad(0), np.deg2rad(38)],
                guess=np.deg2rad(20),
            )

        super().__init__(
            width=width,
            ply=ply,
            veneer_thickness=veneer_thickness,
            wheelbase=wheelbase,
            length=length,
        )

        self.back_pocket = me.Point(r"P_{back_pocket}")
        self.back_pocket.set_pos(self.origin, -0.5 * self.length * self.frame.x)

        self.tail_frame = me.ReferenceFrame(r"A_{tail}")
        self.tail_frame.orient_axis(
            self.frame,
            self.frame.z,
            self.tail_inclination.symbol,
        )

    def _calculate_mass(self):
        """Calculate and instantiate the deck's mass-related attributes."""
        mass_tail_nose_semicircle = (
            sm.Rational(1, 8) * sm.pi * self.width**sm.Integer(2)
            * self.area_density
        )
        self.mass_tail_semicircle = Container(
            symbol=sm.Symbol(r"m_{tail_semi}"),
            value=mass_tail_nose_semicircle,
        )
        self.mass_nose_semicircle = Container(
            symbol=sm.Symbol(r"m_{nose_semi}"),
            value=mass_tail_nose_semicircle,
        )

        mass_tail_nose_rectangle = (
            (self.tail_length - sm.Rational(1, 2) * self.width) * self.width
            * self.area_density
        )
        self.mass_tail_rectangle = Container(
            symbol=sm.Symbol(r"m_{tail_rect}"),
            value=mass_tail_nose_rectangle,
        )
        self.mass_nose_rectangle = Container(
            symbol=sm.Symbol(r"m_{nose_rect}"),
            value=mass_tail_nose_rectangle,
        )

        self.mass_deck_rectangle = Container(
            symbol=sm.Symbol(r"m_{deck_rec}"),
            value=self.wheelbase * self.width * self.area_density,
        )
        self.mass = Container(
            symbol=sm.Symbol(r"m_{deck}"),
            value=(
                self.mass_tail_semicircle + self.mass_tail_rectangle
                + self.mass_deck_rectangle + self.mass_nose_rectangle
                + self.mass_nose_semicircle
            ),
        )

    def _calculate_inertia(self):
        """Calculate and instantiate the deck's inertia-related attributes."""
        self.inertia = Container(
            symbol=sm.Symbol(r"I_{deck}")
        )

    def __repr__(self) -> str:
        """Formatted representation of the segment deck."""
        return (
            f"{self.__class__.__name__}(width={self.width}), ply={self.ply}, "
            f"veneer_thickness={self.veneer_thickness}, "
            f"wheelbase={self.wheelbase}, length={self.length}, "
            f"tail_length={self.tail_length}, "
            f"tail_inclination={self.tail_inclination})"
        )


class Axle(ModelObject):
    """A skateboard truck's axle."""

    def __init__(self, *, diameter: float = 0.008, width: float = 0.21) -> None:
        """Initialize the axle instance."""

        # Constants
        self.diameter = Container(symbol=sm.Symbol(r"d_{axle}"), value=diameter)
        self.width = Container(symbol=sm.Symbol(r"w_{axle}"), value=width)

        # Mechanics
        self.origin = me.Point(r"O_{axle}")
        self.frame = me.ReferenceFrame(r"A_{axle}")
        self.mass_center = me.Point(r"C_{axle}")

        self._calculate_mass()
        self._calculate_inertia()

    def _calculate_mass(self) -> None:
        """Calculate and instantiate the axle's mass-related attributes."""
        self.mass = Container(
            symbol=sm.Symbol(r"m_{axle}"),
            value=(
                sm.pi
                * (sm.S.Half * self.diameter) ** sm.Integer(2)
                * self.width
                * Steel.density
            ),
        )
        self.mass_center.set_pos(self.origin, 0)

    def _calculate_inertia(self):
        """Calculate and instantiate the axle's inertia-related attributes."""
        self.inertia = Container(
            symbol=sm.Symbol(r"I_{axle}"),
            value=inertia_of_cylinder(
                self.frame,
                self.mass,
                axis=self.frame.z,
                radius=sm.Rational(1, 2) * self.diameter,
                length=self.width,
            ),
        )

    def __repr__(self) -> str:
        """Formatted representation of the axle."""
        return (
            f"{self.__class__.__name__}(diameter={self.diameter}, "
            f"width={self.width})"
        )


class Wheel(ModelObject):
    """A skateboard wheel."""

    def __init__(
        self,
        axle: Axle,
        *,
        width: float = 0.031,
        radius: float | None = None,
    ) -> None:
        """Initialize the wheel instance."""

        # Related components
        self.axle = axle

        # Constants
        self.width = Container(symbol=sm.Symbol(r"w_{wheel}"), value=width)

        # Static parameter variables
        if radius is not None:
            self.radius = Container(symbol=sm.Symbol(r"r_{wheel}"), value=radius)
        else:
            self.radius = Container(
                symbol=sm.Symbol(r"r_{wheel}"),
                bounds=(0.5 * self.axle.diameter.value + 0.001, 0.3),
                guess=0.0245,
            )

        # Mechanics
        self.origin = me.Point(r"O_{wheel}")
        self.frame = me.ReferenceFrame(r"A_{wheel}")
        self.mass_center = me.Point(r"C_{wheel}")

        self._calculate_mass()
        self._calculate_inertia()

    def _calculate_mass(self) -> None:
        """Calculate and instantiate the wheel's mass-related attributes."""
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
        self.mass_center.set_pos(self.origin, 0)

    def _calculate_inertia(self):
        """Calculate and instantiate the wheel's inertia-related attributes."""
        self.inertia = Container(
            symbol=sm.Symbol(r"I_{wheel}"),
            value=inertia_of_cylinder(
                self.frame,
                self.mass,
                axis=self.frame.z,
                radius=self.radius,
                length=self.width,
            ),
        )

    def __repr__(self) -> str:
        """Formatted representation of the wheel."""
        return (
            f"{self.__class__.__name__}({self.axle}, "
            f"width={self.width}, radius={self.radius})"
        )


class Truck(ModelObject):
    """A skateboard's truck."""

    def __init__(
        self,
        axle: Axle,
        wheels: Wheel,
        *,
        height: float | None = None
    ) -> None:
        """Initialize the truck instance."""

        # Related components
        self.axle = axle
        self.wheels = wheels

        # Static parameter variables
        if height is not None:
            self.height = Container(
                symbol=sm.Symbol(r"h_{truck}", value=height),
            )
        else:
            self.height = Container(
                symbol=sm.Symbol(r"h_{truck}"), bounds=[0.045, 0.3], guess=0.053,
            )

        # Mechanics
        self.origin = me.Point(r"O_{truck}")
        self.frame = me.ReferenceFrame(r"A_{truck}")
        self.mass_center = me.Point(r"C_{truck}")

        self._calculate_mass()
        self._calculate_inertia()

    def _calculate_mass(self) -> None:
        """Calculate and instantiate the truck's mass-related attributes."""
        self.mass = Container(
            symbol=sm.Symbol(r"m_{truck}"),
            value=(
                (sm.Float(0.366) - self.axle.mass) * self.height / sm.Float(0.053)
            ),  # truck mass scales linear with height compared to measured truck
        )

        mass_center_offset = -sm.Rational(1, 3) * self.height * self.frame.y
        self.mass_center.set_pos(self.origin, mass_center_offset)

    def _calculate_inertia(self):
        """Calculate and instantiate the truck's inertia-related attributes."""
        self.inertia = Container(
            symbol=sm.Symbol(r"I_{truck}"),
            value=inertia_of_isosceles_triangular_prism(
                self.frame,
                self.mass,
                axis=self.frame.z,
                base=sm.Float(0.053),
                height=self.height,
                length=self.axle.width,
            ),
        )

    def __repr__(self) -> str:
        """Formatted representation of the truck."""
        return (
            f"{self.__class__.__name__}({self.axle}, {self.wheel}"
            f"height={self.height})"
        )


@unique
class SegmentShape(Enum):
    """Simple shapes from which objects can be constructed."""

    Semicircle = auto()
    Cuboid = auto()
    Triangle = auto()
    Cylinder = auto()


class Skateboard(ModelObject):
    """A whole skateboard consisting of a deck, two trucks, and four wheels."""

    def __init__(self, *, deck: DeckBase, trucks: Truck):
        """Initialize the skateboard instance."""
        self.deck = deck
        self.trucks = trucks

        self.origin = me.Point(r"O_{skateboard}")
        self.frame = me.ReferenceFrame(r"A_{skateboard}")

        # self._set_point_positions()
        self._calculate_mass()
        # self._calculate_inertia()

    @property
    def axles(self):
        """Utility accessor to get the skateboard's truck's axles."""
        return self.trucks.axle

    @property
    def wheels(self):
        """Utility accessor to get the skateboard's wheels."""
        return self.trucks.wheels

    def _set_point_positions(self):
        raise NotImplementedError

    def _calculate_mass(self):
        """Calculate and instantiate the skateboard's mass-related attributes."""
        self.mass = Container(
            symbol=sm.Symbol(r"m_{skateboard}"),
            value=(
                self.deck.mass + 2*self.trucks.mass + 2*self.axles.mass
                + 4*self.wheels.mass
            ),
        )

    def _calculate_inertia(self):
        """Calculate and instantiate the skateboard's inertia-related attributes."""
        raise NotImplementedError

    def __repr__(self) -> str:
        """Formatted representation of the skateboard."""
        return f"{self.__class__.__name__}({self.deck}, {self.trucks})"
