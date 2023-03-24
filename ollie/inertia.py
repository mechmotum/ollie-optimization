"""Utilies for calculating inertias."""

import sympy as sm
import sympy.physics.mechanics as me

from ollie.container import Container


def inertia_of_cuboid(
    frame: me.ReferenceFrame,
    mass: Container,
    *,
    x_dim: Container,
    y_dim: Container,
    z_dim: Container,
) -> me.Dyadic:
    """Calculate the inertia of a cuboid."""
    two = sm.Integer(2)
    twelfth = sm.Rational(1, 12)
    return me.inertia(
        frame,
        twelfth * mass * (y_dim**two + z_dim**two),
        twelfth * mass * (z_dim**two + x_dim**two),
        twelfth * mass * (x_dim**two + y_dim**two),
    )


def inertia_of_cylinder(
    frame: me.ReferenceFrame,
    mass: Container,
    *,
    axis: me.Vector,
    radius: Container,
    length: Container,
) -> me.Dyadic:
    """Calculate the inertia of a cylinder."""
    if axis not in frame:
        msg = f"Axis {axis} must be a unit vector of frame {frame}."
        raise ValueError(msg)
    two = sm.Integer(2)
    three = sm.Integer(3)
    half = sm.Rational(1, 2)
    twelfth = sm.Rational(1, 12)
    parallel = half * mass * radius**two
    perpendicular = twelfth * mass * (height**two + three*radius**two)
    return me.inertia(
        frame,
        parallel if axis is frame.x else perpendicular,
        parallel if axis is frame.y else perpendicular,
        parallel if axis is frame.z else perpendicular,
    )
