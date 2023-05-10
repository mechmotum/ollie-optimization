"""Utilies for calculating inertias."""

import sympy as sm
import sympy.physics.mechanics as me

from ollie.container import Container


def parallel_axis_theorem(
    dyadic: me.Dyadic,
    mass: Container,
    frame: me.ReferenceFrame,
    vector: me.Vector,
) -> me.Dyadic:
    """Calculate a new moment of inertia using the parallel axis theorem."""
    dyadic = dyadic.express(frame).to_matrix(frame)
    Ixx = dyadic[0, 0]
    Iyy = dyadic[1, 1]
    Izz = dyadic[2, 2]
    two = sm.Integer(2)
    return me.inertia(
        frame,
        Ixx + mass * vector.dot(frame.x)**two,
        Iyy + mass * vector.dot(frame.y)**two,
        Izz + mass * vector.dot(frame.z)**two,
    )


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
    perpendicular = twelfth * mass * (length**two + three*radius**two)
    return me.inertia(
        frame,
        parallel if axis is frame.x else perpendicular,
        parallel if axis is frame.y else perpendicular,
        parallel if axis is frame.z else perpendicular,
    )


def inertia_of_isosceles_triangular_prism(
    frame: me.ReferenceFrame,
    mass: Container,
    *,
    axis: me.Vector,
    base: Container,
    height: Container,
    length: Container,
) -> me.Dyadic:
    """Calculate the inertia of an isosceles triangular prism."""
    if axis not in frame:
        msg = f"Axis {axis} must be a unit vector of frame {frame}."
        raise ValueError(msg)

    two = sm.Integer(2)
    thirty_sixth = sm.Rational(1, 36)
    parallel = thirty_sixth * mass * (base**two + height**two)
    perpendicular = 0  # Assume zero as not important for ollie calculations
    return me.inertia(
        frame,
        parallel if axis is frame.x else perpendicular,
        parallel if axis is frame.y else perpendicular,
        parallel if axis is frame.z else perpendicular,
    )

