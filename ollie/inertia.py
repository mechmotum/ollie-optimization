"""Utilies for calculating inertias."""

import sympy.physics.mechanics as me

from ollie.container import Container


def inertia_of_cuboid(
    frame: me.ReferenceFrame,
    mass: Container,
    x_dim: Container,
    y_dim: Container,
    z_dim: Container,
) -> :
    """Calculate the inertia of a cuboid."""
    two = Integer(2)
    twelfth = sm.Rational(1, 12)
    return me.inertia(
        frame,
        twelfth * mass * (y_dim**two + z_dim**two),
        twelfth * mass * (z_dim**two + x_dim**two),
        twelfth * mass * (x_dim**two + y_dim**two),
    )
