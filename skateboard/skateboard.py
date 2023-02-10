from abc import ABC, abstractmethod
from enum import Enum, unique
from typing import List

import sympy as sm
import sympy.physics.mechanics as me


@unique
class SegmentShape(Enum):
    Semicircle
    Cuboid
    Triangle
    Cylinder


class SkateboardBase(ABC):

    @abstractmethod
    def _calculate_masses(self) -> List[sm.Expr]:
        pass

    @abstractmethod
    def _calculate_inertias(self) -> List[sm.Expr]:
        pass

    @abstractmethod
    def _calculate_center_of_mass(self) -> me.Point:
        pass


class FlatSkateboard(SkateboardBase):
    pass


class SegmentSkateboard(SkateboardBase):
    """"""

    def __init__(
        self,
        *,
        deck_width: float = 0.21,
        wheel_width: float = 0.031,
        ply: int = 7,
        pu_density: float = 1130.0,
        maple_density: float = 705.0,
        steel_density: float = 7700.0,
        glue_specific_mass: float = 0.210,
        axle_diameter: float = 0.008,
        veneer_thickness: float = 0.0016,
        global_frame: me.ReferenceFrame = me.ReferenceFrame('N'),
    ):
        """Initialize immutable skateboard properties."""

        # Constant variables
        self._deck_width = deck_width
        self._wheel_width = wheel_width
        self._ply =  ply
        self._pu_density = pu_density
        self._maple_density = maple_density
        self._steel_density = steel_density
        self._glue_specific_mass = glue_specific_mass
        self._axle_diameter = axle_diameter
        self._veneer_thickness = veneer_thickness

        # Optimization variables
        self.wheelbase = sm.Symbol('l_wb')
        self.tail_length = sm.Symbol('l_t')
        self.deck_length = sm.Symbol('l_d')
        self.tail_inclination = sm.Symbol('phi')
        self.truck_height = sm.Symbol('h_tr')
        self.wheel_radius = sm.Symbol('r_w')

        # State variables 
        self.backfoot_position = me.dynamicsymbols('s1')
        self.frontfoot_position = me.dynamicsymbol('s2')
        
        self.x_position = me.dynamicsymbols('x_s')
        self.y_position = me.dynamicsymbols('y_s')
        self.angle = me.dynamicsymbols('th_s')

        # Calculate values
        self.masses = self._calculate_masses
        self.inertias = self._calculate_inertias

        # Frames
        self.global_frame = global_frame
        self.frame = me.ReferenceFrame('A')
        self.frame.orient_axis(self.global_frame, self.global_frame.z, self.angle)

        # Unpack calculate values
        (
            self.mass_tail_round,
            self.mass_tail_rectangle,
            self.mass_deck,
            self.mass_nose_rectangle,
            self.mass_nose_round,
            self.mass_rear_truck,
            self.mass_front_truck,
            self.mass_rear_axle,
            self.mass_rear_wheels,
            self.mass_front_axle,
            self.mass_front_wheels,
        ) = self.masses
        (
            self.inertia_total,
            self.inertia_center_of_mass,
            self.inertia_steiner,
        ) = self.inertias

    @property
    def deck_width(self) -> float:
        """Width of the skateboard deck in cm."""
        return self._deck_width
       

    @property
    def wheel_width(self) -> float:
        """Width of the skateboard wheels in cm."""
        return self._wheel_width

    @property
    def ply(self) -> int:
        """Number of plies in the skateboard deck construction."""
        return self._ply

    @property
    def pu_density(self) -> float:
        """"""
        return self._pu_density

    @property
    def maple_density(self) -> float:
        """Density of maple wood in kg/m^3"""
        return self._maple_density

    @property
    def steel_density(self) -> float:
        """Density of steel in kg/m^3"""
        return self._steel_density

    @property
    def glue_specific_mass(self) -> float:
        """Mass of glue per unit area in kg/m^2."""
        return self._glue_specific_mass

    @property
    def axle_diameter(self) -> float:
        """Diameter of the skateboard axles in cm."""
        return self._axle_diameter

    @property
    def veneer_thickness(self) -> float:
        """Thickness of each veneer in cm."""
        return self._veneer_thickness

    def _calculate_masses(self) -> List[sm.Expr]:
        """Calculate masses of each individual component of the segmented skateboard.

        The partitioning of the skateboarding is done like this and have indices
        in the list of expressions:

           _   _   ___________   _   _
         /  | | | |           | | | |  \
        | 0 | |1| |     2     | |3| | 4 | 
         \ _| |_| |___________| |_| |_ /

        0\                  /4
         1\_______2________/3
           5 \/         \/ 6
           7 O 8      9 O 10

        Components
        ==========

        0: tail round
        1: tail rectangle
        2: deck
        3: nose rectangle
        4: nose round
        5: rear truck
        6: front truck
        7: rear axle
        8: rear wheels
        9: front axle
        10: front wheels

        """
        deck_thickness = self.ply * self.veneer_tickness
        # 50% of glue evaporates while drying only between plies glue is applied
        mass_per_area = deck_thickness * self.maple_density + (self.glue_specific_mass/2*(n_ply-2))

        # Area of wooden components
        mass_tail_round = (1/2) * (1/4) * sm.pi * self.deck_width**2 *mass_per_area  # 1/4 pi d^2
        mass_tail_rectangle = (self.tail_length - (self.deck_width/2)) * self.deck_width * mass_per_area
        mass_deck = self.deck_length * self.deck_width *mass_per_area
        mass_nose_rectangle = mass_tail_rectangle
        mass_nose_round = mass_tail_round
        axle_mass = sm.pi * (self.axle_diameter/2)**2 * self.deck_width * self.density_steel
        mass_truck = (0.366 - axle_mass) * self.truck_height/0.053  #truck mass scales linear with height compared to measured truck
        wheel_mass = self.pu_density * sm.pi * self.wheel_width * ((2*self.wheel_radius)**2 - self.axle_diameter**2) / 4  # V=pi*h*(D^2-d^2)/4
        
        return [mass_tail_round, mass_tail_rectangle, mass_deck, mass_nose_rectangle, mass_nose_round, mass_truck, mass_truck, axle_mass, 2*wheel_mass, axle_mass, 2*wheel_mass]

    def _calculate_inertias(self, major_dimensions: List[List], com_points: List[me.Point]) -> me.Point:
        """"""
        I_com = []
        I_steiner = []

        for i, (mass, major_dimension, com_point) in enumerate(zip(self.masses, major_dimensions. com_points)):
            if shape[i] == SegmentShape.Semicircle:
                I_com.append(((1/4) - (16/(9 * sm.pi**2))) * mass * (major_dimensions/2)**2)
            elif shape[i] == SegmentShape.Cuboid:
                I_com.append((mass/12) * (major_dimension**2 + major_dimensions[1]**2))
            elif shape[i] == SegmentShape.Triangle:
                #I_com.append(0)
                s = sm.sqrt((major_dimension/2)**2+major_dimensions[1]**2)
                beta = 2*sm.asin((major_dimension/2)/s)
                I_com.append((mass/2) * s**2 * (1 - (2/3) * sm.sin(beta)))
            elif shape[i] == SegmentShape.Cylinder:
                I_com.append((1/2) * mass * (major_dimensions/2)**2)

        I_steiner.append(sm.trigsimp(mass*sm.sqrt(com_points.pos_from(com).dot(A.x)**2+com_points.pos_from(com).dot(A.y)**2)**2))
        I_tot = sum(I_com)+sum(I_steiner)
        return I_tot, I_com, I_steiner

    # def _calculate_center_of_mass(self) -> me.Point:
    #     """"""

    #    mass, com_points, reference_point):
    # # Function gets vector position from reference_point
    # # Then assigns COM location to point COM
    # # Returns position vector of COM points relative to COM
    # r_m = []
    # for i, x in enumerate(com_points):
    #     r_m.append(x.pos_from(reference_point)*mass[i])
    # COM_skateboard = me.Point('COM_skateboard')
    # COM_skateboard.set_pos(reference_point, (sum(r_m)/sum(mass)))

    # return COM_skateboard
    #     return None

    def _construct_skateboard(self):
        """
        Constructs a skateboard with horizontal position x, vertical position y, and angle with respect to the inertial frame.
        """
        #%% Free floating acc
        origin              = me.Point('O')
        com                 = origin.locatenew('centre of mass', self.x_position * self.global_frame.x+self.y_position*self.global_frame.y)
        middle_deck         = com.locatenew('middle deck', d_com_ * self.frame.y)
        back_truck          = middle_deck.locatenew('back trucks at deck', -self.wheelbase/2 * self.frame.x)
        front_truck         = back_truck.locatenew('front trucks at deck', self.wheelbase * self.frame.x)
        back_wheel          = back_truck.locatenew('back wheels', -self.truck_height * self.frame.y)
        back_wheel_contact  = back_wheel.locatenew('contact back wheels', -self.wheel_radius * self.global_frame.y)
        front_wheel         = front_truck.locatenew('front wheels', -self.truck_height * self.frame.y)
        front_wheel_contact = front_wheel.locatenew('contact back wheels', -self.wheel_radius * self.global_frame.y)
        back_pocket         = middle_deck.locatenew('back pocket', -self.deck_length/2 * self.frame.x)
        tail                = back_pocket.locatenew('tail', -self.tail_length * B.x)
        back_foot           = tail.locatenew('back foot', self.backfoot_position * B.x)
        front_foot          = back_pocket.locatenew('front foot', self.frontfoot_position * self.frame.x)
        front_pocket        = back_pocket.locatenew('front pocket', self.deck_length * self.frame.x)
        nose                = front_pocket.locatenew('nose', self.tail_length * C.x)
        objective           = back_wheel_contact.locatenew('objective', self.wheelbase/2 * self.frame.x)


    def TMT_method(self, com_coordinates, mass_matrix, generalized_coordinates, com_applied_loads):
        """
        ROUGH, NOT DONE YET
        """
        #Freefloating EOM
        q = generalized_coordinates
        dq = q.diff(me.dynamicsymbols(_t))
        ddq= dq_b.diff(me.dynamicsymbols(_t))

        positions = com_coordinates

        dx = positions.diff(me.dynamicsymbols(_t))
        Tij = dx.jacobian(dq)
        Tji = Tij.transpose()
        ddx = dx.diff(me.dynamicsymbols(_t))
        gk  = dx.jacobian(q)*dq

        TMT = Tji*mass_matrix*Tij
        Fi = sm.Matrix([com_applied_loads.dot(self.global_frame.x), com_applied_loads.dot(self.global_frame.y)-m_s_*g,(bf_b.pos_from(com_b).cross(F1)+ff_b.pos_from(com_b).cross(F2)).dot(N.z)])
        Fg = Tji*(Fi-Mij*gk)

        #(Fi_b == Fg_b)

        ddq_eq_b = d2s(TMT_b.LUsolve(Fg_b))

        accSB = {ddx_s: ddq_eq_b[0],
                ddy_s: ddq_eq_b[1],
                ddth_s:ddq_eq_b[2],
                }


class SoftplusSkateboard(SkateboardBase):
    pass
