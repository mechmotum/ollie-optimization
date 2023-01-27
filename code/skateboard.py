# in skateboard.py

from abc import ABC, abstractmethod
from functools import cached_property

class SkateboardBase(ABC):

    @abstractmethod
    def mass(self) -> list[sm.Expr]:
    def inertia(self) ->list[sm.Expr]:
    def centre_of_mass(self) -> me.Point:
        pass


class FlatSkateboard(SkateboardBase):
    pass


class SegmentSkateboard(SkateboardBase):
    def __init__(self, *, 
                deck_width:         float = 0.21, 
                wheel_width:        float = 0.031,
                ply:                float = 7.0, 
                pu_density:         float = 1130.0, 
                maple_density:      float = 705.0, 
                steel_density:      float = 7700.0, 
                glue_specificmass:  float = 0.210, 
                axle_diameter:      float = 0.008, 
                veneer_thickness:   float = 0.0016
                ):

        # Constant variables
        self._deck_width = deck_width
        self.wheel_width = wheel_width
        self.ply =  ply           
        self.pu_density = pu_density          
        self.maple_density = maple_density          
        self.steel_density = steel_density     
        self.glue_specificmass = glue_specificmass         
        self.axle_diameter = axle_diameter          
        self.veneer_thickness = veneer_thickness       

        wheelbase, tail_length, deck_length, tail_inclination, truck_height, wheel_radius = sm.symbols('l_wb, l_t, l_d, phi, h_tr, r_w')
        # Optimization variables
        self.wheelbase = wheelbase
        self.tail_length = tail_length
        self.deck_length = deck_length
        self.tail_inclination = tail_inclination
        self.truck_height = truck_height
        self.wheel_radius = wheel_radius
        self.I_tot, self.I_com, self.I_steiner = self.inertia
    @property
    def deck_width(self) -> float:
        return self._deck_width
       
    # Do this for all constant variables to make them immutable
        

    @cached_property # No parenthesis / input for this type   # Use self.variable to acces them. 
    def mass(self) -> sm.Expr:
        """
        Creates self.mass with a list of sympy expressions of the mass of each individual component.

        The partitioning of the skateboarding is done like this and have indices in the list of expressions:

           _   _   ___________   _   _
         /  | | | |           | | | |  \
        | 0 | |1| |     2     | |3| | 4 | 
         \ _| |_| |___________| |_| |_ /

        0\                  /4
         1\_______2________/3
           5 \/         \/ 6
           7 O 8      9 O 10

        """
        deck_thickness = self.ply * self.veneer_tickness
        #50% of glue evaporates while dryinh, only between plies glue is applied
        mass_per_area = deck_thickness * self.maple_density + (self.glue_specificmass/2*(n_ply-2)) 

        # Area of wooden components
        mass_tail_round = (1/2) * (1/4) * sm.pi * self.deck_width**2 *mass_per_area # 1/4 pi d^2
        mass_tail_rectangle = (self.tail_length - (self.deck_width/2)) * self.deck_width * mass_per_area
        mass_deck = self.deck_length * self.deck_width *mass_per_area
        mass_nose_rectangle = mass_tail_rectangle
        mass_nose_round = mass_tail_round
        axle_mass = sm.pi * (self.axle_diameter/2)**2 * self.deck_width * self.density_steel
        mass_truck = (0.366- axle_mass) * self.truck_height/0.053 #truck mass scales linear with height compared to measured truck
        wheel_mass = self.pu_density * sm.pi * self.wheel_width * ((2*self.wheel_radius)**2 - self.axle_diameter**2) / 4  # V=pi*h*(D^2-d^2)/4
        
        return [mass_tail_round, mass_tail_rectangle, mass_deck, mass_nose_rectangle, mass_nose_round, mass_truck, mass_truck, axle_mass, 2*wheel_mass,axle_mass, 2*wheel_mass]

    @cached_property 
    def inertia(self,major_dimensions: list[list], com_points: list[me.Point]) -> me.Point: 
        """ """
        mass = self.mass
        I_com = []
        I_steiner = []

        for i in range(len(mass)):
            if shape[i] == 'semicircle':
                I_com.append(((1/4)-(16/(9*sm.pi**2)))*mass[i]*(major_dimensions[i]/2)**2)

            if shape[i] == 'cuboid':
                I_com.append((mass[i]/12) * (major_dimensions[i]
                            [0]**2 + major_dimensions[i][1]**2))

            if shape[i] == 'triangle':
                #I_com.append(0)
                s = sm.sqrt((major_dimensions[i][0]/2)**2+major_dimensions[i][1]**2)
                beta = 2*sm.asin((major_dimensions[i][0]/2)/s)
                I_com.append((mass[i]/2)*s**2*(1-(2/3)*sm.sin(beta)))

            if shape[i] == 'cylinder':
                I_com.append((1/2)*mass[i]*(major_dimensions[i]/2)**2)

        I_steiner.append(sm.trigsimp(mass[i]*sm.sqrt(com_points[i].pos_from(\
                com).dot(A.x)**2+com_points[i].pos_from(com).dot(A.y)**2)**2))


        I_tot = sum(I_com)+sum(I_steiner)
        return I_tot, I_com, I_steiner




class SoftplusSkateboard(SkateboardBase):
    pass

"""

"""
# multiphase_ollie_optimization.py

from skateboard import Skateboard

