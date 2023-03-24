""""""

import pycollo

from ollie import Axle, Wheel, Truck, FlatDeck, Skateboard, ForceHuman, Ollie


deck = FlatDeck()
axle = Axle()
wheel = Wheel(axle=axle)
truck = Truck(axle=axle, wheels=wheel)
skateboard = Skateboard(deck=deck, trucks=truck)

human = ForceHuman()
ollie = Ollie(
    skateboard=skateboard,
    human=human,
    phases=[
        "preparation",  # both wheels contact ground
        "pre-pop",  # NEED PHASE: only rear wheel contact ground (no front wheel normal force)
        "pop",  # NEED BOUNDARY: tail contacting the ground
        "upward motion",  # NEED PHASE: flight
        "downward motion",  # NEED PHASE: flight
        "landing",  # NEED BOUNDARY: wheel touching ground
    ],
)
# print(ollie.phases)

# ocp = pycollo.OptimalControlProblem(name="Maximum height ollie")

# for ollie_phase in ollie.phases:
#     phase = ocp.new_phase(name=ollie_phase.name)
#     phase.state_variables = ollie_phase.state_variables
#     phase.control_variables = ollie_phase.control_variables
