"""Maximum height ollie optimization using a simple flat skateboard.

Solving this problem involves find the optimal control of the forces applied by
the front and rear foot on the deck as well as the following optimal skateboard
parameters:
- wheelbase
- board length
- wheel diameter
- truck height

"""

import pycollo
from ollie import Axle, Wheel, Truck, FlatDeck, Skateboard, PointMassHuman, Ollie


deck = FlatDeck()
axle = Axle()
wheel = Wheel(axle=axle)
truck = Truck(axle=axle, wheels=wheel)
skateboard = Skateboard(deck=deck, trucks=truck)

human = PointMassHuman()
ollie = Ollie(
    skateboard=skateboard,
    controller=human,
    phases=[
        "preparation",  # both wheels contact ground <BOUNDARY>
        "pre-pop",  # NEED PHASE: only rear wheel contact ground (no front wheel normal force) <PHASE>
        "pop",  # NEED BOUNDARY: tail contacting the ground <BOUNDARY>
        "upward motion",  # NEED PHASE: flight <PHASE>
        "peak height",  # maximum ollie height <BOUNDARY>
        "downward motion",  # NEED PHASE: flight <PHASE>
        "landing",  # NEED BOUNDARY: wheel touching ground <BOUNDARY>
    ],
)

ocp = pycollo.OptimalControlProblem(name="Maximum height ollie")

for ollie_phase in ollie.phases:
    phase = ocp.new_phase(name=ollie_phase.name)
    phase.state_variables = ollie_phase.state_variables
    phase.control_variables = ollie_phase.control_variables

    phase.mesh.number_mesh_sections = ollie_phase.number_mesh_sections

ocp.parameter_variables = ollie.parameter_variables
ocp.bounds.parameter_variables = ollie.parameter_variable_bounds
ocp.guess.parameter_variables = ollie.parameter_variable_guesses

ocp.auxiliary_data = ollie.auxiliary_data
ocp.objective_function = ollie.objective_function

# Settings
problem.settings.nlp_tolerance = 1e-8
problem.settings.mesh_tolerance = 1e-3
problem.settings.max_nlp_iterations = 10000

problem.initialize()
problem.solve()
