import time
from dataclasses import dataclass
import dolfin

# from mshr import *
import matplotlib.pyplot as plt
import numpy as np
from dolfin import *

from FOM import FOM
#from ROM import ROM

# ---------- FEniCS parameters ---------
parameters["reorder_dofs_serial"] = False

# ----------- FOM parameters -----------
@dataclass
class Mandel:
	# M_biot = Biot's constant
	M_biot: float = 1.75e+7 #2.5e+12
	c_biot: float = 1.0 / M_biot

	# alpha_biot = b_biot = Biot's modulo
	alpha_biot: float = 1.0
	viscosity_biot: float = 1.0e-3
	K_biot: float = 1.0e-13
	density_biot: float = 1.0

	# Traction
	traction_x_biot: float = 0.0
	traction_y_biot: float = -1.0e+7

	# Solid parameters
	density_structure: float = 1.0
	lame_coefficient_mu: float = 1.0e+8
	poisson_ratio_nu: float = 0.2
	lame_coefficient_lambda: float = (2. * poisson_ratio_nu * lame_coefficient_mu) / (1.0 - 2. * poisson_ratio_nu)

# start time 
t = 0.
# end time 
T = 5.0e+6
# time step size
dt = 1000.

n_timesteps = int(T / dt)
# dt = T / n_timesteps

# ----------- ROM parameters -----------
REL_ERROR_TOL = 1e-2
MAX_ITERATIONS = 100
TOTAL_ENERGY = {
    "primal": {
        "velocity": 1 - 1e-4,
        "pressure": 1 - 1e-6,
    },
}

fom = FOM(t, T, dt, Mandel())
fom.solve_primal(force_recompute=False)
