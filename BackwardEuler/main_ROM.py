import time
from dataclasses import dataclass

import dolfin

# from mshr import *
import matplotlib.pyplot as plt
import numpy as np
from dolfin import *

from FOM import FOM
from ROM import ROM

# ---------- FEniCS parameters ---------
parameters["reorder_dofs_serial"] = False

# ----------- FOM parameters -----------


@dataclass
class Mandel:
    # M_biot = Biot's constant
    M_biot: float = 1.75e7  # 2.5e+12
    c_biot: float = 1.0 / M_biot

    # alpha_biot = b_biot = Biot's modulo
    alpha_biot: float = 1.0
    viscosity_biot: float = 1.0e-3
    K_biot: float = 1.0e-13
    density_biot: float = 1.0

    # Traction
    traction_x_biot: float = 0.0
    traction_y_biot: float = -1.0e7

    # Solid parameters
    density_structure: float = 1.0
    lame_coefficient_mu: float = 1.0e8
    poisson_ratio_nu: float = 0.2
    lame_coefficient_lambda: float = (2.0 * poisson_ratio_nu * lame_coefficient_mu) / (
        1.0 - 2.0 * poisson_ratio_nu
    )


# start time
t = 0.0
# end time
T = 5.0e6
# time step size
dt = 1000  # 5.e6/20  # 1000.0

n_timesteps = int(T / dt)
# dt = T / n_timesteps

# ----------- ROM parameters -----------
REL_ERROR_TOL = 1e-2
MAX_ITERATIONS = 100
TOTAL_ENERGY = {
    "primal": {
        "displacement": 1 - 1e-2,
        "pressure": 1 - 1e-2,
    },
    "dual": {
        "displacement": 1 - 1e-8,
        "pressure": 1 - 1e-8,
    },
}

# ----------- FOM -----------
start_time_fom = time.time()
fom = FOM(t, T, dt, Mandel())
fom.solve_primal(force_recompute=False)
fom.solve_dual(force_recompute=False)
end_time_fom = time.time()

fom.solve_functional_trajectory()
# fom.plot_bottom_solution()


# ----------- ROM -----------
start_time_rom = time.time()
rom = ROM(
    fom,
    REL_ERROR_TOL=REL_ERROR_TOL,
    MAX_ITERATIONS=MAX_ITERATIONS,
    TOTAL_ENERGY=TOTAL_ENERGY,
)

# POD
rom.init_POD()

# compute reduced matrices
print("starting matrix reduction")
rom.update_matrices(matrix_type="primal")
rom.update_matrices(matrix_type="dual")
rom.update_matrices(matrix_type="estimator")
print("finished matrix reduction")

rom.solve_primal()
end_time_rom = time.time()
rom.solve_dual()


# post processing
rom.update_matrices_plotting()

# ----------- ROM Error -----------
rom.compute_error()
rom.error_estimate_dual_fom_reduced()
rom.error_estimate()
rom.solve_functional_trajectory()
rom.plot_bottom_solution()

# ----------- Results -----------
print("FOM time: ", end_time_fom - start_time_fom)
print("ROM time: ", end_time_rom - start_time_rom)
print("Speedup:  ", (end_time_fom - start_time_fom) / (end_time_rom - start_time_rom))
