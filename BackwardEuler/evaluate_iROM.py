import time
from dataclasses import dataclass

import dolfin

# from mshr import *
import matplotlib.pyplot as plt
import numpy as np
from dolfin import *

from tabulate import tabulate


from FOM import FOM
from iROM import iROM

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
REL_ERROR_TOL = .1e-2
MAX_ITERATIONS = 1000
PARENT_SLAB_SIZE = int(n_timesteps/1)
TOTAL_ENERGY = {
    "primal": {
        "displacement": 1 - 1e-4,
        "pressure": 1 - 1e-10,
    },
    "dual": {
        "displacement": 1 - 1e-8,
        "pressure": 1 - 1e-8,
    },
}

# ----------- FOM -----------
fom = FOM(t, T, dt, Mandel(), goal="endtime")
start_time_fom = time.time()
fom.solve_primal(force_recompute=False)
end_time_fom = time.time()
fom.solve_dual(force_recompute=False)

fom.solve_functional_trajectory()
# fom.plot_bottom_solution()

REL_ERROR_TOLERANCES = [0.1e-2, 1.e-2, 2.e-2, 5.0e-2, 10.e-2, 20.e-2]

result_matrix = np.zeros((len(REL_ERROR_TOLERANCES), 6), dtype=object)


for i, relative_error in enumerate(REL_ERROR_TOLERANCES):
    print("########################################## ")
    print(f"Relative error tolerance: {relative_error}")
    print("########################################## ")
    # ----------- ROM -----------
    rom = iROM(
        fom,
        REL_ERROR_TOL=relative_error,
        MAX_ITERATIONS=MAX_ITERATIONS,
        PARENT_SLAB_SIZE=PARENT_SLAB_SIZE,
        TOTAL_ENERGY=TOTAL_ENERGY,
        PLOTTING=False,
    )

    # POD
    rom.init_POD()
    start_time_rom = time.time()
    rom.run_parent_slab() # parent-slabbing MORe DWR
    end_time_rom = time.time()

    # post processing
    rom.update_matrices_plotting()

    # ----------- ROM Error -----------
    rom.compute_error() # used for plotting of error NOT cost functionals
    rom.solve_functional_trajectory()
    rom.plot_bottom_solution()

    # ----------- Results -----------
    time_FOM = end_time_fom - start_time_fom
    time_FOM = 163. #46.5
    print("WARNING: FOM time is set to 163 seconds from previous run.")
    time_iROM = end_time_rom - start_time_rom

    J_h_t = fom.functional_values
    J_r_t = rom.functional_values

    temporal_interval_error = rom.errors

    true_error = np.abs(np.sum(J_h_t-J_r_t))
    true_abs_error = np.sum(np.abs(J_h_t-J_r_t))
    estimated_error = np.abs(np.sum(temporal_interval_error))
    estimated_abs_error = np.sum(np.abs(temporal_interval_error))
    effectivity = true_error/estimated_error

    # relative error
    result_matrix[i, 0] = 100* np.abs(np.sum(J_h_t) - np.sum(J_r_t))/np.abs(np.sum(J_h_t))
    # speedup
    result_matrix[i, 1] = time_FOM/time_iROM
    # fom solves
    result_matrix[i, 2] = rom.fom_solves
    # size of ROM
    result_matrix[i, 3] = str(rom.POD["primal"]["displacement"]["basis"].shape[1]) + " / " + str(rom.POD["primal"]["pressure"]["basis"].shape[1]) + " + " + str(rom.POD["dual"]["displacement"]["basis"].shape[1]) + " / " + str(rom.POD["dual"]["pressure"]["basis"].shape[1])
    # effectivity index
    result_matrix[i, 4] = effectivity
    # indicator index
    result_matrix[i, 5] = true_abs_error/estimated_abs_error

    rom.plots_for_paper()

    quit()


header_legend= ["relative error [%]", "speedup", "FOM solves", "size ROM", "effectivity", "indicator"]


table = tabulate(result_matrix, headers=header_legend,
                 showindex=100*np.array(REL_ERROR_TOLERANCES))
print(table)

table = tabulate(result_matrix, headers=header_legend,
                 showindex=100*np.array(REL_ERROR_TOLERANCES), tablefmt="latex")
print(table)


