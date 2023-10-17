import time
from dataclasses import dataclass

import dolfin

# from mshr import *
import matplotlib.pyplot as plt
import numpy as np
from dolfin import *

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
REL_ERROR_TOL = 1e-2
MAX_ITERATIONS = 1000
PARENT_SLAB_SIZE = int(n_timesteps / 1)
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
fom = FOM(t, T, dt, Mandel())
start_time_fom = time.time()
fom.solve_primal(force_recompute=False)
end_time_fom = time.time()
fom.solve_dual(force_recompute=False)

fom.solve_functional_trajectory()
# fom.plot_bottom_solution()


# ----------- ROM -----------
rom = iROM(
    fom,
    REL_ERROR_TOL=REL_ERROR_TOL,
    MAX_ITERATIONS=MAX_ITERATIONS,
    PARENT_SLAB_SIZE=PARENT_SLAB_SIZE,
    TOTAL_ENERGY=TOTAL_ENERGY,
)

# POD
rom.init_POD()
start_time_rom = time.time()
rom.run_parent_slab()  # parent-slabbing MORe DWR
end_time_rom = time.time()

# post processing
rom.update_matrices_plotting()

# ----------- ROM Error -----------
rom.compute_error()  # used for plotting of error NOT cost functionals
rom.solve_functional_trajectory()
rom.plot_bottom_solution()

# ----------- Results -----------
time_FOM = end_time_fom - start_time_fom
time_FOM = 163.0  # 46.5
time_iROM = end_time_rom - start_time_rom

print("FOM time:             " + str(time_FOM))
print("iROM time:            " + str(time_iROM))
print(
    "speedup: act/max:     "
    + str(time_FOM / time_iROM)
    + " / "
    + str((fom.dofs["time"] - 1) / rom.fom_solves)
)
print(
    "Size ROM: u/p         "
    + str(rom.POD["primal"]["displacement"]["basis"].shape[1])
    + " / "
    + str(rom.POD["primal"]["pressure"]["basis"].shape[1])
)
print(
    "Size ROM - dual: u/p  "
    + str(rom.POD["dual"]["displacement"]["basis"].shape[1])
    + " / "
    + str(rom.POD["dual"]["pressure"]["basis"].shape[1])
)
print("FOM solves:           " + str(rom.fom_solves) + " / " + str(fom.dofs["time"] - 1))

# %% Computing reduced and full Cost functional

# J_h_t = np.empty([slab_properties["n_total"], 1])
# for i in range(slab_properties["n_total"]):
#     J_h_t[i] = primal_solutions_slab["value"][i].dot(dual_rhs_no_bc[i])

J_h_t = fom.functional_values
J_r_t = rom.functional_values

temporal_interval_error = rom.errors

true_error = np.abs(np.sum(J_h_t - J_r_t))
true_abs_error = np.sum(np.abs(J_h_t - J_r_t))
estimated_error = np.abs(np.sum(temporal_interval_error))
estimated_abs_error = np.sum(np.abs(temporal_interval_error))
effectivity = true_error / estimated_error

print("J_h:                 " + str(np.sum(J_h_t)))
print("J_r:                 " + str(np.sum(J_r_t)))
print("|J(u_h) - J(u_r)|/|J(u_h)| =", np.abs(np.sum(J_h_t) - np.sum(J_r_t)) / np.abs(np.sum(J_h_t)))
print("true error:          " + str(true_error))
print("estimated error:     " + str(estimated_error))
print("effectivity index:   " + str(effectivity))
# print(" ")
print("true abs error:      " + str(true_abs_error))
print("estimated abs error: " + str(estimated_abs_error))
print("inidicator index:    " + str(true_abs_error / estimated_abs_error))

# # %% error calculation


# temporal_interval_error_relative_fom = (J_h_t - J_r_t)/J_h_t

# real_max_error = np.max(np.abs(temporal_interval_error_relative_fom))
# real_max_error_index = np.argmax(np.abs(temporal_interval_error_relative_fom))

# estimated_max_error = np.max(np.abs(temporal_interval_error_relative))
# estimated_max_error_index = np.argmax(np.abs(temporal_interval_error_relative))

# print(f"Largest estimated error at: {estimated_max_error_index} with: {estimated_max_error}")
# print(f"Largest real error at:      {real_max_error_index} with: {real_max_error}")
# print(f"We instead estimated:                 {np.abs(temporal_interval_error_relative)[real_max_error_index]}")


# # %% error metric

# true_tol = np.abs((J_h_t - J_r_t)/J_h_t) > tol_rel
# esti_tol = np.abs(temporal_interval_error_relative) > tol_rel

# if np.sum(true_tol) == np.sum(esti_tol):
#     print("estimator works perfectly")
# else:
#     from sklearn.metrics import confusion_matrix
#     confusion_matrix = confusion_matrix(true_tol.astype(int), esti_tol.astype(int))
#     eltl, egtl, eltg, egtg = confusion_matrix.ravel()
#     # n_slabs=100

#     print(f"(error > tol & esti < tol): {eltg} ({round(100 * eltg / slab_properties['n_total'],1)} %)  (very bad)")
#     print(f"(error < tol & esti > tol): {egtl} ({round(100 * egtl / slab_properties['n_total'],1)} %)  (bad)")
#     print(f"(error > tol & esti > tol): {egtg} ({round(100 * egtg / slab_properties['n_total'],1)} %)  (good)")
#     print(f"(error < tol & esti < tol): {eltl} ({round(100 * eltl / slab_properties['n_total'],1)} %)  (good)")
