# from iROM import iROM
# from FOM import FOM
import os
import re
import time
from dataclasses import dataclass

import dolfin

# from mshr import *
import matplotlib.pyplot as plt
import numpy as np
from dolfin import *
from tabulate import tabulate

# import logging

# configure logger
# logging.basicConfig(
#     level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
# )


@dataclass
class Footing:
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
    traction_y_biot: float = 0.0
    traction_z_biot: float = -1.0e7

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
dt = 1000.0  # 5.e6/20  # 1000.0

n_timesteps = int(T / dt)
# dt = T / n_timesteps

time_points = np.arange(t, T + dt, dt)

# ----------- ROM parameters -----------
REL_ERROR_TOL = 1e-2
MAX_ITERATIONS = 200  # 1000
MIN_ITERATIONS = 20
PARENT_SLAB_SIZE = int(n_timesteps / 1)
TOTAL_ENERGY = {
    "primal": {
        "displacement": 1 - 1e-7,
        "pressure": 1 - 1e-11,
    },
    "dual": {
        "displacement": 1 - 1e-9,
        "pressure": 1 - 1e-9,
    },
}

FONT_SIZE_AXIS = 15
FONT_LABEL_SIZE = 13

# # Footing
# problem_name = "Footing"
# goal = "point"
# MESH_REFINEMENTS = 2
# direct_solve = False if MESH_REFINEMENTS > 1 else True
# SOLVER_TOL = 0.0 if direct_solve else 5.0e-8
# REL_ERROR_TOLERANCES = [0.1e-2, 0.5e-2, 1.0e-2, 2.0e-2, 5.0e-2, 10.0e-2, 20.0e-2]

# Mandel
problem_name = "Mandel"
goal ="mean"
MESH_REFINEMENTS = 1
direct_solve = True
SOLVER_TOL = 0.0
REL_ERROR_TOLERANCES = [0.1e-2, 0.5e-2, 1.e-2, 2.e-2, 5.0e-2, 10.e-2, 20.e-2]


# REL_ERROR_TOL = .5e-2


pattern = r"plot_data_goal_" + goal + "_" + r"\d{6}\.npz"
for i, REL_ERROR_TOL in enumerate(REL_ERROR_TOLERANCES):
    SAVE_DIR = "results/"
    # check if SAVE_DIR exists
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        raise ValueError("SAVE_DIR does not exist")

    files = os.listdir(SAVE_DIR)
    files = [
        SAVE_DIR + f
        for f in files
        if os.path.isfile(os.path.join(SAVE_DIR, f)) and re.match(pattern, f)
    ]

    parameters = np.array(
        [
            dt,
            T,
            problem_name,
            MESH_REFINEMENTS,
            direct_solve,
            SOLVER_TOL,
            REL_ERROR_TOL,
        ]
    )

    exists = False
    for file in files:
        tmp = np.load(file, allow_pickle=True)
        # print(parameters)
        # print(tmp["parameters"])
        if np.array_equal(parameters, tmp["parameters"]):
            functional_FOM = tmp["functional"]
            functional_values_FOM = tmp["functional_values_FOM"]
            functional_ROM = tmp["functional"]
            functional_values_ROM = tmp["functional_values"]
            iterations_infos = tmp["iterations_infos"]
            REL_ERROR_TOL = tmp["REL_ERROR_TOL"]
            parent_slabs = tmp["parent_slabs"]
            goal = tmp["goal"]
            print(f"Loaded {file}")
            exists = True
            break
    if not exists:
        raise ValueError("No file found with the given parameters")

    # ---------------------------------
    # Error over iterations
    # ---------------------------------

    fom_cf = np.sum(functional_values_FOM)  # np.sum(fom.functional_values)

    # print("iteration infos:", iterations_infos)
    error_cf = np.abs(fom_cf - np.array(iterations_infos[0]["functional"])) / np.abs(fom_cf)

    # print(error_cf.shape)
    # print(np.array(iterations_infos[0]["error"]).shape)
    plt.semilogy(
        100 * error_cf,
        label="$e^{\\mathrm{rel}}$: error",
        linewidth=3,
    )
    plt.semilogy(
        100 * np.array(iterations_infos[0]["error"]),
        label="$\eta^{\\mathrm{rel}}$: estimate",
        linestyle=":",
        linewidth=3,
        color="red",
    )
    plt.xlabel("#iterations", fontsize = FONT_SIZE_AXIS)
    plt.ylabel("relative error [%]", fontsize = FONT_SIZE_AXIS)
    plt.legend(fontsize = FONT_LABEL_SIZE)
    plt.grid()
    # set the font size of the tick labels
    plt.tick_params(axis="both", which="major", labelsize=13)

    # plt.show()
    name = f"images/tol={REL_ERROR_TOL}_goal_{goal}_error_over_iterations.eps"
    plt.savefig(name, format="eps")
    plt.clf()
    # ---------------------------------
    # Cost functional over iterations
    # ---------------------------------

    plt.plot(np.array(iterations_infos[0]["functional"]), label="cost functional")
    plt.plot(
        [1, len(iterations_infos[0]["functional"])],
        [fom_cf, fom_cf],
        color="green",
        linestyle="--",
        linewidth=4,
    )
    plt.xlabel("#iterations", fontsize = FONT_SIZE_AXIS)
    plt.ylabel("cost functional", fontsize = FONT_SIZE_AXIS)
    plt.grid()
    # set the font size of the tick labels
    plt.tick_params(axis="both", which="major", labelsize=13)
    # plt.show()

    name = f"images/tol={REL_ERROR_TOL}_goal_{goal}_cost_functional_iterations.eps"
    plt.savefig(name, format="eps")
    plt.clf()

    # ---------------------------------
    # POD size over iterations
    # ---------------------------------

    plt.plot(
        np.array(iterations_infos[0]["POD_size"]["primal"]["displacement"]) + 0.25,
        label="primal displacement", 
        linewidth=3,
        color="#1f77b4",# , linestyle="--",marker="x",
    )

    plt.plot(
        np.array(iterations_infos[0]["POD_size"]["primal"]["pressure"]) + 0.25,
        label="primal pressure",  
        linewidth=3,
        color="#ff7f0e",# , linestyle="--",marker="x",
    )

    plt.plot(
        np.array(iterations_infos[0]["POD_size"]["dual"]["displacement"]),
        label="dual displacement",  
        linewidth=3, 
        linestyle=":",
        color="#1f77b4",
        #marker="o",fillstyle="none",
    )

    plt.plot(
        np.array(iterations_infos[0]["POD_size"]["dual"]["pressure"]),
        label="dual pressure",  
        linewidth=3,
        linestyle=":",
        color="#ff7f0e",
        #marker="o",fillstyle="none",
    )

    plt.xlabel("#iterations", fontsize = FONT_SIZE_AXIS)
    plt.ylabel("POD basis size", fontsize = FONT_SIZE_AXIS)
    plt.legend(fontsize = FONT_LABEL_SIZE-2)
    plt.grid()
    # set the font size of the tick labels
    plt.tick_params(axis="both", which="major", labelsize=13)

    # plt.show()

    name = f"images/tol={REL_ERROR_TOL}_goal_{goal}_POD_size.eps"
    plt.savefig(name, format="eps")
    plt.clf()

    # ---------------------------------
    # Functional over time
    # ---------------------------------

    plt.plot(
        time_points[1:],
        functional_values_FOM,
        label="FOM",
        linewidth=3,
    )
    plt.plot(
        time_points[1:],
        functional_values_ROM,
        label="ROM",
        linestyle=":",
        linewidth=3,
        color="red",
    )
    plt.xlabel("time $t$ [s]", fontsize = FONT_SIZE_AXIS)
    plt.ylabel("$J(U(t))$", fontsize = FONT_SIZE_AXIS)
    # logarithmic y axis
    # plt.yscale("log")
    plt.legend(fontsize = FONT_LABEL_SIZE)
    plt.grid()
    # set the font size of the tick labels
    plt.tick_params(axis="both", which="major", labelsize=13)

    # plt.show()
    name = f"images/tol={REL_ERROR_TOL}_goal_{goal}_functional_over_time.eps"
    plt.savefig(name, format="eps")
    plt.clf()
