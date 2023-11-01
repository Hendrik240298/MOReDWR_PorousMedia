from iROM import iROM
from FOM import FOM
import argparse
import logging
import os
import re
import time
from dataclasses import dataclass

import dolfin

# from mshr import *
import matplotlib.pyplot as plt
import numpy as np
import yaml
from dolfin import *
from tabulate import tabulate

# configure logger
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
)


def save_for_plot(ROM, FOM):
    pattern = r"plot_data_goal_" + ROM.fom.goal + "_" + r"\d{6}\.npz"
    files = os.listdir(ROM.fom.SAVE_DIR)
    files = [
        ROM.fom.SAVE_DIR + f
        for f in files
        if os.path.isfile(os.path.join(ROM.fom.SAVE_DIR, f)) and re.match(pattern, f)
    ]

    parameters = np.array(
        [
            FOM.dt,
            FOM.T,
            FOM.problem_name,
            FOM.MESH_REFINEMENTS,
            FOM.direct_solve,
            FOM.SOLVER_TOL,
            ROM.REL_ERROR_TOL,
        ]
    )

    for file in files:
        tmp = np.load(file, allow_pickle=True)
        if np.array_equal(parameters, tmp["parameters"]):
            np.savez(
                file,
                functional_FOM=ROM.fom.functional,
                functional_values_FOM=ROM.fom.functional_values,
                functional=ROM.functional,
                functional_values=ROM.functional_values,
                iterations_infos=ROM.iterations_infos,
                REL_ERROR_TOL=ROM.REL_ERROR_TOL,
                parent_slabs=ROM.parent_slabs,
                goal=ROM.fom.goal,
                parameters=parameters,
                compression=True,
            )
            print(f"Overwrite {file}")
            return

    file_name = "results/plot_data_goal_" + ROM.fom.goal + "_" + str(len(files)).zfill(6) + ".npz"
    np.savez(
        file_name,
        functional_FOM=ROM.fom.functional,
        functional_values_FOM=ROM.fom.functional_values,
        functional=ROM.functional,
        functional_values=ROM.functional_values,
        iterations_infos=ROM.iterations_infos,
        REL_ERROR_TOL=ROM.REL_ERROR_TOL,
        parent_slabs=ROM.parent_slabs,
        goal=ROM.fom.goal,
        parameters=parameters,
        compression=True,
    )
    print(f"Saved as {file_name}")


# ---------- FEniCS parameters ---------
parameters["reorder_dofs_serial"] = False

# ---------- Parameter file ------------

parser = argparse.ArgumentParser(description="Input file to specify the problem.")
parser.add_argument("yaml_config", nargs="?", help="Path/Name to the YAML config file")

# parse the arguments
args = parser.parse_args()

# ATTENTION: No sanity check for yaml config exists yet!
if args.yaml_config is None:
    logging.info("No YAML config file was specified. Thus standard config 'config.yaml' is used.")
    config_file = "config/config.yaml"
else:
    config_file = args.yaml_config

with open(config_file, "r") as f:
    config = yaml.safe_load(f)

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


# ----------- Time parameters -----------
# start time
t = config["FOM"]["start_time"]  # 0.0
# end time
T = config["FOM"]["end_time"]  # 5.0e6
# time step size
dt = config["FOM"]["delta_t"]  # 1000.0  # 5.e6/20  # 1000.0

n_timesteps = int(T / dt)
# dt = T / n_timesteps

logging.debug(f"t: {t}, T: {T}, dt: {dt}")

# ----------- ROM parameters -----------
REL_ERROR_TOL = 0.1e-2
MAX_ITERATIONS = 300  # 1000
MIN_ITERATIONS = 20
PARENT_SLAB_SIZE = int(n_timesteps / 1)
TOTAL_ENERGY = {
    "primal": {
        "displacement": 1 - config["ROM"]["total_energy"]["primal"]["displacement"],
        "pressure": 1 - config["ROM"]["total_energy"]["primal"]["pressure"],
    },
    "dual": {
        "displacement": 1 - config["ROM"]["total_energy"]["dual"]["displacement"],
        "pressure": 1 - config["ROM"]["total_energy"]["dual"]["pressure"],
    },
}

# ----------- FOM -----------
if config["Problem"]["type"] == "Mandel":
    fom = FOM(config, Mandel())
elif config["Problem"]["type"] == "Footing":
    fom = FOM(config, Footing())
else:
    raise ValueError(f"Problem type {config['Problem']['type']} not recognized")

start_time_fom = time.time()
recomputed_primal_fom = fom.solve_primal(force_recompute=config["FOM"]["force_recompute"]["primal"])
end_time_fom = time.time()
time_FOM = end_time_fom - start_time_fom

if recomputed_primal_fom:
    # save FOM time to file
    fom.save_time(time_FOM)

fom.solve_dual(force_recompute=config["FOM"]["force_recompute"]["dual"])

fom.solve_functional_trajectory()
# fom.plot_bottom_solution()

# [0.1e-2 , 1.e-2, 2.e-2, 5.0e-2, 10.e-2, 20.e-2]
REL_ERROR_TOLERANCES = [0.1e-2, 0.5e-2, 1.0e-2, 2.0e-2, 5.0e-2, 10.0e-2, 20.0e-2]
PLOTTING = False

result_matrix = np.zeros((len(REL_ERROR_TOLERANCES), 6), dtype=object)

for i, relative_error in enumerate(REL_ERROR_TOLERANCES):
    print("########################################## ")
    print(f"Relative error tolerance: {relative_error}")
    print("########################################## ")
    # ----------- ROM -----------
    rom = iROM(
        fom,
        REL_ERROR_TOL=relative_error,
        MAX_ITERATIONS=config["ROM"]["max_iterations"],
        MIN_ITERATIONS=config["ROM"]["min_iterations"],
        PARENT_SLAB_SIZE=PARENT_SLAB_SIZE,
        TOTAL_ENERGY=TOTAL_ENERGY,
        PLOTTING=False,
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
    if fom.problem_name == "Mandel":
        rom.plot_bottom_solution()

    # ----------- ROM save -----------
    save_for_plot(rom, fom)

    # ----------- Results -----------
    time_FOM = fom.load_time()
    time_iROM = end_time_rom - start_time_rom
    print("Time FOM: ", time_FOM)
    print("Time iROM: ", time_iROM)

    J_h_t = fom.functional_values
    J_r_t = rom.functional_values

    J_h = fom.functional
    J_r = rom.functional

    temporal_interval_error = rom.errors

    true_error = np.abs(J_h - J_r)  # np.abs(np.sum(J_h_t-J_r_t))
    print("True error: ", true_error)
    true_abs_error = None
    if fom.goal == "mean" or fom.goal == "point":
        true_abs_error = np.sum(np.abs(J_h_t - J_r_t))
    estimated_error = np.abs(np.sum(temporal_interval_error))
    print("Estimated error: ", estimated_error)
    estimated_abs_error = np.sum(np.abs(temporal_interval_error))
    effectivity = true_error / estimated_error

    # relative error
    # 100* np.abs(np.sum(J_h_t) - np.sum(J_r_t))/np.abs(np.sum(J_h_t))
    result_matrix[i, 0] = 100 * np.abs(J_h - J_r) / np.abs(J_h)
    # speedup
    result_matrix[i, 1] = time_FOM / time_iROM
    # fom solves
    result_matrix[i, 2] = rom.fom_solves
    # size of ROM
    result_matrix[i, 3] = (
        str(rom.POD["primal"]["displacement"]["basis"].shape[1])
        + " / "
        + str(rom.POD["primal"]["pressure"]["basis"].shape[1])
        + " + "
        + str(rom.POD["dual"]["displacement"]["basis"].shape[1])
        + " / "
        + str(rom.POD["dual"]["pressure"]["basis"].shape[1])
    )
    # effectivity index
    result_matrix[i, 4] = effectivity

    if fom.goal == "mean" or fom.goal == "point":
        # indicator index
        result_matrix[i, 5] = true_abs_error / estimated_abs_error

    if PLOTTING:
        rom.plots_for_paper()

header_legend = [
    "relative error [%]",
    "speedup",
    "FOM solves",
    "size ROM",
    "effectivity",
    "indicator",
]

print(result_matrix)

table = tabulate(
    result_matrix, headers=header_legend, showindex=True  # 100 * np.array(REL_ERROR_TOLERANCES)
)
print(table)

table = tabulate(
    result_matrix,
    headers=header_legend,
    showindex=True,  # 100 * np.array(REL_ERROR_TOLERANCES),
    tablefmt="latex",
)
print(table)
