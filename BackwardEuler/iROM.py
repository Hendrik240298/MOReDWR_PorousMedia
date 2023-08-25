""" ------------ IMPLEMENTATION of ROM ------------
"""
import math
import os
import re
import time
from multiprocessing import Process, Queue

import dolfin
import matplotlib.pyplot as plt
import numpy as np
import scipy
from dolfin import *
from fenics import *
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from petsc4py import PETSc
from slepc4py import SLEPc
from tqdm import tqdm


class iROM:
    # constructor
    def __init__(
        self,
        fom,
        REL_ERROR_TOL=1e-2,
        MAX_ITERATIONS=100,
        PARENT_SLAB_SIZE=1,
        TOTAL_ENERGY={
            "primal": {"displacement": 1, "pressure": 1},
        },
        PLOTTING=True
    ):
        self.fom = fom
        self.REL_ERROR_TOL = REL_ERROR_TOL
        self.MAX_ITERATIONS = MAX_ITERATIONS
        self.PLOTTING = PLOTTING

        self.fom.matrix.update(
            {
                "estimate": {
                    "system_matrix": self.fom.matrix["primal"]["system_matrix"],
                    "rhs_matrix": self.fom.matrix["primal"]["rhs_matrix"],
                }
            }
        )
        self.POD = {
            "primal": {
                "displacement": {
                    "basis": np.empty((self.fom.dofs["displacement"], 0)),
                    "sigs": None,
                    "energy": 0.0,
                    "bunch": np.empty((self.fom.dofs["displacement"], 0)),
                    "bunch_size": 1,
                    "TOL_ENERGY": TOTAL_ENERGY["primal"]["displacement"],
                },
                "pressure": {
                    "basis": np.empty((self.fom.dofs["pressure"], 0)),
                    "sigs": None,
                    "energy": 0.0,
                    "bunch": np.empty((self.fom.dofs["pressure"], 0)),
                    "bunch_size": 1,
                    "TOL_ENERGY": TOTAL_ENERGY["primal"]["pressure"],
                },
            },
            "dual": {
                "displacement": {
                    "basis": np.empty((self.fom.dofs["displacement"], 0)),
                    "sigs": None,
                    "energy": 0.0,
                    "bunch": np.empty((self.fom.dofs["displacement"], 0)),
                    "bunch_size": 1,
                    "TOL_ENERGY": TOTAL_ENERGY["dual"]["displacement"],
                },
                "pressure": {
                    "basis": np.empty((self.fom.dofs["pressure"], 0)),
                    "sigs": None,
                    "energy": 0.0,
                    "bunch": np.empty((self.fom.dofs["pressure"], 0)),
                    "bunch_size": 1,
                    "TOL_ENERGY": TOTAL_ENERGY["dual"]["pressure"],
                },
            },
        }

        self.matrix = {
            "primal": {"system_matrix": None, "rhs_matrix": None},
            "dual": {"system_matrix": None, "rhs_matrix": None},
            "estimator": {"system_matrix": None, "rhs_matrix": None},
        }
        self.vector = {
            "primal": {"traction": None},
            "dual": {"pressure_down": None},
            "estimator": {"traction": None},
        }

        self.solution = {
            "primal": {"displacement": None, "pressure": None},
            "dual": {"displacement": None, "pressure": None},
        }


        # initialize parent_slabs
        self.parent_slabs = []

        index_start = 1
        while index_start + PARENT_SLAB_SIZE <= self.fom.dofs["time"]:
            self.parent_slabs.append(
                {
                    "start": index_start,
                    "end": index_start + PARENT_SLAB_SIZE,
                    "steps": PARENT_SLAB_SIZE,
                    "solution": {
                        "primal": {
                            "displacement": None,
                            "pressure": None,
                        },
                        "dual": {
                            "displacement": None,
                            "pressure": None,
                        },
                    },
                    "initial_condition": None,
                    "functional": np.zeros((PARENT_SLAB_SIZE,)),
                }
            )
            index_start += PARENT_SLAB_SIZE

        if self.parent_slabs[-1]["end"] < self.fom.dofs["time"]:
            self.parent_slabs.append(
                {
                    "start": self.parent_slabs[-1]["end"],
                    "end": self.fom.dofs["time"] - 1,
                    "steps": self.fom.dofs["time"] - 1 - self.parent_slabs[-1]["end"],
                    "solution": {
                        "primal": {
                            "displacement": None,
                            "pressure": None,
                        },
                        "dual": {
                            "displacement": None,
                            "pressure": None,
                        },
                    },
                    "initial_condition": None,
                    "functional": np.zeros((self.fom.dofs["time"] - self.parent_slabs[-1]["end"],)),
                }
            )

        for parent_slab in self.parent_slabs:
            print(f"parent slab: {parent_slab['start']} - {parent_slab['end']}")

        self.timings = {
            "iPOD": 0.0,
            "primal_ROM": 0.0,
            "dual_ROM": 0.0,
            "error_estimate": 0.0,
            "enrichment": 0.0,
            "run": 0.0,
        }

        iterations_infos = {
            "error": [],
            "POD_size": {
                "primal": {"displacement": [], "pressure": []},
                "dual": {"displacement": [], "pressure": []},
            },
            "functional": [],
            }

        self.iterations_infos = [iterations_infos.copy() for _ in range(len(self.parent_slabs))]

        self.functional_values = np.zeros((self.fom.dofs["time"] - 1,))

    def compute_error(self):
        error = {
            "displacement": np.zeros((self.fom.dofs["time"] - 1,)),
            "pressure": np.zeros((self.fom.dofs["time"] - 1,)),
        }

        # check fom.Y and rom.solution shape
        print("fom.Y shape: ", self.fom.Y["primal"]["displacement"].shape)
        print("rom.solution shape: ", self.solution["primal"]["displacement"].shape)

        for i in range(self.fom.dofs["time"] - 1):
            projected_solution_disp = self.project_vector(
                self.solution["primal"]["displacement"][:, i],
                type="primal",
                quantity="displacement",
            )
            projected_solution_pres = self.project_vector(
                self.solution["primal"]["pressure"][:, i], type="primal", quantity="pressure"
            )
            error["displacement"][i] = np.linalg.norm(
                projected_solution_disp - self.fom.Y["primal"]["displacement"][:, i]
            ) / np.linalg.norm(self.fom.Y["primal"]["displacement"][:, i])
            error["pressure"][i] = np.linalg.norm(
                projected_solution_pres - self.fom.Y["primal"]["pressure"][:, i]
            ) / np.linalg.norm(self.fom.Y["primal"]["pressure"][:, i])

        if self.PLOTTING:
            # plot results in subplots
            fig, ax = plt.subplots(2, 1, figsize=(10, 10))
            ax[0].semilogy(self.fom.time_points[1:], error["displacement"] * 100, label="displacement")
            ax[0].set_xlabel("time")
            ax[0].set_ylabel("displacement - error [%]")
            ax[0].grid()
            ax[1].semilogy(self.fom.time_points[1:], error["pressure"] * 100, label="pressure")
            ax[1].set_xlabel("time")
            ax[1].set_ylabel("pressure - error [%]")
            ax[1].grid()
            plt.show()

    def solve_functional_trajectory(self):
        self.functional_values = np.zeros((self.fom.dofs["time"] - 1,))
        for i in range(self.fom.dofs["time"] - 1):
            self.functional_values[i] = (
                self.vector["primal"]["pressure_down"].dot(
                    self.solution["primal"]["pressure"][:, i + 1]
                )
                * self.fom.dt
            )

        self.functional = np.sum(self.functional_values)
        print(f"Cost functional - FOM : {self.fom.functional:.4e}")
        print(f"Cost functional - ROM : {self.functional:.4e}")
        print(f"Error:                  {np.abs(self.functional - self.fom.functional):.4e}")
        print(f"Estimate:               {np.abs(np.sum(self.errors)):.4e}")

        print(
            f"Relative Error [%]:     {100* np.abs(self.functional - self.fom.functional) / np.abs(self.fom.functional):.4e}"
        )
        print(f"Relative Estimate [%] : {100* np.sum(np.abs(self.relative_errors)):.4e}")

        # print cost functional trajectory
        if self.PLOTTING:
            plt.plot(self.fom.time_points[1:], self.fom.functional_values, label="FOM")
            plt.plot(self.fom.time_points[1:], self.functional_values, linestyle="--", label="ROM")

            plt.xlabel("time")
            plt.ylabel("cost functional")
            plt.legend()
            plt.grid()
            plt.show()

            # ---------------------------------
            # ------ Plot relative error ------
            # ---------------------------------
            plt.semilogy(
                self.fom.time_points[1:],
                100
                * np.abs(self.fom.functional_values - self.functional_values)
                / np.abs(self.fom.functional_values),
                label="relative error",
            )

            plt.semilogy(
                self.fom.time_points[1:],
                100 * self.fom.dt * self.relative_errors,
                label="relative estimate",
            )

            plt.xlabel("time")
            plt.ylabel("relative error [%]")
            plt.legend()
            plt.grid()
            plt.show()

            # ---------------------------------
            # ------ Plot absolute error ------
            # ---------------------------------

            plt.semilogy(
                self.fom.time_points[1:],
                np.abs(self.fom.functional_values - self.functional_values),
                label="absolute error",
            )

            plt.semilogy(
                self.fom.time_points[1:],
                np.abs(self.errors),
                label="absolute estimate",
            )

            plt.xlabel("time")
            plt.ylabel("absolute error")
            plt.legend()
            plt.grid()
            plt.show()

    def plot_bottom_solution(self):
        times = self.fom.special_times.copy()
        for i, t in enumerate(self.fom.time_points):
            if np.abs(t - times[0]) <= 1e-4:
                times.pop(0)
                if self.PLOTTING:
                    plt.plot(
                        self.fom.bottom_x_u,
                        self.bottom_matrix_u.dot(self.solution["primal"]["displacement"][:, i]),
                        label=f"t = {t} (FOM)",
                    )
                    plt.plot(
                        self.fom.bottom_x_u,
                        self.bottom_matrix_u.dot(self.solution["primal"]["displacement"][:, i]),
                        linestyle="--",
                        label=f"t = {t} (ROM)",
                    )
                if len(times) == 0:
                    break
        if self.PLOTTING:
            plt.xlabel("x")
            plt.ylabel(r"$u_x(x,0)$")
            plt.title("x-Displacement at bottom boundary")
            plt.legend()
            plt.show()

        times = self.fom.special_times.copy()
        for i, t in enumerate(self.fom.time_points):
            if np.abs(t - times[0]) <= 1e-4:
                times.pop(0)
                if self.PLOTTING:
                    plt.plot(
                        self.fom.bottom_x_p,
                        self.bottom_matrix_p.dot(self.solution["primal"]["pressure"][:, i]),
                        label=f"t = {t} (FOM)",
                    )
                    plt.plot(
                        self.fom.bottom_x_p,
                        self.bottom_matrix_p.dot(self.solution["primal"]["pressure"][:, i]),
                        linestyle="--",
                        label=f"t = {t} (ROM)",
                    )
                if len(times) == 0:
                    break

        if self.PLOTTING:
            plt.xlabel("x")
            plt.ylabel(r"$p(x,0)$")
            plt.title("Pressure at bottom boundary")
            plt.legend()
            plt.show()

    def plots_for_paper(self):

        # ---------------------------------
        # Error over iterations
        # ---------------------------------

        fom_cf = np.sum(self.fom.functional_values)

        error_cf = np.abs(fom_cf - np.array(self.iterations_infos[0]["functional"]))/np.abs(fom_cf)
        plt.semilogy(
            100*np.array(self.iterations_infos[0]["error"]),
            label="estimated error",
        )
        plt.semilogy(
            100*error_cf,
            label="true error",
        )
        plt.xlabel("#iterations")
        plt.ylabel("error [%]")
        plt.legend()
        plt.grid()
        if self.PLOTTING:
            plt.show()
        else:
            name=f"images/tol={self.REL_ERROR_TOL}_parent_slab={self.parent_slabs[0]['steps']}_error_over_iterations.eps"
            plt.savefig(name, format="eps")
            plt.clf()

        # ---------------------------------
        # Cost functional over iterations
        # ---------------------------------

        plt.plot(
            np.array(self.iterations_infos[0]["functional"]),
            label="cost functional"
        )
        plt.plot(
            [1, len(self.iterations_infos[0]["functional"])],[fom_cf,fom_cf], color="green", linestyle="--",
        )
        plt.xlabel("#iterations")
        plt.ylabel("cost functional")
        plt.grid()
        if self.PLOTTING:
            plt.show()
        else:
            name=f"images/tol={self.REL_ERROR_TOL}_parent_slab={self.parent_slabs[0]['steps']}_cost_functional_iterations.eps"
            plt.savefig(name, format="eps")
            plt.clf()

        # ---------------------------------
        # POD size over iterations
        # ---------------------------------

        plt.plot(
            np.array(self.iterations_infos[0]["POD_size"]["primal"]["displacement"])+0.25,
            label="primal displacement"#, linestyle="--",marker="x",
        )

        plt.plot(
            np.array(self.iterations_infos[0]["POD_size"]["primal"]["pressure"])+0.25,
            label="primal pressure"#, linestyle="--",marker="x",
        )

        plt.plot(
            np.array(self.iterations_infos[0]["POD_size"]["dual"]["displacement"]),
            label="dual displacement"#,linestyle="--",marker="o",fillstyle="none",
        )

        plt.plot(
            np.array(self.iterations_infos[0]["POD_size"]["dual"]["pressure"]),
            label="dual pressure"#,linestyle="--",marker="o",fillstyle="none",
        )

        plt.xlabel("#iterations")
        plt.ylabel("POD basis size")
        plt.legend()
        plt.grid()
        if self.PLOTTING:
            plt.show()
        else:
            name=f"images/tol={self.REL_ERROR_TOL}_parent_slab={self.parent_slabs[0]['steps']}_POD_size.eps"
            plt.savefig(name, format="eps")
            plt.clf()

        # ---------------------------------
        # Functional over time
        # ---------------------------------

        plt.plot(
            self.fom.time_points[1:],
            self.fom.functional_values,
            label="FOM",
        )
        plt.plot(
            self.fom.time_points[1:],
            self.functional_values,
            label="ROM", linestyle="--",
        )
        plt.xlabel("time")
        plt.ylabel("cost functional")
        plt.legend()
        plt.grid()
        if self.PLOTTING:
            plt.show()
        else:
            name=f"images/tol={self.REL_ERROR_TOL}_parent_slab={self.parent_slabs[0]['steps']}_functional_over_time.eps"
            plt.savefig(name, format="eps")
            plt.clf()




    def iPOD(self, snapshot, type, quantity):
        # type is either "primal" or "dual"

        self.POD[type][quantity]["bunch"] = np.hstack(
            (self.POD[type][quantity]["bunch"], snapshot.reshape(-1, 1))
        )

        # add energy of new snapshot to total energy
        self.POD[type][quantity]["energy"] += np.dot((snapshot), (snapshot))

        # check bunch_matrix size to decide if to update POD
        if self.POD[type][quantity]["bunch"].shape[1] == self.POD[type][quantity]["bunch_size"]:
            # initialize POD with first bunch matrix
            if self.POD[type][quantity]["basis"].shape[1] == 0:
                (
                    self.POD[type][quantity]["basis"],
                    self.POD[type][quantity]["sigs"],
                    _,
                ) = scipy.linalg.svd(self.POD[type][quantity]["bunch"], full_matrices=False)

                # compute the number of POD modes to be kept
                r = 0
                while (
                    np.dot(
                        self.POD[type][quantity]["sigs"][0:r], self.POD[type][quantity]["sigs"][0:r]
                    )
                    <= self.POD[type][quantity]["energy"] * self.POD[type][quantity]["TOL_ENERGY"]
                ) and (r <= np.shape(self.POD[type][quantity]["sigs"])[0]):
                    r += 1

                self.POD[type][quantity]["sigs"] = self.POD[type][quantity]["sigs"][0:r]
                self.POD[type][quantity]["basis"] = self.POD[type][quantity]["basis"][:, 0:r]
            # update POD with  bunch matrix
            else:
                M = np.dot(self.POD[type][quantity]["basis"].T, self.POD[type][quantity]["bunch"])
                P = self.POD[type][quantity]["bunch"] - np.dot(self.POD[type][quantity]["basis"], M)

                Q_p, R_p = scipy.linalg.qr(P, mode="economic")
                Q_q = np.hstack((self.POD[type][quantity]["basis"], Q_p))

                S0 = np.vstack(
                    (
                        np.diag(self.POD[type][quantity]["sigs"]),
                        np.zeros((np.shape(R_p)[0], np.shape(self.POD[type][quantity]["sigs"])[0])),
                    )
                )
                MR_p = np.vstack((M, R_p))
                K = np.hstack((S0, MR_p))

                # check the orthogonality of Q_q heuristically
                if np.inner(Q_q[:, 0], Q_q[:, -1]) >= 1e-10:
                    Q_q, R_q = scipy.linalg.qr(Q_q, mode="economic")
                    K = np.matmul(R_q, K)

                # inner SVD of K
                U_k, S_k, _ = scipy.linalg.svd(K, full_matrices=False)

                # compute the number of POD modes to be kept
                r = self.POD[type][quantity]["basis"].shape[1] 
                while (
                    np.dot(S_k[0:r], S_k[0:r])
                    <= self.POD[type][quantity]["energy"] * self.POD[type][quantity]["TOL_ENERGY"]
                ) and (r < np.shape(S_k)[0]):
                    r += 1

                self.POD[type][quantity]["sigs"] = S_k[0:r]
                self.POD[type][quantity]["basis"] = np.matmul(Q_q, U_k[:, 0:r])

            # empty bunch matrix after update
            self.POD[type][quantity]["bunch"] = np.empty([self.fom.dofs[quantity], 0])

    def reduce_vector(self, vector, type, quantity):
        return np.dot(self.POD[type][quantity]["basis"].T, vector)

    def project_vector(self, vector, type, quantity):
        return np.dot(self.POD[type][quantity]["basis"], vector)

    def reduce_matrix(self, matrix, type, quantity0, quantity1):
        """
        A_N = Z_{q0}^T A Z_{q1}

        REMARK:
        - A_N is reduced submatrix
        """

        reduced_matrix = self.POD[type][quantity0]["basis"].T.dot(
            matrix.dot(self.POD[type][quantity1]["basis"])
        )
        return reduced_matrix

    def reduce_matrix_different_spaces(self, matrix, type0, type1, quantity0, quantity1):
        """
        A_N = Z_{q0}^T A Z_{q1}

        REMARK:
        - A_N is reduced submatrix
        """

        reduced_matrix = self.POD[type0][quantity0]["basis"].T.dot(
            matrix.dot(self.POD[type1][quantity1]["basis"])
        )
        return reduced_matrix

    def init_POD(self):
        # primal POD
        time_points = self.fom.time_points[:]

        for i, t in enumerate(time_points[:1]):
            for quantity in self.fom.quantities:
                self.iPOD(self.fom.Y["primal"][quantity][:, i+1], type="primal", quantity=quantity)
            
        print(
            "DISPLACEMENT primal POD size:   ", self.POD["primal"]["displacement"]["basis"].shape[1]
        )
        print("PRESSURE primal POD size:   ", self.POD["primal"]["pressure"]["basis"].shape[1])

        # dual POD brought to you by iPOD
        for i, t in enumerate(time_points[:1]):
            for quantity in self.fom.quantities:
                self.iPOD(self.fom.Y["dual"][quantity][:, i+1], type="dual", quantity=quantity)

        print("DISPLACEMENT dual POD size:   ", self.POD["dual"]["displacement"]["basis"].shape[1])
        print("PRESSURE dual POD size:   ", self.POD["dual"]["pressure"]["basis"].shape[1])

        # for i in range(self.POD["primal"]["displacement"]["basis"].shape[1]):
        #     u, p = self.fom.U_n.split()
        #     u.vector().set_local(self.POD["primal"]["displacement"]["basis"][:,i])
        #     c = plot(sqrt(dot(u, u)), title=f"{i}th displacement POD magnitude")
        #     plt.colorbar(c, orientation="horizontal")
        #     plt.show()

        # for i in range(self.POD["primal"]["pressure"]["basis"].shape[1]):
        #     u, p = self.fom.U_n.split()
        # self.fom.U_n.vector().set_local(np.concatenate(
        #     (
        #         self.POD["primal"]["displacement"]["basis"][:,0],
        #         self.POD["primal"]["pressure"]["basis"][:,i]
        #     )
        # ))
        #     c = plot(p, title=f"{i}th pressure POD magnitude")
        #     plt.colorbar(c, orientation="horizontal")
        #     plt.show()

        # TODO: save POD basis as vtks instead of plotting them

    def update_matrices(self, matrix_type):
        assert matrix_type in ["primal", "dual", "estimator"], f"Matrix type {matrix_type} not valid"

        if matrix_type == "primal":
            # primal rhs
            self.vector["primal"]["traction"] = self.reduce_vector(
                self.fom.vector["primal"]["traction"], "primal", "displacement"
            )

            # cost functional rhs
            self.vector["primal"]["pressure_down"] = self.reduce_vector(
                self.fom.vector["primal"]["pressure_down"], "primal", "pressure"
            )

            # * primal sub system matrices
            matrix_stress = self.reduce_matrix(
                self.fom.matrix["primal"]["stress"],
                type="primal",
                quantity0="displacement",
                quantity1="displacement",
            )
            matrix_elasto_pressure = self.reduce_matrix(
                self.fom.matrix["primal"]["elasto_pressure"],
                type="primal",
                quantity0="displacement",
                quantity1="pressure",
            )
            matrix_laplace = self.reduce_matrix(
                self.fom.matrix["primal"]["laplace"],
                type="primal",
                quantity0="pressure",
                quantity1="pressure",
            )
            matrix_time_displacement = self.reduce_matrix(
                self.fom.matrix["primal"]["time_displacement"],
                type="primal",
                quantity0="pressure",
                quantity1="displacement",
            )
            matrix_time_pressure = self.reduce_matrix(
                self.fom.matrix["primal"]["time_pressure"],
                type="primal",
                quantity0="pressure",
                quantity1="pressure",
            )

            # build primal system matrix from blocks
            self.matrix["primal"]["system_matrix"] = np.block(
                [
                    [matrix_stress, matrix_elasto_pressure],
                    [matrix_time_displacement, matrix_time_pressure + matrix_laplace],
                ]
            )

            # numpy lu decomposition of system matrix
            self.lu_primal, self.piv_primal = scipy.linalg.lu_factor(self.matrix["primal"]["system_matrix"])

            # build primal rhs matrix from blocks
            self.matrix["primal"]["rhs_matrix"] = np.block(
                [
                    [np.zeros_like(matrix_stress), np.zeros_like(matrix_elasto_pressure)],
                    [matrix_time_displacement, matrix_time_pressure],
                ]
            )


        if matrix_type == "dual":
            # dual rhs
            self.vector["dual"]["pressure_down"] = self.reduce_vector(
                self.fom.vector["primal"]["pressure_down"], "dual", "pressure"
            )

            # * dual sub system matrices
            matrix_stress_dual = self.reduce_matrix(
                self.fom.matrix["primal"]["stress"].transpose(),
                type="dual",
                quantity0="displacement",
                quantity1="displacement",
            )
            matrix_elasto_pressure_dual = self.reduce_matrix(
                self.fom.matrix["primal"]["elasto_pressure"].transpose(),
                type="dual",
                quantity0="pressure",
                quantity1="displacement",
            )
            matrix_laplace_dual = self.reduce_matrix(
                self.fom.matrix["primal"]["laplace"].transpose(),
                type="dual",
                quantity0="pressure",
                quantity1="pressure",
            )
            matrix_time_displacement_dual = self.reduce_matrix(
                self.fom.matrix["primal"]["time_displacement"].transpose(),
                type="dual",
                quantity0="displacement",
                quantity1="pressure",
            )
            matrix_time_pressure_dual = self.reduce_matrix(
                self.fom.matrix["primal"]["time_pressure"].transpose(),
                type="dual",
                quantity0="pressure",
                quantity1="pressure",
            )

            # build dual system matrix from blocks
            self.matrix["dual"]["system_matrix"] = np.block(
                [
                    [matrix_stress_dual, matrix_time_displacement_dual],
                    [matrix_elasto_pressure_dual, matrix_time_pressure_dual + matrix_laplace_dual],
                ]
            )

            # numpy lu decomposition of system matrix
            self.lu_dual, self.piv_dual = scipy.linalg.lu_factor(self.matrix["dual"]["system_matrix"])

            # build dual rhs matrix from blocks
            self.matrix["dual"]["rhs_matrix"] = np.block(
                [
                    [np.zeros_like(matrix_stress_dual), matrix_time_displacement_dual],
                    [np.zeros_like(matrix_elasto_pressure_dual), matrix_time_pressure_dual],
                ]
            )

        if matrix_type == "estimator":
            # error estimate primal rhs projected in dual space
            self.vector["estimator"]["traction"] = np.concatenate(
                (
                    self.reduce_vector(
                        self.fom.vector["primal"]["traction"], type="dual", quantity="displacement"
                    ),
                    np.zeros((self.POD["dual"]["pressure"]["basis"].shape[1],)),
                )
            )

            # * estimator sub system matrices
            matrix_stress_estimator = self.reduce_matrix_different_spaces(
                self.fom.matrix["primal"]["stress"],
                type0="dual",
                type1="primal",
                quantity0="displacement",
                quantity1="displacement",
            )
            matrix_elasto_pressure_estimator = self.reduce_matrix_different_spaces(
                self.fom.matrix["primal"]["elasto_pressure"],
                type0="dual",
                type1="primal",
                quantity0="displacement",
                quantity1="pressure",
            )
            matrix_laplace_estimator = self.reduce_matrix_different_spaces(
                self.fom.matrix["primal"]["laplace"],
                type0="dual",
                type1="primal",
                quantity0="pressure",
                quantity1="pressure",
            )
            matrix_time_displacement_estimator = self.reduce_matrix_different_spaces(
                self.fom.matrix["primal"]["time_displacement"],
                type0="dual",
                type1="primal",
                quantity0="pressure",
                quantity1="displacement",
            )
            matrix_time_pressure_estimator = self.reduce_matrix_different_spaces(
                self.fom.matrix["primal"]["time_pressure"],
                type0="dual",
                type1="primal",
                quantity0="pressure",
                quantity1="pressure",
            )   


            # build estimator system matrix from blocks
            # build system matrix from blocks
            self.matrix["estimator"]["system_matrix"] = np.block(
                [
                    [matrix_stress_estimator, matrix_elasto_pressure_estimator],
                    [
                        matrix_time_displacement_estimator,
                        matrix_time_pressure_estimator + matrix_laplace_estimator,
                    ],
                ]
            )

            # build estimator rhs matrix from blocks
            self.matrix["estimator"]["rhs_matrix"] = np.block(
                [
                    [
                        np.zeros_like(matrix_stress_estimator),
                        np.zeros_like(matrix_elasto_pressure_estimator),
                    ],
                    [matrix_time_displacement_estimator, matrix_time_pressure_estimator],
                ]
            )


    def update_matrices_plotting(self):
        # reduce matrices for evaluation of the solution at the bottom boundary
        self.bottom_matrix_u = self.fom.bottom_matrix_u.dot(
            self.POD["primal"]["displacement"]["basis"]
        )
        self.bottom_matrix_p = self.fom.bottom_matrix_p.dot(self.POD["primal"]["pressure"]["basis"])


    def solve_primal(self):
        self.solution["primal"]["displacement"] = np.zeros(
            (self.POD["primal"]["displacement"]["basis"].shape[1], self.fom.dofs["time"])
        )

        self.solution["primal"]["pressure"] = np.zeros(
            (self.POD["primal"]["pressure"]["basis"].shape[1], self.fom.dofs["time"])
        )

        # old timestep
        solution = np.concatenate(
            (
                self.solution["primal"]["displacement"][:, 0],
                self.solution["primal"]["pressure"][:, 0],
            )
        )

        for i, t in enumerate(tqdm(self.fom.time_points[1:])):
            # print("#-----------------------------------------------#")
            # print(f"t = {t:.4f}")
            n = i + 1

            solution = scipy.linalg.lu_solve((self.lu_primal, self.piv_primal),
                self.matrix["primal"]["rhs_matrix"].dot(solution)
                + np.concatenate(
                    (
                        self.vector["primal"]["traction"],
                        np.zeros((self.POD["primal"]["pressure"]["basis"].shape[1],)),
                    )
                ),
            )

            self.solution["primal"]["displacement"][:, n] = solution[
                : self.POD["primal"]["displacement"]["basis"].shape[1]
            ]
            self.solution["primal"]["pressure"][:, n] = solution[
                self.POD["primal"]["displacement"]["basis"].shape[1] :
            ]

        self.save_vtk()
        # plt.semilogy(self.DEBUG_RES)
        # plt.show()

    def solve_dual(self):
        print("Solving dual ROM ...")
        self.solution["dual"]["displacement"] = np.zeros(
            (self.POD["dual"]["displacement"]["basis"].shape[1], self.fom.dofs["time"])
        )
        self.solution["dual"]["pressure"] = np.zeros(
            (self.POD["dual"]["pressure"]["basis"].shape[1], self.fom.dofs["time"])
        )

        dual_solution = np.concatenate(
            (
                self.solution["dual"]["displacement"][:, -1],
                self.solution["dual"]["pressure"][:, -1],
            )
        )

        for i, t in tqdm(list(enumerate(self.fom.time_points[:-1]))[::-1]):
            n = i
            # print(f"i = {i}, t = {t:.4f}")
            # LES: A^T * U = dt * J + B^T * U^{n+1}
            dual_rhs = self.matrix["dual"]["rhs_matrix"].dot(dual_solution)
            dual_rhs[self.POD["dual"]["displacement"]["basis"].shape[1] :] += (
                self.fom.dt * self.vector["dual"]["pressure_down"]
            )
            dual_solution = scipy.linalg.lu_solve((self.lu_dual, self.piv_dual), dual_rhs)
            # dual_solution = np.linalg.solve(self.matrix["dual"]["system_matrix"], dual_rhs)

            self.solution["dual"]["displacement"][:, n] = dual_solution[
                : self.POD["dual"]["displacement"]["basis"].shape[1]
            ]
            self.solution["dual"]["pressure"][:, n] = dual_solution[
                self.POD["dual"]["displacement"]["basis"].shape[1] :
            ]

            IF_PLOT = False
            if (i == 0 or i == 4999) and IF_PLOT:
                # plot dual solution
                u, p = self.fom.U_n.split()
                self.fom.U_n.vector().set_local(
                    np.concatenate(
                        (
                            self.project_vector(
                                self.solution["dual"]["displacement"][:, n],
                                type="dual",
                                quantity="displacement",
                            ),
                            self.project_vector(
                                self.solution["dual"]["pressure"][:, n],
                                type="dual",
                                quantity="pressure",
                            ),
                        )
                    )
                )

                # plot u and p in a subplot
                plt.subplot(2, 1, 1)
                c = plot(u, title=f"u - ROM")
                plt.colorbar(c, orientation="horizontal")
                plt.subplot(2, 1, 2)
                c = plot(p, title=f"p - ROM")
                plt.colorbar(c, orientation="horizontal")
                plt.show()

    def error_estimate(self):
        print("Estimating error in reduced spaces (OG)...")
        self.errors = np.zeros((self.fom.dofs["time"] - 1,))
        self.relative_errors = np.zeros((self.fom.dofs["time"] - 1,))
        self.functional_values = np.zeros((self.fom.dofs["time"] - 1,))

        for i in range(1, self.fom.dofs["time"]):
            # print("Estimating error for time step: ", i)
            solution = np.concatenate(
                (
                    self.solution["primal"]["displacement"][:, i],
                    self.solution["primal"]["pressure"][:, i],
                )
            )

            old_solution = np.concatenate(
                (
                    self.solution["primal"]["displacement"][:, i - 1],
                    self.solution["primal"]["pressure"][:, i - 1],
                )
            )

            dual_sol = np.concatenate(
                (
                    self.solution["dual"]["displacement"][:, i - 1],
                    self.solution["dual"]["pressure"][:, i - 1],
                )
            )

            # - A*U^n + F + B*U^{n-1}
            AU = self.matrix["estimator"]["system_matrix"].dot(solution)
            BU_old = self.matrix["estimator"]["rhs_matrix"].dot(old_solution)

            # print shapes
            # print("AU:                               ", AU.shape)
            # print("BU_old:                           ", BU_old.shape)
            # print("self.vector[estimator][traction]: ", self.vector["estimator"]["traction"].shape)

            primal_res = -AU + self.vector["estimator"]["traction"] + BU_old

            # print(f"Primal sol norm   : {np.linalg.norm(solution)}")
            # print(f"Primal res for {i}: {np.linalg.norm(primal_res)}")

            self.errors[i - 1] = np.dot(dual_sol, primal_res)

            self.functional_values[i - 1] = self.vector["primal"]["pressure_down"].dot(
                self.solution["primal"]["pressure"][:, i]
            ) * self.fom.dt

            if np.abs(self.errors[i-1] + self.functional_values[i-1]) < 1e-12:
                self.relative_errors[i - 1] = self.errors[i - 1] / 1e-12
            else:
                self.relative_errors[i - 1] = self.errors[i - 1] / (
                    self.errors[i - 1] + self.functional_values[i - 1]
                )

    def solve_primal_time_step_slab(self, old_solution):
        return scipy.linalg.lu_solve((self.lu_primal, self.piv_primal),
            self.matrix["primal"]["rhs_matrix"].dot(old_solution)
            + np.concatenate(
                (
                    self.vector["primal"]["traction"],
                    np.zeros((self.POD["primal"]["pressure"]["basis"].shape[1],)),
                )
            ),
        )

    def solve_dual_time_step_slab(self, old_solution):
        dual_rhs = self.matrix["dual"]["rhs_matrix"].dot(old_solution)
        dual_rhs[self.POD["dual"]["displacement"]["basis"].shape[1] :] += (
            self.fom.dt * self.vector["dual"]["pressure_down"]
        )
        # solve with lu deccomposition 
        return scipy.linalg.lu_solve((self.lu_dual, self.piv_dual), dual_rhs)

        # return np.linalg.solve(self.matrix["dual"]["system_matrix"], dual_rhs)

    def validate(self):
        self.solve_primal()
        self.solve_dual()
        self.error_estimate()

    def solve_primal_parent_slab(self, index_parent_slab):
        execution_time = time.time()
        # initialize solution matrices for displacement and pressure
        self.parent_slabs[index_parent_slab]["solution"]["primal"]["displacement"] = np.zeros(
            (self.POD["primal"]["displacement"]["basis"].shape[1],
             self.parent_slabs[index_parent_slab]["steps"])
        )
        self.parent_slabs[index_parent_slab]["solution"]["primal"]["pressure"] = np.zeros(
            (self.POD["primal"]["pressure"]["basis"].shape[1],
             self.parent_slabs[index_parent_slab]["steps"])
        )

        # read in intial condition from parent_slab element
        old_solution = self.parent_slabs[index_parent_slab]["initial_condition"]

        for i in range(
            self.parent_slabs[index_parent_slab]["start"],
            self.parent_slabs[index_parent_slab]["end"],
        ):
            n = i - self.parent_slabs[index_parent_slab]["start"]
            old_solution = self.solve_primal_time_step_slab(old_solution)
            # old_solution = self.parent_slabs[index_parent_slab]["solution"]["primal"][:, n]
            self.parent_slabs[index_parent_slab]["solution"]["primal"]["displacement"][:,n] = old_solution[
                : self.POD["primal"]["displacement"]["basis"].shape[1]
            ]
            self.parent_slabs[index_parent_slab]["solution"]["primal"]["pressure"][:,n] = old_solution[
                self.POD["primal"]["displacement"]["basis"].shape[1] :
            ]

        self.timings["primal_ROM"] += time.time() - execution_time

    def solve_dual_parent_slab(self, index_parent_slab):
        execution_time = time.time()
        # initialize solution matrices for displacement and pressure
        self.parent_slabs[index_parent_slab]["solution"]["dual"]["displacement"] = np.zeros(
            (self.POD["dual"]["displacement"]["basis"].shape[1],
             self.parent_slabs[index_parent_slab]["steps"])
        )
        self.parent_slabs[index_parent_slab]["solution"]["dual"]["pressure"] = np.zeros(
            (self.POD["dual"]["pressure"]["basis"].shape[1],
             self.parent_slabs[index_parent_slab]["steps"])
        )
        # zero initial condition for dual
        old_solution = np.zeros((self.POD["dual"]["displacement"]["basis"].shape[1] + self.POD["dual"]["pressure"]["basis"].shape[1],))

        for i in reversed(
            range(
                self.parent_slabs[index_parent_slab]["start"],
                self.parent_slabs[index_parent_slab]["end"],  # -1 is DEBUG,
            )
        ):
            n = i - self.parent_slabs[index_parent_slab]["start"]
            old_solution = self.solve_dual_time_step_slab(old_solution)
            self.parent_slabs[index_parent_slab]["solution"]["dual"]["displacement"][:,n] = old_solution[
                : self.POD["dual"]["displacement"]["basis"].shape[1]
            ]
            self.parent_slabs[index_parent_slab]["solution"]["dual"]["pressure"][:,n] = old_solution[
                self.POD["dual"]["displacement"]["basis"].shape[1] :
            ] 
        self.timings["dual_ROM"] += time.time() - execution_time

    def error_estimate_parent_slab(self, index_parent_slab):
        execution_time = time.time()
        if self.PLOTTING:
            print("num steps: ", self.parent_slabs[index_parent_slab]["steps"])
        errors = np.zeros((self.parent_slabs[index_parent_slab]["steps"],))
        relative_errors = np.zeros(
            (self.parent_slabs[index_parent_slab]["steps"],))
        self.parent_slabs[index_parent_slab]["functional"] = np.zeros(
            (self.parent_slabs[index_parent_slab]["steps"],)
        )

        for i in range(self.parent_slabs[index_parent_slab]["steps"]):
            # print("Estimating error for time step: ", i)
            solution = np.concatenate(
                (
                    self.parent_slabs[index_parent_slab]["solution"]["primal"]["displacement"][:, i],
                    self.parent_slabs[index_parent_slab]["solution"]["primal"]["pressure"][:, i],
                )
            )

            old_solution = None
            if i >= 1:
                old_solution = np.concatenate(
                    (
                        self.parent_slabs[index_parent_slab]["solution"]["primal"]["displacement"][:, i - 1],
                        self.parent_slabs[index_parent_slab]["solution"]["primal"]["pressure"][:, i - 1],
                    )
                )
            else:
                old_solution = self.parent_slabs[index_parent_slab]["initial_condition"]

            dual_sol = np.concatenate(
                (
                    self.parent_slabs[index_parent_slab]["solution"]["dual"]["displacement"][:, i - 1],
                    self.parent_slabs[index_parent_slab]["solution"]["dual"]["pressure"][:, i - 1],
                )
            )

            # - A*U^n + F + B*U^{n-1}
            AU = self.matrix["estimator"]["system_matrix"].dot(solution)
            BU_old = self.matrix["estimator"]["rhs_matrix"].dot(old_solution)
            primal_res = -AU + self.vector["estimator"]["traction"] + BU_old

            errors[i] = np.dot(dual_sol, primal_res)

            self.parent_slabs[index_parent_slab]["functional"][i] = self.vector["primal"]["pressure_down"].dot(
                self.parent_slabs[index_parent_slab]["solution"]["primal"]["pressure"][:, i]
            ) * self.fom.dt

            # check if dividing by zero
            if np.abs(errors[i] + self.parent_slabs[index_parent_slab]["functional"][i]) < 1e-12:
                relative_errors[i] = errors[i] / 1e-12
            else:
                relative_errors[i] = errors[i] / (
                    errors[i] +
                    self.parent_slabs[index_parent_slab]["functional"][i]
                )    

            # total relative error on entire parent slab
            slab_relative_error = np.abs(np.sum(errors) / np.sum(errors + self.parent_slabs[index_parent_slab]["functional"]))

        # # ORIGINAL
        i_max = np.argmax(np.abs(relative_errors))
        i_max_abs =  np.argmax(np.abs(errors))
        # print(f"First estim: {relative_errors[0:10]}")
        # print(f"i_max: {i_max}")
        error = {
            "relative": relative_errors,
            "absolute": errors,
            "max": np.abs(relative_errors[i_max]),
            "i_max": i_max,
            "max_abs": np.abs(errors[i_max_abs]),
            "i_max_abs": i_max_abs,
            "slab_relative_error": slab_relative_error
        }
        self.timings["error_estimate"] += time.time() - execution_time
        return error

    def enrich_parent_slab(self, index_parent_slab, i_max):
        # COMMENT: Right now I have implemented without forward and backward of primal and dual
        #          with new solution to get a better last solution for dual

        execution_time = time.time()
        # print(self.parent_slabs[index_parent_slab]["initial_condition"].shape)
        projected_last_initial = np.concatenate(    
            (
                self.project_vector(
                    self.parent_slabs[index_parent_slab]["initial_condition"][:self.POD["primal"]["displacement"]["basis"].shape[1]],
                    type="primal",
                    quantity="displacement"
                ),
                self.project_vector(
                    self.parent_slabs[index_parent_slab]["initial_condition"][self.POD["primal"]["displacement"]["basis"].shape[1]:],
                    type="primal",
                    quantity="pressure"
                ),
            )
        )

        # DEBUG WITH FULL IC
        projected_last_initial = self.DEBUG_FULL_IC

        if i_max == 0:
            # if we need to enrich at first step choose IC as last solution
            last_solution = projected_last_initial
        else:
            last_solution = np.concatenate( 
                (
                    self.project_vector(
                        self.parent_slabs[index_parent_slab]["solution"]["primal"]["displacement"][:, i_max - 1],
                        type="primal",
                        quantity="displacement"
                    ),
                    self.project_vector(
                        self.parent_slabs[index_parent_slab]["solution"]["primal"]["pressure"][:, i_max - 1],
                        type="primal",
                        quantity="pressure"
                    ),
                )
            )
            
        # enrich primal
        new_solution = self.fom.solve_primal_time_step(
            last_solution[:self.fom.dofs["displacement"]], 
            last_solution[self.fom.dofs["displacement"]:]
        )
        
        for i, quantity in enumerate(self.fom.quantities):
            self.iPOD(new_solution[i], type="primal", quantity=quantity)
        self.update_matrices(matrix_type="primal")

        # enrich dual
        # check if dual[i_max] is already last solution if
        # true: use zeros
        # false: use dual[i_max+1]
        last_dual_solution = None
        if i_max == self.parent_slabs[index_parent_slab]["steps"] - 1:
            last_dual_solution = np.zeros(
                (self.fom.dofs["displacement"] + self.fom.dofs["pressure"],)
            )
        else:
            last_dual_solution = np.concatenate(
                (
                    self.project_vector(
                        self.parent_slabs[index_parent_slab]["solution"]["dual"]["displacement"][:, i_max + 1],
                        type="dual",
                        quantity="displacement"
                    ),
                    self.project_vector(
                        self.parent_slabs[index_parent_slab]["solution"]["dual"]["pressure"][:, i_max + 1],
                        type="dual",
                        quantity="pressure"
                    ),  
                )
            )
            
        new_dual_solution = self.fom.solve_dual_time_step(
            last_dual_solution[:self.fom.dofs["displacement"]],
            last_dual_solution[self.fom.dofs["displacement"]:]
        )
        
        for i, quantity in enumerate(self.fom.quantities):
            self.iPOD(new_dual_solution[i], type="dual", quantity=quantity)
        self.update_matrices(matrix_type="dual")

        # update estimate components
        self.update_matrices(matrix_type="estimator")

        # update the initial condition to new basis
        self.parent_slabs[index_parent_slab]["initial_condition"] = np.concatenate(
            (
                self.reduce_vector(
                    projected_last_initial[:self.fom.dofs["displacement"]], type="primal", quantity="displacement"
                ),
                self.reduce_vector(
                    projected_last_initial[self.fom.dofs["displacement"]:], type="primal", quantity="pressure"
                ),
            )
        )
        self.timings["enrichment"] += time.time() - execution_time


    # relative error:
    # \sum_i {eta_i / (eta_i + J(u_r)_i)}

    # Gegenvorschlag:
    # \sum_i {eta_i / \sum_j (eta_j + J(u_r)_j)}

    def run_parent_slab(self):
        execution_time = time.time()
        self.init_POD()

        # update reduced matrices
        self.update_matrices("primal")
        self.update_matrices("dual")
        self.update_matrices("estimator")

        iteration = 1
        self.fom_solves = 0
        max_error_iteration = []
        self.parent_slabs[0]["initial_condition"] = np.zeros(
            (self.POD["primal"]["displacement"]["basis"].shape[1]\
             + self.POD["primal"]["pressure"]["basis"].shape[1] ,)
        )

        self.DEBUG_FULL_IC = np.zeros((self.POD["primal"]["displacement"]["basis"].shape[0]\
                                    + self.POD["primal"]["pressure"]["basis"].shape[0],))

        for index_ps, parent_slab in enumerate(self.parent_slabs):
            print(
                f"====== PARENT SLAB: {index_ps} n=({self.parent_slabs[index_ps]['start']}, {self.parent_slabs[index_ps]['end']}) , t=({self.fom.time_points[self.parent_slabs[index_ps]['start']]:.2}, {self.fom.time_points[np.min([self.parent_slabs[index_ps]['end'],len(self.fom.time_points)-1])]:.2}) ======"
            )
            while iteration <= self.MAX_ITERATIONS:
                if self.PLOTTING:
                    print(f"====== Iteration: {iteration} ======")
                    print(
                        f"Bases: {self.POD['primal']['displacement']['basis'].shape[1]}+{self.POD['primal']['pressure']['basis'].shape[1]} / {self.POD['dual']['displacement']['basis'].shape[1]} + {self.POD['dual']['pressure']['basis'].shape[1]}"
                    )
                # 1. Solve primal ROM
                self.solve_primal_parent_slab(index_ps)

                # 2. Solve dual ROM
                self.solve_dual_parent_slab(index_ps)

                # 3. Evaluate DWR error estimator
                estimate = self.error_estimate_parent_slab(index_ps)

                max_error_iteration.append(estimate["max"])

                plot_investigation = True

                if plot_investigation:
                    self.iterations_infos[index_ps]["error"].append(estimate["slab_relative_error"])
                    self.iterations_infos[index_ps]["POD_size"]["primal"]["displacement"].append(self.POD["primal"]["displacement"]["basis"].shape[1])
                    self.iterations_infos[index_ps]["POD_size"]["primal"]["pressure"].append(self.POD["primal"]["pressure"]["basis"].shape[1])
                    self.iterations_infos[index_ps]["POD_size"]["dual"]["displacement"].append(self.POD["dual"]["displacement"]["basis"].shape[1])
                    self.iterations_infos[index_ps]["POD_size"]["dual"]["pressure"].append(self.POD["dual"]["pressure"]["basis"].shape[1])
                    self.iterations_infos[index_ps]["functional"].append(np.sum(self.parent_slabs[index_ps]["functional"]))

                if self.PLOTTING:
                    print(f"Termination variable: {estimate['slab_relative_error']:.5}")

                # 4. If relative error is too large, then solve primal and dual FOM on
                # time step with largest error
                # if estimate["max"] <= self.REL_ERROR_TOL:
                if estimate["slab_relative_error"] <= self.REL_ERROR_TOL:
                    break
                else:
                    # import random
                    # estimate['i_max'] = random.randint(0, self.parent_slabs[index_ps]["steps"] - 1)
                    
                    worst_index =  estimate["i_max_abs"] # estimate['i_max']
                    
                    if self.PLOTTING:
                        print(
                            f"Enrich for largest error @ (i={worst_index}, t={self.fom.time_points[worst_index + self.parent_slabs[index_ps]['start']]:.2}): {estimate['max_abs']:.5}"
                        )
                    self.enrich_parent_slab(index_ps, worst_index)
                    self.fom_solves += 2

                iteration += 1
            iteration = 1
            print("\n")
            if index_ps < len(self.parent_slabs) - 1:
                self.parent_slabs[index_ps + 1]["initial_condition"] = np.concatenate(
                    (
                        self.parent_slabs[index_ps]["solution"]["primal"]["displacement"][:, -1],
                        self.parent_slabs[index_ps]["solution"]["primal"]["pressure"][:, -1],
                    )
                )
                
                # DEBUG by trying full IC
                self.DEBUG_FULL_IC = np.concatenate(
                    (
                        self.project_vector(
                            self.parent_slabs[index_ps]["solution"]["primal"]["displacement"][:, -1],
                            type="primal",
                            quantity="displacement",
                        ),
                        self.project_vector(
                            self.parent_slabs[index_ps]["solution"]["primal"]["pressure"][:, -1],
                            type="primal",
                            quantity="pressure",
                        ),
                    )
                )

        self.validate()

        self.timings["run"] += time.time() - execution_time
        #print(f"Total FOM solves: {self.fom_solves}")

    def save_vtk(self):
        folder = f"paraview/{self.fom.dt}_{self.fom.T}_{self.fom.problem_name}/ROM"

        if not os.path.exists(folder):
            os.makedirs(folder)

        print("Starting saving ROM vtk files...")

        for i, t in list(enumerate(self.fom.time_points))[::50]:
            vtk_displacement = File(f"{folder}/displacement_{str(i)}.pvd")
            vtk_pressure = File(f"{folder}/pressure_{str(i)}.pvd")

            sol_displacement = self.project_vector(
                self.solution["primal"]["displacement"][:, i],
                type="primal",
                quantity="displacement",
            )
            sol_pressure = self.project_vector(
                self.solution["primal"]["pressure"][:, i], type="primal", quantity="pressure"
            )
            u, p = self.fom.U_n.split()

            self.fom.U_n.vector().set_local(
                np.concatenate(
                    (
                        sol_displacement,
                        sol_pressure,
                    )
                )
            )

            u.rename("displacement", "solution")
            p.rename("pressure", "solution")
            vtk_displacement.write(u)
            vtk_pressure.write(p)

        print("Done.")

  