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


class ROM:
    # constructor
    def __init__(
        self,
        fom,
        REL_ERROR_TOL=1e-2,
        MAX_ITERATIONS=100,
        TOTAL_ENERGY={
            "primal": {"displacement": 1, "pressure": 1},
        },
    ):
        self.fom = fom
        self.REL_ERROR_TOL = REL_ERROR_TOL
        self.MAX_ITERATIONS = MAX_ITERATIONS

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
        }

        self.matrix = {
            "primal": {"system_matrix": None, "rhs_matrix": None},
        }
        self.vector = {
            "primal": {"traction": None},
        }

        self.solution = {"primal": {"displacement": None, "pressure": None}}
        self.functional_values = np.zeros((self.fom.dofs["time"] - 1,))

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

    # TODO: consider also supremizer here
    def reduce_matrix(self, matrix, type, quantity0, quantity1):
        """
        OLD:
            |   A_uu    |   A_up    |
        A = | --------- | --------- |
            |   A_pu    |   A_pp    |

        NEW: (is always better - B.S.)

        A_N = Z_{q0}^T A Z_{q1}

        REMARK:
        - A_N is reduced submatrix, c.f. OLD A
        """

        reduced_matrix = self.POD[type][quantity0]["basis"].T.dot(
            matrix.dot(self.POD[type][quantity1]["basis"])
        )
        return reduced_matrix

    def update_matrices(self, type):
        # TODO: update when matrix are known

        # self.

        for key in self.matrix[type].keys():
            self.matrix[type][key] = self.reduce_matrix(self.fom.matrix[type][key], type)

    def update_rhs(self, type):
        self.RHS[type] = np.dot(self.POD[type]["basis"].T, self.fom.RHS)

    def init_POD(self):
        # primal POD
        time_points = self.fom.time_points[:]

        for i, t in enumerate(time_points):
            self.iPOD(self.fom.Y["displacement"][:, i], type="primal", quantity="displacement")
            self.iPOD(self.fom.Y["pressure"][:, i], type="primal", quantity="pressure")

        print("DISPLACEMENT POD size:   ", self.POD["primal"]["displacement"]["basis"].shape[1])
        print("PRESSURE POD size:   ", self.POD["primal"]["pressure"]["basis"].shape[1])

        # for i in range(self.POD["primal"]["velocity"]["basis"].shape[1]):
        #     v, p = self.fom.U_n.split()
        #     v.vector().set_local(self.POD["primal"]["velocity"]["basis"][:,i])
        #     c = plot(sqrt(dot(v, v)), title="Velocity")
        #     plt.colorbar(c, orientation="horizontal")
        #     plt.show()


    def compute_reduced_matrices(self):
        self.vector["primal"]["traction"] = self.reduce_vector(self.fom.vector["primal"]["traction"], "primal", "displacement")

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

        # build system matrix from blocks
        self.matrix["primal"]["system_matrix"] = np.block(
            [
                [matrix_stress, matrix_elasto_pressure],
                [matrix_time_displacement, matrix_time_pressure + matrix_laplace],
            ]
        )

        # TODO: factorize reduced system matrix
        #self.solve_factorized_primal = scipy.sparse.linalg.factorized(self.matrix["primal"]["system_matrix"])

        # build rhs matrix from blocks
        self.matrix["primal"]["rhs_matrix"] = np.block(
            [
                [np.zeros_like(matrix_stress), np.zeros_like(matrix_elasto_pressure)],
                [matrix_time_displacement, matrix_time_pressure],
            ]
        )

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

        for i, t in enumerate(self.fom.time_points[1:]):
            print("#-----------------------------------------------#")
            print(f"t = {t:.4f}")
            n = i + 1

            solution = np.linalg.solve(
                self.matrix["primal"]["system_matrix"],
                self.matrix["primal"]["rhs_matrix"].dot(solution) \
                    + np.concatenate((self.vector["primal"]["traction"], np.zeros((self.POD["primal"]["pressure"]["basis"].shape[1],))))
            )

            self.solution["primal"]["displacement"][:, n] = solution[ : self.POD["primal"]["displacement"]["basis"].shape[1]]
            self.solution["primal"]["pressure"][:, n] = solution[self.POD["primal"]["displacement"]["basis"].shape[1] : ]

            # print("BREAKING ROM LOOP FOR DEBUGGING")
            # break

            """
            if i % 500 == 0:

                            # DEBUG HF: remove lifting to see resiudal
                sol_velocity = self.project_vector(
                    self.solution["primal"]["velocity"][:, i], type="primal", quantity="velocity"
                ) + self.lifting["velocity"]
                sol_pressure = self.project_vector(
                    self.solution["primal"]["pressure"][:, i], type="primal", quantity="pressure"
                )
                
                v, p = self.fom.U_n.split()

                self.fom.U_n.vector().set_local(
                    np.concatenate(
                        (
                            sol_velocity,
                            sol_pressure,
                        )
                    )
                )


                # subplot for velocuty and pressure
                plt.figure(figsize=(8, 5.5))
                plt.subplot(2, 1, 1)
                c = plot(sqrt(dot(v, v)), title=f"Velocity @ t={t:.2}")
                plt.colorbar(c, orientation="horizontal")
                plt.subplot(2, 1, 2)
                c = plot(p, title=f"Pressure @ t={t:.2}")
                plt.colorbar(c, orientation="horizontal")

                plt.show()
            """
                
        # plt.semilogy(self.DEBUG_RES)
        # plt.show()

    def run_parent_slab(self):
        execution_time = time.time()
        self.init_POD()

        # update reduced matrices
        self.update_matrices("primal")
        self.update_matrices("dual")
        self.update_matrices("estimate")

        # update reduced rhs
        self.update_rhs("primal")
        self.update_rhs("dual")

        iteration = 1
        fom_solves = 0
        max_error_iteration = []
        self.parent_slabs[0]["initial_condition"] = np.zeros(
            (self.POD["primal"]["basis"].shape[1],)
        )

        self.DEBUG_FULL_IC = np.zeros((self.POD["primal"]["basis"].shape[0],))

        for index_ps, parent_slab in enumerate(self.parent_slabs):
            print(
                f"====== PARENT SLAB: {index_ps} n=({self.parent_slabs[index_ps]['start']}, {self.parent_slabs[index_ps]['end']}) , t=({self.fom.time_points[self.parent_slabs[index_ps]['start']]:.2}, {self.fom.time_points[np.min([self.parent_slabs[index_ps]['end'],len(self.fom.time_points)-1])]:.2}) ======"
            )
            while iteration <= self.MAX_ITERATIONS:
                print(f"====== Iteration: {iteration} ======")
                print(
                    f"Bases: {self.POD['primal']['basis'].shape[1]} / {self.POD['dual']['basis'].shape[1]}"
                )
                # 1. Solve primal ROM
                self.solve_primal_parent_slab(index_ps)

                # 2. Solve dual ROM
                self.solve_dual_parent_slab(index_ps)

                # 3. Evaluate DWR error estimator
                estimate = self.error_estimate_parent_slab(index_ps)

                max_error_iteration.append(estimate["max"])

                # 4. If relative error is too large, then solve primal and dual FOM on
                # time step with largest error
                if estimate["max"] <= self.REL_ERROR_TOL:
                    break
                else:
                    print(
                        f"Enrich for largest error @ (i={estimate['i_max']}, t={self.fom.time_points[estimate['i_max'] + self.parent_slabs[index_ps]['start']]:.2}): {estimate['max']:.5}"
                    )
                    self.enrich_parent_slab(index_ps, estimate["i_max"])
                    fom_solves += 2

                iteration += 1
            iteration = 1
            print("\n")
            if index_ps < len(self.parent_slabs) - 1:
                self.parent_slabs[index_ps + 1]["initial_condition"] = self.parent_slabs[index_ps][
                    "solution"
                ]["primal"][:, -1]

                # DEBUG by trying full IC
                self.DEBUG_FULL_IC = self.project_vector(
                    self.parent_slabs[index_ps]["solution"]["primal"][:, -1], type="primal"
                )

        self.validate()

        self.timings["run"] += time.time() - execution_time
        print(f"Total FOM solves: {fom_solves}")

    def save_vtk(self):
        folder = f"paraview/{self.fom.dt}_{self.fom.T}_{self.fom.theta}_{float(self.fom.nu)}/ROM"

        if not os.path.exists(folder):
            os.makedirs(folder)

        lifting = self.lifting["velocity"]

        for i, t in list(enumerate(self.fom.time_points))[::10]:
            print(f"PLOT {i}-th solution")
            vtk_velocity = File(f"{folder}/velocity_{str(i)}.pvd")
            vtk_pressure = File(f"{folder}/pressure_{str(i)}.pvd")

            # DEBUG HF: remove lifting to see resiudal
            sol_velocity = self.project_vector(
                self.solution["primal"]["velocity"][:, i], type="primal", quantity="velocity"
            ) + self.lifting["velocity"]
            sol_pressure = self.project_vector(
                self.solution["primal"]["pressure"][:, i], type="primal", quantity="pressure"
            )
            v, p = self.fom.U_n.split()

            self.fom.U_n.vector().set_local(
                np.concatenate(
                    (
                        sol_velocity,
                        sol_pressure,
                    )
                )
            )

            vtk_velocity << v
            vtk_pressure << p

            # subplot for velocuty and pressure
            plt.figure(figsize=(8, 5.5))
            plt.subplot(2, 1, 1)
            c = plot(sqrt(dot(v, v)), title=f"Velocity @ t={t:.2}")
            plt.colorbar(c, orientation="horizontal")
            plt.subplot(2, 1, 2)
            c = plot(p, title=f"Pressure @ t={t:.2}")
            plt.colorbar(c, orientation="horizontal")

            plt.show()


    def compute_drag_lift(self):
        offset = 100

        self.drag_force = np.zeros((self.fom.dofs["time"],))
        self.lift_force = np.zeros((self.fom.dofs["time"],))

        for i, t in list(enumerate(self.fom.time_points)):
            sol_velocity = self.project_vector(
                self.solution["primal"]["velocity"][:, i], type="primal", quantity="velocity"
            ) + self.lifting["velocity"]
            sol_pressure = self.project_vector(
                self.solution["primal"]["pressure"][:, i], type="primal", quantity="pressure"
            )
            self.drag_force[i], self.lift_force[i] = self.fom.compute_drag_lift_time_step(sol_velocity, sol_pressure)


        # plot results in subplots 
        # fig, ax = plt.subplots(2, 1, figsize=(10, 10))
        # ax[0].plot(self.fom.time_points[offset:], self.drag_force[offset:], label="drag")
        # ax[0].set_xlabel("time")
        # ax[0].set_ylabel("drag")
        # ax[0].grid()
        # ax[1].plot(self.fom.time_points[offset:], self.lift_force[offset:], label="lift")
        # ax[1].set_xlabel("time")
        # ax[1].set_ylabel("lift")
        # ax[1].grid()
        # plt.show()

        # subplot for velocuty and pressure
        # plt.figure(figsize=(8, 5.5))
        plt.subplot(2, 2, 1)
        plt.plot(self.fom.time_points[offset:], self.drag_force[offset:], label="drag - ROM")
        plt.legend()
        plt.grid()
        plt.subplot(2, 2, 2)
        plt.plot(self.fom.time_points[offset:], self.fom.drag_force[offset:], label="drag - FOM")
        plt.legend()
        plt.grid()
        plt.subplot(2, 2, 3)
        plt.plot(self.fom.time_points[offset:], self.lift_force[offset:], label="lift - ROM")
        plt.legend()
        plt.grid()
        plt.subplot(2, 2, 4)
        plt.plot(self.fom.time_points[offset:], self.fom.lift_force[offset:], label="lidt - FOM")
        plt.legend()
        plt.grid()

        plt.show()
