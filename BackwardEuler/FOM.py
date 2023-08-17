import math
import os
import random
import re
import time
from multiprocessing import Pool

import dolfin

# from mshr import *
import matplotlib.pyplot as plt
import numpy as np
import rich.console
import rich.table
import scipy
from dolfin import *


class FOM:
    # constructor
    def __init__(self, t, T, dt, problem):
        self.t = t
        self.T = T
        self.dt = dt
        self.problem = problem

        # for each variable of the object problem, add this variable to the FOM
        for key, value in problem.__dict__.items():
            setattr(self, key, value)

        self.time_points = np.arange(self.t, self.T + self.dt, self.dt)
        print(f"FIRST/LATEST TIME POINT:    {self.time_points[0]}/{self.time_points[-1]}")
        print(f"NUMBER OF TIME POINTS:      {self.time_points.shape[0]}")

        # get class name of problem
        self.problem_name = problem.__class__.__name__
        print(f"Problem name: {self.problem_name}")

        self.mesh = None
        if self.problem_name == "Mandel":
            self.mesh = RectangleMesh(Point(0.0, 0.0), Point(100.0, 20.0), 16, 16)
            self.dim = self.mesh.geometry().dim()

            # plt.title("Mesh")
            # plot(self.mesh)
            # plt.show()
        else:
            raise NotImplementedError("Only Mandel problem implemented so far.")

        element = {
            "u": VectorElement("Lagrange", self.mesh.ufl_cell(), 2),
            "p": FiniteElement("Lagrange", self.mesh.ufl_cell(), 1),
        }
        self.V = FunctionSpace(self.mesh, MixedElement(*element.values()))

        self.dofs = {
            "displacement": self.V.sub(0).dim(),
            "pressure": self.V.sub(1).dim(),
            "time": self.time_points.shape[0],
        }

        if self.problem_name == "Mandel":
            left = CompiledSubDomain("near(x[0], 0.) && on_boundary")
            right = CompiledSubDomain("near(x[0], 100.) && on_boundary")
            down = CompiledSubDomain("near(x[1], 0.) && on_boundary")
            up = CompiledSubDomain("near(x[1], 20.) && on_boundary")

            facet_marker = MeshFunction("size_t", self.mesh, 1)
            facet_marker.set_all(0)
            left.mark(facet_marker, 1)
            right.mark(facet_marker, 2)
            down.mark(facet_marker, 3)
            up.mark(facet_marker, 4)

            self.ds_up = Measure("ds", subdomain_data=facet_marker, subdomain_id=4)
            # boundary for cost functional
            self.ds_down = Measure("ds", subdomain_data=facet_marker, subdomain_id=3)

            bc_left = DirichletBC(self.V.sub(0).sub(0), Constant(0.0), left)  # left: u_x = 0
            bc_right = DirichletBC(self.V.sub(1), Constant(0.0), right)  # right:  p = 0
            bc_down = DirichletBC(self.V.sub(0).sub(1), Constant(0.0), down)  # down: u_y = 0
            self.bc = [bc_left, bc_right, bc_down]
        else:
            raise NotImplementedError("Only Mandel problem implemented so far.")

        # Define trial and test functions and function at old time step
        U = TrialFunction(self.V)
        Phi = TestFunction(self.V)
        self.Uh = Function(self.V)  # current solution
        self.U_n = Function(self.V)  # old timestep solution

        # Split functions into velocity and pressure components
        u, p = split(U)
        phi_u, phi_p = split(Phi)
        u_n, p_n = split(self.U_n)

        self.matrix = {
            "primal": {},
        }

        self.vector = {
            "primal": {},
        }

        if self.problem_name == "Mandel":
            # Stress function
            def stress(u):
                return self.lame_coefficient_lambda * div(u) * Identity(
                    self.dim
                ) + self.lame_coefficient_mu * (grad(u) + grad(u).T)

            n = FacetNormal(self.mesh)

            self.matrix["primal"]["stress"] = scipy.sparse.csr_matrix(
                as_backend_type(assemble(inner(stress(u), grad(phi_u)) * dx))
                .mat()
                .getValuesCSR()[::-1],
                shape=(
                    self.dofs["displacement"] + self.dofs["pressure"],
                    self.dofs["displacement"] + self.dofs["pressure"],
                ),
            )[: self.dofs["displacement"], : self.dofs["displacement"]]

            self.matrix["primal"]["elasto_pressure"] = scipy.sparse.csr_matrix(
                as_backend_type(
                    assemble(
                        -self.alpha_biot * inner(p * Identity(self.dim), grad(phi_u)) * dx
                        + self.alpha_biot * inner(p * n, phi_u) * self.ds_up
                    )
                )
                .mat()
                .getValuesCSR()[::-1],
                shape=(
                    self.dofs["displacement"] + self.dofs["pressure"],
                    self.dofs["displacement"] + self.dofs["pressure"],
                ),
            )[: self.dofs["displacement"], self.dofs["displacement"] :]

            self.matrix["primal"]["laplace"] = scipy.sparse.csr_matrix(
                as_backend_type(
                    assemble(
                        dt * (self.K_biot / self.viscosity_biot) * inner(grad(p), grad(phi_p)) * dx
                    )
                )
                .mat()
                .getValuesCSR()[::-1],
                shape=(
                    self.dofs["displacement"] + self.dofs["pressure"],
                    self.dofs["displacement"] + self.dofs["pressure"],
                ),
            )[self.dofs["displacement"] :, self.dofs["displacement"] :]

            self.matrix["primal"]["time_pressure"] = scipy.sparse.csr_matrix(
                as_backend_type(assemble(self.c_biot * p * phi_p * dx)).mat().getValuesCSR()[::-1],
                shape=(
                    self.dofs["displacement"] + self.dofs["pressure"],
                    self.dofs["displacement"] + self.dofs["pressure"],
                ),
            )[self.dofs["displacement"] :, self.dofs["displacement"] :]

            self.matrix["primal"]["time_displacement"] = scipy.sparse.csr_matrix(
                as_backend_type(assemble(self.alpha_biot * div(u) * phi_p * dx))
                .mat()
                .getValuesCSR()[::-1],
                shape=(
                    self.dofs["displacement"] + self.dofs["pressure"],
                    self.dofs["displacement"] + self.dofs["pressure"],
                ),
            )[self.dofs["displacement"] :, : self.dofs["displacement"]]

            self.vector["primal"]["traction"] = np.array(
                assemble(Constant(self.traction_y_biot) * Phi[1] * self.ds_up)
            )[: self.dofs["displacement"]]

            # pressure vector for cost functional (paper JR p. 25)
            self.vector["primal"]["pressure_down"] = np.array(
                assemble(Phi[self.dim] * self.ds_down)
            )[self.dofs["displacement"] :]

            # build system matrix
            self.matrix["primal"]["system_matrix"] = scipy.sparse.csr_matrix(
                (
                    self.dofs["displacement"] + self.dofs["pressure"],
                    self.dofs["displacement"] + self.dofs["pressure"],
                )
            )
            self.matrix["primal"]["system_matrix"][
                : self.dofs["displacement"], : self.dofs["displacement"]
            ] = self.matrix["primal"]["stress"]
            self.matrix["primal"]["system_matrix"][
                : self.dofs["displacement"], self.dofs["displacement"] :
            ] = self.matrix["primal"]["elasto_pressure"]
            self.matrix["primal"]["system_matrix"][
                self.dofs["displacement"] :, self.dofs["displacement"] :
            ] = self.matrix["primal"]["laplace"]
            self.matrix["primal"]["system_matrix"][
                self.dofs["displacement"] :, : self.dofs["displacement"]
            ] = self.matrix["primal"]["time_displacement"]
            self.matrix["primal"]["system_matrix"][
                self.dofs["displacement"] :, self.dofs["displacement"] :
            ] += self.matrix["primal"]["time_pressure"]
            # apply BC
            self.boundary_dof_vector = np.zeros(
                (self.dofs["displacement"] + self.dofs["pressure"],)
            )
            for _bc in self.bc:
                for i, val in _bc.get_boundary_values().items():
                    assert val == 0.0, "Only homogeneous Dirichlet BCs are supported so far."
                    self.boundary_dof_vector[i] = 1.0
            # apply homogeneous Dirichlet BC to system matrix
            self.matrix["primal"]["system_matrix"] = self.matrix["primal"][
                "system_matrix"
            ].multiply((1.0 - self.boundary_dof_vector).reshape(-1, 1)) + scipy.sparse.diags(
                self.boundary_dof_vector
            )
            self.solve_factorized_primal = scipy.sparse.linalg.factorized(
                self.matrix["primal"]["system_matrix"].tocsc()
            )  # NOTE: LU factorization is dense

            # build rhs matrix
            self.matrix["primal"]["rhs_matrix"] = scipy.sparse.csr_matrix(
                (
                    self.dofs["displacement"] + self.dofs["pressure"],
                    self.dofs["displacement"] + self.dofs["pressure"],
                )
            )
            self.matrix["primal"]["rhs_matrix"][
                self.dofs["displacement"] :, self.dofs["displacement"] :
            ] = self.matrix["primal"]["time_pressure"]
            self.matrix["primal"]["rhs_matrix"][
                self.dofs["displacement"] :, : self.dofs["displacement"]
            ] = self.matrix["primal"]["time_displacement"]

            # get all x-displacement DoFs and coordinates at bottom boundary
            self.bottom_dofs_u = []
            self.bottom_x_u = []
            for i, dof in enumerate(self.V.tabulate_dof_coordinates()[: self.V.sub(0).sub(0).dim()]):
                if dof[1] == 0.0:
                    self.bottom_dofs_u.append(i)
                    self.bottom_x_u.append(dof[0])

            # sort bottom_dofs_u and bottom_x_u by x-coordinate
            self.bottom_dofs_u, self.bottom_x_u = zip(
               *sorted(zip(self.bottom_dofs_u, self.bottom_x_u), key=lambda x: x[1])
            ) 

            self.bottom_matrix_u = scipy.sparse.csr_matrix(
                (
                    len(self.bottom_dofs_u),
                    self.dofs["displacement"],
                )
            )
            for i, dof in enumerate(self.bottom_dofs_u):
                self.bottom_matrix_u[i, dof] = 1.0

            # get all pressure DoFs and coordinates at bottom boundary
            self.bottom_dofs_p = []
            self.bottom_x_p = []
            for i, dof in enumerate(self.V.tabulate_dof_coordinates()[self.dofs["displacement"] :]):
                if dof[1] == 0.0:
                    self.bottom_dofs_p.append(i)
                    self.bottom_x_p.append(dof[0])
            self.bottom_matrix_p = scipy.sparse.csr_matrix(
                (
                    len(self.bottom_dofs_p),
                    self.dofs["pressure"],
                )
            )
            for i, dof in enumerate(self.bottom_dofs_p):
                self.bottom_matrix_p[i, dof] = 1.0

            # times for plotting solution at bottom boundary
            self.special_times = [1000., 5000., 10000., 100000.,  500000., 5000000.]
        else:
            raise NotImplementedError("Only Mandel problem implemented so far.")

        # define snapshot matrix
        self.Y = {
            "displacement": np.zeros((self.dofs["displacement"], self.dofs["time"])),
            "pressure": np.zeros((self.dofs["pressure"], self.dofs["time"])),
        }

        # define functional values
        self.functional_values = np.zeros((self.dofs["time"] - 1,))

        # IO data
        self.SAVE_DIR = "results/"

    def save_solution(self):
        pattern = r"solution_\d{6}\.npz"
        files = os.listdir(self.SAVE_DIR)
        files = [
            self.SAVE_DIR + f
            for f in files
            if os.path.isfile(os.path.join(self.SAVE_DIR, f)) and re.match(pattern, f)
        ]

        parameters = np.array(
            [
                self.dt,
                self.T,
                self.problem_name,
                *[float(x) for x in self.problem.__dict__.values()],
            ]
        )

        for file in files:
            tmp = np.load(file, allow_pickle=True)
            if np.array_equal(parameters, tmp["parameters"]):
                np.savez(
                    file,
                    displacement=self.Y["displacement"],
                    pressure=self.Y["pressure"],
                    parameters=parameters,
                    compression=True,
                )
                print(f"Overwrite {file}")
                return

        file_name = "results/solution_" + str(len(files)).zfill(6) + ".npz"
        np.savez(
            file_name,
            displacement=self.Y["displacement"],
            pressure=self.Y["pressure"],
            parameters=parameters,
            compression=True,
        )
        print(f"Saved as {file_name}")

    def load_solution(self):
        pattern = r"solution_\d{6}\.npz"

        # check if self.SAVE_DIR exists
        if not os.path.exists(self.SAVE_DIR):
            os.makedirs(self.SAVE_DIR)
            return False

        files = os.listdir(self.SAVE_DIR)
        files = [
            self.SAVE_DIR + f
            for f in files
            if os.path.isfile(os.path.join(self.SAVE_DIR, f)) and re.match(pattern, f)
        ]

        parameters = np.array(
            [
                self.dt,
                self.T,
                self.problem_name,
                *[float(x) for x in self.problem.__dict__.values()],
            ]
        )

        for file in files:
            tmp = np.load(file)
            if np.array_equal(parameters, tmp["parameters"]):
                self.Y["displacement"] = tmp["displacement"]
                self.Y["pressure"] = tmp["pressure"]
                print(f"Loaded {file}")
                return True
        return False

    def solve_functional_trajectory(self):
        self.functional_values = np.zeros((self.dofs["time"] - 1,))
        for i in range(self.dofs["time"] - 1):
            self.functional_values[i] = self.vector["primal"]["pressure_down"].dot(
                self.Y["pressure"][:, i + 1]
            )

        self.functional = np.sum(self.functional_values) * self.dt
        # print cost functional in scientific notation
        # print(f"Cost functional: {self.functional:.4e}")

        # # print cost functional trajectory
        # plt.plot(self.time_points[1:], self.functional_values)
        # plt.show()

    def plot_bottom_solution(self):
        times = self.special_times.copy()
        for i, t in enumerate(self.time_points):
            if np.abs(t-times[0]) <= 1e-4:
                times.pop(0)
                plt.plot(self.bottom_x_u, self.bottom_matrix_u.dot(self.Y["displacement"][:, i]), label=f"t = {t}")
                if len(times) == 0:
                    break
        plt.xlabel("x")
        plt.ylabel(r"$u_x(x,0)$")
        plt.title("x-Displacement at bottom boundary")
        plt.legend()
        plt.show()

        times = self.special_times.copy()
        for i, t in enumerate(self.time_points):
            if np.abs(t-times[0]) <= 1e-4:
                times.pop(0)
                plt.plot(self.bottom_x_p, self.bottom_matrix_p.dot(self.Y["pressure"][:, i]), label=f"t = {t}")
                if len(times) == 0:
                    break
        plt.xlabel("x")
        plt.ylabel(r"$p(x,0)$")
        plt.title("Pressure at bottom boundary")
        plt.legend()
        plt.show()

    # Solve one time_step
    def solve_primal_time_step(self, u_n_vector, p_n_vector):
        old_solution = np.concatenate((u_n_vector, p_n_vector))

        rhs = self.matrix["primal"]["rhs_matrix"].dot(old_solution)
        rhs[: self.dofs["displacement"]] += self.vector["primal"]["traction"]

        # apply homogeneous Dirichlet BC to right hand side
        rhs = rhs * (1.0 - self.boundary_dof_vector)

        solution = self.solve_factorized_primal(rhs)

        return solution[: self.dofs["displacement"]], solution[self.dofs["displacement"] :]

    def solve_dual_time_step(self, u_n_vector, z_n_vector):
        pass  # todo

    # Solve time trajectory
    def solve_primal(self, force_recompute=False):
        if not force_recompute:
            if self.load_solution():
                return

        # zero initial condition
        self.Y["displacement"][:, 0] = np.zeros((self.dofs["displacement"],))
        self.Y["pressure"][:, 0] = np.zeros((self.dofs["pressure"],))

        for i, t in enumerate(self.time_points[1:]):
            n = i + 1
            print(f"\n\nt = {round(t,5)}:\n===============")
            self.Y["displacement"][:, n], self.Y["pressure"][:, n] = self.solve_primal_time_step(
                self.Y["displacement"][:, n - 1], self.Y["pressure"][:, n - 1]
            )

        self.save_solution()
        self.save_vtk()

    def save_vtk(self):
        folder = f"paraview/{self.dt}_{self.T}_{self.problem_name}/FOM"
        # check if folder exists else create it 
        if not os.path.exists(folder):
            os.makedirs(folder)


        print("Starting saving FOM vtk files...")

        if not os.path.exists(folder):
            os.makedirs(folder)


        # only each 10th time step
        for i, t in list(enumerate(self.time_points))[::10]:
            vtk_displacement = File(f"{folder}/displacement_{str(i)}.pvd")
            vtk_pressure = File(f"{folder}/pressure_{str(i)}.pvd")

            u, p = self.U_n.split()

            self.U_n.vector().set_local(np.concatenate(
                (
                    self.Y["displacement"][:, i],
                    self.Y["pressure"][:, i]
                )
            ))

            u.rename("displacement", "solution")
            p.rename("pressure", "solution")
            vtk_displacement.write(u)
            vtk_pressure.write(p)
        print("Done.")
