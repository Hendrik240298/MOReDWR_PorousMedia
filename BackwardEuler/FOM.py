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
            self.mesh = RectangleMesh(Point(0., 0.), Point(100., 20.), 16, 16)
            self.dim = self.mesh.geometry().dim()

            # plt.title("Mesh")
            # plot(self.mesh)
            # plt.show()
        else: 
            raise NotImplementedError("Only Mandel problem implemented so far.")
            
        element = {
            "u" : VectorElement("Lagrange", self.mesh.ufl_cell(), 2),
            "p" : FiniteElement("Lagrange", self.mesh.ufl_cell(), 1),
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

            bc_left = DirichletBC(self.V.sub(0).sub(0), Constant(0.), left) # left: u_x = 0
            bc_right = DirichletBC(self.V.sub(1), Constant(0.), right)      # right:  p = 0
            bc_down = DirichletBC(self.V.sub(0).sub(1), Constant(0.), down) # down: u_y = 0
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
                return self.lame_coefficient_lambda*div(u)*Identity(self.dim) + self.lame_coefficient_mu*(grad(u) + grad(u).T)

            n = FacetNormal(self.mesh)

            self.matrix["primal"]["stress"] = scipy.sparse.csr_matrix(
                as_backend_type(assemble(inner(stress(u), grad(phi_u))*dx)).mat().getValuesCSR()[::-1],
                shape=(
                    self.dofs["displacement"] + self.dofs["pressure"],
                    self.dofs["displacement"] + self.dofs["pressure"],
                ),
            )[: self.dofs["displacement"], : self.dofs["displacement"]]

            self.matrix["primal"]["elasto_pressure"] = scipy.sparse.csr_matrix(
                as_backend_type(assemble(- self.alpha_biot*inner(p*Identity(self.dim), grad(phi_u))*dx + self.alpha_biot*inner(p*n, phi_u)*self.ds_up)).mat().getValuesCSR()[::-1],
                shape=(
                    self.dofs["displacement"] + self.dofs["pressure"],
                    self.dofs["displacement"] + self.dofs["pressure"],
                ),
            )[: self.dofs["displacement"], self.dofs["displacement"] : ]

            self.matrix["primal"]["laplace"] = scipy.sparse.csr_matrix(
                as_backend_type(assemble(dt*(self.K_biot/self.viscosity_biot)*inner(grad(p), grad(phi_p))*dx)).mat().getValuesCSR()[::-1],
                shape=(
                    self.dofs["displacement"] + self.dofs["pressure"],
                    self.dofs["displacement"] + self.dofs["pressure"],
                ),
            )[self.dofs["displacement"] :, self.dofs["displacement"] :]

            self.matrix["primal"]["time_pressure"] = scipy.sparse.csr_matrix(
                as_backend_type(assemble(self.c_biot*p*phi_p*dx)).mat().getValuesCSR()[::-1],
                shape=(
                    self.dofs["displacement"] + self.dofs["pressure"],
                    self.dofs["displacement"] + self.dofs["pressure"],
                ),
            )[self.dofs["displacement"] :, self.dofs["displacement"] :]

            self.matrix["primal"]["time_displacement"] = scipy.sparse.csr_matrix(
                as_backend_type(assemble(self.alpha_biot*div(u)*phi_p*dx)).mat().getValuesCSR()[::-1],
                shape=(
                    self.dofs["displacement"] + self.dofs["pressure"],
                    self.dofs["displacement"] + self.dofs["pressure"],
                ),
            )[self.dofs["displacement"] :, : self.dofs["displacement"]]

            self.vector["primal"]["traction"] = np.array(assemble(Constant(self.traction_y_biot)*Phi[1]*self.ds_up))[: self.dofs["displacement"]]

            # build system matrix
            self.matrix["primal"]["system_matrix"] = scipy.sparse.csr_matrix(
                (
                    self.dofs["displacement"] + self.dofs["pressure"],
                    self.dofs["displacement"] + self.dofs["pressure"],
                )
            )
            self.matrix["primal"]["system_matrix"][: self.dofs["displacement"], : self.dofs["displacement"]] = self.matrix["primal"]["stress"]
            self.matrix["primal"]["system_matrix"][: self.dofs["displacement"], self.dofs["displacement"] :] = self.matrix["primal"]["elasto_pressure"]
            self.matrix["primal"]["system_matrix"][self.dofs["displacement"] :, self.dofs["displacement"] :] = self.matrix["primal"]["laplace"]
            self.matrix["primal"]["system_matrix"][self.dofs["displacement"] :, : self.dofs["displacement"]] = self.matrix["primal"]["time_displacement"]
            self.matrix["primal"]["system_matrix"][self.dofs["displacement"] :, self.dofs["displacement"] :] += self.matrix["primal"]["time_pressure"]
            # apply BC
            self.boundary_dof_vector = np.zeros((self.dofs["displacement"]+self.dofs["pressure"],))
            for _bc in self.bc:
                for i, val in _bc.get_boundary_values().items():
                    assert val == 0., "Only homogeneous Dirichlet BCs are supported so far."
                    self.boundary_dof_vector[i] = 1.
            self.matrix["primal"]["system_matrix"] = self.matrix["primal"]["system_matrix"].multiply((1.-self.boundary_dof_vector).reshape(-1,1)) + scipy.sparse.diags(self.boundary_dof_vector)
            self.solve_factorized_primal = scipy.sparse.linalg.factorized(self.matrix["primal"]["system_matrix"].tocsc())

            # build rhs matrix
            self.matrix["primal"]["rhs_matrix"] = scipy.sparse.csr_matrix(
                (
                    self.dofs["displacement"] + self.dofs["pressure"],
                    self.dofs["displacement"] + self.dofs["pressure"],
                )
            )
            self.matrix["primal"]["rhs_matrix"][self.dofs["displacement"] :, self.dofs["displacement"] :] = self.matrix["primal"]["time_pressure"]
            self.matrix["primal"]["rhs_matrix"][self.dofs["displacement"] :, : self.dofs["displacement"]] = self.matrix["primal"]["time_displacement"]
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

        parameters = np.array([self.dt, self.T, self.problem_name, *[float(x) for x in self.problem.__dict__.values()]])

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

        parameters = np.array([self.dt, self.T, self.problem_name, *[float(x) for x in self.problem.__dict__.values()]])

        for file in files:
            tmp = np.load(file)
            if np.array_equal(parameters, tmp["parameters"]):
                self.Y["displacement"] = tmp["displacement"]
                self.Y["pressure"] = tmp["pressure"]
                print(f"Loaded {file}")
                return True
        return False

    def solve_functional_trajectory(self):
        pass  # TODO

    # assemble system matrices and rhs
    def assemble_system(self, force_recompute=False):
        pass  # TODO

    def assemble_lifting_matrices(self, lifting):
        solution = np.concatenate(
            (
                lifting["velocity"],
                np.zeros((self.dofs["pressure"],)),
            )
        )
        U_lifting = Function(self.V)  # lifting function
        U_lifting.vector().set_local(solution)
        dv, _ = split(U_lifting)

        U = TrialFunction(self.V)
        Phi_U = TestFunctions(self.V)

        # Split functions into velocity and pressure components
        v, _ = split(U)
        phi_v, _ = Phi_U

        # assemble Frechet derivative C^l := C(l)
        self.lifting_matrix = scipy.sparse.csr_matrix(
            as_backend_type(
                assemble(dot(dot(grad(v), dv) + dot(grad(dv), v), phi_v) * dx)
            )
            .mat()
            .getValuesCSR()[::-1],
            shape=(
                self.dofs["velocity"] + self.dofs["pressure"],
                self.dofs["velocity"] + self.dofs["pressure"],
            ),
        )[: self.dofs["velocity"], : self.dofs["velocity"]]

        # assemble convection term evaluated at lifting
        self.lifting_rhs = (
            # DEBUG HF: remove duplicate
            # -np.array(
            #     self.dt * self.nu * self.matrix["primal"]["laplace"].dot(lifting["velocity"]),
            #     dtype=np.float64,
            # )
            -self.dt
            * np.array(
                assemble(
                    Constant(self.nu) * inner(grad(dv), grad(phi_v)) * dx
                    + dot(dot(grad(dv), dv), phi_v) * dx
                )
            )[: self.dofs["velocity"]]
        )
        print("Assembled lifting matrices + rhs")

    def assemble_linear_operators(self):
        self.velocity_lin_operator_theta = (
            self.matrix["primal"]["mass"]
            + float(self.dt * self.theta * self.nu) * self.matrix["primal"]["laplace"]
            + float(self.dt * self.theta) * self.lifting_matrix
        )
        self.velocity_lin_operator_one_minus_theta = (
            self.matrix["primal"]["mass"]
            - float(self.dt * (1.0 - self.theta) * self.nu) * self.matrix["primal"]["laplace"]
            - float(self.dt * (1.0 - self.theta)) * self.lifting_matrix
        )
        self.pressure_lin_operator = -self.dt * self.matrix["primal"]["pressure"]

        print("Assembled linear operators")

    # Solve one time_step
    def solve_primal_time_step(self, u_n_vector, p_n_vector):
        old_solution = np.concatenate((u_n_vector, p_n_vector))
        
        rhs = self.matrix["primal"]["rhs_matrix"].dot(old_solution)
        rhs[: self.dofs["displacement"]] += self.vector["primal"]["traction"]

        # apply homogeneous Dirichlet BC to right hand side
        rhs = rhs * (1. - self.boundary_dof_vector)
        
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

        print("Starting saving vtk files...")
        print("   TODO: save pressure as vtk")

        if not os.path.exists(folder):
            os.makedirs(folder)

        xdmffile_u = XDMFFile(f"{folder}/displacement.xdmf")

        for i, t in enumerate(self.time_points):
            vtk_displacement = File(f"{folder}/displacement_{str(i)}.pvd")
            vtk_pressure = File(f"{folder}/pressure_{str(i)}.pvd")

            # self.U_n.vector().set_local(solution)
            u, p = self.U_n.split()
            # v.vector().vec()[:self.dofs["velocity"]] = self.Y["velocity"][:, i]

            u.vector().set_local(self.Y["displacement"][:, i])
            p.vector().set_local(self.Y["pressure"][:, i])

            #
            xdmffile_u.write(u, t)

            # c = plot(sqrt(dot(v, v)), title="Velocity")
            # plt.colorbar(c, orientation="horizontal")
            # plt.show()

            u.rename("displacement", "solution")
            p.rename("pressure", "solution")
            vtk_displacement.write(u)

            # v.rename('velocity', 'solution')
            # p.rename('pressure', 'solution')

            # vtk_velocity << v
            # vtk_pressure << p

            # vtkfile.write(v, "velocity")
            # # vtkfile.write(p, "pressure")
        
        print("Done.")
