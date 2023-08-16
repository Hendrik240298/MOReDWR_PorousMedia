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

            plt.title("Mesh")
            plot(self.mesh)
            plt.show()
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

        parameters = np.array([self.dt, self.T, self.theta, float(self.nu)])

        for file in files:
            tmp = np.load(file, allow_pickle=True)
            if np.array_equal(parameters, tmp["parameters"]):
                np.savez(
                    file,
                    velocity=self.Y["velocity"],
                    pressure=self.Y["pressure"],
                    parameters=parameters,
                    compression=True,
                )
                print(f"Overwrite {file}")
                return

        file_name = "results/solution_" + str(len(files)).zfill(6) + ".npz"
        np.savez(
            file_name,
            velocity=self.Y["velocity"],
            pressure=self.Y["pressure"],
            parameters=parameters,
            compression=True,
        )
        print(f"Saved as {file_name}")

    def load_solution(self):
        pattern = r"solution_\d{6}\.npz"
        files = os.listdir(self.SAVE_DIR)
        files = [
            self.SAVE_DIR + f
            for f in files
            if os.path.isfile(os.path.join(self.SAVE_DIR, f)) and re.match(pattern, f)
        ]

        parameters = np.array([self.dt, self.T, self.theta, float(self.nu)])

        for file in files:
            tmp = np.load(file)
            if np.array_equal(parameters, tmp["parameters"]):
                self.Y["velocity"] = tmp["velocity"]
                self.Y["pressure"] = tmp["pressure"]
                print(f"Loaded {file}")

                # for i, _ in enumerate(self.time_points):
                #     v, p = self.U.split()
                #     v.vector().set_local(self.Y["velocity"][:, i])
                #     c = plot(sqrt(dot(v, v)), title="Velocity")
                #     plt.colorbar(c, orientation="horizontal")
                #     plt.show()

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
    def solve_primal_time_step(self, v_n_vector, p_n_vector):
        # Newton update
        dU = Function(self.V)
        # self.U_n.vector()[:] = u_n_vector.
        old_solution = np.concatenate((v_n_vector, p_n_vector))
        self.U_n.vector().set_local(old_solution)
        # Newton table
        newton_table = rich.table.Table(title="Newton solver")
        newton_table.add_column("Step", justify="right")
        newton_table.add_column("Residuum", justify="right")
        newton_table.add_column("Residuum fraction", justify="right")
        newton_table.add_column("Assembled matrix", justify="center")
        newton_table.add_column("Linesearch steps", justify="right")

        # Newton iteration
        system_matrix = None
        system_rhs = assemble(-self.F)
        for _bc in self.bc_homogen:
            _bc.apply(system_rhs)
        newton_residuum = np.linalg.norm(np.array(system_rhs), ord=np.Inf)

        newton_step = 1

        if newton_residuum < self.NEWTON_TOL:
            print(f"Newton residuum: {newton_residuum}")

        # Newton loop
        while newton_residuum > self.NEWTON_TOL and newton_step < self.MAX_N_NEWTON_STEPS:
            old_newton_residuum = newton_residuum

            system_rhs = assemble(-self.F)
            for _bc in self.bc_homogen:
                _bc.apply(system_rhs)
            newton_residuum = np.linalg.norm(np.array(system_rhs), ord=np.Inf)

            if newton_residuum < self.NEWTON_TOL:
                # print(f"Newton residuum: {newton_residuum}")
                newton_table.add_row("-", f"{newton_residuum:.4e}", "-", "-", "-")

                console = rich.console.Console()
                console.print(newton_table)
                break

            if newton_residuum / old_newton_residuum > self.NONLINEAR_RHO:
                # For debugging the derivative of the variational formulation:
                # assert np.max(np.abs((assemble(A)-assemble(J)).array())) == 0., f"Handcoded derivative and auto-diff generated derivative should be the same. Difference is {np.max(np.abs((assemble(A)-assemble(J)).array()))}."
                system_matrix = assemble(self.A)
                for _bc in self.bc_homogen:
                    _bc.apply(system_matrix, system_rhs)

            solve(system_matrix, dU.vector(), system_rhs)

            for linesearch_step in range(self.MAX_N_LINE_SEARCH_STEPS):
                self.U.vector()[:] += dU.vector()[:]

                system_rhs = assemble(-self.F)
                for _bc in self.bc_homogen:
                    _bc.apply(system_rhs)
                new_newton_residuum = np.linalg.norm(np.array(system_rhs), ord=np.Inf)

                if new_newton_residuum < newton_residuum:
                    break
                else:
                    self.U.vector()[:] -= dU.vector()[:]

                dU.vector()[:] *= self.LINE_SEARCH_DAMPING

            assembled_matrix = newton_residuum / old_newton_residuum > self.NONLINEAR_RHO
            # print(f"Newton step: {newton_step} | Newton residuum: {newton_residuum} | Residuum fraction: {newton_residuum/old_newton_residuum } | Assembled matrix: {assembled_matrix} | Linesearch steps: {linesearch_step}")
            newton_table.add_row(
                str(newton_step),
                f"{newton_residuum:.4e}",
                f"{round(newton_residuum/old_newton_residuum, 4):#.4f}",
                str(assembled_matrix),
                str(linesearch_step),
            )
            newton_step += 1

        v, p = self.U.split()

        # c = plot(sqrt(dot(v, v)), title="Velocity")
        # plt.colorbar(c, orientation="horizontal")
        # plt.show()

        print("v shape = ", v.vector().get_local()[: self._V.dim()].shape)
        print("p shape = ", p.vector().get_local()[self._V.dim() :].shape)
        return v.vector().get_local()[: self._V.dim()], p.vector().get_local()[self._V.dim() :]

    def solve_dual_time_step(self, u_n_vector, z_n_vector):
        pass  # todo

    # Solve time trajectory
    def solve_primal(self, force_recompute=False):
        if not force_recompute:
            if self.load_solution():
                return

        # reset solution
        self.U.vector()[:] = 0.0
        # apply boundary conditions to initial condition
        for _bc in self.bc:
            _bc.apply(self.U.vector())
        v, p = self.U.split()
        self.Y["velocity"][:, 0] = np.zeros_like(v.vector().get_local()[: self.dofs["velocity"]])
        self.Y["pressure"][:, 0] = np.zeros_like(p.vector().get_local()[self.dofs["velocity"] :])

        print(f"MAX PRESSURE: {np.max(self.Y['pressure'][:, 0])}")

        for i, t in enumerate(self.time_points[1:]):
            n = i + 1
            print(f"\n\nt = {round(t,5)}:\n==========")
            self.Y["velocity"][:, n], self.Y["pressure"][:, n] = self.solve_primal_time_step(
                self.Y["velocity"][:, n - 1], self.Y["pressure"][:, n - 1]
            )

        self.save_solution()
        self.save_vtk()

    def save_vtk(self):
        folder = f"paraview/{self.dt}_{self.T}_{self.theta}_{float(self.nu)}/FOM"

        if not os.path.exists(folder):
            os.makedirs(folder)

        xdmffile_u = XDMFFile(f"{folder}/velocity.xdmf")

        for i, t in enumerate(self.time_points):
            vtk_velocity = File(f"{folder}/velocity_{str(i)}.pvd")
            vtk_pressure = File(f"{folder}/pressure_{str(i)}.pvd")

            # self.U_n.vector().set_local(solution)
            v, p = self.U_n.split()
            # v.vector().vec()[:self.dofs["velocity"]] = self.Y["velocity"][:, i]

            v.vector().set_local(self.Y["velocity"][:, i])
            p.vector().set_local(self.Y["pressure"][:, i])

            #
            xdmffile_u.write(v, t)

            # c = plot(sqrt(dot(v, v)), title="Velocity")
            # plt.colorbar(c, orientation="horizontal")
            # plt.show()

            v.rename("velocity", "solution")
            p.rename("pressure", "solution")
            vtk_velocity.write(v)

            # v.rename('velocity', 'solution')
            # p.rename('pressure', 'solution')

            # vtk_velocity << v
            # vtk_pressure << p

            # vtkfile.write(v, "velocity")
            # # vtkfile.write(p, "pressure")
