import math
import os
import random
import re
import time
from multiprocessing import Pool

import dolfin

import matplotlib.pyplot as plt
import numpy as np
import rich.console
import rich.table
import scipy
from dolfin import *
from tqdm import tqdm

class SpaceFE:
    form = {}
    matrix = {}

    def __init__(self, mesh, V, problem):
        self.mesh = mesh
        self.V = V
        self.dim = self.mesh.geometry().dim()
        self.problem = problem
        self.problem_name = problem.__class__.__name__

        # for each variable of the object problem, add this variable to the object SpaceFE
        for key, value in problem.__dict__.items():
            setattr(self, key, value)
        
        #self.dofs = self.V.tabulate_dof_coordinates().reshape((-1, self.dim))
        self.n_dofs = {
            "total": self.V.dim(),
            "displacement": self.V.sub(0).dim(),
            "pressure": self.V.sub(1).dim()
        }
        # For debugging:
        # self.print_dofs()
        self.compute_bc()
        self.assemble_matrices()
        self.build_system_matrix()

    def print_dofs(self):
        print("\nSpace DoFs:")
        for dof, dof_x in zip(self.V.dofmap().dofs(), self.dofs):
          print(dof, ':', dof_x)

    def compute_bc(self):
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

            bc_left = DirichletBC(self.V.sub(0).sub(0), Constant(0.0), left)    # left: u_x = 0
            bc_right = DirichletBC(self.V.sub(1), Constant(0.0), right)         # right:  p = 0
            bc_down = DirichletBC(self.V.sub(0).sub(1), Constant(0.0), down)    # down: u_y = 0
            self.bc = [bc_left, bc_right, bc_down]
        else:
            raise NotImplementedError("Only Mandel problem implemented so far.")

    def assemble_matrices(self):
        # Define trial and test functions and function at old time step
        U = TrialFunction(self.V)
        Phi = TestFunction(self.V)

        # Split functions into velocity and pressure components
        u, p = split(U)
        phi_u, phi_p = split(Phi)

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
                    self.n_dofs["displacement"] + self.n_dofs["pressure"],
                    self.n_dofs["displacement"] + self.n_dofs["pressure"],
                ),
            )[: self.n_dofs["displacement"], : self.n_dofs["displacement"]]

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
                    self.n_dofs["displacement"] + self.n_dofs["pressure"],
                    self.n_dofs["displacement"] + self.n_dofs["pressure"],
                ),
            )[: self.n_dofs["displacement"], self.n_dofs["displacement"] :]

            self.matrix["primal"]["laplace"] = scipy.sparse.csr_matrix(
                as_backend_type(
                    assemble(
                        (self.K_biot / self.viscosity_biot) * inner(grad(p), grad(phi_p)) * dx
                    )
                )
                .mat()
                .getValuesCSR()[::-1],
                shape=(
                    self.n_dofs["displacement"] + self.n_dofs["pressure"],
                    self.n_dofs["displacement"] + self.n_dofs["pressure"],
                ),
            )[self.n_dofs["displacement"] :, self.n_dofs["displacement"] :]

            self.matrix["primal"]["time_pressure"] = scipy.sparse.csr_matrix(
                as_backend_type(assemble(self.c_biot * p * phi_p * dx)).mat().getValuesCSR()[::-1],
                shape=(
                    self.n_dofs["displacement"] + self.n_dofs["pressure"],
                    self.n_dofs["displacement"] + self.n_dofs["pressure"],
                ),
            )[self.n_dofs["displacement"] :, self.n_dofs["displacement"] :]

            self.matrix["primal"]["time_displacement"] = scipy.sparse.csr_matrix(
                as_backend_type(assemble(self.alpha_biot * div(u) * phi_p * dx))
                .mat()
                .getValuesCSR()[::-1],
                shape=(
                    self.n_dofs["displacement"] + self.n_dofs["pressure"],
                    self.n_dofs["displacement"] + self.n_dofs["pressure"],
                ),
            )[self.n_dofs["displacement"] :, : self.n_dofs["displacement"]]

            self.vector["primal"]["traction"] = np.array(
                assemble(Constant(self.traction_y_biot) * Phi[1] * self.ds_up)
            )[: self.n_dofs["displacement"]]
            # the same vector, but not truncated
            self.vector["primal"]["traction_full"] = np.array(
                assemble(Constant(self.traction_y_biot) * Phi[1] * self.ds_up)
            )

            # pressure vector for cost functional (paper JR p. 25)
            self.vector["primal"]["pressure_down"] = np.array(
                assemble(Phi[self.dim] * self.ds_down)
            )[self.n_dofs["displacement"] :]

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
                    self.n_dofs["displacement"],
                )
            )
            for i, dof in enumerate(self.bottom_dofs_u):
                self.bottom_matrix_u[i, dof] = 1.0

            # get all pressure DoFs and coordinates at bottom boundary
            self.bottom_dofs_p = []
            self.bottom_x_p = []
            for i, dof in enumerate(self.V.tabulate_dof_coordinates()[self.n_dofs["displacement"] :]):
                if dof[1] == 0.0:
                    self.bottom_dofs_p.append(i)
                    self.bottom_x_p.append(dof[0])
            self.bottom_matrix_p = scipy.sparse.csr_matrix(
                (
                    len(self.bottom_dofs_p),
                    self.n_dofs["pressure"],
                )
            )
            for i, dof in enumerate(self.bottom_dofs_p):
                self.bottom_matrix_p[i, dof] = 1.0
        else:
            raise NotImplementedError("Only Mandel problem implemented so far.")

    def build_system_matrix(self):
        if self.problem_name == "Mandel":
            # build system matrix
            # all the terms that have a \partial_t term
            self.matrix["primal"]["time_derivative_matrix"] = scipy.sparse.csr_matrix(
                (
                    self.n_dofs["displacement"] + self.n_dofs["pressure"],
                    self.n_dofs["displacement"] + self.n_dofs["pressure"],
                )
            )
            
            # all the terms without any time derivatives
            self.matrix["primal"]["time_mass_matrix"] = scipy.sparse.csr_matrix(
                (
                    self.n_dofs["displacement"] + self.n_dofs["pressure"],
                    self.n_dofs["displacement"] + self.n_dofs["pressure"],
                )
            )

            # NO time-derivatives:
            self.matrix["primal"]["time_mass_matrix"][
                : self.n_dofs["displacement"], : self.n_dofs["displacement"]
            ] += self.matrix["primal"]["stress"]
            self.matrix["primal"]["time_mass_matrix"][
                : self.n_dofs["displacement"], self.n_dofs["displacement"] :
            ] += self.matrix["primal"]["elasto_pressure"]
            self.matrix["primal"]["time_mass_matrix"][
                self.n_dofs["displacement"] :, self.n_dofs["displacement"] :
            ] += self.matrix["primal"]["laplace"]
            
            # WITH time-derivative
            self.matrix["primal"]["time_derivative_matrix"][
                self.n_dofs["displacement"] :, : self.n_dofs["displacement"]
            ] = self.matrix["primal"]["time_displacement"]
            self.matrix["primal"]["time_derivative_matrix"][
                self.n_dofs["displacement"] :, self.n_dofs["displacement"] :
            ] += self.matrix["primal"]["time_pressure"]

            # prepare spatial BC
            self.boundary_dof_vector = np.zeros(
                (self.n_dofs["displacement"] + self.n_dofs["pressure"],)
            )
            for _bc in self.bc:
                for i, val in _bc.get_boundary_values().items():
                    assert val == 0.0, "Only homogeneous Dirichlet BCs are supported so far."
                    self.boundary_dof_vector[i] = 1.0
        else:
            raise NotImplementedError("Only Mandel problem implemented so far.")
        
class TimeFE:
    form = { "primal": {} }
    matrix = { "primal": {} }
    vector = { "primal": {} }

    def __init__(self, mesh, V):
        self.mesh = mesh
        self.V = V
        self.dofs = self.V.tabulate_dof_coordinates().reshape((-1, 1))
        self.n_dofs = self.dofs.shape[0]
        # For debugging:
        # self.print_dofs()
        self.assemble_matrices()

    def print_dofs(self):
        print("\nTime DoFs:")
        for dof, dof_t in zip(self.V.dofmap().dofs(), self.dofs):
            print(dof, ':', dof_t)

    def assemble_matrices(self):
        initial_time = CompiledSubDomain('near(x[0], t0)', t0=self.dofs[0,0])
        interior_facets = CompiledSubDomain("!on_boundary")
        boundary_marker = MeshFunction("size_t", self.mesh, 0)
        boundary_marker.set_all(0)
        initial_time.mark(boundary_marker, 1)
        interior_facets.mark(boundary_marker, 2)

        # Measure for the initial time
        d0 = Measure('ds', domain=self.mesh, subdomain_data=boundary_marker, subdomain_id=1)
        dS = Measure('dS', domain=self.mesh, subdomain_data=boundary_marker, subdomain_id=2)

        u = TrialFunction(self.V)
        phi = TestFunction(self.V)

        # NOTE: FEniCS has weird definitions for '+' and '-' (https://fenicsproject.discourse.group/t/integrating-over-an-interior-surface/247/3)
        self.form["primal"]["derivative"] = grad(u)[0]*phi*dx + (u('-')-u('+'))*phi('-')*dS + u('+')*phi('+')*d0
        self.form["primal"]["mass"] = u*phi*dx

        for (name, _form) in self.form["primal"].items():
            self.matrix["primal"][name] = scipy.sparse.csr_matrix(
              as_backend_type(assemble(_form)).mat().getValuesCSR()[::-1],
              shape=(self.n_dofs, self.n_dofs)
            )

        self.vector["primal"]["traction_full"] = np.array(
            assemble(Constant(1.) * phi * dx)
        )

        # # manually create matrix for initial condition (assuming Gauss-Lobatto quadrature in time)
        # self.matrix["primal"]["initial"] = scipy.sparse.csr_matrix(
        #     (
        #         self.n_dofs,
        #         self.n_dofs,
        #     )
        # )
        # self.matrix["primal"]["initial"][0, self.n_dofs - 1] = 1.0

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

        # get class name of problem
        self.problem_name = problem.__class__.__name__
        print(f"Problem name: {self.problem_name}")

        space_mesh = None
        if self.problem_name == "Mandel":
            space_mesh = RectangleMesh(Point(0.0, 0.0), Point(100.0, 20.0), 16, 16)
        
            # plt.title("Mesh")
            # plot(self.mesh)
            # plt.show()
        else:
            raise NotImplementedError("Only Mandel problem implemented so far.")

        element = {
            "u": VectorElement("Lagrange", space_mesh.ufl_cell(), 2),
            "p": FiniteElement("Lagrange", space_mesh.ufl_cell(), 1),
        }
        space_V = FunctionSpace(space_mesh, MixedElement(*element.values()))
        
        ####################
        # Spatial FE object:
        self.Space = SpaceFE(space_mesh, space_V, self.problem)
        self.U_n = Function(self.Space.V)

        # self.dofs = {
        #     "displacement": Space.V.sub(0).dim(),
        #     "pressure": Space.V.sub(1).dim(),
        #     "time": self.time_points.shape[0],
        # }

        start_time = self.t
        end_time = self.T
        slab_size = self.dt
        self.slabs = [(start_time, start_time+slab_size)]
        while self.slabs[-1][1] < end_time - 1e-8:
            self.slabs.append((self.slabs[-1][1], self.slabs[-1][1]+slab_size))

        r = 0 # polynomial degree in time
        n_time = 1 # number of temporal elements
        time_mesh = IntervalMesh(n_time, self.slabs[0][0], self.slabs[0][1]) # Start time: slab[0], End time: slab[1] = slab[0]+slab_size
        time_V = FunctionSpace(time_mesh, 'DG', r)

        #####################
        # Temporal FE object:
        self.Time = TimeFE(time_mesh, time_V)
        total_temporal_dofs = self.Time.n_dofs * len(self.slabs)
        self.time_points = np.zeros((total_temporal_dofs+1,))
        self.time_points[0] = self.t
        for i in range(len(self.slabs)):
            self.time_points[i*self.Time.n_dofs+1:(i+1)*self.Time.n_dofs+1] = self.Time.dofs + self.slabs[i][0]

        print(f"Time DoFs: {total_temporal_dofs:,} ({self.Time.n_dofs:,} DoFs x {len(self.slabs):,} slabs)")
        print(f"Space DoFs: {self.Space.n_dofs['total']:,} ({self.Space.n_dofs['displacement']:,} u + {self.Space.n_dofs['pressure']:,} p)")
        print(f"Total DoFs: {self.Space.n_dofs['total'] * total_temporal_dofs:,}")

        if self.problem_name == "Mandel":
            # times for plotting solution at bottom boundary
            self.special_times = [1000., 5000., 10000., 100000.,  500000., 5000000.]
        else:
            raise NotImplementedError("Only Mandel problem implemented so far.")
        
        ###########################
        # Space-time system matrix
        system_matrix = scipy.sparse.kron(self.Time.matrix["primal"]["derivative"], self.Space.matrix["primal"]["time_derivative_matrix"]) + scipy.sparse.kron(self.Time.matrix["primal"]["mass"], self.Space.matrix["primal"]["time_mass_matrix"])

        # apply bc in space-time
        self.dofs_at_boundary =  np.kron(np.ones((self.Time.dofs.shape[0],1)), self.Space.boundary_dof_vector.reshape(-1,1)).flatten()

        system_matrix = system_matrix.multiply((1.-self.dofs_at_boundary).reshape(-1,1)) + scipy.sparse.diags(self.dofs_at_boundary)
        self.rhs_traction = np.kron(self.Time.vector["primal"]["traction_full"], self.Space.vector["primal"]["traction_full"])
        
        self.solve_factorized_primal = scipy.sparse.linalg.factorized(
            system_matrix.tocsc()
        )  # NOTE: LU factorization is dense

        # define snapshot matrix
        self.Y = {
            "displacement": np.zeros((self.Space.n_dofs["displacement"], total_temporal_dofs+1)),
            "pressure": np.zeros((self.Space.n_dofs["pressure"], total_temporal_dofs+1)),
        }

        # define functional values
        self.functional_values = np.zeros((total_temporal_dofs,))

        # IO data
        self.SAVE_DIR = "results/"

    def save_solution(self):
        # check if self.SAVE_DIR exists
        if not os.path.exists(self.SAVE_DIR):
            os.makedirs(self.SAVE_DIR)

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
        for i in range(self.functional_values.shape[0]):
            self.functional_values[i] = self.Space.vector["primal"]["pressure_down"].dot(
                self.Y["pressure"][:, i + 1]
            )

        # TODO: adapt this for dG(r) with r > 0 
        self.functional = np.sum(self.functional_values) * self.dt
        # print cost functional in scientific notation
        # print(f"Cost functional: {self.functional:.4e}")

        # print cost functional trajectory
        print(self.time_points)
        print(self.functional_values)
        plt.plot(self.time_points[1:], self.functional_values)
        plt.show()

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
    def solve_primal_time_step(self, u0_vector, p0_vector):
        initial_solution = np.concatenate((u0_vector, p0_vector))

        rhs = np.zeros((self.Space.n_dofs["total"]*self.Time.n_dofs,))
        rhs[: self.Space.n_dofs["total"]] = self.Space.matrix["primal"]["time_derivative_matrix"].dot(initial_solution)
        
        rhs += self.rhs_traction

        # apply homogeneous Dirichlet BC to right hand side
        rhs = rhs * (1. - self.dofs_at_boundary)

        solution = self.solve_factorized_primal(rhs)

        solutions_displacement = []
        solutions_pressure = []
        for i in range(self.Time.n_dofs):
            solutions_displacement.append(solution[i*self.Space.n_dofs["total"]: i*self.Space.n_dofs["total"]+self.Space.n_dofs["displacement"]])
            solutions_pressure.append(solution[i*self.Space.n_dofs["total"]+self.Space.n_dofs["displacement"] : (i+1)*self.Space.n_dofs["total"]])

        return solutions_displacement, solutions_pressure

    def solve_dual_time_step(self, u_n_vector, z_n_vector):
        pass  # todo

    # Solve time trajectory
    def solve_primal(self, force_recompute=False):
        if not force_recompute:
            if self.load_solution():
                return

        # zero initial condition
        self.Y["displacement"][:, 0] = np.zeros((self.Space.n_dofs["displacement"],))
        self.Y["pressure"][:, 0] = np.zeros((self.Space.n_dofs["pressure"],))

        print("Solving primal FOM:")
        for i, _ in enumerate(tqdm(self.slabs)):
            n = i * self.Time.n_dofs
            #print(f"\n\nt = {round(t,5)}:\n===============")
            solutions_displacement, solutions_pressure = self.solve_primal_time_step(
                self.Y["displacement"][:, n], self.Y["pressure"][:, n]
            )
            for j in range(self.Time.n_dofs):
                self.Y["displacement"][:, n+j+1] = solutions_displacement[j]
                self.Y["pressure"][:, n+j+1] = solutions_pressure[j]

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
