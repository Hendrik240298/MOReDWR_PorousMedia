import logging
import math
import os
import pickle
import random
import re
import time
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import petsc4py
import pyamg
import rich.console
import rich.table
import scipy
from dolfin import *
from mpi4py import MPI
# from mumps import DMumpsContext
from petsc4py import PETSc
from tqdm import tqdm

# configure logger
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# GMRES callback, c.f.
# https://stackoverflow.com/questions/33512081/getting-the-number-of-iterations-of-scipys-gmres-iterative-method


class gmres_counter(object):
    def __init__(self, disp=False):
        self._disp = disp
        self.niter = 0

    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print("iter %3i\trk = %s" % (self.niter, str(rk)))


class FOM:
    # constructor
    def __init__(self, config, problem, quantities=["displacement", "pressure"]):
        self.config = config
        self.t = self.config["FOM"]["start_time"]
        self.T = self.config["FOM"]["end_time"]
        self.dt = self.config["FOM"]["delta_t"]
        self.problem = problem
        # ORDERING IS IMPORTANT FOR ROM!
        self.quantities = quantities
        goal = self.config["Problem"]["goal"]
        if problem.__class__.__name__ == "Mandel":
            assert goal in [
                "mean",
                "endtime",
            ], "goal must be either 'mean' or 'endtime' for Mandel problem"
        elif problem.__class__.__name__ == "Footing":
            assert goal in ["point"], "goal must be 'point' for Footing problem"
        self.goal = goal

        self.direct_solve = True
        self.SOLVER_TOL = 0.0

        # for each variable of the object problem, add this variable to the FOM
        for key, value in problem.__dict__.items():
            setattr(self, key, value)

        self.time_points = np.arange(self.t, self.T + self.dt, self.dt)
        logging.info(f"FIRST/LATEST TIME POINT:    {self.time_points[0]}/{self.time_points[-1]}")
        logging.info(f"NUMBER OF TIME POINTS:      {self.time_points.shape[0]}")

        # get class name of problem
        self.problem_name = problem.__class__.__name__
        logging.info(f"Problem name: {self.problem_name}")

        self.mesh = None
        self.MESH_REFINEMENTS = self.config["FOM"]["mesh_refinement"]
        if self.problem_name == "Mandel":
            self.mesh = RectangleMesh(
                Point(0.0, 0.0),
                Point(100.0, 20.0),
                self.MESH_REFINEMENTS * 5 * 16,
                self.MESH_REFINEMENTS * 16,
            )
            self.dim = self.mesh.geometry().dim()

            # plt.figure(figsize=(50,21))
            # plot(self.mesh)
            # plt.axis("off")
            # plt.savefig("mesh.svg",  bbox_inches='tight') #,transparent=True)
            # plt.clf()

            # plt.title("Mesh")
            # plot(self.mesh)
            # plt.show()
        elif self.problem_name == "Footing":
            self.mesh = BoxMesh(
                Point(-32.0, -32.0, 0.0),
                Point(32.0, 32.0, 64.0),
                self.MESH_REFINEMENTS * 8,
                self.MESH_REFINEMENTS * 8,
                self.MESH_REFINEMENTS * 8,
            )  # 24, 24, 24)
            self.dim = self.mesh.geometry().dim()
        else:
            raise NotImplementedError("Only Mandel and Footing problem implemented so far.")

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

        logging.info(f"DOFS: {self.dofs}")

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
        elif self.problem_name == "Footing":
            compression = CompiledSubDomain(
                "near(x[2], 64.) && x[0] >= -16. && x[0] <= 16. && x[1] >= -16. && x[1] <= 16. && on_boundary"
            )
            dirichlet = CompiledSubDomain("near(x[2], 0.) && on_boundary")
            neumann = CompiledSubDomain(
                "( near(x[0], -32.) || near(x[0], 32.) || near(x[1], -32.) || near(x[1], 32.) || (near(x[2], 64.) && (x[0] <= -16. || x[0] >= 16. || x[1] <= -16. || x[1] >= 16.) ) ) && on_boundary"
            )

            facet_marker = MeshFunction("size_t", self.mesh, 1)
            facet_marker.set_all(0)
            neumann.mark(facet_marker, 3)
            compression.mark(facet_marker, 1)
            dirichlet.mark(facet_marker, 2)

            self.ds_compression = Measure("ds", subdomain_data=facet_marker, subdomain_id=1)
            self.ds_neumann = Measure("ds", subdomain_data=facet_marker, subdomain_id=3)

            bc_down_x = DirichletBC(
                self.V.sub(0).sub(0), Constant(0.0), dirichlet
            )  # dirichlet: u_x = 0
            bc_down_y = DirichletBC(
                self.V.sub(0).sub(1), Constant(0.0), dirichlet
            )  # dirichlet: u_y = 0
            bc_down_z = DirichletBC(
                self.V.sub(0).sub(2), Constant(0.0), dirichlet
            )  # dirichlet: u_z = 0
            bc_compression_p = DirichletBC(
                self.V.sub(1), Constant(0.0), dirichlet
            )  # dirichlet: p = 0
            self.bc = [bc_down_x, bc_down_y, bc_down_z, bc_compression_p]

            # for \Gamma_{compression} integral workaround: create FE vector
            # which is 1 on \Gamma_{compression} and 0 else
            def compression_boundary(x, on_boundary):
                return (
                    on_boundary
                    and x[2] > 64.0 - 1e-14
                    and (np.abs(x[0]) <= 16.0 and np.abs(x[1]) <= 16.0)
                )

            bc_compression = DirichletBC(self.V, Constant((0, 0, 1, 0)), compression_boundary)

            self.indicator_compression = Function(self.V)
            bc_compression.apply(self.indicator_compression.vector())
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
            "dual": {},
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
                        self.dt
                        * (self.K_biot / self.viscosity_biot)
                        * inner(grad(p), grad(phi_p))
                        * dx
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

            self.vector["primal"]["traction_full_vector"] = np.array(
                assemble(Constant(self.traction_y_biot) * Phi[1] * self.ds_up)
            )

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

            # system matrix wo boundary conditions
            self.matrix["primal"]["system_matrix_no_bc"] = self.matrix["primal"][
                "system_matrix"
            ].copy()

            # dual system matrix as transposed primal system matrix
            self.matrix["dual"]["system_matrix"] = (
                self.matrix["primal"]["system_matrix"].transpose().copy()
            )

            # NOTE: only use this for mean goal functional
            if self.goal == "mean":
                self.matrix["dual"]["rhs_matrix"] = scipy.sparse.csr_matrix(
                    as_backend_type(assemble(p * phi_p * self.ds_down)).mat().getValuesCSR()[::-1],
                    shape=(
                        self.dofs["displacement"] + self.dofs["pressure"],
                        self.dofs["displacement"] + self.dofs["pressure"],
                    ),
                )

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

            # apply homogeneous Dirichlet BC to dual system matrix
            self.matrix["dual"]["system_matrix"] = self.matrix["dual"]["system_matrix"].multiply(
                (1.0 - self.boundary_dof_vector).reshape(-1, 1)
            ) + scipy.sparse.diags(self.boundary_dof_vector)

            self.solve_factorized_primal = scipy.sparse.linalg.factorized(
                self.matrix["primal"]["system_matrix"].tocsc()
            )  # NOTE: LU factorization is dense

            self.solve_factorized_dual = scipy.sparse.linalg.factorized(
                self.matrix["dual"]["system_matrix"].tocsc()
            )

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
            for i, dof in enumerate(
                self.V.tabulate_dof_coordinates()[: self.V.sub(0).sub(0).dim()]
            ):
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
            self.special_times = [1000.0, 5000.0, 10000.0, 100000.0, 500000.0, 5000000.0]
        elif self.problem_name == "Footing":
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
                        + self.alpha_biot
                        * inner(
                            self.indicator_compression[2] * p * Constant((0.0, 0.0, 1.0)), phi_u
                        )
                        * dx
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
                        self.dt
                        * (self.K_biot / self.viscosity_biot)
                        * inner(grad(p), grad(phi_p))
                        * dx
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
                assemble(
                    Constant(self.traction_z_biot) * inner(self.indicator_compression, Phi) * dx
                )
            )[: self.dofs["displacement"]]

            self.vector["primal"]["traction_full_vector"] = np.array(
                assemble(
                    Constant(self.traction_z_biot) * inner(self.indicator_compression, Phi) * dx
                )
            )

            if self.goal == "point":
                # vector for cost functional
                # compression_midpoint_p = {"DoF (full)": -1, "DoF (p)": -1, "x": 0., "y": 0., "z": 64.}
                # for i, dof in enumerate(
                #     self.V.tabulate_dof_coordinates()[self.dofs["displacement"] : ]
                # ):
                #     if dof[2] == 64.0 and dof[0] == 0.0 and dof[1] == 0.0:
                #         compression_midpoint_p["DoF (full)"] = i + self.dofs["displacement"]
                #         compression_midpoint_p["DoF (p)"] = i
                # self.vector["primal"]["point"] = np.zeros((self.dofs["pressure"],))
                # self.vector["primal"]["point"][compression_midpoint_p["DoF (p)"]] = 1.0
                # self.vector["primal"]["point_full_vector"] = np.zeros((self.dofs["displacement"]+self.dofs["pressure"],))
                # self.vector["primal"]["point_full_vector"][compression_midpoint_p["DoF (full)"]] = 1.0

                # QoI: pressure integral over \Gamma_{compression}
                self.vector["primal"]["point"] = np.array(
                    assemble(self.indicator_compression[2] * phi_p * dx)
                )[self.dofs["displacement"] :]
                self.vector["primal"]["point_full_vector"] = np.array(
                    assemble(self.indicator_compression[2] * phi_p * dx)
                )

            # build system matrix
            self.matrix["primal"]["system_matrix"] = scipy.sparse.csr_matrix(
                (
                    self.dofs["displacement"] + self.dofs["pressure"],
                    self.dofs["displacement"] + self.dofs["pressure"],
                )
            )

            # primal stress to system matrix
            logging.debug("primal stress to system matrix")
            self.matrix["primal"]["system_matrix"] = self.set_sub_matrix(
                self.matrix["primal"]["system_matrix"], self.matrix["primal"]["stress"], "top-left"
            )

            # primal elasto pressure to system matrix
            logging.debug("primal elasto pressure to system matrix")
            self.matrix["primal"]["system_matrix"] = self.set_sub_matrix(
                self.matrix["primal"]["system_matrix"],
                self.matrix["primal"]["elasto_pressure"],
                "top-right",
            )

            # primal laplace + time pressure to system matrix
            logging.debug("primal laplace + time pressure to system matrix")
            self.matrix["primal"]["system_matrix"] = self.set_sub_matrix(
                self.matrix["primal"]["system_matrix"],
                self.matrix["primal"]["laplace"] + self.matrix["primal"]["time_pressure"],
                "bottom-right",
            )
            # primal time displacement to system matrix
            logging.debug("primal time displacement to system matrix")
            self.matrix["primal"]["system_matrix"] = self.set_sub_matrix(
                self.matrix["primal"]["system_matrix"],
                self.matrix["primal"]["time_displacement"],
                "bottom-left",
            )

            # system matrix wo boundary conditions
            logging.debug("copy system matrix to system matrix no bc")
            self.matrix["primal"]["system_matrix_no_bc"] = self.matrix["primal"][
                "system_matrix"
            ].copy()

            # dual system matrix as transposed primal system matrix
            logging.debug("dual system matrix as transposed primal system matrix")
            self.matrix["dual"]["system_matrix"] = (
                self.matrix["primal"]["system_matrix"].transpose().copy()
            )

            # apply BC
            logging.debug("prepare BC")
            self.boundary_dof_vector = np.zeros(
                (self.dofs["displacement"] + self.dofs["pressure"],)
            )
            for _bc in self.bc:
                for i, val in _bc.get_boundary_values().items():
                    assert val == 0.0, "Only homogeneous Dirichlet BCs are supported so far."
                    self.boundary_dof_vector[i] = 1.0

            # apply homogeneous Dirichlet BC to system matrix
            logging.debug("apply BC to primal system matrix")

            self.matrix["primal"]["system_matrix"] = self.matrix["primal"][
                "system_matrix"
            ].multiply((1.0 - self.boundary_dof_vector).reshape(-1, 1)) + scipy.sparse.diags(
                self.boundary_dof_vector
            )

            # apply homogeneous Dirichlet BC to dual system matrix
            logging.debug("apply BC to dual system matrix")
            self.matrix["dual"]["system_matrix"] = self.matrix["dual"]["system_matrix"].multiply(
                (1.0 - self.boundary_dof_vector).reshape(-1, 1)
            ) + scipy.sparse.diags(self.boundary_dof_vector)

            if self.MESH_REFINEMENTS > 1:
                self.direct_solve = False

            # iterative solver tolerance
            if not self.direct_solve:
                self.SOLVER_TOL = self.config["FOM"]["solver"]["tolerance"]

            if self.direct_solve:
                logging.debug("factorize primal system matrix with factorized")
                self.solve_factorized_primal = scipy.sparse.linalg.factorized(
                    self.matrix["primal"]["system_matrix"].tocsc()
                )  # NOTE: LU factorization is dense

                logging.debug("factorize dual system matrix")
                self.solve_factorized_dual = scipy.sparse.linalg.factorized(
                    self.matrix["dual"]["system_matrix"].tocsc()
                )
            else:
                # build preconditioner
                # TODO: Dict with primal and dual preconditioner
                logging.debug("build preconditioner")

                preconditioner_matrix = {}
                self.preconditioner = {}

                E = {}
                D = {}
                E["primal"] = scipy.sparse.tril(self.matrix["primal"]["system_matrix"], k=-1)
                D["primal"] = scipy.sparse.diags(
                    self.matrix["primal"]["system_matrix"].diagonal()
                ).tocsr()

                E["dual"] = scipy.sparse.tril(self.matrix["dual"]["system_matrix"], k=-1)
                D["dual"] = scipy.sparse.diags(
                    self.matrix["dual"]["system_matrix"].diagonal()
                ).tocsr()

                D_inv = {}
                D_inv["primal"] = scipy.sparse.diags(
                    1.0 / self.matrix["primal"]["system_matrix"].diagonal()
                ).tocsr()
                D_inv["dual"] = scipy.sparse.diags(
                    1.0 / self.matrix["dual"]["system_matrix"].diagonal()
                ).tocsr()

                preconditioner_x = {}
                # jacobi
                preconditioner_x["primal"] = lambda x: D_inv["primal"].dot(x)
                preconditioner_x["dual"] = lambda x: D_inv["dual"].dot(x)

                # self.primal_ilu = scipy.sparse.linalg.spilu(self.matrix["primal"]["system_matrix"].tocsc(),
                #                                             drop_tol=1e-12,
                #                                             fill_factor=1000)

                # SOR preconditioner
                omega = 0.5
                preconditioner_SOR = {}
                preconditioner_SOR["primal"] = (
                    1 / omega * (D["primal"] + omega * E["primal"])
                ).tocsr()
                preconditioner_SOR["dual"] = (1 / omega * (D["dual"] + omega * E["dual"])).tocsr()

                preconditioner_SOR_x = {}
                preconditioner_SOR_x["primal"] = lambda x: scipy.sparse.linalg.spsolve_triangular(
                    preconditioner_SOR["primal"], x
                )
                preconditioner_SOR_x["dual"] = lambda x: scipy.sparse.linalg.spsolve_triangular(
                    preconditioner_SOR["dual"], x
                )

                self.preconditioner["primal_SOR"] = scipy.sparse.linalg.LinearOperator(
                    self.matrix["primal"]["system_matrix"].shape, preconditioner_SOR_x["primal"]
                )
                self.preconditioner["dual_SOR"] = scipy.sparse.linalg.LinearOperator(
                    self.matrix["dual"]["system_matrix"].shape, preconditioner_SOR_x["dual"]
                )

                # AMG preconditioner
                ml = {}
                ml["primal_RS"] = pyamg.ruge_stuben_solver(self.matrix["primal"]["system_matrix"])
                # pyamg.ruge_stuben_solver(self.matrix["dual"]["system_matrix"])
                ml["dual_RS"] = pyamg.ruge_stuben_solver(self.matrix["dual"]["system_matrix"])
                ml["primal_SA"] = pyamg.smoothed_aggregation_solver(
                    self.matrix["primal"]["system_matrix"]
                )
                ml["dual_SA"] = pyamg.smoothed_aggregation_solver(
                    self.matrix["dual"]["system_matrix"]
                )

                # print(ml["primal"])
                # print(ml["dual"])

                ml_x = {}
                ml_x["primal_RS"] = lambda x: ml["primal_RS"].solve(x, tol=1e-10)
                ml_x["dual_RS"] = lambda x: ml["dual_RS"].solve(x, tol=1e-10)
                ml_x["primal_SA"] = lambda x: ml["primal_SA"].solve(x, tol=1e-10)
                ml_x["dual_SA"] = lambda x: ml["dual_SA"].solve(x, tol=1e-10)

                # sanity check if preconditioner and backup preconditioner
                # differ
                if (
                    self.config["FOM"]["solver"]["preconditioner"]
                    == self.config["FOM"]["solver"]["preconditioner_backup"]
                ):
                    raise ValueError("Preconditioner and backup preconditioner are the same.")

                # choose preconditioner between  "jacobi", "SOR", "AMG_RS", "AMG_SA"
                # jacobi
                if self.config["FOM"]["solver"]["preconditioner"]["type"] == "jacobi":
                    self.preconditioner["primal"] = scipy.sparse.linalg.LinearOperator(
                        self.matrix["primal"]["system_matrix"].shape, preconditioner_x["primal"]
                    )
                    self.preconditioner["dual"] = scipy.sparse.linalg.LinearOperator(
                        self.matrix["dual"]["system_matrix"].shape, preconditioner_x["dual"]
                    )
                # Ruge-Stuben AMG
                elif self.config["FOM"]["solver"]["preconditioner"]["type"] == "AMG_RS":
                    self.preconditioner["primal"] = scipy.sparse.linalg.LinearOperator(
                        self.matrix["primal"]["system_matrix"].shape, ml_x["primal_RS"]
                    )
                    self.preconditioner["dual"] = scipy.sparse.linalg.LinearOperator(
                        self.matrix["dual"]["system_matrix"].shape, ml_x["dual_RS"]
                    )
                # Smoothed Aggregation AMG
                elif self.config["FOM"]["solver"]["preconditioner"]["type"] == "AMG_SA":
                    self.preconditioner["primal"] = scipy.sparse.linalg.LinearOperator(
                        self.matrix["primal"]["system_matrix"].shape, ml_x["primal_SA"]
                    )
                    self.preconditioner["dual"] = scipy.sparse.linalg.LinearOperator(
                        self.matrix["dual"]["system_matrix"].shape, ml_x["dual_SA"]
                    )
                else:
                    raise ValueError("Preconditioner type does not exist.")

                # choose backup preconditioner between  "jacobi", "SOR", "AMG_RS", "AMG_SA"
                # jacobi
                if self.config["FOM"]["solver"]["preconditioner_backup"]["type"] == "jacobi":
                    self.preconditioner["primal_backup"] = scipy.sparse.linalg.LinearOperator(
                        self.matrix["primal"]["system_matrix"].shape, preconditioner_x["primal"]
                    )
                    self.preconditioner["dual_backup"] = scipy.sparse.linalg.LinearOperator(
                        self.matrix["dual"]["system_matrix"].shape, preconditioner_x["dual"]
                    )
                # Ruge-Stuben AMG
                elif self.config["FOM"]["solver"]["preconditioner_backup"]["type"] == "AMG_RS":
                    self.preconditioner["primal_backup"] = scipy.sparse.linalg.LinearOperator(
                        self.matrix["primal"]["system_matrix"].shape, ml_x["primal_RS"]
                    )
                    self.preconditioner["dual_backup"] = scipy.sparse.linalg.LinearOperator(
                        self.matrix["dual"]["system_matrix"].shape, ml_x["dual_RS"]
                    )
                # Smoothed Aggregation AMG
                elif self.config["FOM"]["solver"]["preconditioner_backup"]["type"] == "AMG_SA":
                    self.preconditioner["primal_backup"] = scipy.sparse.linalg.LinearOperator(
                        self.matrix["primal"]["system_matrix"].shape, ml_x["primal_SA"]
                    )
                    self.preconditioner["dual_backup"] = scipy.sparse.linalg.LinearOperator(
                        self.matrix["dual"]["system_matrix"].shape, ml_x["dual_SA"]
                    )
                else:
                    raise ValueError("Backup preconditioner type does not exist.")

                logging.info(
                    f"Preconditioner: {self.config['FOM']['solver']['preconditioner']['type']}"
                )
                logging.info(
                    f"Backup preconditioner: {self.config['FOM']['solver']['preconditioner_backup']['type']}"
                )

                self.solver_results = {
                    "iterations": [],
                    "wall_time": [],
                }

            # build rhs matrix
            self.matrix["primal"]["rhs_matrix"] = scipy.sparse.csr_matrix(
                (
                    self.dofs["displacement"] + self.dofs["pressure"],
                    self.dofs["displacement"] + self.dofs["pressure"],
                )
            )

            logging.debug("primal time pressure to rhs matrix")
            self.matrix["primal"]["rhs_matrix"] = self.set_sub_matrix(
                self.matrix["primal"]["rhs_matrix"],
                self.matrix["primal"]["time_pressure"],
                "bottom-right",
            )
            logging.debug("primal time displacement to rhs matrix")
            self.matrix["primal"]["rhs_matrix"] = self.set_sub_matrix(
                self.matrix["primal"]["rhs_matrix"],
                self.matrix["primal"]["time_displacement"],
                "bottom-left",
            )

            # self.matrix["primal"]["rhs_matrix"][
            #     self.dofs["displacement"] :, self.dofs["displacement"] :
            # ] = self.matrix["primal"]["time_pressure"]
            # self.matrix["primal"]["rhs_matrix"][
            #     self.dofs["displacement"] :, : self.dofs["displacement"]
            # ] = self.matrix["primal"]["time_displacement"]
        else:
            raise NotImplementedError("Only Mandel and Footing problem implemented so far.")

        # define snapshot matrix
        self.Y = {
            "primal": {
                "displacement": np.zeros((self.dofs["displacement"], self.dofs["time"])),
                "pressure": np.zeros((self.dofs["pressure"], self.dofs["time"])),
            },
            "dual": {
                "displacement": np.zeros((self.dofs["displacement"], self.dofs["time"])),
                "pressure": np.zeros((self.dofs["pressure"], self.dofs["time"])),
            },
        }

        # define functional values
        self.functional_values = np.zeros((self.dofs["time"] - 1,))

        # IO data
        self.SAVE_DIR = self.config["INFRASTRUCTURE"]["safe_directory"]

    def set_sub_matrix(self, matrix, sub_matrix, block):
        matrix = matrix.tolil()
        sub_matrix = sub_matrix.tolil()

        if block == "top-left":
            for i in range(self.dofs["displacement"]):
                # logging.debug(f"row {i} of {self.dofs['displacement']}")
                matrix[i, : self.dofs["displacement"]] = sub_matrix[i, :]
            # matrix[: self.dofs["displacement"], : self.dofs["displacement"]] = sub_matrix
        elif block == "top-right":
            for i in range(self.dofs["displacement"]):
                matrix[i, self.dofs["displacement"] :] = sub_matrix[i, :]
            # matrix[: self.dofs["displacement"], self.dofs["displacement"] :] = sub_matrix
        elif block == "bottom-left":
            for i in range(self.dofs["pressure"]):
                matrix[self.dofs["displacement"] + i, : self.dofs["displacement"]] = sub_matrix[
                    i, :
                ]
            # matrix[self.dofs["displacement"] :, : self.dofs["displacement"]] = sub_matrix
        elif block == "bottom-right":
            for i in range(self.dofs["pressure"]):
                matrix[self.dofs["displacement"] + i, self.dofs["displacement"] :] = sub_matrix[
                    i, :
                ]
            # matrix[self.dofs["displacement"] :, self.dofs["displacement"] :] = sub_matrix
        else:
            raise NotImplementedError(f"Block {block} does not exist.")

        return matrix.tocsr()

    def save_time(self, computation_time):
        pattern = r"time_goal_" + self.goal + "_" + r"\d{6}\.npz"
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
                self.MESH_REFINEMENTS,
                self.direct_solve,
                self.SOLVER_TOL,
            ]
        )

        for file in files:
            tmp = np.load(file, allow_pickle=True)
            if np.array_equal(parameters, tmp["parameters"]):
                np.savez(
                    file,
                    time=computation_time,
                    parameters=parameters,
                    compression=False,
                )
                return

        file_name = "results/time_goal_" + self.goal + "_" + str(len(files)).zfill(6) + ".npz"
        np.savez(
            file_name,
            time=computation_time,
            parameters=parameters,
            compression=False,
        )

    def load_time(self):
        pattern = r"time_goal_" + self.goal + "_" + r"\d{6}\.npz"

        # check if self.SAVE_DIR exists
        if not os.path.exists(self.SAVE_DIR):
            os.makedirs(self.SAVE_DIR)
            raise Exception("No time data available.")

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
                self.MESH_REFINEMENTS,
                self.direct_solve,
                self.SOLVER_TOL,
            ]
        )

        for file in files:
            tmp = np.load(file)
            if np.array_equal(parameters, tmp["parameters"]):
                computation_time = tmp["time"]
                return computation_time
        raise Exception("No time data available.")

    def save_solution(self, solution_type="primal"):
        pattern = r"solution_" + solution_type + "_goal_" + self.goal + "_" + r"\d{6}\.npz"
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
                self.MESH_REFINEMENTS,
                self.direct_solve,
                self.SOLVER_TOL,
            ]
        )

        for file in files:
            tmp = np.load(file, allow_pickle=True)
            if np.array_equal(parameters, tmp["parameters"]):
                np.savez(
                    file,
                    displacement=self.Y[solution_type]["displacement"],
                    pressure=self.Y[solution_type]["pressure"],
                    parameters=parameters,
                    compression=True,
                )
                print(f"Overwrite {file}")
                return

        file_name = (
            "results/solution_"
            + solution_type
            + "_goal_"
            + self.goal
            + "_"
            + str(len(files)).zfill(6)
            + ".npz"
        )
        np.savez(
            file_name,
            displacement=self.Y[solution_type]["displacement"],
            pressure=self.Y[solution_type]["pressure"],
            parameters=parameters,
            compression=True,
        )
        print(f"Saved as {file_name}")

    def load_solution(self, solution_type="primal"):
        pattern = r"solution_" + solution_type + "_goal_" + self.goal + "_" + r"\d{6}\.npz"

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
                self.MESH_REFINEMENTS,
                self.direct_solve,
                self.SOLVER_TOL,
            ]
        )

        for file in files:
            tmp = np.load(file)
            if np.array_equal(parameters, tmp["parameters"]):
                self.Y[solution_type]["displacement"] = tmp["displacement"]
                self.Y[solution_type]["pressure"] = tmp["pressure"]
                logging.info(f"Loaded {file}")
                return True
        return False

    def solve_functional_trajectory(self):
        if self.goal == "mean":
            self.functional_values = np.zeros((self.dofs["time"] - 1,))
            for i in range(self.dofs["time"] - 1):
                self.functional_values[i] = (
                    self.vector["primal"]["pressure_down"].dot(
                        self.Y["primal"]["pressure"][:, i + 1]
                    )
                    * self.dt
                )
            self.functional = np.sum(self.functional_values)
        elif self.goal == "endtime":
            self.functional_values = None
            self.functional = self.vector["primal"]["pressure_down"].dot(
                self.Y["primal"]["pressure"][:, -1]
            )
        elif self.goal == "point":
            self.functional_values = np.zeros((self.dofs["time"] - 1,))
            for i in range(self.dofs["time"] - 1):
                self.functional_values[i] = (
                    self.vector["primal"]["point"].dot(self.Y["primal"]["pressure"][:, i + 1])
                    * self.dt
                )
            self.functional = np.sum(self.functional_values)

        # print cost functional in scientific notation
        # print(f"Cost functional: {self.functional:.4e}")

        # # print cost functional trajectory
        # plt.plot(self.time_points[1:], self.functional_values)
        # plt.show()

    def plot_bottom_solution(self):
        times = self.special_times.copy()
        for i, t in enumerate(self.time_points):
            if np.abs(t - times[0]) <= 1e-4:
                times.pop(0)
                plt.plot(
                    self.bottom_x_u,
                    self.bottom_matrix_u.dot(self.Y["primal"]["displacement"][:, i]),
                    label=f"t = {t}",
                )
                if len(times) == 0:
                    break
        plt.xlabel("x")
        plt.ylabel(r"$u_x(x,0)$")
        plt.title("x-Displacement at bottom boundary")
        plt.legend()
        plt.show()

        times = self.special_times.copy()
        for i, t in enumerate(self.time_points):
            if np.abs(t - times[0]) <= 1e-4:
                times.pop(0)
                plt.plot(
                    self.bottom_x_p,
                    self.bottom_matrix_p.dot(self.Y["primal"]["pressure"][:, i]),
                    label=f"t = {t}",
                )
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

        # solve primal system
        if self.direct_solve:
            solution = self.solve_factorized_primal(rhs)
        else:
            # GMRES with jacobi preconditioner
            counter = gmres_counter()
            start_time = time.time()
            solution, exit_code = scipy.sparse.linalg.gmres(
                self.matrix["primal"]["system_matrix"],
                rhs,
                M=self.preconditioner["primal"],
                x0=old_solution,
                tol=self.SOLVER_TOL,
                maxiter=self.config["FOM"]["solver"]["preconditioner"]["max_iter"],
                restart=self.config["FOM"]["solver"]["preconditioner"]["restart"],
                callback=counter,
            )
            end_time = time.time() - start_time
            self.solver_results["iterations"].append(counter.niter)
            self.solver_results["wall_time"].append(end_time)

            if exit_code != 0:
                logging.info("Primal GMRES did not converge. Try with backup preconditioner.")
                counter_backup = gmres_counter(disp=True)
                solution, exit_code = scipy.sparse.linalg.gmres(
                    self.matrix["primal"]["system_matrix"],
                    rhs,
                    M=self.preconditioner["primal_backup"],
                    x0=solution,
                    tol=self.SOLVER_TOL,
                    maxiter=self.config["FOM"]["solver"]["preconditioner_backup"]["max_iter"],
                    restart=self.config["FOM"]["solver"]["preconditioner_backup"]["restart"],
                    callback=counter_backup,
                )
                if exit_code == 0:
                    logging.info(f"Backup converged in {counter_backup.niter} iterations.")

            # throw exepction if exit code unequal zero
            if exit_code != 0:
                raise Exception("GMRES did not converge.")

        # # compare timings
        # logging.debug(f"Direct solver:    {end_time_direct}")
        # logging.debug(f"Iterative solver: {end_time_iter}")

        # # compare solution and solution_iter
        # logging.debug(f"Differnce solution:  {np.linalg.norm(solution-solution_iter)}")
        # logging.debug(f"Difference relative: {np.linalg.norm(solution-solution_iter)/np.linalg.norm(solution)}")

        return solution[: self.dofs["displacement"]], solution[self.dofs["displacement"] :]

    # Solve time trajectory
    def solve_primal(self, force_recompute=False):
        if not force_recompute:
            if self.load_solution(solution_type="primal"):
                return False

        # zero initial condition
        self.Y["primal"]["displacement"][:, 0] = np.zeros((self.dofs["displacement"],))
        self.Y["primal"]["pressure"][:, 0] = np.zeros((self.dofs["pressure"],))

        for i, t in enumerate(tqdm(self.time_points[1:])):
            n = i + 1
            # print(f"\nt = {round(t,5)}:\n===============")
            (
                self.Y["primal"]["displacement"][:, n],
                self.Y["primal"]["pressure"][:, n],
            ) = self.solve_primal_time_step(
                self.Y["primal"]["displacement"][:, n - 1], self.Y["primal"]["pressure"][:, n - 1]
            )

        self.save_solution(solution_type="primal")
        # self.save_vtk()
        # save solver results
        if not self.direct_solve:
            with open(self.SAVE_DIR + self.config["INFRASTRUCTURE"]["name"] + ".pkl", "wb") as f:
                pickle.dump(self.solver_results, f)
        return True

    def solve_dual_time_step(self, z_u_nn_vector, z_p_nn_vector):
        """
        INPUT:
            z_u_nn_vector: dual displacement at time n+1
            z_p_nn_vector: dual pressure at time n+1
        OUTPUT:
            z_u_n_vector:  dual displacement at time n
            z_p_n_vector:  dual pressure at time n
        """

        # old dual solution
        old_dual_solution = np.concatenate((z_u_nn_vector, z_p_nn_vector))
        # primal solution

        # ATTENTION: NO NEED FOR PRIMAL SOLUTION FOR LINEAR PROBLEMS

        # old dual solution on rhs (NO self.dt NEEDED! ELSE SOLUTION blowups)
        dual_rhs = self.matrix["primal"]["rhs_matrix"].transpose().dot(old_dual_solution)

        # derivative of CF on rhs
        if self.goal == "mean":
            dual_rhs[self.dofs["displacement"] :] += (
                self.dt * self.vector["primal"]["pressure_down"]
            )
        elif self.goal == "point":
            dual_rhs[self.dofs["displacement"] :] += self.dt * self.vector["primal"]["point"]

        # apply homogeneous Dirichlet BC to right hand side
        dual_rhs *= 1.0 - self.boundary_dof_vector

        # solve dual system
        if self.direct_solve:
            dual_solution = self.solve_factorized_dual(dual_rhs)
        else:
            # GMRES with jacobi preconditioner
            counter = gmres_counter()
            dual_solution, exit_code = scipy.sparse.linalg.gmres(
                self.matrix["dual"]["system_matrix"],
                dual_rhs,
                M=self.preconditioner["dual"],
                x0=old_dual_solution,
                tol=self.SOLVER_TOL,
                maxiter=self.config["FOM"]["solver"]["preconditioner"]["max_iter"],
                restart=self.config["FOM"]["solver"]["preconditioner"]["restart"],
                callback=counter,
            )

            # if do not converge, first try with backup preconditioner
            if exit_code != 0:
                logging.info("Dual GMRES did not converge. Try with backup preconditioner.")
                counter_backup = gmres_counter(disp=True)
                dual_solution, exit_code = scipy.sparse.linalg.gmres(
                    self.matrix["dual"]["system_matrix"],
                    dual_rhs,
                    M=self.preconditioner["dual_backup"],
                    x0=dual_solution,
                    tol=self.SOLVER_TOL,
                    maxiter=self.config["FOM"]["solver"]["preconditioner_backup"]["max_iter"],
                    restart=self.config["FOM"]["solver"]["preconditioner_backup"]["restart"],
                    callback=counter_backup,
                )
                if exit_code == 0:
                    logging.info(f"Backup converged in {counter_backup.niter} iterations.")

            # throw exepction if exit code unequal zero
            if exit_code != 0:
                raise Exception("GMRES did not converge.")

        # compare solution and solution_iter
        # logging.debug(f"Differnce solution:  {np.linalg.norm(dual_solution-dual_solution_iter)}")
        # logging.debug(f"Difference relative: {np.linalg.norm(dual_solution-dual_solution_iter)/np.linalg.norm(dual_solution)}")

        # # plot dual solution
        # u, p = self.U_n.split()
        # self.U_n.vector().set_local(
        #     dual_solution
        # )

        # # plot u and p in a subplot
        # plt.subplot(2, 1, 1)
        # c = plot(u, title=f"u")
        # plt.colorbar(c, orientation="horizontal")
        # plt.subplot(2, 1, 2)
        # c = plot(p, title=f"p")
        # plt.colorbar(c, orientation="horizontal")
        # plt.show()

        # c = plot(p, title=f"dua")
        # plt.colorbar(c, orientation="horizontal")
        # plt.show()

        # c = plot(p, title=f"{i}th pressure POD magnitude")
        # plt.colorbar(c, orientation="horizontal")
        # plt.show()

        return (
            dual_solution[: self.dofs["displacement"]],
            dual_solution[self.dofs["displacement"] :],
        )

    def solve_dual(self, force_recompute=False):
        if not force_recompute:
            if self.load_solution(solution_type="dual"):
                return

        logging.info("Solve dual FOM problem...")

        self.Y["dual"]["displacement"][:, -1] = np.zeros((self.dofs["displacement"],))
        self.Y["dual"]["pressure"][:, -1] = np.zeros((self.dofs["pressure"],))

        if self.goal == "endtime":
            for i, dof in enumerate(self.bottom_dofs_p):
                self.Y["dual"]["pressure"][:, -1][i] = 1.0

        # print(np.linalg.norm(self.Y["dual"]["displacement"][:, -1]))
        # print(np.linalg.norm(self.Y["dual"]["pressure"][:, -1]))

        for i, t in tqdm(list(enumerate(self.time_points[:-1]))[::-1]):
            n = i
            # print(f"\n n = {n}; t = {round(t,5)}")
            (
                self.Y["dual"]["displacement"][:, n],
                self.Y["dual"]["pressure"][:, n],
            ) = self.solve_dual_time_step(
                self.Y["dual"]["displacement"][:, n + 1], self.Y["dual"]["pressure"][:, n + 1]
            )

            IF_PLOT = False
            if (i == 0 or i == 4999) and IF_PLOT:
                # plot dual solution
                u, p = self.U_n.split()
                self.U_n.vector().set_local(
                    np.concatenate(
                        (self.Y["dual"]["displacement"][:, n], self.Y["dual"]["pressure"][:, n])
                    )
                )
                # plot u and p in a subplot
                plt.subplot(2, 1, 1)
                c = plot(u, title=f"u")
                plt.colorbar(c, orientation="horizontal")
                plt.subplot(2, 1, 2)
                c = plot(p, title=f"p")
                plt.colorbar(c, orientation="horizontal")
                plt.show()

        self.save_solution(solution_type="dual")

        self.save_vtk(type="dual")

    def save_vtk(self, type="primal"):
        folder = f"paraview/{self.dt}_{self.T}_{self.problem_name}_{self.goal}/FOM"
        # check if folder exists else create it
        if not os.path.exists(folder):
            os.makedirs(folder)

        print(f"Starting saving {type} FOM vtk files...")

        if not os.path.exists(folder):
            os.makedirs(folder)

        # only each 10th time step
        for i, t in list(enumerate(self.time_points))[::25]:
            # print(f"{type}: i = {i}")
            # print(np.linalg.norm(self.Y[type]["displacement"][:, i]))
            # print(np.linalg.norm(self.Y[type]["pressure"][:, i]))
            vtk_displacement = File(f"{folder}/{type}_displacement_{str(i)}.pvd")
            vtk_pressure = File(f"{folder}/{type}_pressure_{str(i)}.pvd")

            u, p = self.U_n.split()

            self.U_n.vector().set_local(
                np.concatenate((self.Y[type]["displacement"][:, i], self.Y[type]["pressure"][:, i]))
            )

            u.rename("displacement", "solution")
            p.rename("pressure", "solution")
            vtk_displacement.write(u)
            vtk_pressure.write(p)
        print("Done.")
