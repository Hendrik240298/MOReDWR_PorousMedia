import dolfin
import os

##############
# PARAMETERS #
##############
# M_biot = Biot's constant
M_biot = 0. #1.75e+7 #2.5e+12
c_biot = 0. #1.0 / M_biot

# alpha_biot = b_biot = Biot's modulo
alpha_biot = 1.0
viscosity_biot = 1.0e-3
K_biot = 1.0e-6
density_biot = 1.0

# Traction
traction_x_biot = 0.0
traction_y_biot = 0.0 #-1.0e+7
traction_z_biot = 0.1

# Solid parameters
density_structure = 1.0
youngs_modulus_E = 3.0e+4
poisson_ratio_nu = 0.2
lame_coefficient_mu = youngs_modulus_E / (2. * (1. + poisson_ratio_nu))
lame_coefficient_lambda = (2. * poisson_ratio_nu * lame_coefficient_mu) / (1.0 - 2. * poisson_ratio_nu)
print(f"λ = {lame_coefficient_mu}, μ = {lame_coefficient_lambda}")

import matplotlib.pyplot as plt
from dolfin import *
import numpy as np
import time

set_log_active(False) # turn off FEniCS logging

mesh = BoxMesh(Point(-32.,-32.,0.), Point(32.,32.,64.), 16, 16, 16)
dim = mesh.geometry().dim()
# plot(mesh)
# plt.show()

# times = [1000., 5000., 10000., 100000.,  500000., 5000000.]
# x_linspace = np.linspace(0., 100., 16)
# solutions = {"u_x": {}, "p": {}}

cpu_start_time = time.time()

# start time
t = 0.
# end time
T = 0.5 #5.0e+6
# time step size
k = 0.001 #0.01 #1000.
# time step number
time_step_number = 0

element = {
  "u" : VectorElement("Lagrange" , mesh.ufl_cell(), 2),
  "p" : FiniteElement("Lagrange" , mesh.ufl_cell(), 1),
}
V = FunctionSpace(mesh, MixedElement(*element.values()))
print(f"Number of spatial DoFs: {V.dim()} ({V.sub(0).dim()} + {V.sub(1).dim()})")

compression = CompiledSubDomain("near(x[2], 64.) && x[0] >= -16. && x[0] <= 16. && x[1] >= -16. && x[1] <= 16. && on_boundary")
dirichlet = CompiledSubDomain("near(x[2], 0.) && on_boundary")
neumann = CompiledSubDomain("( near(x[0], -32.) || near(x[0], 32.) || near(x[1], -32.) || near(x[1], 32.) || (near(x[2], 64.) && (x[0] <= -16. || x[0] >= 16. || x[1] <= -16. || x[1] >= 16.) ) ) && on_boundary")

facet_marker = MeshFunction("size_t", mesh, 1)
facet_marker.set_all(0)
neumann.mark(facet_marker, 3)
compression.mark(facet_marker, 1)
dirichlet.mark(facet_marker, 2)

# save facet marker to file
File("facet_marker.pvd") << facet_marker

ds_compression = Measure("ds", subdomain_data=facet_marker, subdomain_id=1)
ds_neumann = Measure("ds", subdomain_data=facet_marker, subdomain_id=3)

bc_down_x = DirichletBC(V.sub(0).sub(0), Constant(0.), dirichlet) # dirichlet: u_x = 0
bc_down_y = DirichletBC(V.sub(0).sub(1), Constant(0.), dirichlet) # dirichlet: u_y = 0
bc_down_z = DirichletBC(V.sub(0).sub(2), Constant(0.), dirichlet) # dirichlet: u_z = 0
bcs = [bc_down_x, bc_down_y, bc_down_z]

# zero initial condition for u and p
U_0 = Constant((0.,)*(dim+1))
# U_n = (u_n, p_n): solution from last time step
U_n = interpolate(U_0, V)

# variational problem
U = TrialFunction(V)
Phi = TestFunction(V)

# split functions into displacement and pressure components
u, p = split(U)
phi_u, phi_p = split(Phi)
u_n, p_n = split(U_n)

# Stress function
def stress(u):
    return lame_coefficient_lambda*div(u)*Identity(dim) + lame_coefficient_mu*(grad(u) + grad(u).T)

n = FacetNormal(mesh)
A_u = inner(stress(u), grad(phi_u))*dx - alpha_biot*inner(p*Identity(dim), grad(phi_u))*dx + alpha_biot*inner(p*n, phi_u)*ds_compression + alpha_biot*inner(p*n, phi_u)*ds_neumann
A_p = alpha_biot*div(u)*phi_p*dx + k*(K_biot/viscosity_biot)*inner(grad(p), grad(phi_p))*dx #+ c_biot*p*phi_p*dx
L = Constant(traction_z_biot)*Phi[2]*ds_compression + alpha_biot*div(u_n)*phi_p*dx #+ c_biot*p_n*phi_p*dx

# Time-stepping
Uh = Function(V) # Uh = U_{n+1}: current solution

folder = f"output"
if not os.path.exists(folder):
    os.makedirs(folder)

while(t+k <= T+1e-8):
    # Update current time
    t += k

    print("Time step number:", time_step_number, "; Time:", round(t,5))

    # Compute solution
    solve_start_time = time.time()
    solve(A_u+A_p == L, Uh, bcs, solver_parameters={'linear_solver':'mumps'})
    print(f"    Solve Time: {round(time.time() - solve_start_time, 5)} s")
    
    vtk_displacement = File(f"{folder}/displacement_{str(time_step_number)}.pvd")
    vtk_pressure = File(f"{folder}/pressure_{str(time_step_number)}.pvd")
    _u, _p = Uh.split()
    _u.rename("displacement", "solution")
    _p.rename("pressure", "solution")
    vtk_displacement.write(_u)
    vtk_pressure.write(_p)

    # for _t in times:
    #   if np.abs(_t-t) <= 1e-4:
    #     solutions['u_x'][_t] = [Uh(x, 0.)[0] for x in x_linspace]
    #     solutions['p'][_t] = [Uh(x, 0.)[dim] for x in x_linspace]

    # if time_step_number % 1 == 0:
      # print("Time step number:", time_step_number, "; Time:", t)
      # Plot solution
      # plt.title("u_x")
      # c = plot(Uh[0])
      # plt.colorbar(c)
      # plt.show()

      # plt.title("u_y")
      # c = plot(Uh[1])
      # plt.colorbar(c)
      # plt.show()

      # plt.title("p")
      # c = plot(Uh[2])
      # plt.colorbar(c)
      # plt.show()

    # Update previous solution
    U_n.assign(Uh)
    time_step_number += 1

cpu_time = round(time.time() - cpu_start_time, 5)
print(f"CPU Time: {cpu_time} s \n\n")

# plt.figure(figsize=(10,6))
# plt.title("x-Displacement")
# for _t in times:
#   plt.plot(x_linspace, solutions["u_x"][_t], label=f"t = {_t}")
# plt.legend()
# plt.show()

# plt.figure(figsize=(10,6))
# plt.title("Pressure")
# for _t in times:
#   plt.plot(x_linspace, solutions["p"][_t], label=f"t = {_t}")
# plt.legend()
# plt.show()

