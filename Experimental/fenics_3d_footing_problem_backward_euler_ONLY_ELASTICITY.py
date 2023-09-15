import dolfin
from dolfin import *
import os
from ufl import replace, nabla_div, nabla_grad
import matplotlib.pyplot as plt
import numpy as np
import time

##############
# PARAMETERS #
##############

# Traction
traction_x_biot = 0.0
traction_y_biot = 0.0
traction_z_biot = -1.0e+7

# Solid parameters
poisson_ratio_nu = 0.2
lame_coefficient_mu = 1.0e+8
lame_coefficient_lambda = (2. * poisson_ratio_nu * lame_coefficient_mu) / (1.0 - 2. * poisson_ratio_nu)
print(f"λ = {lame_coefficient_mu}, μ = {lame_coefficient_lambda}")


set_log_active(False) # turn off FEniCS logging

mesh = BoxMesh(Point(-32.,-32.,0.), Point(32.,32.,64.), 8, 8, 8)
dim = mesh.geometry().dim()

cpu_start_time = time.time()

V = VectorFunctionSpace(mesh, "CG", 1)
print(f"Number of spatial DoFs: {V.dim()}")

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

bc_down = DirichletBC(V, Constant((0.,0.,0.)), dirichlet) # dirichlet: u = 0
bcs = [bc_down]

# variational problem
U = TrialFunction(V)
Phi = TestFunction(V)

# Stress function
def stress(u):
    return lame_coefficient_lambda*nabla_div(u)*Identity(dim) + lame_coefficient_mu*(nabla_grad(u) + nabla_grad(u).T)

n = FacetNormal(mesh)
A_u = inner(stress(U), grad(Phi))*dx 
L = inner(Constant((traction_x_biot, traction_y_biot, traction_z_biot)), Phi)*ds_compression

Uh = Function(V) # Uh = U_{n+1}: current solution

folder = f"output_elasto"
if not os.path.exists(folder):
    os.makedirs(folder)

# Compute solution
solve_start_time = time.time()
solve(A_u == L, Uh, bcs) #, solver_parameters={'linear_solver':'mumps'})
print(f"    Solve Time: {round(time.time() - solve_start_time, 5)} s")
print(f"    u(0,0,64) = {Uh(0.,0.,64.)}")
print(f"    u(1,1,64) = {Uh(1.,1.,64.)}")
print(f"    u_max = {Uh.vector().max()}")

vtk_displacement = File(f"{folder}/displacement.pvd")
Uh.rename("displacement", "solution")
vtk_displacement.write(Uh)

cpu_time = round(time.time() - cpu_start_time, 5)
print(f"CPU Time: {cpu_time} s \n\n")
