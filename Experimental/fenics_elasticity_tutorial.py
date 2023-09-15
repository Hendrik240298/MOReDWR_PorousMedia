from fenics import *
from ufl import nabla_div, nabla_grad
import numpy as np

mu = 1
rho = 1
delta = 0.2
gamma = 0.4*delta**2
beta = 1.25
lambda_ = beta
g = gamma

# Create mesh and define function space
mesh = BoxMesh(Point(-32, -32, 0), Point(32, 32, 64), 8, 8, 8)

compression = CompiledSubDomain("near(x[2], 64.) && x[0] >= -16. && x[0] <= 16. && x[1] >= -16. && x[1] <= 16. && on_boundary")
dirichlet = CompiledSubDomain("near(x[2], 0.) && on_boundary")
neumann = CompiledSubDomain("( near(x[0], -32.) || near(x[0], 32.) || near(x[1], -32.) || near(x[1], 32.) || (near(x[2], 64.) && (x[0] <= -16. || x[0] >= 16. || x[1] <= -16. || x[1] >= 16.) ) ) && on_boundary")

facet_marker = MeshFunction("size_t", mesh, 1)
facet_marker.set_all(0)
neumann.mark(facet_marker, 3)
compression.mark(facet_marker, 1)
dirichlet.mark(facet_marker, 2)

ds_compression = Measure("ds", subdomain_data=facet_marker, subdomain_id=1)

V = VectorFunctionSpace(mesh, 'P', 1)

# Define boundary condition
tol = 1E-14

def clamped_boundary(x, on_boundary):
    return on_boundary and x[2] < tol

def compression_boundary(x, on_boundary):
    return on_boundary and x[2] > 64. - tol and (np.abs(x[0]) <= 16. and np.abs(x[1]) <= 16.)

bc = DirichletBC(V, Constant((0, 0, 0)), clamped_boundary)
bc_compression = DirichletBC(V, Constant((0, 0, -1.e8)), compression_boundary)

f = Function(V) 
bc_compression.apply(f.vector())

# Define strain and stress

def epsilon(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)
    #return sym(nabla_grad(u))

def sigma(u):
    return lambda_*nabla_div(u)*Identity(d) + 2*mu*epsilon(u)

# Define variational problem
u = TrialFunction(V)
d = u.geometric_dimension()  # space dimension
v = TestFunction(V)
T = Constant((0, 0, -1.e8))
a = inner(sigma(u), epsilon(v))*dx
L = dot(f, v)*dx #dot(T, v)*ds_compression

# Compute solution
u = Function(V)
solve(a == L, u, bc)

vtkfile = File('elasticity_solution.pvd')
vtkfile << u

