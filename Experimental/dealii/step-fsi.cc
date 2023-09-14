/**
 Thomas Wick 
 RICAM Linz
 Date: May 13, 2016
 E-mail: thomas.wick@ricam.oeaw.ac.at

 ///
 This code is a modification of 
 the ANS article open-source version:

 http://media.archnumsoft.org/10305/

 while replacing the fluid-structure equations 
 by the Biot Lame-Navier system.

 If you use the code or code-pieces, it would be very nice to cite 
 the above paper http://media.archnumsoft.org/10305/
 
 T. Wick; Solving Monolithic Fluid-Structure Interaction Problems
 in Arbitrary Lagrangian Eulerian Coordinates with the deal.II Library,
 Archive of Numerical Software, Vol. 1 (2013), pp. 1-19


 ///
 Specific features:
 1. The coupled Biot system is solved monolithically
 2. In order to solve nonlinear extension, a Newton 
    solver with quasi-Newton steps and line search techniques 
    is implemented.
 3. The current code computes the Mandel benchmark problem.
    The results are provided in the accompanying 
    report:

     Numerical results to Mandel's benchmark in porous media
     using the Biot equations

 4. Disclaimer: At several places there might be 
    relicts from the original FSI code (ALE transformations,
    different material ids, three FE functions (velocity, displacement, pressure)). 
    Just do not worry about them.


 ///
 This code is based deal.II version 8.3.0.
 Corresponding modifications (if applicable) to 
 version 8.4.1 are recommended.

 
 ///
 This code is licensed under the "GNU GPL version 2 or later". See
 license.txt or https://www.gnu.org/licenses/gpl-2.0.html

 Copyright 2012-2016: Thomas Wick 


*/



// Include files
//--------------

// The first step, as always, is to include
// the functionality of these 
// deal.II library files and some C++ header
// files.

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/timer.h>  

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
//#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
// From deal.II 9.x.x
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
//#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/solution_transfer.h>


// C++
#include <fstream>
#include <sstream>

// At the end of this top-matter, we import
// all deal.II names into the global
// namespace:				
using namespace dealii;


void print_as_numpy_arrays_high_resolution(SparseMatrix<double> &matrix,
					                                  std::ostream &     out,
                                            const unsigned int precision)
{
  AssertThrow(out.fail() == false, ExcIO());

  out.precision(precision);
  out.setf(std::ios::scientific, std::ios::floatfield);

  std::vector<int> rows;
  std::vector<int> columns;
  std::vector<double> values;
  rows.reserve(matrix.n_nonzero_elements());
  columns.reserve(matrix.n_nonzero_elements());
  values.reserve(matrix.n_nonzero_elements());

  SparseMatrixIterators::Iterator< double, false > it = matrix.begin();
  for (unsigned int i = 0; i < matrix.m(); i++) {
 	 for (it = matrix.begin(i); it != matrix.end(i); ++it) {
 		rows.push_back(i);
 		columns.push_back(it->column());
		values.push_back(matrix.el(i,it->column()));
 	 }
  }

  for (auto d : values)
    out << d << ' ';
  out << '\n';

  for (auto r : rows)
    out << r << ' ';
  out << '\n';

  for (auto c : columns)
    out << c << ' ';
  out << '\n';
  out << std::flush;

  AssertThrow(out.fail() == false, ExcIO());
}


// First, we define tensors to access 
// certain variables in an easier way in 
// the assembling routines.  
namespace Tensors
{    
  template <int dim> 
    inline
    Tensor<2,dim> 
    get_pI (unsigned int q,
	    std::vector<Vector<double> > old_solution_values)
    {
      Tensor<2,dim> tmp;
      tmp[0][0] =  old_solution_values[q](dim+dim);
      tmp[1][1] =  old_solution_values[q](dim+dim);
      tmp[2][2] =  old_solution_values[q](dim+dim);
      

      return tmp;      
    }

  template <int dim> 
    inline
    Tensor<2,dim> 
    get_pI_LinP (const double phi_i_p)
    {
      Tensor<2,dim> tmp;
      tmp.clear();
      tmp[0][0] = phi_i_p;    
      tmp[1][1] = phi_i_p;
      tmp[2][2] = phi_i_p;
      
      return tmp;
   }

 template <int dim> 
   inline
   Tensor<1,dim> 
   get_grad_p (unsigned int q,
	       std::vector<std::vector<Tensor<1,dim> > > old_solution_grads)	 
   {     
     Tensor<1,dim> grad_p;     
     grad_p[0] =  old_solution_grads[q][dim+dim][0];
     grad_p[1] =  old_solution_grads[q][dim+dim][1];
     grad_p[2] =  old_solution_grads[q][dim+dim][2];
      
     return grad_p;
   }

 template <int dim> 
  inline
  Tensor<1,dim> 
  get_grad_p_LinP (const Tensor<1,dim> phi_i_grad_p)	 
    {
      Tensor<1,dim> grad_p;      
      grad_p[0] =  phi_i_grad_p[0];
      grad_p[1] =  phi_i_grad_p[1];
      grad_p[2] =  phi_i_grad_p[2];
	   
      return grad_p;
   }

 template <int dim> 
   inline
   Tensor<2,dim> 
   get_grad_u (unsigned int q,
	       std::vector<std::vector<Tensor<1,dim> > > old_solution_grads)	 
   {   
      Tensor<2,dim> structure_continuation;     
      structure_continuation[0][0] = old_solution_grads[q][dim][0];
      structure_continuation[0][1] = old_solution_grads[q][dim][1];
      structure_continuation[0][2] = old_solution_grads[q][dim][2];

      structure_continuation[1][0] = old_solution_grads[q][dim+1][0];
      structure_continuation[1][1] = old_solution_grads[q][dim+1][1];
      structure_continuation[1][2] = old_solution_grads[q][dim+1][2];

      structure_continuation[2][0] = old_solution_grads[q][dim+2][0];
      structure_continuation[2][1] = old_solution_grads[q][dim+2][1];
      structure_continuation[2][2] = old_solution_grads[q][dim+2][2];

      return structure_continuation;
   }


template <int dim> 
inline
double
get_divergence_u_LinU (const Tensor<2,dim> phi_i_grads_u)
{
  return (phi_i_grads_u[0][0] + phi_i_grads_u[1][1] + phi_i_grads_u[2][2]);
}




  template <int dim> 
    inline
    Tensor<2,dim> 
    get_Identity ()
    {   
      Tensor<2,dim> identity;
      identity[0][0] = 1.0;
      identity[0][1] = 0.0;
      identity[0][2] = 0.0;

      identity[1][0] = 0.0;
      identity[1][1] = 1.0;
      identity[1][2] = 0.0;  

      identity[2][0] = 0.0;
      identity[2][1] = 0.0;
      identity[2][2] = 1.0;  
   
      return identity;      
   }


 template <int dim> 
 inline
 Tensor<1,dim> 
 get_v (unsigned int q,
	std::vector<Vector<double> > old_solution_values)
    {
      Tensor<1,dim> v;	    
      v[0] = old_solution_values[q](0);
      v[1] = old_solution_values[q](1);
      v[2] = old_solution_values[q](2);
      
      return v;    
   }



 template <int dim> 
 inline
 Tensor<1,dim> 
 get_u (unsigned int q,
	std::vector<Vector<double> > old_solution_values)
   {
     Tensor<1,dim> u;     
     u[0] = old_solution_values[q](dim);
     u[1] = old_solution_values[q](dim+1);
     u[2] = old_solution_values[q](dim+2);
     
     return u;          
   }

 template <int dim> 
   inline
   Tensor<1,dim> 
   get_u_LinU (const Tensor<1,dim> phi_i_u)
   {
     Tensor<1,dim> tmp;     
     tmp[0] = phi_i_u[0];
     tmp[1] = phi_i_u[1];
     tmp[2] = phi_i_u[2];
     
     return tmp;    
   }
 


}


// Class for initial values
  template <int dim>
  class InitialValues : public Function<dim>
  {
    public:
      InitialValues () : Function<dim>(dim+dim+1) {}

      virtual double value (const Point<dim>   &p,
			    const unsigned int  component = 0) const;

      virtual void vector_value (const Point<dim> &p,
				 Vector<double>   &value) const;

  };


  template <int dim>
  double
  InitialValues<dim>::value (const Point<dim>  &p,
			     const unsigned int component) const
  {
    // Only pressure
    if (component == 6)   
      {
	return 0.0; // Initial pressure if there is a wish: e.g., 2.6e+7;
      }
    
    return 0.0;
  }


  template <int dim>
  void
  InitialValues<dim>::vector_value (const Point<dim> &p,
				    Vector<double>   &values) const
  {
    for (unsigned int comp=0; comp<this->n_components; ++comp)
      values (comp) = InitialValues<dim>::value (p, comp);
  }





// In the next class, we define the main problem at hand.

// The  program is organized as follows. First, we set up
// runtime parameters and the system as done in other deal.II tutorial steps. 
// Then, we assemble
// the system matrix (Jacobian of Newton's method) 
// and system right hand side (residual of Newton's method) for the non-linear
// system. Two functions for the boundary values are provided because
// we are only supposed to apply boundary values in the first Newton step. In the
// subsequent Newton steps all Dirichlet values have to be equal zero.
// Afterwards, the routines for solving the linear 
// system and the Newton iteration are self-explaining. The following
// function is standard in deal.II tutorial steps:
// writing the solutions to graphical output. 
// The last three functions provide the framework to compute 
// functional values of interest.     
template <int dim>
class Biot_Problem 
{
public:
  
  Biot_Problem (const unsigned int degree);
  ~Biot_Problem (); 
  void run ();
  
private:
  
  // Setup of material parameters, time-stepping scheme
  // spatial grid, etc.
  void set_runtime_parameters ();

  // Create system matrix, rhs and distribute degrees of freedom.
  void setup_system ();

  // Assemble left and right hand side for Newton's method
  void assemble_system_matrix ();   
  void assemble_system_rhs ();
  
  // Boundary conditions (bc)
  void set_initial_bc ();
  void set_newton_bc ();
  
  // Linear solver
  void solve ();

  // Nonlinear solver
  void newton_iteration();

  // Graphical visualization of output			  
  void output_results (const unsigned int refinement_cycle,
		       const Vector<double> solution) const;

  // Evaluation of functional values   
  double compute_point_value (Point<dim> p,
			      const unsigned int component) const;
  
  void compute_functional_values (); 

  const unsigned int   degree;
  
  Triangulation<dim>   triangulation_old;
  Triangulation<dim>   triangulation;
  FESystem<dim>        fe;
  DoFHandler<dim>      dof_handler;

  AffineConstraints<double>     constraints;
  
  SparsityPattern      sparsity_pattern; 
  SparseMatrix<double> system_matrix; 
  
  Vector<double> solution, newton_update, old_timestep_solution;
  Vector<double> system_rhs;

  
  TimerOutput         timer;
  
  // Global variables for timestepping scheme   
  unsigned int timestep_number;
  unsigned int max_no_timesteps;  
  double timestep, theta, time; 
  std::string time_stepping_scheme;


  double force_structure_x_biot, force_structure_y_biot;	  
  double force_structure_x, force_structure_y;	  
  
  // Biot parameters
  double c_biot, alpha_biot, viscosity_biot, 
    K_biot, density_biot;

  double gravity_x, gravity_y, volume_source, traction_x, traction_y, traction_z, 
    traction_x_biot, traction_y_biot, traction_z_biot;
  
  
  // Structure parameters
  double density_structure; 
  double lame_coefficient_mu, lame_coefficient_lambda, poisson_ratio_nu;  

  
  

};


// The constructor of this class is comparable 
// to other tutorials steps, e.g., step-22, and step-31. 
// We are going to use the following finite element discretization: 
// Q_1^c for the fluid (not used here!!!), 
// Q_2^c for the structure, 
// Q_1^c for the pressure
template <int dim>
Biot_Problem<dim>::Biot_Problem (const unsigned int degree)
                :
                degree (degree),
		triangulation (Triangulation<dim>::maximum_smoothing),
                fe (FE_Q<dim>(degree), dim,                    
		    FE_Q<dim>(degree+1), dim,		    
		    FE_Q<dim>(degree), 1),
                dof_handler (triangulation),
		timer (std::cout, TimerOutput::summary, TimerOutput::cpu_times)		
{}


// This is the standard destructor.
template <int dim>
Biot_Problem<dim>::~Biot_Problem () 
{}


// In this method, we set up runtime parameters that 
// could also come from a paramter file. 
template <int dim>
void Biot_Problem<dim>::set_runtime_parameters ()
{
  // Biot parameters:
  // M_biot = Biot's constant
  double M_biot = 2.5e+12; 
  c_biot = 1.0/M_biot;
  
  
  // alpha_biot = b_biot = Biot's modulo
  alpha_biot = 1.0; 
  viscosity_biot = 1.0e-3; 
  K_biot = 1.0e-13; 
  density_biot = 1.0;



  // Outer forces
  gravity_x = 0.0;
  gravity_y = 0.0;
  volume_source = 0.0;

  force_structure_x_biot = 0.0;
  force_structure_y_biot = 0.0;

  // Elasticity right hand side 
  // should be zero in Mandel's problem
  force_structure_x = 0.0;
  force_structure_y = 0.0;

  // Traction 
  traction_x_biot = 0.0;
  traction_y_biot = 0.0;
  traction_z_biot = -1.0e+7;

  traction_x = 0.0;
  traction_y = 0.0;
  traction_z = 0.0;



  // Solid parameters
  density_structure = 1.0; 
  lame_coefficient_mu = 1.0e+8; 
  poisson_ratio_nu = 0.2; 

  lame_coefficient_lambda =  (2 * poisson_ratio_nu * lame_coefficient_mu)/
    (1.0 - 2 * poisson_ratio_nu);

  
  // Timestepping schemes
  //BE, CN, CN_shifted
  time_stepping_scheme = "BE";

  // Timestep size:
  timestep = 1.0e+3;

  // Compute 5000 time steps, which 
  // corresponds to the final time 5.0e+6s 
  max_no_timesteps = 5001; 
 

 
  // A variable to count the number of time steps
  timestep_number = 1;

  // Counts total time  
  time = 0;
 
  // Here, we choose a time-stepping scheme that
  // is based on finite differences:
  // BE         = backward Euler scheme 
  // CN         = Crank-Nicolson scheme
  // CN_shifted = time-shifted Crank-Nicolson scheme 
  // For further properties of these schemes,
  // we refer to standard literature.
  if (time_stepping_scheme == "BE")
    theta = 1.0;
  else if (time_stepping_scheme == "CN")
    theta = 0.5;
  else if (time_stepping_scheme == "CN_shifted")
    theta = 0.5 + timestep;
  else 
    std::cout << "No such timestepping scheme" << std::endl;

  std::string grid_name;
  //grid_name  = "rectangle_mandel.inp"; 
    grid_name  = "unit_cube_64.inp"; 
  
  GridIn<dim> grid_in;
  grid_in.attach_triangulation (triangulation);
  std::ifstream input_file(grid_name.c_str());      
  //Assert (dim==2, ExcInternalError());
  grid_in.read_ucd (input_file); 
    
  triangulation.refine_global (3); 
  
  // TODO: adapt to simplex mesh
  //GridGenerator::convert_hypercube_to_simplex_mesh(triangulation_old,triangulation ); 	
  //triangulation.refine_global(3);
  //GridGenerator::subdivided_hyper_rectangle_with_simplices(triangulation, {8, 8, 8}, Point<3>(-32., -32., 0), Point<3>(32., 32., 64.));
  
  // Set boundary ids for \Gamma compression
  typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();


  unsigned int number_of_vertices;
  for (; cell!=endc; ++cell)
    { 

      for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
	{
	  
	  if (cell->face(face)->at_boundary() && 		  
	      (cell->face(face)->boundary_id() == 1) 
	      )
	    {

	  number_of_vertices = 1;
	  // set material_ids
	  for (unsigned int vertex=0;
	       vertex < GeometryInfo<dim>::vertices_per_cell;
	       ++vertex)
	    {
	      //std::cout << "Bin drin" << std::endl;
	      Tensor<1,dim> my_vertex = (cell->vertex(vertex));
	      if (my_vertex[0] <= 16.0 && my_vertex[0] >= -16.0 && my_vertex[2] == 64.0 &&
		  my_vertex[1] <= 16.0 && my_vertex[1] >= -16.0 && my_vertex[2] == 64.0
		  ) 
		{
		  //std::cout << "Bin drin" << std::endl;
		number_of_vertices++;
		cell->face(face)->set_boundary_id(6);  
		}
	  }


//      if (number_of_vertices == GeometryInfo<dim>::vertices_per_cell)
//	{
//	  //std::cout << GeometryInfo<dim>::vertices_per_cell << std::endl;
//	  //std::cout << "Bin drin" << std::endl;
//	cell->set_boundary_id(6);    
//	}
//      else
//	cell->set_boundary_id(1); 

	    }
	  
	}
      
    }
  


}



// This function is similar to many deal.II tuturial steps.
template <int dim>
void Biot_Problem<dim>::setup_system ()
{
  TimerOutput::Scope timer_section(timer, "Setup system.");

  // We set runtime parameters to drive the problem.
  // These parameters could also be read from a parameter file that
  // can be handled by the ParameterHandler object (see step-19)
  set_runtime_parameters ();

  system_matrix.clear ();
  
  dof_handler.distribute_dofs (fe);  
  DoFRenumbering::Cuthill_McKee (dof_handler);

  // We are dealing with 5 components for this 
  // two-dimensional Biot problem
  // (the first component is actually unnecessary 
  // and a relict from the original fluid-structure 
  // interaction code. But this component 
  // might be used for future extensions with other equations.
  // velocity in x and y:                0
  // solid displacement in x and y:      1
  // scalar pressure field:              2
  std::vector<unsigned int> block_component (7,0);
  block_component[dim] = 1;
  block_component[dim+1] = 1;
  block_component[dim+2] = 1;
  block_component[dim+dim] = 2;
 
  DoFRenumbering::component_wise (dof_handler, block_component);

  {				 
    constraints.clear ();
    set_newton_bc ();
    DoFTools::make_hanging_node_constraints (dof_handler,
					     constraints);
  }
  constraints.close ();
  
  std::vector<unsigned int> dofs_per_block (3);
  dofs_per_block = DoFTools::count_dofs_per_fe_block (dof_handler, block_component); 
  const unsigned int n_v = dofs_per_block[0],
    n_u = dofs_per_block[1],
    n_p =  dofs_per_block[2];

  std::cout << "Cells:\t"
            << triangulation.n_active_cells()
            << std::endl  	  
            << "DoFs:\t"
            << dof_handler.n_dofs()
            << " (" << n_v << '+' << n_u << '+' << n_p <<  ')'
            << std::endl;


 
      
 {
    // BlockDynamicSparsityPattern csp (3,3);
    DynamicSparsityPattern csp (dof_handler.n_dofs()); //3,3);

    // csp.block(0,0).reinit (n_v, n_v);
    // csp.block(0,1).reinit (n_v, n_u);
    // csp.block(0,2).reinit (n_v, n_p);
  
    // csp.block(1,0).reinit (n_u, n_v);
    // csp.block(1,1).reinit (n_u, n_u);
    // csp.block(1,2).reinit (n_u, n_p);
  
    // csp.block(2,0).reinit (n_p, n_v);
    // csp.block(2,1).reinit (n_p, n_u);
    // csp.block(2,2).reinit (n_p, n_p);
 
    // csp.collect_sizes();    
  

    DoFTools::make_sparsity_pattern (dof_handler, csp, constraints, false);

    sparsity_pattern.copy_from (csp);
  }
 
 system_matrix.reinit (sparsity_pattern);

  // Actual solution at time step n
  solution.reinit (dof_handler.n_dofs()); //3);
  // solution.block(0).reinit (n_v);
  // solution.block(1).reinit (n_u);
  // solution.block(2).reinit (n_p);
 
  // solution.collect_sizes ();
 
  // Old timestep solution at time step n-1
  old_timestep_solution.reinit (dof_handler.n_dofs()); //3);
  // old_timestep_solution.block(0).reinit (n_v);
  // old_timestep_solution.block(1).reinit (n_u);
  // old_timestep_solution.block(2).reinit (n_p);
 
  // old_timestep_solution.collect_sizes ();


  // Updates for Newton's method
  newton_update.reinit (dof_handler.n_dofs()); //3);
  // newton_update.block(0).reinit (n_v);
  // newton_update.block(1).reinit (n_u);
  // newton_update.block(2).reinit (n_p);
 
  // newton_update.collect_sizes ();
 
  // Residual for  Newton's method
  system_rhs.reinit (dof_handler.n_dofs());
  // system_rhs.block(0).reinit (n_v);
  // system_rhs.block(1).reinit (n_u);
  // system_rhs.block(2).reinit (n_p);

  // system_rhs.collect_sizes ();

   
}


// In this function, we assemble the Jacobian matrix
// for the Newton iteration. 
//
// Assembling of the inner most loop is treated with help of 
// the fe.system_to_component_index(j).first function from
// the library. 
// Using this function makes the assembling process much faster
// than running over all local degrees of freedom. 
template <int dim>
void Biot_Problem<dim>::assemble_system_matrix ()
{
  TimerOutput::Scope timer_section(timer, "Assemble Matrix.");
  system_matrix=0;
     
  QGauss<dim>   quadrature_formula(degree+2);  
  QGauss<dim-1> face_quadrature_formula(degree+2);

  FEValues<dim> fe_values (fe, quadrature_formula,
                           update_values    |
                           update_quadrature_points  |
                           update_JxW_values |
                           update_gradients);
  
  FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula, 
				    update_values         | update_quadrature_points  |
				    update_normal_vectors | update_gradients |
				    update_JxW_values);
   
  const unsigned int   dofs_per_cell   = fe.dofs_per_cell;
  
  const unsigned int   n_q_points      = quadrature_formula.size();
  const unsigned int n_face_q_points   = face_quadrature_formula.size();

  FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);

  std::vector<unsigned int> local_dof_indices (dofs_per_cell); 
		

  // Now, we are going to use the 
  // FEValuesExtractors to determine
  // the four principle variables
  const FEValuesExtractors::Vector velocities (0);
  const FEValuesExtractors::Vector displacements (dim); // 2
  const FEValuesExtractors::Scalar pressure (dim+dim); // 4

  // We declare Vectors and Tensors for 
  // the solutions at the previous Newton iteration:
  std::vector<Vector<double> > old_solution_values (n_q_points, 
				 		    Vector<double>(dim+dim+1));

  std::vector<std::vector<Tensor<1,dim> > > old_solution_grads (n_q_points, 
								std::vector<Tensor<1,dim> > (dim+dim+1));

  std::vector<Vector<double> >  old_solution_face_values (n_face_q_points, 
							  Vector<double>(dim+dim+1));
       
  std::vector<std::vector<Tensor<1,dim> > > old_solution_face_grads (n_face_q_points, 
								     std::vector<Tensor<1,dim> > (dim+dim+1));
    

  // We declare Vectors and Tensors for 
  // the solution at the previous time step:
   std::vector<Vector<double> > old_timestep_solution_values (n_q_points, 
				 		    Vector<double>(dim+dim+1));


  std::vector<std::vector<Tensor<1,dim> > > old_timestep_solution_grads (n_q_points, 
  					  std::vector<Tensor<1,dim> > (dim+dim+1));


  std::vector<Vector<double> >   old_timestep_solution_face_values (n_face_q_points, 
								    Vector<double>(dim+dim+1));
  
    
  std::vector<std::vector<Tensor<1,dim> > >  old_timestep_solution_face_grads (n_face_q_points, 
									       std::vector<Tensor<1,dim> > (dim+dim+1));
   
  // Declaring test functions:
  std::vector<Tensor<1,dim> > phi_i_v (dofs_per_cell); 
  std::vector<Tensor<2,dim> > phi_i_grads_v(dofs_per_cell);
  std::vector<double>         phi_i_p(dofs_per_cell); 
  std::vector<Tensor<1,dim> > phi_i_grads_p (dofs_per_cell);
  std::vector<Tensor<1,dim> > phi_i_u (dofs_per_cell); 
  std::vector<Tensor<2,dim> > phi_i_grads_u(dofs_per_cell);

  // This is the identity matrix in two dimensions:
  const Tensor<2,dim> Identity = Tensors
    ::get_Identity<dim> ();
 				     				   
  typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
  

  unsigned int cell_counter = 0;
  for (; cell!=endc; ++cell)
    { 
      fe_values.reinit (cell);
      local_matrix = 0;
      
      
      // Old Newton iteration values
      fe_values.get_function_values (solution, old_solution_values);
      fe_values.get_function_gradients (solution, old_solution_grads);
      
      // Old_timestep_solution values
      fe_values.get_function_values (old_timestep_solution, old_timestep_solution_values);
      fe_values.get_function_gradients (old_timestep_solution, old_timestep_solution_grads);
      
      // pay-zone with Biot // 1
      if (cell->material_id() == 1)
	{
	  for (unsigned int q=0; q<n_q_points; ++q)
	    {	      
	      for (unsigned int k=0; k<dofs_per_cell; ++k)
		{
		  phi_i_v[k]       = fe_values[velocities].value (k, q);
		  phi_i_grads_v[k] = fe_values[velocities].gradient (k, q);
		  phi_i_p[k]       = fe_values[pressure].value (k, q);	
		  phi_i_grads_p[k] = fe_values[pressure].gradient (k, q);
		  phi_i_u[k]       = fe_values[displacements].value (k, q);
		  phi_i_grads_u[k] = fe_values[displacements].gradient (k, q);
		}
	      
	      	    
	      	      
	      for (unsigned int i=0; i<dofs_per_cell; ++i)
		{	    	     		
		  const Tensor<2,dim> pI_LinP = Tensors
		    ::get_pI_LinP<dim> (phi_i_p[i]);
		  
   
		  // STVK: Green-Lagrange strain tensor derivatives
		  const Tensor<2,dim> E_LinU = 0.5 * (phi_i_grads_u[i] + transpose(phi_i_grads_u[i]));
		  
		  const double tr_E_LinU = trace(E_LinU);
		  
		       
		  // STVK
		  // Piola-kirchhoff stress structure STVK linearized in all directions 		  
		  Tensor<2,dim> piola_kirchhoff_stress_structure_STVK_LinALL;
		  piola_kirchhoff_stress_structure_STVK_LinALL = lame_coefficient_lambda * 
		    tr_E_LinU * Identity + 2 * lame_coefficient_mu * E_LinU;
		    
		      
		  const double divergence_u_LinU = Tensors
		    ::get_divergence_u_LinU<dim> (phi_i_grads_u[i]);
			   
		  for (unsigned int j=0; j<dofs_per_cell; ++j)
		    {
		      // Biot-Lame-Navier
		      const unsigned int comp_j = fe.system_to_component_index(j).first; 
		      if (comp_j == 0 || comp_j == 1 || comp_j == 2)
			{
			  // This first block should be zero and is a remaining 
			  // relict from the original FSI code.
			  local_matrix(j,i) += (
						phi_i_v[i] * phi_i_v[j]
						) *  fe_values.JxW(q);			  
			}		     
		      else if (comp_j == 3 || comp_j == 4 || comp_j == 5)
			{
			  // Elasticity part
			  local_matrix(j,i) += (scalar_product(piola_kirchhoff_stress_structure_STVK_LinALL, 
							       phi_i_grads_u[j])
						+  alpha_biot * scalar_product(-pI_LinP, phi_i_grads_u[j])	  
						) * fe_values.JxW(q);      	



			}
		      else if (comp_j == 6)
			{
			  // Pressure equation
			  local_matrix(j,i) += (c_biot * phi_i_p[i] * phi_i_p[j] + 
						alpha_biot * divergence_u_LinU * phi_i_p[j] +
						 timestep * theta * 1.0/viscosity_biot  
						  * K_biot * phi_i_grads_p[i] * phi_i_grads_p[j]	
						) * fe_values.JxW(q);      

			} // end comp_j
		      
		    }  // end j dofs
		  	     
		}  // end i dofs	
	      
	    }  // end n_q_points 


	  // upper traction (pay zone) if used only for Mandels problem
	  for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
	    {
	      if (cell->face(face)->at_boundary() && 		  
		  (cell->face(face)->boundary_id() == 6) 
		  )
		{
		  
		  fe_face_values.reinit (cell, face);
		  
		  fe_face_values.get_function_values (solution, old_solution_face_values);
		  fe_face_values.get_function_values (old_timestep_solution, old_timestep_solution_face_values);
		
	 
		  for (unsigned int q=0; q<n_face_q_points; ++q)
		    {	
		      //Tensor<1,dim> neumann_value;
		      //neumann_value[0] = traction_x_biot;
		      //neumann_value[1] = traction_y_biot;

		      for (unsigned int k=0; k<dofs_per_cell; ++k)
			{
			  //phi_i_v[k]       = fe_face_values[velocities].value (k, q);
			  //phi_i_grads_v[k] = fe_face_values[velocities].gradient (k, q);
			  phi_i_u[k]         = fe_face_values[displacements].value (k, q);
			  //phi_i_grads_u[k] = fe_face_values[displacements].gradient (k, q);
			  phi_i_p[k]         = fe_face_values[pressure].value (k, q);	
			  //phi_i_grads_p[k] = fe_face_values[pressure].gradient (k, q);
			}
			

		      for (unsigned int i=0; i<dofs_per_cell; ++i)
			{
			  double fluid_pressure = phi_i_p[i];
			  
			  for (unsigned int j=0; j<dofs_per_cell; ++j)
			    {	
			      
			      const unsigned int comp_j = fe.system_to_component_index(j).first; 
			      if (comp_j == 3 || comp_j == 4 || comp_j == 5)
				{  
				  local_matrix(j,i) -=  (// zero in system matrix : neumann_value * fe_face_values[displacements].value (i, q)
						    - alpha_biot * fluid_pressure * fe_face_values.normal_vector(q) * 
						    phi_i_u[j]
						    ) * fe_face_values.JxW(q);					   
				} // end comp_j

			    } // end j
			 
			}   // end i
		      
		    }  // end face_n_q_points                                 

		}  // end boundary id

	    }  // end face integrals



	  
	  cell->get_dof_indices (local_dof_indices);
	  constraints.distribute_local_to_global (local_matrix, local_dof_indices,
						  system_matrix);



	} 
      // non-pay zone
      else if (cell->material_id() == 0)
	{
	  // Second example: augmented Mandel's test, but 
	  // not implemented in this code.
	  abort();
	  
	} 
      

      cell_counter++;
    
    }  // end cell
  
  
}



// In this function we assemble the semi-linear 
// of the right hand side of Newton's method (its residual).
// The framework is in principal the same as for the 
// system matrix.
template <int dim>
void
Biot_Problem<dim>::assemble_system_rhs ()
{
  TimerOutput::Scope timer_section(timer, "Assemble Rhs.");
  system_rhs=0;
  
  QGauss<dim>   quadrature_formula(degree+2);
  QGauss<dim-1> face_quadrature_formula(degree+2);

  FEValues<dim> fe_values (fe, quadrature_formula,
                           update_values    |
                           update_quadrature_points  |
                           update_JxW_values |
                           update_gradients);

  FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula, 
				    update_values         | update_quadrature_points  |
				    update_normal_vectors | update_gradients |
				    update_JxW_values);

  const unsigned int   dofs_per_cell   = fe.dofs_per_cell;
  
  const unsigned int   n_q_points      = quadrature_formula.size();
  const unsigned int n_face_q_points   = face_quadrature_formula.size();
 
  Vector<double>       local_rhs (dofs_per_cell);

  std::vector<unsigned int> local_dof_indices (dofs_per_cell);
  
  const FEValuesExtractors::Vector velocities (0);
  const FEValuesExtractors::Vector displacements (dim); 
  const FEValuesExtractors::Scalar pressure (dim+dim); 
 
  std::vector<Vector<double> > 
    old_solution_values (n_q_points, Vector<double>(dim+dim+1));

  std::vector<std::vector<Tensor<1,dim> > > 
    old_solution_grads (n_q_points, std::vector<Tensor<1,dim> > (dim+dim+1));


  std::vector<Vector<double> > 
    old_solution_face_values (n_face_q_points, Vector<double>(dim+dim+1));
  
  std::vector<std::vector<Tensor<1,dim> > > 
    old_solution_face_grads (n_face_q_points, std::vector<Tensor<1,dim> > (dim+dim+1));
  
  std::vector<Vector<double> > 
    old_timestep_solution_values (n_q_points, Vector<double>(dim+dim+1));

  std::vector<std::vector<Tensor<1,dim> > > 
    old_timestep_solution_grads (n_q_points, std::vector<Tensor<1,dim> > (dim+dim+1));

  std::vector<Vector<double> > 
    old_timestep_solution_face_values (n_face_q_points, Vector<double>(dim+dim+1));
     
  std::vector<std::vector<Tensor<1,dim> > > 
    old_timestep_solution_face_grads (n_face_q_points, std::vector<Tensor<1,dim> > (dim+dim+1));
   
  const Tensor<2,dim> Identity = Tensors
    ::get_Identity<dim> ();

  
  typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();

  unsigned int cell_counter = 0;
  for (; cell!=endc; ++cell)
    { 

      fe_values.reinit (cell);	 
      local_rhs = 0;   	
      
      
      // old Newton iteration
      fe_values.get_function_values (solution, old_solution_values);
      fe_values.get_function_gradients (solution, old_solution_grads);
            
      // old timestep iteration
      fe_values.get_function_values (old_timestep_solution, old_timestep_solution_values);
      fe_values.get_function_gradients (old_timestep_solution, old_timestep_solution_grads);
      
      // pay-zone with Biot  // 1
      if (cell->material_id() == 1)
	{
	  for (unsigned int q=0; q<n_q_points; ++q)
	    {	
	      const Tensor<2,dim> pI = Tensors	
		::get_pI<dim> (q, old_solution_values);
	 		 	      
	      const Tensor<1,dim> grad_p = Tensors 
		::get_grad_p<dim> (q, old_solution_grads);

	      const Tensor<1,dim> v = Tensors
		::get_v<dim> (q, old_solution_values);
	  
	      const Tensor<2,dim> grad_u = Tensors 
		::get_grad_u<dim> (q, old_solution_grads);
	      
	      
	      const Tensor<2,dim> E = 0.5 * (grad_u + transpose(grad_u));
	      double tr_E = trace(E);

	      Tensor<2,dim> stress_term;
	      stress_term.clear();
	      stress_term = lame_coefficient_lambda * tr_E * Identity +
		2 * lame_coefficient_mu * E;
					  


	      Tensor<2,dim> fluid_pressure;
	      fluid_pressure.clear();
	      fluid_pressure = (-pI);

//	      Tensor<1,dim> g_biot;
//	      g_biot.clear();
//	      g_biot[0] = gravity_x;
//	      g_biot[1] = gravity_y;
//
//	      Tensor<1,dim> old_timestep_g_biot;
//	      old_timestep_g_biot.clear();
//	      old_timestep_g_biot[0] = gravity_x;
//	      old_timestep_g_biot[1] = gravity_y;
//
	      double q_biot = volume_source;
	      
	      const double divergence_u = old_solution_grads[q][dim][0] +  old_solution_grads[q][dim+1][1] +  old_solution_grads[q][dim+2][2];

	      const double old_timestep_divergence_u = 
		old_timestep_solution_grads[q][dim][0] +  old_timestep_solution_grads[q][dim+1][1] +  old_timestep_solution_grads[q][dim+2][2];


	       
	      //Tensor<1,dim> structure_force;
	      //structure_force.clear();
	      //structure_force[0] = force_structure_x_biot;
	      //structure_force[1] = force_structure_y_biot;
	      
	      	      
	      for (unsigned int i=0; i<dofs_per_cell; ++i)
		{
		  // Biot-Navier-Lame
		  const unsigned int comp_i = fe.system_to_component_index(i).first; 
		  if (comp_i == 0 || comp_i == 1 || comp_i == 2)
		    { 
		      const Tensor<1,dim> phi_i_v = fe_values[velocities].value (i, q);

		      // Currently without any use.
		      local_rhs(i) -=  (
					v * phi_i_v
					) * fe_values.JxW(q);    

		      
		    }		
		  else if (comp_i == 3 || comp_i == 4 || comp_i == 5)
		    {
		      const Tensor<1,dim> phi_i_u       = fe_values[displacements].value (i, q);
		      const Tensor<2,dim> phi_i_grads_u = fe_values[displacements].gradient (i, q);

		      local_rhs(i) -= (scalar_product(stress_term,phi_i_grads_u)   
				       + alpha_biot * scalar_product(fluid_pressure, phi_i_grads_u)	
				       // Right hand side (e.g. gravitation)
				       //- structure_force * phi_i_u   
				       ) * fe_values.JxW(q);    
		    }
		  else if (comp_i == 6)
		    {
		      const double phi_i_p = fe_values[pressure].value (i, q);
		      const Tensor<1,dim> phi_i_grads_p = fe_values[pressure].gradient (i, q);
		      local_rhs(i) -= (c_biot * (old_solution_values[q](dim+dim) 
				       	 - old_timestep_solution_values[q](dim+dim)) * phi_i_p +
				        alpha_biot * (divergence_u - old_timestep_divergence_u) * phi_i_p +
				        timestep * theta * 1.0/viscosity_biot  
				         * K_biot * grad_p * phi_i_grads_p
				       - timestep *  theta * q_biot * phi_i_p
				       // Right hand side values
				       //- timestep * theta * density_biot * 1.0/viscosity_biot 
				       //  * K_biot * g_biot * phi_i_grads_p
				       ) * fe_values.JxW(q);  
		      
		    } // end comp_i
		  	  
		} // end i	
	      		   
	    } // end n_q_points 
	  
	  // upper traction (pay zone) if used only for Mandels problem
	  for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
	    {
	      if (cell->face(face)->at_boundary() && 		  
		  (cell->face(face)->boundary_id() == 6) 
		  )
		{
		  
		  fe_face_values.reinit (cell, face);
		  
		  fe_face_values.get_function_values (solution, old_solution_face_values);
		  fe_face_values.get_function_values (old_timestep_solution, old_timestep_solution_face_values);
		
	 
		  for (unsigned int q=0; q<n_face_q_points; ++q)
		    {	
		      Tensor<1,dim> neumann_value;
		      neumann_value[0] = traction_x_biot;
		      neumann_value[1] = traction_y_biot;
		      neumann_value[2] = traction_z_biot;
			
		      //double fluid_pressure = old_timestep_solution_face_values[q](dim+dim);
		      double fluid_pressure = old_solution_face_values[q](dim+dim);

		      for (unsigned int i=0; i<dofs_per_cell; ++i)
			{
			  const unsigned int comp_i = fe.system_to_component_index(i).first; 
			  if (comp_i == 3 || comp_i == 4 || comp_i == 5)
			    {  
			      local_rhs(i) +=  (neumann_value * fe_face_values[displacements].value (i, q)
						- alpha_biot * fluid_pressure * fe_face_values.normal_vector(q) * 
						fe_face_values[displacements].value (i, q)
						) * fe_face_values.JxW(q);					   
			    }
			  // end i
			}  
		      // end face_n_q_points    
		    }                                     
		} 
	    }  // end face integrals



	  cell->get_dof_indices (local_dof_indices);
	  constraints.distribute_local_to_global (local_rhs, local_dof_indices,
						  system_rhs);
	  
	// end if (for material id 0)  
	}   
      // non-pay zone
      else if (cell->material_id() == 0)
	{
	  abort();
	}   
      
      cell_counter++;

    }  // end cell
      
}


// Here, we impose boundary conditions
// for the whole system. 
template <int dim>
void
Biot_Problem<dim>::set_initial_bc ()
{ 
    std::map<unsigned int,double> boundary_values;  
    std::vector<bool> component_mask (dim+dim+1, true);
    // (Scalar) pressure
    component_mask[dim+dim] = false; // false  
 
    component_mask[dim]       = true;
    component_mask[dim+1]     = true;
    component_mask[dim+2]     = true;
    component_mask[dim+dim]   = true;
    VectorTools::interpolate_boundary_values (dof_handler,
					      0,
					      ZeroFunction<dim>(dim+dim+1),  
					      boundary_values,
					      component_mask);  

    component_mask[dim]       = false;
    component_mask[dim+1]     = false;
    component_mask[dim+2]     = false;
    component_mask[dim+dim]   = false;
    VectorTools::interpolate_boundary_values (dof_handler,
					      1,
					      ZeroFunction<dim>(dim+dim+1),					
					      boundary_values,
					      component_mask);

 
//    component_mask[dim]     =  false;
//    component_mask[dim+1]   =  true;
//    component_mask[dim+dim] =  false;
//    VectorTools::interpolate_boundary_values (dof_handler,
//                                              2,
//					      ZeroFunction<dim>(dim+dim+1), 					 
//                                              boundary_values,
//                                              component_mask);
// 
//
//    component_mask[dim]     = false;
//    component_mask[dim+1]   = false; 
//    component_mask[dim+dim] = false;  // false  
//    VectorTools::interpolate_boundary_values (dof_handler,
//					      3,
//					      ZeroFunction<dim>(dim+dim+1),  
//					      boundary_values,
//					      component_mask);
//    
    

    
    for (typename std::map<unsigned int, double>::const_iterator
	   i = boundary_values.begin();
	 i != boundary_values.end();
	 ++i)
      solution(i->first) = i->second;
    
}

// This function applies boundary conditions 
// to the Newton iteration steps. For all variables that
// have non-homogeneous Dirichlet conditions on some (or all) parts
// of the outer boundary, we apply zero-Dirichlet
// conditions, now. 
template <int dim>
void
Biot_Problem<dim>::set_newton_bc ()
{
    std::vector<bool> component_mask (dim+dim+1, true);


    component_mask[dim]       = true;
    component_mask[dim+1]     = true;
    component_mask[dim+2]     = true;
    component_mask[dim+dim]   = true;  
    VectorTools::interpolate_boundary_values (dof_handler,
					      0,
					      ZeroFunction<dim>(dim+dim+1), 
					      constraints,				
					      component_mask);  

    component_mask[dim]       = false;
    component_mask[dim+1]     = false;
    component_mask[dim+2]     = false;
    component_mask[dim+dim]   = false;
    VectorTools::interpolate_boundary_values (dof_handler,
					      1,
					      ZeroFunction<dim>(dim+dim+1),  
					      constraints,
					      component_mask);

 
//    component_mask[dim]       = false;
//    component_mask[dim+1]     = true;
//    component_mask[dim+dim]   = false;
//    VectorTools::interpolate_boundary_values (dof_handler,
//                                              2,
//					      ZeroFunction<dim>(dim+dim+1),  
//					      constraints,
//                                              component_mask);
// 
//
//    component_mask[dim]     = false;
//    component_mask[dim+1]   = false; 
//    component_mask[dim+dim] = false;  
//
//    VectorTools::interpolate_boundary_values (dof_handler,
//					      3,
//					      ZeroFunction<dim>(dim+dim+1),  
//					      constraints,		
//					      component_mask);
//
   

    

}  

// In this function, we solve the linear systems
// inside the nonlinear Newton iteration. We just
// use a direct solver from UMFPACK.
template <int dim>
void 
Biot_Problem<dim>::solve () 
{
  TimerOutput::Scope timer_section(timer, "Solve linear system.");
  Vector<double> sol, rhs;    
  sol = newton_update;    
  rhs = system_rhs;
  
  SparseDirectUMFPACK A_direct;
  A_direct.factorize(system_matrix);     
  A_direct.vmult(sol,rhs); 
  newton_update = sol;
  
  constraints.distribute (newton_update);
  
}

// This is the Newton iteration to solve the 
// non-linear system of equations. First, we declare some
// standard parameters of the solution method. Addionally,
// we also implement an easy line search algorithm. 
template <int dim>
void Biot_Problem<dim>::newton_iteration () 
					       
{ 
  Timer timer_newton;
  const double lower_bound_newton_residuum = 1.0e-4; 
  const unsigned int max_no_newton_steps  = 50;

  // Decision whether the system matrix should be build
  // at each Newton step
  const double nonlinear_rho = 0.1; 
 
  // Line search parameters
  unsigned int line_search_step;
  const unsigned int  max_no_line_search_steps = 10;
  const double line_search_damping = 0.6;
  double new_newton_residuum;
  
  // Application of the initial boundary conditions to the 
  // variational equations:
  set_initial_bc ();
  assemble_system_rhs();

  double newton_residuum = system_rhs.linfty_norm(); 
  double old_newton_residuum= newton_residuum;
  unsigned int newton_step = 1;
   
  if (newton_residuum < lower_bound_newton_residuum)
    {
      std::cout << '\t' 
		<< std::scientific 
		<< newton_residuum 
		<< std::endl;     
    }
  
  while (newton_residuum > lower_bound_newton_residuum &&
	 newton_step < max_no_newton_steps)
    {
      timer_newton.start();
      old_newton_residuum = newton_residuum;
      
      assemble_system_rhs();
      newton_residuum = system_rhs.linfty_norm();

      if (newton_residuum < lower_bound_newton_residuum)
	{
	  std::cout << '\t' 
		    << std::scientific 
		    << newton_residuum << std::endl;
	  break;
	}
  
      // Check if matrix needs to re-build. If not 
      // then we save some time and perform quasi-Newton steps.
      if (newton_residuum/old_newton_residuum > nonlinear_rho)
	assemble_system_matrix ();	

      // Solve Ax = b
      solve ();	  
        
      line_search_step = 0;	  
      for ( ; 
	    line_search_step < max_no_line_search_steps; 
	    ++line_search_step)
	{	     					 
	  solution += newton_update;
	  
	  assemble_system_rhs ();			
	  new_newton_residuum = system_rhs.linfty_norm();
	  
	  if (new_newton_residuum < newton_residuum)
	      break;
	  else 	  
	    solution -= newton_update;
	  
	  newton_update *= line_search_damping;
	}	   
     
      timer_newton.stop();
      
      std::cout << std::setprecision(5) <<newton_step << '\t' 
		<< std::scientific << newton_residuum << '\t'
		<< std::scientific << newton_residuum/old_newton_residuum  <<'\t' ;
      if (newton_residuum/old_newton_residuum > nonlinear_rho)
	std::cout << "r" << '\t' ;
      else 
	std::cout << " " << '\t' ;
      std::cout << line_search_step  << '\t' 
		<< std::scientific << timer_newton.cpu_time()
		<< std::endl;


      // Updates
      timer_newton.reset();
      newton_step++;      
    }
}

// This function is known from almost all other 
// tutorial steps in deal.II.
template <int dim>
void
Biot_Problem<dim>::output_results (const unsigned int refinement_cycle,
			      const Vector<double> output_vector)  const
{

  std::vector<std::string> solution_names;
  solution_names.push_back ("x_velo");
  solution_names.push_back ("y_velo");
  solution_names.push_back ("z_velo"); 
  solution_names.push_back ("x_dis");
  solution_names.push_back ("y_dis");
  solution_names.push_back ("z_dis");
  solution_names.push_back ("p_fluid");
   
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation
    (dim+dim+1, DataComponentInterpretation::component_is_scalar);


  DataOut<dim> data_out;
  data_out.attach_dof_handler (dof_handler);  
   
  data_out.add_data_vector (output_vector, solution_names,
			    DataOut<dim>::type_dof_data,			   
			    data_component_interpretation);
  

  data_out.build_patches (0);

  std::string filename_basis;
  filename_basis  = "solution_"; 
   
  std::ostringstream filename_vtk;
  std::ostringstream filename_ucd;

  std::cout << "------------------" << std::endl;
  std::cout << "Write solution" << std::endl;
  std::cout << "------------------" << std::endl;
  std::cout << std::endl;

  filename_vtk << filename_basis
	   << Utilities::int_to_string (refinement_cycle, 5)
	   << ".vtk";

  filename_ucd << filename_basis
	   << Utilities::int_to_string (refinement_cycle, 5)
	   << ".ucd";
  
  std::ofstream output_vtk (filename_vtk.str().c_str());
  data_out.write_vtk (output_vtk);

}

// With help of this function, we extract 
// point values for a certain component from our
// discrete solution. We use it to gain the 
// displacements of the structure in the x- and y-directions.
template <int dim>
double Biot_Problem<dim>::compute_point_value (Point<dim> p, 
					       const unsigned int component) const  
{
 
  Vector<double> tmp_vector(dim+dim+1);
  VectorTools::point_value (dof_handler, 
			    solution, 
			    p, 
			    tmp_vector);
  
  return tmp_vector(component);
}


// Here, we compute the four quantities of interest
template<int dim>
void Biot_Problem<dim>::compute_functional_values()
{
  double x1, x2, x3, x4, x5, p1,p2,p3,p4,p5;
  x1 = compute_point_value(Point<dim>(0.0,0.0,64.0), dim);
  //  x2 = compute_point_value(Point<dim>(25.0,0.0), dim);
  //  x3 = compute_point_value(Point<dim>(50.0,0.0), dim);
  //  x4 = compute_point_value(Point<dim>(75.0,0.0), dim);
  //  x5 = compute_point_value(Point<dim>(100.0,0.0), dim);

  //  double y1, y2, y3, y4;
//  y1 = compute_point_value(Point<dim>(0.0,1.0), dim+1);
//  y2 = compute_point_value(Point<dim>(0.25,1.0), dim+1);
//  y3 = compute_point_value(Point<dim>(0.5,1.0), dim+1);
//  y4 = compute_point_value(Point<dim>(0.75,1.0), dim+1);

  p1 = compute_point_value(Point<dim>(0.0,0.0,64.0),  dim+dim);
  //  p2 = compute_point_value(Point<dim>(25.0,0.0), dim+dim);
  //  p3 = compute_point_value(Point<dim>(50.0,0.0), dim+dim);
  //  p4 = compute_point_value(Point<dim>(75.0,0.0), dim+dim);
  //  p5 = compute_point_value(Point<dim>(100.0,0.0),dim+dim);

  std::cout << "------------------" << std::endl;
  std::cout << "DisX1:  " << x1  << std::endl;
//  std::cout << "DisX2:  " << x2  << std::endl;
//  std::cout << "DisX3:  " << x3  << std::endl;
//  std::cout << "DisX4:  " << x4  << std::endl;
//  std::cout << "DisX5:  " << x5  << std::endl;
  std::cout << "------------------" << std::endl;
  std::cout << "DisP1:  " << p1  << std::endl;
//  std::cout << "DisP2:  " << p2  << std::endl;
//  std::cout << "DisP3:  " << p3  << std::endl;
//  std::cout << "DisP4:  " << p4  << std::endl;
//  std::cout << "DisP5:  " << p5  << std::endl;
  std::cout << "------------------" << std::endl;


//  std::cout << "DisY1:  " << y1  << std::endl;
//  std::cout << "DisY2:  " << y2  << std::endl;
//  std::cout << "DisY3:  " << y3  << std::endl;
//  std::cout << "DisY4:  " << y4  << std::endl;
//  std::cout << "------------------" << std::endl;
//






  
  std::cout << std::endl;
}

// As usual, we have to call the run method. It handles
// the output stream to the terminal.
// Second, we define some output skip that is necessary 
// (and really useful) to avoid to much printing 
// of solutions. For large time dependent problems it is 
// sufficient to print only each tenth solution. 
// Third, we perform the time stepping scheme of 
// the solution process.
template <int dim>
void Biot_Problem<dim>::run () 
{  
  setup_system();

  std::cout << "\n==============================" 
	    << "====================================="  << std::endl;
  std::cout << "Parameters\n" 
	    << "==========\n"
	    << "Density structure: "   <<  density_structure << "\n"  
	    << "Lame coeff. mu:    "   <<  lame_coefficient_mu << "\n"
	    << std::endl;
  // More output can be printed here if wished


   {
      AffineConstraints<double> constraints;
      constraints.close();

      std::vector<bool> component_mask (dim+dim+1, true);
      VectorTools::project (dof_handler,
			    constraints,
			    QGauss<dim>(degree+2),
			    InitialValues<dim>(),
			    solution
			    );

      output_results (0,solution);
    }


 
  const unsigned int output_skip = 1;
  do
    { 
      std::cout << "Timestep " << timestep_number 
		<< " (" << time_stepping_scheme 
		<< ")" <<    ": " << time
		<< " (" << timestep << ")"
		<< "\n==============================" 
		<< "=====================================" 
		<< std::endl; 
      
      std::cout << std::endl;
      
      // Compute next time step
      old_timestep_solution = solution;
      newton_iteration ();   
      time += timestep;
	
      // Compute functional values
      std::cout << std::endl;
      compute_functional_values();
      
      // Write solutions 
      if ((timestep_number % output_skip == 0))
	output_results (timestep_number,solution);
      
      
      ++timestep_number;

    }
  while (timestep_number <= max_no_timesteps+1);
  
  
}

// The main function looks almost the same
// as in all other deal.II tuturial steps. 
int main () 
{
  try
    {
      deallog.depth_console (0);

      //Biot_Problem<2> biot_problem(1);
      Biot_Problem<3> biot_problem(1);
      biot_problem.run ();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      
      return 1;
    }
  catch (...) 
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}




