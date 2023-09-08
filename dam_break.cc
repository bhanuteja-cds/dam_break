#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/generic_linear_algebra.h>

namespace LA
{
    #if defined(DEAL_II_WITH_PETSC) && !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
    using namespace dealii::LinearAlgebraPETSc;
    #  define USE_PETSC_LA
    #elif defined(DEAL_II_WITH_TRILINOS)
    using namespace dealii::LinearAlgebraTrilinos;
    #else
    #  error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
    #endif
}

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/grid/manifold.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/lac/sparse_direct.h>

#include <deal.II/base/tensor_function.h>

#include <deal.II/lac/sparse_ilu.h>

#include <iostream>
#include <fstream>
#include <sstream>

namespace Navierstokes
{
    using namespace dealii;
    
    template <int dim>
    class StokesProblem
    {        
    private:
        MPI_Comm                                  mpi_communicator;
        double deltat = 0.0005;
        double totaltime = 20;
        double viscosity_fluid = 0.001, density_fluid = 750.0;
        double viscosity_air = 0.0001, density_air = 1;
        double viscosity1 = 0.0, density1 = 0.0;
        int meshrefinement = 0;

        int degree;
        parallel::distributed::Triangulation<dim> triangulation;
        LA::MPI::SparseMatrix                     ns_system_matrix;
        LA::MPI::SparseMatrix                     vof_system_matrix;
        DoFHandler<dim>                           dof_handler, vof_dof_handler;
        FESystem<dim>                             fe, fevof;
        LA::MPI::Vector                           lr_solution, lr_vof_solution, lr_zero_solution;
        LA::MPI::Vector                           lo_system_rhs, lo_vof_system_rhs, lo_initial_condition_vof;
        AffineConstraints<double>                 vofconstraints, stokesconstraints;
        IndexSet                                  owned_partitioning_stokes, owned_partitioning_vof;
        IndexSet                                  relevant_partitioning_stokes, relevant_partitioning_vof;
        ConditionalOStream                        pcout;
        TimerOutput                               computing_timer;
        
    public:
        void setup_stokessystem();
        void setup_vofsystem();
        void resetup_stokessystem();
        void resetup_vofsystem();
        void assemble_stokessystem();
        void assemble_vofsystem();
        void solve_stokes();
        double compute_errors();
        double compute_quantity();
        double compute_quantity_under_vof1();
        void compute_masses();
        void compute_fluid_mass_cut045();
        void compute_fluid_mass_cut050();
        void compute_fluid_mass_cut060();
        void compute_fluid_mass_cut070();
        void compute_fluid_mass_cut080();
        void compute_fluid_mass_cut090();
        void compute_fluid_mass_cut098();
        void compute_test_area();
        void compute_area_dealii();
        void output_results (int);
        void timeloop();
        
        StokesProblem(int degreein)
        :
        mpi_communicator (MPI_COMM_WORLD),
        degree(degreein),
        triangulation (mpi_communicator),
        dof_handler(triangulation),
        vof_dof_handler(triangulation),
        fe(FE_Q<dim>(degree+1), dim, FE_Q<dim>(degree), 1),
//         fe(FE_Q<dim>(degree+1), dim, FE_DGQ<dim>(degree), 1),
        fevof(FE_Q<dim>(degree+1), 1),
        pcout (std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)),
        computing_timer (mpi_communicator, pcout, TimerOutput::summary, TimerOutput::wall_times)
        {      
            pcout << "stokes constructor success...."<< std::endl;
        }
    };
    //=====================================================  
    template <int dim>
    class RightHandSide : public Function<dim>
    {
    public:
        RightHandSide () : Function<dim>(dim+1)
        {}
        virtual void vector_value(const Point<dim> &, Vector<double> &value) const;
    };
    
    template <int dim>
    void
    RightHandSide<dim>::vector_value(const Point<dim> &,  Vector<double> &values) const
    {
        values[0] = 0.0;  //test
        values[1] = -9.8; //gravity
        values[2] = 0.0;
    }
    //==================================================  
    template <int dim>
    class TractionBoundaryValues : public Function<dim>
    {
    public:
        TractionBoundaryValues () : Function<dim>(dim)
        {}
        virtual void vector_value_list(const std::vector<Point<dim>> &points, std::vector<Vector<double>> &values) const;
    };
    
    template <int dim>
    void  TractionBoundaryValues<dim>::vector_value_list(const std::vector<Point<dim>> &points, std::vector<Vector<double>> &values) const
    {
        //       const double time = this->get_time();
        for (unsigned int p=0; p<points.size(); ++p)
        {
            values[p][0] = 0.0;
            values[p][1] = 0.0;
        }    
    }
    //===========================================================
    template <int dim>
    class VofRightHandSide : public Function<dim>
    {
    public:
        VofRightHandSide() : Function<dim>(1) {}
        virtual double value(const Point<dim> &, const unsigned int component = 0) const override;
    };
    
    template <int dim>
    double VofRightHandSide<dim>::value(const Point<dim> &, const unsigned int) const
    {
        return 0.0;
    }
    //==========================================
    template <int dim>
    class InitialValues : public Function<dim>
    {
    public:
        int fac = 100;
        InitialValues () : Function<dim>(1) {}
        virtual double value (const Point<dim> &p, const unsigned int component = 0) const;
    };
    
    template <int dim>
    double InitialValues<dim>::value (const Point<dim> &p, const unsigned int) const
    {
//         if(p[0]<=2.1 && p[1] <= 1.8)
//             return 1.0;
//         else
//             return 0.0;
        
        if(p[0]<=2.5/fac && p[1] <= 1.8/fac)
            return 1.0;
        else
            return 0.0;
        

//         if(p[0]<=2.1/fac)
//             return 1.0;
//         else
//             return 0.0;
        
    }
    //==============================================================  
    template <int dim>
    void StokesProblem<dim>::setup_stokessystem()
    {  
        TimerOutput::Scope t(computing_timer, "setup_stokessystem");
        pcout <<"in setup_stokessystem "<<std::endl;
        GridIn<dim> grid_in;
        grid_in.attach_triangulation(triangulation);
        std::ifstream input_file("vof_2D_hex.msh");
        grid_in.read_msh(input_file);
        triangulation.refine_global (meshrefinement);
        dof_handler.distribute_dofs(fe);
        
        pcout << "   Number of active cells: "
        << triangulation.n_active_cells()
        << std::endl
        << "   Total number of cells: "
        << triangulation.n_cells()
        << std::endl;
        pcout << "   Number of degrees of freedom: "
        << dof_handler.n_dofs()
        << std::endl;
        pcout<< "Reynolds number " << density_fluid*1*0.05/viscosity_fluid << std::endl;
        std::vector<unsigned int> block_component(dim+1,0);
        block_component[dim] = 1;
//         std::vector<types::global_dof_index> dofs_per_block (2);
//         DoFTools::count_dofs_per_block (dof_handler, dofs_per_block, block_component);
        const std::vector<types::global_dof_index> dofs_per_block = DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
        const unsigned int n_u = dofs_per_block[0],  n_p = dofs_per_block[1];
        pcout << " (" << n_u << '+' << n_p << ')' << std::endl;
        pcout << "dofspercell "<< fe.dofs_per_cell << std::endl;
        
        owned_partitioning_stokes = dof_handler.locally_owned_dofs();
        DoFTools::extract_locally_relevant_dofs (dof_handler, relevant_partitioning_stokes);
        
        {
            stokesconstraints.clear();
            stokesconstraints.reinit(relevant_partitioning_stokes);
            DoFTools::make_hanging_node_constraints (dof_handler, stokesconstraints);
            std::set<types::boundary_id> no_normal_flux_boundaries;
            no_normal_flux_boundaries.insert (101);
            no_normal_flux_boundaries.insert (102);
            no_normal_flux_boundaries.insert (103);
            no_normal_flux_boundaries.insert (104);
            const FEValuesExtractors::Vector velocities(0);
            const FEValuesExtractors::Scalar pressure(dim);        
            ComponentMask velocities_mask = fe.component_mask(velocities);
            ComponentMask pressure_mask = fe.component_mask(pressure);
            VectorTools::compute_no_normal_flux_constraints (dof_handler, 0, no_normal_flux_boundaries, stokesconstraints);
//             VectorTools::interpolate_boundary_values (dof_handler, 101, ZeroFunction<dim>(dim+1), stokesconstraints, velocities_mask);
//             VectorTools::interpolate_boundary_values (dof_handler, 103, ZeroFunction<dim>(dim+1), stokesconstraints, velocities_mask);
//             VectorTools::interpolate_boundary_values (dof_handler, 103, ZeroFunction<dim>(dim+1), stokesconstraints, velocities_mask);
//             VectorTools::interpolate_boundary_values (dof_handler, 104, ZeroFunction<dim>(dim+1), stokesconstraints, velocities_mask);
            stokesconstraints.close();
        }
        
        ns_system_matrix.clear();        
        DynamicSparsityPattern dsp (relevant_partitioning_stokes);
        DoFTools::make_sparsity_pattern(dof_handler, dsp, stokesconstraints, false);
        SparsityTools::distribute_sparsity_pattern (dsp, dof_handler.locally_owned_dofs(), mpi_communicator, relevant_partitioning_stokes);
        ns_system_matrix.reinit (owned_partitioning_stokes, owned_partitioning_stokes, dsp, mpi_communicator);        
        lr_solution.reinit(owned_partitioning_stokes, relevant_partitioning_stokes, mpi_communicator);
        lo_system_rhs.reinit(owned_partitioning_stokes, mpi_communicator);
        pcout <<"end of setup_stokessystem "<<std::endl;
    }  
    //========================================================  
    template <int dim>
    void StokesProblem<dim>::setup_vofsystem()
    { 
        TimerOutput::Scope t(computing_timer, "setup_vofsystem");
        pcout <<"in setup_vofsystem "<<std::endl;
        vof_dof_handler.distribute_dofs(fevof);
        owned_partitioning_vof = vof_dof_handler.locally_owned_dofs();
        DoFTools::extract_locally_relevant_dofs (vof_dof_handler, relevant_partitioning_vof);
        
        {
            vofconstraints.clear();
            vofconstraints.reinit(relevant_partitioning_vof);
            //             DoFTools::make_hanging_node_constraints (vof_dof_handler, vofconstraints);
            //             std::set<types::boundary_id> no_normal_flux_boundaries;
            //             no_normal_flux_boundaries.insert (101);
            //             no_normal_flux_boundaries.insert (102);
            //             VectorTools::compute_no_normal_flux_constraints (move_dof_handler, 0, no_normal_flux_boundaries, moveconstraints);
            //             VectorTools::interpolate_boundary_values (move_dof_handler, 104, ZeroFunction<dim>(dim), moveconstraints);
            vofconstraints.close();
        }
        pcout << "Number of vof degrees of freedom: " << vof_dof_handler.n_dofs() << std::endl; 
        vof_system_matrix.clear();
        DynamicSparsityPattern vof_dsp(relevant_partitioning_vof);
        DoFTools::make_sparsity_pattern(vof_dof_handler, vof_dsp, vofconstraints, false);
        SparsityTools::distribute_sparsity_pattern (vof_dsp, vof_dof_handler.locally_owned_dofs(), mpi_communicator, relevant_partitioning_vof);        
        vof_system_matrix.reinit(owned_partitioning_vof, owned_partitioning_vof, vof_dsp, mpi_communicator);
        
        lr_vof_solution.reinit(owned_partitioning_vof, relevant_partitioning_vof, mpi_communicator);
        lr_zero_solution.reinit(owned_partitioning_vof, relevant_partitioning_vof, mpi_communicator);
        lo_vof_system_rhs.reinit(owned_partitioning_vof, mpi_communicator);
        lo_initial_condition_vof.reinit(owned_partitioning_vof, mpi_communicator);
        
        InitialValues<dim> initialcondition;
        VectorTools::interpolate(vof_dof_handler, initialcondition, lo_initial_condition_vof);
        lr_vof_solution = lo_initial_condition_vof;
        pcout <<"end of setup_vofsystem"<<std::endl;
    } 
    //===========================================================
    template <int dim>
    void StokesProblem<dim>::assemble_stokessystem()
    {
        TimerOutput::Scope t(computing_timer, "assemble_stokessystem");
        pcout <<"in assemble_stokessystem "<<std::endl;
        ns_system_matrix=0;
        lo_system_rhs=0;
        
        QGauss<dim>   quadrature_formula(degree+2);
        QGauss<dim-1> face_quadrature_formula(degree+2);
        
        FEValues<dim> fe_values (fe, quadrature_formula,
                                 update_values    |
                                 update_quadrature_points  |
                                 update_JxW_values |
                                 update_gradients);
        
        FEValues<dim> fe_vof_values (fevof, quadrature_formula,
                                     update_values    |
                                     update_gradients |
                                     update_quadrature_points  |
                                     update_JxW_values);
        
        
        FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
                                          update_values    | update_normal_vectors |
                                          update_quadrature_points  | update_JxW_values);
        
        const unsigned int   dofs_per_cell   = fe.dofs_per_cell;
        const unsigned int   n_q_points      = quadrature_formula.size();
        const unsigned int   n_face_q_points = face_quadrature_formula.size();
        
        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(dim); 
        
        FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
        Vector<double>       local_rhs (dofs_per_cell);
        
        std::vector<types::global_dof_index>  local_dof_indices (dofs_per_cell);        
        const RightHandSide<dim>              right_hand_side;
        const TractionBoundaryValues<dim>     traction_boundary_values;
        std::vector<Vector<double>>           rhs_values(n_q_points, Vector<double>(dim+1));
        std::vector<Vector<double>>           neumann_boundary_values(n_face_q_points, Vector<double>(dim+1));        
        std::vector<Tensor<1, dim>>           value_phi_u (dofs_per_cell);    
        std::vector<Tensor<2, dim>>           gradient_phi_u (dofs_per_cell);
        std::vector<SymmetricTensor<2, dim>>  symgrad_phi_u (dofs_per_cell);
        std::vector<double>                   div_phi_u(dofs_per_cell);
        std::vector<double>                   phi_p(dofs_per_cell);        
        std::vector<Tensor<2, dim> >          old_solution_gradients(n_q_points);
        std::vector<Tensor<1, dim> >          old_solution_values(n_q_points);
        std::vector<double>                   vof_values(n_q_points);
        std::vector<Tensor<1, dim> >          vof_values_gradients(n_q_points);
        
        typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(), endc = dof_handler.end();
        typename DoFHandler<dim>::active_cell_iterator vof_cell = vof_dof_handler.begin_active();
        
        for (; cell!=endc; ++cell)
        { 
            if (cell->is_locally_owned())
            {
                fe_values.reinit (cell);
                fe_vof_values.reinit(vof_cell);
                fe_vof_values.get_function_values(lr_vof_solution, vof_values);
//                 fe_vof_values.get_function_gradients(lr_vof_solution, vof_values_gradients);
                fe_values[velocities].get_function_values(lr_solution, old_solution_values);
//                 fe_values[velocities].get_function_gradients(lr_solution, old_solution_gradients);
                
                local_matrix = 0;
                local_rhs = 0;
                right_hand_side.vector_value_list(fe_values.get_quadrature_points(), rhs_values);

                for (unsigned int q=0; q<n_q_points; ++q)
                {
                    viscosity1 = vof_values[q]*viscosity_fluid +(1-vof_values[q])*viscosity_air;
                    density1   = vof_values[q]*density_fluid +(1-vof_values[q])*density_air;                     
                    
                    for (unsigned int k=0; k<dofs_per_cell; ++k)
                    {                        
                        value_phi_u[k]   = fe_values[velocities].value (k, q);
                        gradient_phi_u[k]= fe_values[velocities].gradient (k, q);
                        symgrad_phi_u[k] = fe_values[velocities].symmetric_gradient (k, q);
                        div_phi_u[k]     = fe_values[velocities].divergence (k, q);
                        phi_p[k]         = fe_values[pressure].value (k, q);
                    }
                    
                    for (unsigned int i=0; i<dofs_per_cell; ++i)
                    {                    
                        for (unsigned int j=0; j<dofs_per_cell; ++j)
                        {
//                             local_matrix(i, j) += ((density1*value_phi_u[j]*value_phi_u[i] + 
//                             deltat*density1*(gradient_phi_u[j] * old_solution_values[q])*value_phi_u[i] -
//                              2*deltat*(viscosity_fluid-viscosity_air)*(symgrad_phi_u[j] * vof_values_gradients[q])*value_phi_u[i] +
//                             (2*deltat*viscosity1)*scalar_product(symgrad_phi_u[i], symgrad_phi_u[j])) -
//                             deltat * div_phi_u[i] * phi_p[j] - 
//                             phi_p[i] * div_phi_u[j]) *
//                             fe_values.JxW(q);
                            
//                             local_matrix(i, j) += ((density1*value_phi_u[j]*value_phi_u[i] + 
//                             deltat*density1*(gradient_phi_u[j] * old_solution_values[q])*value_phi_u[i] +
//                             2*deltat*(viscosity_fluid-viscosity_air)*(symgrad_phi_u[j] * vof_values_gradients[q])*value_phi_u[i] +
//                             (2*deltat*viscosity1)*scalar_product(symgrad_phi_u[i], symgrad_phi_u[j])) -
//                             deltat * div_phi_u[i] * phi_p[j] - 
//                             phi_p[i] * div_phi_u[j]) *
//                             fe_values.JxW(q);
                            
                            local_matrix(i, j) += (density1*value_phi_u[j]*value_phi_u[i] + 
                            deltat*density1*(gradient_phi_u[j] * old_solution_values[q])*value_phi_u[i] +
                            2*deltat*viscosity1*scalar_product(symgrad_phi_u[j], symgrad_phi_u[i]) -
                            deltat * phi_p[j] * div_phi_u[i] - 
                            div_phi_u[j] * phi_p[i]) *
                            fe_values.JxW(q);
                        }
                        const unsigned int component_i = fe.system_to_component_index(i).first;
                        
                        local_rhs(i) += (deltat*density1*(fe_values.shape_value(i,q) * rhs_values[q](component_i)) + density1*old_solution_values[q]*value_phi_u[i]) * fe_values.JxW(q);
                        
//                         local_rhs(i) += (deltat*(fe_values.shape_value(i,q) * rhs_values[q](component_i)) + density1*old_solution_values[q]*value_phi_u[i]) * fe_values.JxW(q);

                    } // end of i loop                
                }  // end of quadrature points loop
                
//                 for (unsigned int face_n=0; face_n<GeometryInfo<dim>::faces_per_cell; ++face_n)
//                 {
//                     if (cell->face(face_n)->at_boundary() && (cell->face(face_n)->boundary_id() == 102))
//                     {             
//                         fe_face_values.reinit (cell, face_n);
//                         traction_boundary_values.vector_value_list(fe_face_values.get_quadrature_points(), neumann_boundary_values);
//                         for (unsigned int q=0; q<n_face_q_points; ++q)
//                             for (unsigned int i=0; i<dofs_per_cell; ++i)
//                             {                 
//                                 const unsigned int component_i = fe.system_to_component_index(i).first;
//                                 local_rhs(i) += (fe_face_values.shape_value(i, q) * neumann_boundary_values[q](component_i) * fe_face_values.JxW(q))*deltat/density1;
//                             }
//                     } // end of face if
//                 } // end of face for      
                cell->get_dof_indices (local_dof_indices);         
                stokesconstraints.distribute_local_to_global(local_matrix, local_rhs, local_dof_indices, ns_system_matrix, lo_system_rhs);
            } // end of if cell->locally owned
            vof_cell++;
        } // end of cell loop
        ns_system_matrix.compress (VectorOperation::add);
        lo_system_rhs.compress (VectorOperation::add);
        pcout <<"end of assemble_stokessystem "<<std::endl;
    } // end of assemble system
    //=======================================
    template <int dim>
    void StokesProblem<dim>::assemble_vofsystem()
    {
        TimerOutput::Scope t(computing_timer, "assembly_vofsystem");
        pcout << "in assemble_vofsystem" << std::endl;
        vof_system_matrix=0;
        lo_vof_system_rhs=0;
        
        QGauss<dim>   vof_quadrature_formula(degree+2);
        
        FEValues<dim> fe_vof_values (fevof, vof_quadrature_formula,
                                     update_values  |
                                     update_quadrature_points |
                                     update_JxW_values |
                                     update_gradients | update_hessians);
        
        FEValues<dim> fe_velocity_values (fe, vof_quadrature_formula,
                                          update_values    |
                                          update_quadrature_points  |
                                          update_JxW_values |
                                          update_gradients);
        
        const unsigned int dofs_per_cell = fevof.dofs_per_cell;
        const unsigned int vof_n_q_points = vof_quadrature_formula.size();
        
        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(dim);
        
        FullMatrix<double>                   vof_local_matrix (dofs_per_cell, dofs_per_cell);
        Vector<double>                       vof_local_rhs(dofs_per_cell);        
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
        const VofRightHandSide<dim>          vof_right_hand_side;
        std::vector<double>                  vof_rhs_values(vof_n_q_points);
        std::vector<double>                  old_vof_values(vof_n_q_points);
        std::vector<Tensor<1,dim>>           old_vof_gradients(vof_n_q_points);
        std::vector<Tensor<1,dim>>           nsvelocity_values(vof_n_q_points);
        std::vector<double>                  value_phi_vof(dofs_per_cell);
        std::vector<Tensor<1,dim>>           gradient_phi_vof(dofs_per_cell);
        std::vector<Tensor<2,dim>>           hessian_phi_vof(dofs_per_cell);

        double temp=0;
        double taucell = 0.0;
        double velocity_norm = 0.0;
        double peclet_num = 0.0;

        typename DoFHandler<dim>::active_cell_iterator vof_cell = vof_dof_handler.begin_active(), vof_endc = vof_dof_handler.end();
        typename DoFHandler<dim>::active_cell_iterator cellns = dof_handler.begin_active();

        for (; vof_cell!=vof_endc; ++vof_cell)
        {
            if (vof_cell->is_locally_owned())
            {
                fe_vof_values.reinit(vof_cell);
                fe_velocity_values.reinit(cellns);
                vof_local_matrix = 0;
                vof_local_rhs = 0;
                vof_right_hand_side.value_list(fe_vof_values.get_quadrature_points(), vof_rhs_values);
                fe_velocity_values[velocities].get_function_values(lr_solution, nsvelocity_values);
                fe_vof_values.get_function_gradients(lr_vof_solution, old_vof_gradients);
                fe_vof_values.get_function_values(lr_vof_solution, old_vof_values);
                
                for (unsigned int q_index=0; q_index<vof_n_q_points; ++q_index)
                {
                    for (unsigned int k=0; k<dofs_per_cell; ++k)
                    {                        
                        value_phi_vof[k]   = fe_vof_values.shape_value (k, q_index);
                        gradient_phi_vof[k]= fe_vof_values.shape_grad (k, q_index);
                        hessian_phi_vof[k] = fe_vof_values.shape_hessian (k, q_index);
                    }
                    
                    velocity_norm = nsvelocity_values[q_index].norm();
                    if(temp < 1e-6 || velocity_norm/temp > 1e6)
                    {
                        taucell = 1.75*vof_cell->diameter()/(2*velocity_norm);
                    }
                    else
                    {
                        peclet_num = (velocity_norm*vof_cell->diameter()/2)/temp;
                        taucell = 1.75*vof_cell->diameter()/(2*velocity_norm)*(cosh(peclet_num)/sinh(peclet_num) - 1/peclet_num);
                    }
                    for (unsigned int i=0; i<dofs_per_cell; ++i)            
                    {
                        for (unsigned int j=0; j<dofs_per_cell; ++j)
                        {                             
//                             vof_local_matrix(i,j) += (value_phi_vof[j]*(value_phi_vof[i] + taucell*nsvelocity_values[q_index]*gradient_phi_vof[i]) - deltat*(value_phi_vof[j])*(nsvelocity_values[q_index]*gradient_phi_vof[i]) + deltat*taucell*(nsvelocity_values[q_index]*gradient_phi_vof[j])*(nsvelocity_values[q_index]*gradient_phi_vof[i]))*fe_vof_values.JxW(q_index);
                            
//                             vof_local_matrix(i,j) += (value_phi_vof[j]*(value_phi_vof[i] + taucell*nsvelocity_values[q_index]*gradient_phi_vof[i])*(value_phi_vof[i]+taucell*nsvelocity_values[q_index]*gradient_phi_vof[i]) -                            deltat*value_phi_vof[j]*(nsvelocity_values[q_index]*gradient_phi_vof[i]) + deltat*taucell*(nsvelocity_values[q_index]*gradient_phi_vof[j])*(nsvelocity_values[q_index]*gradient_phi_vof[i]) +
//                             deltat*temp*(gradient_phi_vof[j]*gradient_phi_vof[i]) -deltat*temp*taucell*(trace(hessian_phi_vof[j])*(nsvelocity_values[q_index]*gradient_phi_vof[i]))
//                             )*fe_vof_values.JxW(q_index);  //implicit
                            
                            vof_local_matrix(i,j) += ( value_phi_vof[j]*(value_phi_vof[i]+taucell*nsvelocity_values[q_index]*gradient_phi_vof[i]) -deltat*(nsvelocity_values[q_index]*gradient_phi_vof[i])*value_phi_vof[j] +
                            deltat*(nsvelocity_values[q_index]*gradient_phi_vof[j])*(nsvelocity_values[q_index]*gradient_phi_vof[i])*taucell +
                            deltat*temp*(gradient_phi_vof[j]*gradient_phi_vof[i]) -
                            deltat*temp*taucell*(trace(hessian_phi_vof[j])*(nsvelocity_values[q_index]*gradient_phi_vof[i]))
                            )*fe_vof_values.JxW(q_index);
                        }
                        vof_local_rhs(i) += old_vof_values[q_index]*(value_phi_vof[i] + taucell*(nsvelocity_values[q_index]*gradient_phi_vof[i]))*fe_vof_values.JxW(q_index);
                    }                
                } //end of quadrature points loop
                vof_cell->get_dof_indices(local_dof_indices);                
                vofconstraints.distribute_local_to_global(vof_local_matrix, vof_local_rhs, local_dof_indices, vof_system_matrix, lo_vof_system_rhs);     
            } //end of if vof_cell->is_locally_owned()
            cellns++;
        } //end of cell loop
        vof_system_matrix.compress (VectorOperation::add);
        lo_vof_system_rhs.compress (VectorOperation::add);
        pcout << "end of assemble_vofsystem"<< std::endl;
    }
    //====================================================
    //     template <int dim>
    //     double StokesProblem<dim>::compute_errors()
    //     {        
    //         const ComponentSelectFunction<dim> pressure_mask(dim, dim+1);
    //         const ComponentSelectFunction<dim> velocity_mask(std::make_pair(0,dim), dim+1);        
    //         Vector<double> cellwise_errors(triangulation.n_active_cells());
    //         QGauss<dim> quadrature(4);
    //         VectorTools::integrate_difference (dof_handler, lr_nonlinear_residue, ZeroFunction<dim>(dim+1), cellwise_errors, quadrature, VectorTools::L2_norm, &velocity_mask);
    //         const double u_l2_error = VectorTools::compute_global_error(triangulation, cellwise_errors, VectorTools::L2_norm);
    //         return u_l2_error;
    //     }
            //===========================================================
    template <int dim>
        double StokesProblem<dim>::compute_quantity()
        {      
            Vector<double> cellwise_errors(triangulation.n_active_cells());
            QGauss<dim> quadrature(4);
            VectorTools::integrate_difference(vof_dof_handler, lr_vof_solution, ZeroFunction<dim>(1), cellwise_errors, quadrature, VectorTools::L1_norm);
            const double total_quantity = VectorTools::compute_global_error(triangulation, cellwise_errors, VectorTools::L1_norm);
            return total_quantity;
        }
        //===========================================================
    template <int dim>
        double StokesProblem<dim>::compute_quantity_under_vof1()
        {      
            Vector<double> cellwise_errors(triangulation.n_active_cells());
            QGauss<dim> quadrature(4);
            VectorTools::integrate_difference(vof_dof_handler, lr_vof_solution, ZeroFunction<dim>(1), cellwise_errors, quadrature, VectorTools::L1_norm);
            const double total_quantity = VectorTools::compute_global_error(triangulation, cellwise_errors, VectorTools::L1_norm);
            return total_quantity;
        }
        //=================================================
    template <int dim>
    void StokesProblem<dim>::compute_masses()
    {      
        Vector<double> cellwise_fluid_fractions(triangulation.n_active_cells());
        Vector<double> cellwise_gas_fractions(triangulation.n_active_cells());
        QGauss<dim> quadrature(4);
        VectorTools::integrate_difference (vof_dof_handler, lr_vof_solution, ZeroFunction<dim>(1), cellwise_fluid_fractions, quadrature, VectorTools::mean);
        VectorTools::integrate_difference (vof_dof_handler, lr_vof_solution, ConstantFunction<dim>(1, 1), cellwise_gas_fractions, quadrature, VectorTools::mean);
        const double fluid_mass = VectorTools::compute_global_error(triangulation, cellwise_fluid_fractions, VectorTools::mean);
        const double gas_mass = VectorTools::compute_global_error(triangulation, cellwise_gas_fractions, VectorTools::mean);
        pcout << "fluid_mass is " << -fluid_mass*density_fluid << std::endl;
        pcout << "gas_mass is " << gas_mass*density_air << std::endl;
    }
    //======================================
    template <int dim>
    void StokesProblem<dim>::compute_area_dealii()
    {      
        Vector<double> cellwise_fluid_fractions(triangulation.n_active_cells());
        QGauss<dim> quadrature(4);
        VectorTools::integrate_difference (vof_dof_handler, lr_zero_solution, ConstantFunction<dim>(1, 1), cellwise_fluid_fractions, quadrature, VectorTools::mean);
        const double area = VectorTools::compute_global_error(triangulation, cellwise_fluid_fractions, VectorTools::mean);
        pcout << "area_dealii is " << area << std::endl;
    }
        //=================================================
    template <int dim>
    void StokesProblem<dim>::compute_test_area()
    {      
        QGauss<dim>   vof_quadrature_formula(4);
        FEValues<dim> fe_vof_values (fevof, vof_quadrature_formula,
                                     update_values  |
                                     update_quadrature_points |
                                     update_JxW_values);
        const unsigned int vof_n_q_points = vof_quadrature_formula.size();
//         std::vector<double>                  vof_values(vof_n_q_points);
//         std::vector<Point<dim>> quadrature_points(vof_n_q_points);
        double cell_area = 0.0;
        double local_area = 0.0;
        double total_area = 0.0;
        
        for (typename DoFHandler<dim>::active_cell_iterator  vof_cell = vof_dof_handler.begin_active(); vof_cell != vof_dof_handler.end(); ++vof_cell)
        {
            cell_area = 0.0;
            if (vof_cell->is_locally_owned())
            {
                fe_vof_values.reinit (vof_cell);

                for(unsigned int q=0; q<vof_n_q_points; q++)
                {
                        cell_area += fe_vof_values.JxW(q);
                }
            }
            local_area += cell_area;
        }
        MPI_Allreduce(&local_area, &total_area, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
        pcout << "test_area is " << total_area << std::endl;
    }
      //================================================= 
    template <int dim>
    void StokesProblem<dim>::compute_fluid_mass_cut045()
    { 
        QGauss<dim>   vof_quadrature_formula(4);
        FEValues<dim> fe_vof_values (fevof, vof_quadrature_formula,
                                     update_values  |
                                     update_quadrature_points |
                                     update_JxW_values);
        const unsigned int vof_n_q_points = vof_quadrature_formula.size();
        std::vector<double>                  vof_values(vof_n_q_points);
//         std::vector<Point<dim>> quadrature_points(vof_n_q_points);
        double cell_fraction;
        double local_fluid_mass = 0.0;
        double total_fluid_mass = 0.0;
        
        for (typename DoFHandler<dim>::active_cell_iterator  vof_cell = vof_dof_handler.begin_active(); vof_cell != vof_dof_handler.end(); ++vof_cell)
        {
            cell_fraction = 0.0;
            if (vof_cell->is_locally_owned())
            {
                fe_vof_values.reinit (vof_cell);
                fe_vof_values.get_function_values(lr_vof_solution, vof_values);
//                 quadrature_points = fe_vof_values.get_quadrature_points();
                for(unsigned int q=0; q<vof_n_q_points; q++)
                {
                    if(vof_values[q]>=0.45)
                    {
                        cell_fraction += vof_values[q]*fe_vof_values.JxW(q);
                    }
                }
            }
            local_fluid_mass += cell_fraction;
        }
        MPI_Allreduce(&local_fluid_mass, &total_fluid_mass, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
        pcout << "fluid_mass_cut_045 is " << total_fluid_mass*density_fluid << std::endl;
    }
          //================================================= 
    template <int dim>
    void StokesProblem<dim>::compute_fluid_mass_cut050()
    { 
        QGauss<dim>   vof_quadrature_formula(4);
        FEValues<dim> fe_vof_values (fevof, vof_quadrature_formula,
                                     update_values  |
                                     update_quadrature_points |
                                     update_JxW_values);
        const unsigned int vof_n_q_points = vof_quadrature_formula.size();
        std::vector<double>                  vof_values(vof_n_q_points);
//         std::vector<Point<dim>> quadrature_points(vof_n_q_points);
        double cell_fraction;
        double local_fluid_mass = 0.0;
        double total_fluid_mass = 0.0;
        
        for (typename DoFHandler<dim>::active_cell_iterator  vof_cell = vof_dof_handler.begin_active(); vof_cell != vof_dof_handler.end(); ++vof_cell)
        {
            cell_fraction = 0.0;
            if (vof_cell->is_locally_owned())
            {
                fe_vof_values.reinit (vof_cell);
                fe_vof_values.get_function_values(lr_vof_solution, vof_values);
//                 quadrature_points = fe_vof_values.get_quadrature_points();
                for(unsigned int q=0; q<vof_n_q_points; q++)
                {
                    if(vof_values[q]>=0.50)
                    {
                        cell_fraction += vof_values[q]*fe_vof_values.JxW(q);
                    }
                }
            }
            local_fluid_mass += cell_fraction;
        }
        MPI_Allreduce(&local_fluid_mass, &total_fluid_mass, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
        pcout << "fluid_mass_cut_050 is " << total_fluid_mass*density_fluid << std::endl;
    }
              //================================================= 
    template <int dim>
    void StokesProblem<dim>::compute_fluid_mass_cut060()
    { 
        QGauss<dim>   vof_quadrature_formula(4);
        FEValues<dim> fe_vof_values (fevof, vof_quadrature_formula,
                                     update_values  |
                                     update_quadrature_points |
                                     update_JxW_values);
        const unsigned int vof_n_q_points = vof_quadrature_formula.size();
        std::vector<double>                  vof_values(vof_n_q_points);
//         std::vector<Point<dim>> quadrature_points(vof_n_q_points);
        double cell_fraction;
        double local_fluid_mass = 0.0;
        double total_fluid_mass = 0.0;
        
        for (typename DoFHandler<dim>::active_cell_iterator  vof_cell = vof_dof_handler.begin_active(); vof_cell != vof_dof_handler.end(); ++vof_cell)
        {
            cell_fraction = 0.0;
            if (vof_cell->is_locally_owned())
            {
                fe_vof_values.reinit (vof_cell);
                fe_vof_values.get_function_values(lr_vof_solution, vof_values);
//                 quadrature_points = fe_vof_values.get_quadrature_points();
                for(unsigned int q=0; q<vof_n_q_points; q++)
                {
                    if(vof_values[q]>=0.60)
                    {
                        cell_fraction += vof_values[q]*fe_vof_values.JxW(q);
                    }
                }
            }
            local_fluid_mass += cell_fraction;
        }
        MPI_Allreduce(&local_fluid_mass, &total_fluid_mass, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
        pcout << "fluid_mass_cut_060 is " << total_fluid_mass*density_fluid << std::endl;
    }
              //================================================= 
    template <int dim>
    void StokesProblem<dim>::compute_fluid_mass_cut070()
    { 
        QGauss<dim>   vof_quadrature_formula(4);
        FEValues<dim> fe_vof_values (fevof, vof_quadrature_formula,
                                     update_values  |
                                     update_quadrature_points |
                                     update_JxW_values);
        const unsigned int vof_n_q_points = vof_quadrature_formula.size();
        std::vector<double>                  vof_values(vof_n_q_points);
//         std::vector<Point<dim>> quadrature_points(vof_n_q_points);
        double cell_fraction;
        double local_fluid_mass = 0.0;
        double total_fluid_mass = 0.0;
        
        for (typename DoFHandler<dim>::active_cell_iterator  vof_cell = vof_dof_handler.begin_active(); vof_cell != vof_dof_handler.end(); ++vof_cell)
        {
            cell_fraction = 0.0;
            if (vof_cell->is_locally_owned())
            {
                fe_vof_values.reinit (vof_cell);
                fe_vof_values.get_function_values(lr_vof_solution, vof_values);
//                 quadrature_points = fe_vof_values.get_quadrature_points();
                for(unsigned int q=0; q<vof_n_q_points; q++)
                {
                    if(vof_values[q]>=0.70)
                    {
                        cell_fraction += vof_values[q]*fe_vof_values.JxW(q);
                    }
                }
            }
            local_fluid_mass += cell_fraction;
        }
        MPI_Allreduce(&local_fluid_mass, &total_fluid_mass, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
        pcout << "fluid_mass_cut_070 is " << total_fluid_mass*density_fluid << std::endl;
    }
              //================================================= 
    template <int dim>
    void StokesProblem<dim>::compute_fluid_mass_cut080()
    { 
        QGauss<dim>   vof_quadrature_formula(4);
        FEValues<dim> fe_vof_values (fevof, vof_quadrature_formula,
                                     update_values  |
                                     update_quadrature_points |
                                     update_JxW_values);
        const unsigned int vof_n_q_points = vof_quadrature_formula.size();
        std::vector<double>                  vof_values(vof_n_q_points);
//         std::vector<Point<dim>> quadrature_points(vof_n_q_points);
        double cell_fraction;
        double local_fluid_mass = 0.0;
        double total_fluid_mass = 0.0;
        
        for (typename DoFHandler<dim>::active_cell_iterator  vof_cell = vof_dof_handler.begin_active(); vof_cell != vof_dof_handler.end(); ++vof_cell)
        {
            cell_fraction = 0.0;
            if (vof_cell->is_locally_owned())
            {
                fe_vof_values.reinit (vof_cell);
                fe_vof_values.get_function_values(lr_vof_solution, vof_values);
//                 quadrature_points = fe_vof_values.get_quadrature_points();
                for(unsigned int q=0; q<vof_n_q_points; q++)
                {
                    if(vof_values[q]>=0.80)
                    {
                        cell_fraction += vof_values[q]*fe_vof_values.JxW(q);
                    }
                }
            }
            local_fluid_mass += cell_fraction;
        }
        MPI_Allreduce(&local_fluid_mass, &total_fluid_mass, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
        pcout << "fluid_mass_cut_080 is " << total_fluid_mass*density_fluid << std::endl;
    }
              //================================================= 
    template <int dim>
    void StokesProblem<dim>::compute_fluid_mass_cut090()
    { 
        QGauss<dim>   vof_quadrature_formula(4);
        FEValues<dim> fe_vof_values (fevof, vof_quadrature_formula,
                                     update_values  |
                                     update_quadrature_points |
                                     update_JxW_values);
        const unsigned int vof_n_q_points = vof_quadrature_formula.size();
        std::vector<double>                  vof_values(vof_n_q_points);
//         std::vector<Point<dim>> quadrature_points(vof_n_q_points);
        double cell_fraction;
        double local_fluid_mass = 0.0;
        double total_fluid_mass = 0.0;
        
        for (typename DoFHandler<dim>::active_cell_iterator  vof_cell = vof_dof_handler.begin_active(); vof_cell != vof_dof_handler.end(); ++vof_cell)
        {
            cell_fraction = 0.0;
            if (vof_cell->is_locally_owned())
            {
                fe_vof_values.reinit (vof_cell);
                fe_vof_values.get_function_values(lr_vof_solution, vof_values);
//                 quadrature_points = fe_vof_values.get_quadrature_points();
                for(unsigned int q=0; q<vof_n_q_points; q++)
                {
                    if(vof_values[q]>=0.90)
                    {
                        cell_fraction += vof_values[q]*fe_vof_values.JxW(q);
                    }
                }
            }
            local_fluid_mass += cell_fraction;
        }
        MPI_Allreduce(&local_fluid_mass, &total_fluid_mass, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
        pcout << "fluid_mass_cut_090 is " << total_fluid_mass*density_fluid << std::endl;
    }
              //================================================= 
    template <int dim>
    void StokesProblem<dim>::compute_fluid_mass_cut098()
    { 
        QGauss<dim>   vof_quadrature_formula(4);
        FEValues<dim> fe_vof_values (fevof, vof_quadrature_formula,
                                     update_values  |
                                     update_quadrature_points |
                                     update_JxW_values);
        const unsigned int vof_n_q_points = vof_quadrature_formula.size();
        std::vector<double>                  vof_values(vof_n_q_points);
//         std::vector<Point<dim>> quadrature_points(vof_n_q_points);
        double cell_fraction;
        double local_fluid_mass = 0.0;
        double total_fluid_mass = 0.0;
        
        for (typename DoFHandler<dim>::active_cell_iterator  vof_cell = vof_dof_handler.begin_active(); vof_cell != vof_dof_handler.end(); ++vof_cell)
        {
            cell_fraction = 0.0;
            if (vof_cell->is_locally_owned())
            {
                fe_vof_values.reinit (vof_cell);
                fe_vof_values.get_function_values(lr_vof_solution, vof_values);
//                 quadrature_points = fe_vof_values.get_quadrature_points();
                for(unsigned int q=0; q<vof_n_q_points; q++)
                {
                    if(vof_values[q]>=0.98)
                    {
                        cell_fraction += vof_values[q]*fe_vof_values.JxW(q);
                    }
                }
            }
            local_fluid_mass += cell_fraction;
        }
        MPI_Allreduce(&local_fluid_mass, &total_fluid_mass, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
        pcout << "fluid_mass_cut_098 is " << total_fluid_mass*density_fluid << std::endl;
    }
    //================================================================
    template <int dim>
    void StokesProblem<dim>::solve_stokes()
    {
        pcout <<"in solve_stokes"<<std::endl;
        TimerOutput::Scope t(computing_timer, "solve");
        LA::MPI::Vector  distributed_solution_stokes (owned_partitioning_stokes, mpi_communicator);
        LA::MPI::Vector  distributed_solution_vof_adjusted (owned_partitioning_vof, mpi_communicator);
        
        SolverControl solver_control_stokes (dof_handler.n_dofs(), 1e-12);
        dealii::PETScWrappers::SparseDirectMUMPS solver_stokes(solver_control_stokes, mpi_communicator);
        
        SolverControl solver_control_vof (vof_dof_handler.n_dofs(), 1e-12);
        dealii::PETScWrappers::SparseDirectMUMPS solver_vof(solver_control_vof, mpi_communicator);
        
        solver_stokes.solve (ns_system_matrix, distributed_solution_stokes, lo_system_rhs);
        stokesconstraints.distribute(distributed_solution_stokes);                 
        lr_solution = distributed_solution_stokes;
        
        assemble_vofsystem();
        solver_vof.solve (vof_system_matrix, distributed_solution_vof_adjusted, lo_vof_system_rhs);
        
        for(unsigned int i = distributed_solution_vof_adjusted.local_range().first; i < distributed_solution_vof_adjusted.local_range().second; ++i)
        {
            if(distributed_solution_vof_adjusted(i) > 1.0)
                distributed_solution_vof_adjusted(i) = 1.0;
            else if(distributed_solution_vof_adjusted(i) < 0.0)
                distributed_solution_vof_adjusted(i) = 0.0;
        }
        distributed_solution_vof_adjusted.compress(VectorOperation::insert);
        
        vofconstraints.distribute(distributed_solution_vof_adjusted);
        lr_vof_solution = distributed_solution_vof_adjusted;
//         compute_masses();
//         compute_fluid_mass_cut097();
        pcout <<"end of solve_stokes "<<std::endl;
    }
    //===================================================================
    template <int dim>
    void StokesProblem<dim>::output_results(int timestepnumber)
    {
        TimerOutput::Scope t(computing_timer, "output");
        std::vector<std::string> solution_names (dim, "velocity");
        solution_names.emplace_back ("pressure");
        std::vector<DataComponentInterpretation::DataComponentInterpretation> data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
        data_component_interpretation.push_back (DataComponentInterpretation::component_is_scalar);
        
        DataOut<dim> data_out;
        data_out.attach_dof_handler (dof_handler);
        data_out.add_data_vector (lr_solution, solution_names, DataOut<dim>::type_dof_data, data_component_interpretation);
        
        Vector<float> subdomain (triangulation.n_active_cells());
        for (unsigned int i=0; i<subdomain.size(); ++i)
            subdomain(i) = triangulation.locally_owned_subdomain();
        data_out.add_data_vector (subdomain, "subdomain");
        data_out.add_data_vector(vof_dof_handler, lr_vof_solution, "vof");
        data_out.build_patches ();
        
        std::string filenamebase = "zfs2d_";
        
        const std::string filename = (filenamebase + Utilities::int_to_string (timestepnumber, 3) + "." +Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4));
        std::ofstream output ((filename + ".vtu").c_str());
        data_out.write_vtu (output);
        
        if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
        {
            std::vector<std::string> filenames;
            for (unsigned int i=0; i<Utilities::MPI::n_mpi_processes(mpi_communicator); ++i)
                filenames.push_back (filenamebase + Utilities::int_to_string (timestepnumber, 3) + "." + Utilities::int_to_string (i, 4) + ".vtu");
            
            std::ofstream master_output ((filenamebase + Utilities::int_to_string (timestepnumber, 3) + ".pvtu").c_str());
            data_out.write_pvtu_record (master_output, filenames);
        }
    }
    //==================================================================  
    template <int dim>
    void StokesProblem<dim>::timeloop()
    {      
        double timet = deltat;
        int timestepnumber=0;
        
//         pcout << "Reynolds number approximately is " <<density_fluid *1.0*7.0/100/viscosity_fluid<< std::endl;
        pcout << "deltat is " << deltat << std::endl;
        pcout << "Running for a total time of " << totaltime << std::endl;
        
        while(timet<totaltime)
        {  
            pcout << "=============================" << std::endl;
//             output_results(timestepnumber);
//             pcout << "total scalar quantity = " << compute_quantity() << std::endl;
            compute_area_dealii();
            compute_test_area();
            compute_masses();
            compute_fluid_mass_cut045();
            compute_fluid_mass_cut050();
            compute_fluid_mass_cut060();
            compute_fluid_mass_cut070();
            compute_fluid_mass_cut080();
            compute_fluid_mass_cut090();
            compute_fluid_mass_cut098();
            assemble_stokessystem();
            solve_stokes();
            pcout <<"timet "<<timet <<std::endl;                       
            timet+=deltat;
            timestepnumber++;
        } 
        output_results(timestepnumber);
    }
}  // end of namespace
//====================================================
int main (int argc, char *argv[])
{
    try
    {
        using namespace dealii;
        using namespace Navierstokes;        
        Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);        
        StokesProblem<2> flow_problem(1);
        flow_problem.setup_stokessystem();
        flow_problem.setup_vofsystem();
        flow_problem.timeloop();
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
