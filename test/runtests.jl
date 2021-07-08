using SafeTestsets

const GROUP = get(ENV, "GROUP", "All")
const is_APPVEYOR = Sys.iswindows() && haskey(ENV,"APPVEYOR")
const is_TRAVIS = haskey(ENV,"TRAVIS")

#Start Test Script

@time begin

if GROUP == "All" || GROUP == "OperatorInterface"
    @time @safetestset "Poisson example" begin include("DerivativeOperators/poisson.jl") end
    @time @safetestset "Heat equation example" begin include("DerivativeOperators/heat_equation.jl") end
    @time @safetestset "Robin Boundary Condition Operators" begin include("DerivativeOperators/robin.jl") end
    @time @safetestset "Composite Operators Interface" begin include("DerivativeOperators/composite_operators_interface.jl") end
    @time @safetestset "BC and Coefficient Compositions" begin include("DerivativeOperators/bc_coeff_compositions.jl") end
    @time @safetestset "Derivative Operators Interface" begin include("DerivativeOperators/derivative_operators_interface.jl") end
    @time @safetestset "Ghost Derivative Operators Interface" begin include("DerivativeOperators/ghost_derivative_operators_interface.jl") end
    @time @safetestset "Validate Regular Derivative Operators" begin include("DerivativeOperators/regular_operator_validation.jl") end
    @time @safetestset "Validate and Compare Generic Operators" begin include("DerivativeOperators/generic_operator_validation.jl") end
    @time @safetestset "Validate Boundary Padded Array Concretization" begin include("DerivativeOperators/boundary_padded_array.jl") end
    #@time @safetestset "Validate Higher Dimensional Boundary Extension" begin include("DerivativeOperators/multi_dim_bc_test.jl") end
    @time @safetestset "2nd order check" begin include("DerivativeOperators/2nd_order_check.jl") end
    @time @safetestset "Non-linear Diffusion" begin include("DerivativeOperators/Fast_Diffusion.jl") end
    @time @safetestset "KdV" begin include("DerivativeOperators/KdV.jl") end # 2-Soliton case needs implementation
    @time @safetestset "Heat Equation" begin include("DerivativeOperators/heat_eqn.jl") end
    @time @safetestset "Matrix-Free Operators" begin include("DerivativeOperators/matrixfree.jl") end
    @time @safetestset "Convolutions" begin include("DerivativeOperators/convolutions.jl") end
    @time @safetestset "Differentiation Dimension" begin include("DerivativeOperators/differentiation_dimension.jl") end
    @time @safetestset "Higher Dimensional Concretization" begin include("DerivativeOperators/concretization.jl") end
    @time @safetestset "Coefficient Functions" begin include("DerivativeOperators/coefficient_functions.jl") end
    @time @safetestset "Upwind Operator Interface" begin include("DerivativeOperators/upwind_operators_interface.jl") end
    @time @safetestset "Basic SDO Examples" begin include("DerivativeOperators/BasicSDOExamples.jl") end
    @time @safetestset "3D laplacian Test" begin include("DerivativeOperators/3D_laplacian.jl") end
    # @time @safetestset "Linear Complementarity Problem Examples" begin include("DerivativeOperators/lcp.jl"); include("DerivativeOperators/lcp_split.jl") end
end

if GROUP == "All" || GROUP == "MOLFiniteDifference"
    # @time @safetestset "MOLFiniteDifference Interface" begin include("MOL/MOLtest.jl") end
    @time @safetestset "MOLFiniteDifference Interface: Linear Convection" begin include("MOL/MOL_1D_Linear_Convection.jl") end
    @time @safetestset "MOLFiniteDifference Interface: 1D Linear Diffusion" begin include("MOL/MOL_1D_Linear_Diffusion.jl") end
    @time @safetestset "MOLFiniteDifference Interface: 1D Non-Linear Diffusion" begin include("MOL/MOL_1D_NonLinear_Diffusion.jl") end
    @time @safetestset "MOLFiniteDifference Interface: 2D Diffusion" begin include("MOL/MOL_2D_Diffusion.jl") end
    @time @safetestset "MOLFiniteDifference Interface: 1D HigherOrder" begin include("MOL/MOL_1D_HigherOrder.jl") end
    @time @safetestset "MOLFiniteDifference Interface: 1D Partial DAE" begin include("MOL/MOL_1D_PDAE.jl") end
    @time @safetestset "MOLFiniteDifference Interface: Stationary Nonlinear Problems" begin include("MOL/MOL_NonlinearProblem.jl") end
end

if GROUP == "All" || GROUP == "Misc"
    @time @safetestset "Utilities Tests" begin include("Misc/utils.jl") end
    @time @safetestset "JacVec Operators Interface" begin include("Misc/jacvec_operators.jl") end
    @time @safetestset "JacVec Operator Integration Test" begin include("Misc/jacvec_integration_test.jl") end
end

if !is_APPVEYOR && (GROUP == "All" || GROUP == "Multithreading")
    @time @safetestset "2D and 3D fast multiplication" begin include("DerivativeOperators/2D_3D_fast_multiplication.jl") end
end

if GROUP == "GPU"

end
end
