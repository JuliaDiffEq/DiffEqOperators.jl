# 2D diffusion problem

# Packages and inclusions
using ModelingToolkit,DiffEqOperators,LinearAlgebra,Test,OrdinaryDiffEq, DomainSets
using ModelingToolkit: Differential

# Tests
@testset "Test 00: Dt(u(t,x,y)) ~ Dxx(u(t,x,y)) + Dyy(u(t,x,y))" begin

    # Variables, parameters, and derivatives
    @parameters t x y
    @variables u(..)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2
    Dt = Differential(t)
    t_min= 0.
    t_max = 2.0
    x_min = 0.
    x_max = 2.
    y_min = 0.
    y_max = 2.

    # Analytic solution
    analytic_sol_func(t,x,y) = exp(x+y)*cos(x+y+4t)
    
    # Equation
    eq  = Dt(u(t,x,y)) ~ Dxx(u(t,x,y)) + Dyy(u(t,x,y))

    # Initial and boundary conditions
    bcs = [u(t_min,x,y) ~ analytic_sol_func(t_min,x,y),
           u(t,x_min,y) ~ analytic_sol_func(t,x_min,y),
           u(t,x_max,y) ~ analytic_sol_func(t,x_max,y),
           u(t,x,y_min) ~ analytic_sol_func(t,x,y_min),
           u(t,x,y_max) ~ analytic_sol_func(t,x,y_max)]

    # Space and time domains
    domains = [t ∈ Interval(t_min,t_max),
               x ∈ Interval(x_min,x_max),
               y ∈ Interval(y_min,y_max)]


    # PDE system
    pdesys = PDESystem([eq],bcs,domains,[t,x,y],[u(t,x,y)])

    dx = 0.1; dy = 0.2
    
    # Test against exact solution
    Nx = floor(Int64, (x_max - x_min) / dx) + 1
    Ny = floor(Int64, (y_max - y_min) / dy) + 1
    @variables u[1:Nx,1:Ny](t)
    r_space_x = x_min:dx:x_max
    r_space_y = y_min:dy:y_max
    asf = reshape([analytic_sol_func(t_max,r_space_x[i],r_space_y[j]) for j in 1:Ny for i in 1:Nx],(Nx,Ny))
    # Method of lines discretization
    order = 2
    discretization = MOLFiniteDifference([x=>dx,y=>dy],t;centered_order=order)
    prob = ModelingToolkit.discretize(pdesys,discretization)
    
    # Solution of the ODE system
    sol = solve(prob,Tsit5())
    
    # Test against exact solution
    sol′ = reshape([sol[u[(i-1)*Ny+j]][end] for i in 1:Nx for j in 1:Ny],(Nx,Ny))
    @test asf ≈ sol′ atol=0.4
    
    #Plot
    #using Plots
    #heatmap(sol′)
    #savefig("MOL_Linear_Diffusion_2D_Test00.png")
end

@testset "Test 01: Dt(u(t,x,y)) ~ Dx( a(x,y,u) * Dx(u(t,x,y))) + Dy( a(x,y,u) * Dy(u(t,x,y)))" begin

    # Variables, parameters, and derivatives
    @parameters t x y
    @variables u(..)
    Dx = Differential(x)
    Dy = Differential(y)
    Dt = Differential(t)
    t_min= 0.
    t_max = 2.0
    x_min = 0.
    x_max = 2.
    y_min = 0.
    y_max = 2.

    # Analytic solution
    analytic_sol_func(t,x,y) = exp(x+y)*cos(x+y+4t)
    
    # Equation
    eq = Dt(u(t,x,y)) ~ Dx( (u(t,x,y)^2 / exp(x+y)^2 + sin(x+y+4t)^2)^0.5 * Dx(u(t,x,y))) +
                        Dy( (u(t,x,y)^2 / exp(x+y)^2 + sin(x+y+4t)^2)^0.5 * Dy(u(t,x,y)))

    # Initial and boundary conditions
    bcs = [u(t_min,x,y) ~ analytic_sol_func(t_min,x,y),
           u(t,x_min,y) ~ analytic_sol_func(t,x_min,y),
           u(t,x_max,y) ~ analytic_sol_func(t,x_max,y),
           u(t,x,y_min) ~ analytic_sol_func(t,x,y_min),
           u(t,x,y_max) ~ analytic_sol_func(t,x,y_max)]

    # Space and time domains
    domains = [t ∈ Interval(t_min,t_max),
               x ∈ Interval(x_min,x_max),
               y ∈ Interval(y_min,y_max)]

    # Space and time domains
    pdesys = PDESystem([eq],bcs,domains,[t,x,y],[u(t,x,y)])

    # Method of lines discretization
    dx = 0.1; dy = 0.2
    discretization = MOLFiniteDifference([x=>dx,y=>dy],t)
    prob = ModelingToolkit.discretize(pdesys,discretization)

    # Solution of the ODE system
    sol = solve(prob,Rosenbrock23())

    # Test against exact solution
    Nx = floor(Int64, (x_max - x_min) / dx) + 1
    Ny = floor(Int64, (y_max - y_min) / dy) + 1
    @variables u[1:Nx,1:Ny](t)
    sol′ = reshape([sol[u[(i-1)*Ny+j]][end] for i in 1:Nx for j in 1:Ny],(Nx,Ny))
    r_space_x = x_min:dx:x_max
    r_space_y = y_min:dy:y_max
    asf = reshape([analytic_sol_func(t_max,r_space_x[i],r_space_y[j]) for j in 1:Ny for i in 1:Nx],(Nx,Ny))
    m = max(asf...)
    @test asf / m ≈ sol′ / m  atol=0.4 # TODO: use lower atol when g(x) is improved in MOL_discretize.jl

    #Plot
    #using Plots
    #heatmap(sol′)
    #savefig("MOL_NonLinear_Diffusion_2D_Test01.png")
    
end

