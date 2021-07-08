# 1D Non-Linear Diffusion tests
# See doi:10.1016/j.camwa.2006.12.077

# TODO: Add more complex tests.

# Packages and inclusions
using ModelingToolkit,DiffEqOperators,LinearAlgebra,Test,OrdinaryDiffEq, DomainSets
using ModelingToolkit: Differential

# Tests
@testset "Test 00: Dt(u(t,x)) ~ Dx(u(t,x)^(-1) * Dx(u(t,x)))" begin 
    # Variables, parameters, and derivatives
    @parameters t x
    @variables u(..)
    Dx = Differential(x)
    Dt = Differential(t)
    t_min= 0.
    t_max = 2.
    x_min = 0.
    x_max = 2.
    c = 1.0
    a = 1.0

    # Analytic solution
    analytic_sol_func(t,x) = 2.0 * (c + t) / (a + x)^2
    
    # Equation
    eq = Dt(u(t,x)) ~ Dx(u(t,x)^(-1) * Dx(u(t,x)))

    # Initial and boundary conditions
    bcs = [u(t_min,x) ~ analytic_sol_func(t_min,x),
           u(t,x_min) ~ analytic_sol_func(t,x_min),
           u(t,x_max) ~ analytic_sol_func(t,x_max)]

    # Space and time domains
    domains = [t ∈ Interval(t_min,t_max),
               x ∈ Interval(x_min,x_max)]

    # PDE system
    pdesys = PDESystem([eq],bcs,domains,[t,x],[u(t,x)])

    # Method of lines discretization
    dx = 0.01
    discretization = MOLFiniteDifference([x=>dx],t)
    prob = ModelingToolkit.discretize(pdesys,discretization)

    # Solution of the ODE system
    using OrdinaryDiffEq
    sol = solve(prob,Rosenbrock32())

    # Test against exact solution
    r_space = x_min:dx:x_max
    asf = [analytic_sol_func(t_max,x) for x in r_space]
    Nx = floor(Int64, (x_max - x_min) / dx) + 1
    @variables u[1:Nx](t)
    sol′ = [sol[u[i]][end] for i in 1:Nx]
    @test asf ≈ sol′ atol=0.1

    # Plots
    #using Plots
    #plot(r_space, asf, seriestype = :scatter,label="analytic solution")
    #plot!(r_space, sol′, label="numeric solution")
    #savefig("MOL_NonLinear_Diffusion_1D_Test00.png")
    

end

@testset "Test 01: Dt(u(t,x)) ~ Dx(u(t,x)^2 * Dx(u(t,x)))" begin

    # Variables, parameters, and derivatives
    @parameters t x
    @variables u(..)
    Dx = Differential(x)
    Dt = Differential(t)
    t_min= 0.
    t_max = 2.
    x_min = 0.
    x_max = 2.
    c = 50.0
    h = 0.50

    # Analytic solution
    analytic_sol_func(t,x) = 0.5 * (x + h) / sqrt(c - t)
    
    # Equation
    eq = Dt(u(t,x)) ~ Dx(u(t,x)^2 * Dx(u(t,x)))

    # Initial and boundary conditions
    bcs = [u(t_min,x) ~ analytic_sol_func(t_min,x),
           u(t,x_min) ~ analytic_sol_func(t,x_min),
           u(t,x_max) ~ analytic_sol_func(t,x_max)]

    # Space and time domains
    domains = [t ∈ Interval(t_min,t_max),
               x ∈ Interval(x_min,x_max)]

    # PDE system
    pdesys = PDESystem([eq],bcs,domains,[t,x],[u(t,x)])

    # Method of lines discretization
    dx = 0.01
    discretization = MOLFiniteDifference([x=>dx],t)
    prob = ModelingToolkit.discretize(pdesys,discretization)

    # Solution of the ODE system
    using OrdinaryDiffEq
    sol = solve(prob,Rosenbrock32())

    # Test against exact solution
    r_space = x_min:dx:x_max
    asf = [analytic_sol_func(t_max,x) for x in r_space]
    Nx = floor(Int64, (x_max - x_min) / dx) + 1
    @variables u[1:Nx](t)
    sol′ = [sol[u[i]][end] for i in 1:Nx]
    @test asf ≈ sol′ atol=0.01

    # Plots
    #using Plots
    #plot(r_space, asf, seriestype = :scatter,label="analytic solution")
    #plot!(r_space, sol′, label="numeric solution")
    #savefig("MOL_NonLinear_Diffusion_1D_Test01.png")

end


@testset "Test 02: Dt(u(t,x)) ~ Dx(1. / (1. + u(t,x)^2) * Dx(u(t,x)))" begin

    # Variables, parameters, and derivatives
    @parameters t x
    @variables u(..)
    Dx = Differential(x)
    Dt = Differential(t)
    t_min= 0.
    t_max = 2.
    x_min = 0.
    x_max = 2.

    # Analytic solution
    analytic_sol_func(t,x) = tan(x)

    # Equation
    eq = Dt(u(t,x)) ~ Dx(1. / (1. + u(t,x)^2) * Dx(u(t,x)))

    # Initial and boundary conditions
    bcs = [u(t_min,x) ~ analytic_sol_func(t_min,x),
           u(t,x_min) ~ analytic_sol_func(t,x_min),
           u(t,x_max) ~ analytic_sol_func(t,x_max)]

    # Space and time domains
    domains = [t ∈ Interval(t_min,t_max),
               x ∈ Interval(x_min,x_max)]

    # PDE system
    pdesys = PDESystem([eq],bcs,domains,[t,x],[u(t,x)])

    # Method of lines discretization
    dx = 0.01
    discretization = MOLFiniteDifference([x=>dx],t)
    prob = ModelingToolkit.discretize(pdesys,discretization)

    # Solution of the ODE system
    using OrdinaryDiffEq
    sol = solve(prob,Rosenbrock32()) # TODO: check warnings

    # Test against exact solution
    r_space = x_min:dx:x_max
    asf = [analytic_sol_func(t_max,x) for x in r_space]
    Nx = floor(Int64, (x_max - x_min) / dx) + 1
    @variables u[1:Nx](t)
    sol′ = [sol[u[i]][end] for i in 1:Nx]

    m = max(asf...,sol′...)
    @test asf / m ≈ sol′ / m atol=0.16 # the difference occurs when tan(x) goes to infinite

    # Plots
    #using Plots
    #plot(r_space, asf, seriestype = :scatter,label="Analytic solution")
    #plot!(r_space, sol′, label="Numeric solution")
    #savefig("MOL_NonLinear_Diffusion_1D_Test02.png")

end

@testset "Test 03: Dt(u(t,x)) ~ Dx(1. / (u(t,x)^2 - 1.) * Dx(u(t,x)))" begin

    # Variables, parameters, and derivatives
    @parameters t x
    @variables u(..)
    Dx = Differential(x)
    Dt = Differential(t)
    t_min= 0.
    t_max = 2.
    x_min = 0.
    x_max = 2.

    # Analytic solution
    analytic_sol_func(t,x) = -coth(x)

    # Equation
    eq = Dt(u(t,x)) ~ Dx(1. / (u(t,x)^2 - 1.) * Dx(u(t,x)))

    # Initial and boundary conditions
    bcs = [u(t_min,x) ~ analytic_sol_func(t_min,x),
           u(t,x_min) ~ analytic_sol_func(t,x_min),
           u(t,x_max) ~ analytic_sol_func(t,x_max)]

    # Space and time domains
    domains = [t ∈ Interval(t_min,t_max),
               x ∈ Interval(x_min,x_max)]

    # PDE system
    pdesys = PDESystem([eq],bcs,domains,[t,x],[u(t,x)])

    # Method of lines discretization
    dx = 0.01
    discretization = MOLFiniteDifference([x=>dx],t)
    prob = ModelingToolkit.discretize(pdesys,discretization)

    # Solution of the ODE system
    using OrdinaryDiffEq
    sol = solve(prob,Rosenbrock32())

    # Test against exact solution
    r_space = x_min:dx:x_max
    asf = [analytic_sol_func(t_max,x) for x in r_space]
    Nx = floor(Int64, (x_max - x_min) / dx) + 1
    @variables u[1:Nx](t)
    sol′ = [sol[u[i]][end] for i in 1:Nx]

    m = max(asf...,sol′...)
    @test asf / m ≈ sol′ / m atol=0.16

    # Plots
    #using Plots
    #plot(r_space, asf, seriestype = :scatter,label="Analytic solution")
    #plot!(r_space, sol′, label="Numeric solution")
    #savefig("MOL_NonLinear_Diffusion_1D_Test03.png")

end
