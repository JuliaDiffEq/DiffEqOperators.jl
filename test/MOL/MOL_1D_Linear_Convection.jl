# 1D linear convection problem

# Packages and inclusions
using ModelingToolkit,DiffEqOperators,DiffEqBase,LinearAlgebra,Test, DomainSets


# Tests

@testset "Test 00: Dt(u(t,x)) ~ -Dx(u(t,x))" begin
    # Parameters, variables, and derivatives
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    t_i = 0.0; t_f = 0.6
    x_i = 0.0; x_f = 2.0

    # Analytic solution
    analytic_sol_func(t, x) = (0.5 / (0.2 * sqrt(2.0 * 3.1415))) *
                               exp(-(x - t - 0.75)^2 / (2.0 * 0.2^2))

    # 1D PDE and boundary conditions
    eq  = Dt(u(t, x)) ~ -Dx(u(t, x))
    bcs = [u(0.0, x) ~ analytic_sol_func(0.0, x),
           u(t, x_i) ~ analytic_sol_func(t, x_i),
           u(t, x_f) ~ analytic_sol_func(t, x_f)]

    # Space and time domains
    domains = [t ∈ Interval(t_i, t_f),
               x ∈ Interval(x_i, x_f)]


    # PDE system
    pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    # Method of lines discretization
    dx = 2 / 80
    order = 1
    discretization = MOLFiniteDifference([x => dx], t)
    discretization_upwind = MOLFiniteDifference([x => dx], t; upwind_order = order)

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys, discretization)
    prob_upwind = discretize(pdesys, discretization_upwind)

    # Solve ODE problem
    using OrdinaryDiffEq
    sol = solve(prob, Euler(), dt = .0025, saveat = 0.1)
    sol_upwind = solve(prob_upwind, Euler(), dt = .0025, saveat = 0.1)

    # Test
    x_interval = infimum(domains[2].domain) + dx : dx : supremum(domains[2].domain) - dx
    asf = [analytic_sol_func(t_f, x) for x in x_interval]
    t_f_idx = size(sol)[2]
    m = max(asf...)
    @test sol[:, t_f_idx] ≈ asf / m atol = 0.5;
    @test sol_upwind[:, t_f_idx] / m ≈ asf / m atol = 0.5;

    #using Plots
    #plot(sol[:, t_f_idx])
    #savefig("sol[:, t_f_idx]")
    #plot(sol_upwind[:, t_f_idx])
    #savefig("sol_upwind[:, t_f_idx]")
end

@testset "Test 01: Dt(u(t,x)) ~ -Dx(u(t,x)) + 0.01" begin
   # Parameters, variables, and derivatives
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    t_i = 0.0; t_f = 0.6
    x_i = 0.0; x_f = 2.0

    # Analytic solution
    analytic_sol_func(t, x) = (0.5 / (0.2 * sqrt(2.0 * 3.1415))) *
                               exp(-(x - t - 0.75)^2 / (2.0 * 0.2^2))

    # 1D PDE and boundary conditions
    eq  = Dt(u(t, x)) ~ -Dx(u(t, x)) + 0.01
    bcs = [u(0.0, x) ~ analytic_sol_func(0.0, x),
           u(t, x_i) ~ analytic_sol_func(t, x_i),
           u(t, x_f) ~ analytic_sol_func(t, x_f)]

    # Space and time domains
    domains = [t ∈ Interval(t_i, t_f),
               x ∈ Interval(x_i, x_f)]


    # PDE system
    pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    # Method of lines discretization
    dx = 2 / 80
    order = 1
    discretization = MOLFiniteDifference([x => dx], t)
    discretization_upwind = MOLFiniteDifference([x => dx], t; upwind_order = order)

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys, discretization)
    prob_upwind = discretize(pdesys, discretization_upwind)

    # Solve ODE problem
    using OrdinaryDiffEq
    sol = solve(prob, Euler(), dt = .0025, saveat = 0.1)
    sol_upwind = solve(prob_upwind, Euler(), dt = .0025, saveat = 0.1)

    # Test
    x_interval = infimum(domains[2].domain) + dx : dx : supremum(domains[2].domain) - dx
    asf = [analytic_sol_func(t_f, x) for x in x_interval]
    t_f_idx = size(sol)[2]
    m = max(asf...)
    @test sol[:, t_f_idx] ≈ asf / m atol = 0.5;
    @test sol_upwind[:, t_f_idx] / m ≈ asf / m atol = 0.5;

    #using Plots
    #plot(sol[:, t_f_idx])
    #savefig("sol[:, t_f_idx]")
    #plot(sol_upwind[:, t_f_idx])
    #savefig("sol_upwind[:, t_f_idx]")

end

@testset "Test 02: Dt(u(t,x)) ~ -c * Dx(u(t,x))" begin
    # Parameters, variables, and derivatives
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    c = 1.0
    t_i = 0.0; t_f = 0.6
    x_i = 0.0; x_f = 2.0

    # Analytic solution
    analytic_sol_func(t, x) = (0.5 / (0.2 * sqrt(2.0 * 3.1415))) *
                               exp(-(x - c * t - 0.75)^2 / (2.0 * 0.2^2))

    # 1D PDE and boundary conditions
    eq  = Dt(u(t, x)) ~ -c * Dx(u(t, x))
    bcs = [u(0.0, x) ~ analytic_sol_func(0.0, x),
           u(t, x_i) ~ analytic_sol_func(t, x_i),
           u(t, x_f) ~ analytic_sol_func(t, x_f)]

    # Space and time domains
    domains = [t ∈ Interval(t_i, t_f),
               x ∈ Interval(x_i, x_f)]

    # PDE system
    pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    # Method of lines discretization
    dx = 2 / 80
    order = 1
    discretization = MOLFiniteDifference([x => dx], t)
    discretization_upwind = MOLFiniteDifference([x => dx], t; upwind_order = order)

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys, discretization)
    prob_upwind = discretize(pdesys, discretization_upwind)

    # Solve ODE problem
    using OrdinaryDiffEq
    sol = solve(prob, Euler(), dt = .0025, saveat = 0.1)
    sol_upwind = solve(prob_upwind, Euler(), dt = .0025, saveat = 0.1)

    # Test
    x_interval = infimum(domains[2].domain) + dx : dx : supremum(domains[2].domain) - dx
    asf = [analytic_sol_func(t_f, x) for x in x_interval]
    t_f_idx = size(sol)[2]
    m = max(asf...)
    @test sol[:, t_f_idx] / m ≈ asf / m atol = 0.5;
    @test sol_upwind[:, t_f_idx] / m ≈ asf / m atol = 0.5;
end

#@testset "Test 03: Dt(u(t,x)) ~ -Dx(v(t,x)) * u(t,x) - v(t,x) * Dx(u(t,x)); v(t,x) ~ 1" begin
#    # Parameters, variables, and derivatives
#    @parameters t x
#    @variables u(..) v(..)
#    Dt = Differential(t)
#    Dx = Differential(x)
#    t_i = 0.0; t_f = 0.6
#    x_i = 0.0; x_f = 2.0

#    # Analytic solution
#    analytic_sol_func(t, x) = (0.5 / (0.2 * sqrt(2.0 * 3.1415))) *
#                               exp(-(x - t - 0.75)^2 / (2.0 * 0.2^2))

#    # 1D PDE and boundary conditions
#    eq  = [Dt(u(t, x)) ~ -Dx(v(t,x)) * u(t,x) - v(t,x) * Dx(u(t,x)), 
#           v(t,x) ~ 1.0]
#    bcs = [u(0.0, x) ~ analytic_sol_func(0.0, x),
#           u(t, x_i) ~ analytic_sol_func(t, x_i),
#           u(t, x_f) ~ analytic_sol_func(t, x_f),
#           v(0,x) ~ 1.0,
#           v(t,0) ~ 1.0,
#           v(t,2) ~ 1.0]

#    # Space and time domains
#    domains = [t ∈ Interval(t_i, t_f),
#               x ∈ Interval(x_i, x_f)]

#    # PDE system
#    pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x), v(t, x)])

#    # Method of lines discretization
#    dx = 2 / 80
#    order = 1
#    discretization = MOLFiniteDifference([x => dx], t)
#    discretization_upwind = MOLFiniteDifference([x => dx], t; upwind_order = order)

#    # Convert the PDE problem into an ODE problem
#    prob = discretize(pdesys, discretization)
#    prob_upwind = discretize(pdesys, discretization_upwind)

#    # Solve ODE problem
#    using OrdinaryDiffEq
#    sol = solve(prob, Euler(), dt = .0025, saveat = 0.1)
#    sol_upwind = solve(prob_upwind, Euler(), dt = .0025, saveat = 0.1)

#    # Test
#    x_interval = infimum(domains[2].domain) + dx : dx : supremum(domains[2].domain) - dx
#    asf = [analytic_sol_func(t_f, x) for x in x_interval]
#    t_f_idx = size(sol)[2]
#    m = max(asf...)
#    @test sol[:, t_f_idx] / m ≈ asf / m atol = 0.6;
#    @test sol_upwind[:, t_f_idx] / m ≈ asf / m atol = 0.6;
#end

#@testset "Test 04: Dt(u(t,x)) ~ -Dx(v(t,x)) * u(t,x) - v(t,x) * Dx(u(t,x)); v(t,x) ~ sin(t*x)^2 + cos(t*x)^2 " begin
#    # Parameters, variables, and derivatives
#    @parameters t x
#    @variables u(..) v(..)
#    Dt = Differential(t)
#    Dx = Differential(x)
#    t_i = 0.0; t_f = 0.6
#    x_i = 0.0; x_f = 2.0

#    # Analytic solution
#    analytic_sol_func(t, x) = (0.5 / (0.2 * sqrt(2.0 * 3.1415))) *
#                               exp(-(x - t - 0.75)^2 / (2.0 * 0.2^2))

#    # 1D PDE and boundary conditions
#    eq  = [Dt(u(t, x)) ~ -Dx(v(t,x)) * u(t,x) - v(t,x) * Dx(u(t,x)), 
#           v(t,x) ~ sin(t*x)^2 + cos(t*x)^2]
#    bcs = [u(0.0, x) ~ analytic_sol_func(0.0, x),
#           u(t, x_i) ~ analytic_sol_func(t, x_i),
#           u(t, x_f) ~ analytic_sol_func(t, x_f),
#           v(0,x) ~ 1.0,
#           v(t,0) ~ 1.0,
#           v(t,2) ~ 1.0]

#    # Space and time domains
#    domains = [t ∈ Interval(t_i, t_f),
#               x ∈ Interval(x_i, x_f)]

#    # PDE system
#    pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x), v(t, x)])

#    # Method of lines discretization
#    dx = 2 / 80
#    order = 1
#    discretization = MOLFiniteDifference([x => dx], t)
#    discretization_upwind = MOLFiniteDifference([x => dx], t; upwind_order = order)

#    # Convert the PDE problem into an ODE problem
#    prob = discretize(pdesys, discretization)
#    prob_upwind = discretize(pdesys, discretization_upwind)

#    # Solve ODE problem
#    using OrdinaryDiffEq
#    sol = solve(prob, Euler(), dt = .0025, saveat = 0.1)
#    sol_upwind = solve(prob_upwind, Euler(), dt = .0025, saveat = 0.1)

#    # Test
#    x_interval = infimum(domains[2].domain) + dx : dx : supremum(domains[2].domain) - dx
#    asf = [analytic_sol_func(t_f, x) for x in x_interval]
#    t_f_idx = size(sol)[2]
#    m = max(asf...)
#    @test sol[:, t_f_idx] / m ≈ asf / m atol = 0.5;
#    @test sol_upwind[:, t_f_idx] / m ≈ asf / m atol = 0.5;
#end



