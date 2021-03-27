using ModelingToolkit: operation, istree, arguments
# Method of lines discretization scheme
struct MOLFiniteDifference{T,T2} <: DiffEqBase.AbstractDiscretization
    dxs::T
    time::T2
    upwind_order::Int
    centered_order::Int
end

# Constructors. If no order is specified, both upwind and centered differences will be 2nd order
MOLFiniteDifference(dxs, time = nothing; upwind_order = 1, centered_order = 2) =
    MOLFiniteDifference(dxs, time, upwind_order, centered_order)

function SciMLBase.symbolic_discretize(pdesys::ModelingToolkit.PDESystem,discretization::DiffEqOperators.MOLFiniteDifference)
    t = discretization.time
    nottime = filter(x->x.val != t.val,pdesys.indvars)

    # Discretize space

    space = map(nottime) do x
        xdomain = pdesys.domain[findfirst(d->x.val == d.variables,pdesys.domain)]
        @assert xdomain.domain isa IntervalDomain
        dx = discretization.dxs[findfirst(dxs->x.val == dxs[1].val,discretization.dxs)][2]
        xdomain.domain.lower:dx:xdomain.domain.upper
    end
    tdomain = pdesys.domain[findfirst(d->t.val == d.variables,pdesys.domain)]
    @assert tdomain.domain isa IntervalDomain
    tspan = (tdomain.domain.lower,tdomain.domain.upper)

    # Build symbolic variables
    indices = CartesianIndices(((axes(s)[1] for s in space)...,))
    depvars = map(pdesys.depvars) do u
        [Num(Variable{Symbolics.FnType{Tuple{Any}, Real}}(Base.nameof(ModelingToolkit.operation(u.val)),II.I...))(t) for II in indices]
    end
    spacevals = map(y->[Pair(nottime[i],space[i][y.I[i]]) for i in 1:length(nottime)],indices)

    # Build symbolic maps
    edges = reduce(vcat,[[vcat([Colon() for j in 1:i-1],1,[Colon() for j in i+1:length(nottime)]),
      vcat([Colon() for j in 1:i-1],size(depvars[1],i),[Colon() for j in i+1:length(nottime)])] for i in 1:length(nottime)])

    edgevals = reduce(vcat,[[nottime[i]=>first(space[i]),nottime[i]=>last(space[i])] for i in 1:length(space)])
    edgevars = [[d[e...] for e in edges] for d in depvars]
    depvarmaps = reduce(vcat,[substitute.((pdesys.depvars[i],),edgevals) .=> edgevars[i] for i in 1:length(pdesys.depvars)])
    edgemaps = [spacevals[e...] for e in edges]
    initmaps = substitute.(pdesys.depvars,[t=>tspan[1]])

    # Generate initial conditions and bc equations
    # Assume in the form `u(...) ~ ...` for now
    u0 = []
    bceqs = []
    for bc in pdesys.bcs
        if t.val ∉ ModelingToolkit.arguments(bc.lhs)
            # initial condition
            push!(u0,vec(depvars[findfirst(isequal(bc.lhs),initmaps)] .=> substitute.((bc.rhs,),spacevals)))
        else
            # Algebraic equations for BCs
            i = findfirst(x->isequal(x,bc.lhs),first.(depvarmaps))
            lhs = substitute(bc.lhs,depvarmaps[i])
            rhs = substitute.((bc.rhs,),edgemaps[i])
            push!(bceqs,lhs .~ rhs)
        end
    end
    u0 = reduce(vcat,u0)
    bceqs = reduce(vcat,bceqs)

    # Generate PDE Equations
    interior = indices[2:end-1,2:end-1]
    eqs = vec(map(Base.product(interior,pdesys.eqs)) do p
        i,eq = p
        # [-1,0,1] assumes dx and dy are constant
        central_weights = DiffEqOperators.calculate_weights(discretization.centered_order, 0.0, [-1,0,1])
        neighbor_idxs(i,j) = [i+CartesianIndex([ifelse(l==j,-1,0) for l in 1:length(nottime)]...),i,i+CartesianIndex([ifelse(l==j,1,0) for l in 1:length(nottime)]...)]
        deriv_rules = [(Differential(nottime[j])^2)(pdesys.depvars[k]) => dot(central_weights,depvars[k][neighbor_idxs(i,j)]) for j in 1:length(nottime), k in 1:length(pdesys.depvars)]
        valrules = [pdesys.depvars[k] => depvars[k][i] for k in 1:length(pdesys.depvars)]
        substitute(eq.lhs,vcat(vec(deriv_rules),valrules)) ~ substitute(eq.rhs,vcat(vec(deriv_rules),valrules))
    end)

    # Finalize
    sys = ODESystem(vcat(eqs,unique(bceqs)),t,vec(reduce(vcat,vec(depvars))),Num[])
    sys, u0, tspan
end

function SciMLBase.discretize(pdesys::ModelingToolkit.PDESystem,discretization::DiffEqOperators.MOLFiniteDifference)
    sys, u0, tspan = SciMLBase.symbolic_discretize(pdesys,discretization)
    simpsys = structural_simplify(sys)
    prob = ODEProblem(simpsys,vec(u0),tspan)
end
