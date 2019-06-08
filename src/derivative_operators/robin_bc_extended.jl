
# For a boundary condition, the variables correspond to al*u(0) + bl*u'(0) = cl
struct RobinBC{T}
    al::T
    bl::T
    cl::T
    dx_l::T # should grid size be Real or more general?
    ar::T
    br::T
    cr::T
    dx_r::T

    function RobinBC(al::T, bl::T, cl::T, dx_l::T, ar::T, br::T, cr::T, dx_r::T) where T
        return new{T}(al, bl, cl, dx_l, ar, br, cr, dx_r)
    end
end

struct RobinBCExtended{T,T2<:AbstractVector{T}}
    l::T
    r::T
    u::T2

    function RobinBCExtended(u::T2, al::T, bl::T, cl::T,
                                    dx_l::T, ar::T, br::T, cr::T, dx_r::T) where
                                    {T,T2<:AbstractVector{T}}
        u                    = u
        l = (cl - bl*u[1]/dx_l)*(1/(al-bl/dx_l))
        r = (cr + br*u[length(u)]/dx_r)*(1/(ar+br/dx_r))
        return new{T,T2}(l, r, u)

    end
end


Base.:*(Q::RobinBC,u) = RobinBCExtended(u, Q.al, Q.bl, Q.cl, Q.dx_l, Q.ar, Q.br, Q.cr, Q.dx_r)
Base.length(Q::RobinBCExtended) = length(Q.u) + 2
Base.lastindex(Q::RobinBCExtended) = Base.length(Q)

function Base.getindex(Q::RobinBCExtended,i)
    if i == 1
        return Q.l
    elseif i == length(Q)
        return Q.r
    else
        return Q.u[i-1]
    end
end

function LinearAlgebra.Array(Q::RobinBC, N::Int)
    Q_L = [(-Q.bl/Q.dx_l)/(Q.al-Q.bl/Q.dx_l) transpose(zeros(N-1)); Diagonal(ones(N)); (Q.br/Q.dx_r)/(Q.ar+Q.br/Q.dx_r) transpose(zeros(N-1))]
    Q_b = [Q.cl; zeros(N); Q.cr]
    return (Q_L, Q_b)
end

function LinearAlgebra.Array(Q::RobinBCExtended)
    return [Q.l; Q.u; Q.r]
end


#################################################################################################

#=
(L::DirichletBCExtended)(u,p,t) = L*u
(L::DirichletBCExtended)(du,u,p,t) = mul!(du,L,u)
get_type(::DirichletBCExtended{A,B}) where {A,B} = A

#=
    The Inf opnorm can be calculated easily using the stencil coeffiicents, while other opnorms
    default to compute from the full matrix form.
=#
function LinearAlgebra.opnorm(A::DirichletBCExtended{T,S}, p::Real=2) where {T,S}
    if p == Inf
        sum(abs.(A.stencil_coefs)) / A.dx^A.derivative_order
    else
        opnorm(convert(Array,A), p)
    end
end
=#
