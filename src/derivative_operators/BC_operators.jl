abstract type AbstractBC{T} <: AbstractDiffEqLinearOperator{T} end

# Deepen type tree to support multi layered BCs in the future - a better version of PeriodicBC for example
abstract type AtomicBC{T} <: AbstractBC{T} end
abstract type SingleLayerBC{T} <: AtomicBC{T} end

abstract type MultiDimensionalBC{T, N} <: AbstractBC{T} end
abstract type AbstractBoundaryPaddedArray{T, N} <: AbstractArray{T, N} end
"""
Robin, General, and in general Neumann and Dirichlet BCs are all affine opeartors, meaning that they take the form Q*x = Qa*x + Qb.
"""
abstract type AffineBC{T} <: SingleLayerBC{T} end

struct PeriodicBC{T} <: SingleLayerBC{T}
end

struct MultiDimensionalPeriodicBC{T,D,N} <: MultiDimensionalBC{T,N}
end
"""
  RobinBC(left_coefficients, right_coefficients, [dx_left, dx_right], approximation_order)

-------------------------------------------------------------------------------------

  The variables in l are [αl, βl, γl], and correspond to a BC of the form al*u(0) + bl*u'(0) = cl
  Implements a robin boundary condition operator Q that acts on a vector to give an extended vector as a result
  Referring to (https://github.com/JuliaDiffEq/DiffEqOperators.jl/files/3267835/ghost_node.pdf)
  Write vector b̄₁ as a vertical concatanation with b0 and the rest of the elements of b̄ ₁, denoted b̄`₁, the same with ū into u0 and ū`. b̄`₁ = b̄`_2 = fill(β/Δx, length(stencil)-1)
  Pull out the product of u0 and b0 from the dot product. The stencil used to approximate u` is denoted s. b0 = α+(β/Δx)*s[1]
  Rearrange terms to find a general formula for u0:= -b̄`₁̇⋅ū`/b0 + γ/b0, which is dependent on ū` the robin coefficients and Δx.
  The non identity part of Qa is qa:= -b`₁/b0 = -β.*s[2:end]/(α+β*s[1]/Δx). The constant part is Qb = γ/(α+β*s[1]/Δx)
  do the same at the other boundary (amounts to a flip of s[2:end], with the other set of boundary coeffs)
"""
struct RobinBC{T} <: AffineBC{T}
    a_l::Vector{T}
    b_l::T
    a_r::Vector{T}
    b_r::T
    function RobinBC(l::AbstractVector{T}, r::AbstractVector{T}, dx::AbstractVector{T}, order = 1) where {T}
        αl, βl, γl = l
        αr, βr, γr = r
        dx_l, dx_r = dx

        s = calculate_weights(1, one(T), Array(one(T):convert(T,order+1))) #generate derivative coefficients about the boundary of required approximation order

        a_l = -s[2:end]./(αl*dx_l/βl + s[1])
        a_r = s[end:-1:2]./(αr*dx_r/βr - s[1]) # for other boundary stencil is flippedlr with *opposite sign*

        b_l = γl/(αl+βl*s[1]/dx_l)
        b_r = γr/(αr-βr*s[1]/dx_r)

        return new{T}(a_l, b_l, a_r, b_r)
    end
end



"""
GeneralBC(α_leftboundary, α_rightboundary, [dx_left, dx_right], approximation_order)

-------------------------------------------------------------------------------------

Implements a generalization of the Robin boundary condition, where α is a vector of coefficients.
Represents a condition of the form α[1] + α[2]u[0] + α[3]u'[0] + α[4]u''[0]+... = 0
Implemented in a similar way to the RobinBC (see above).
This time there are multiple stencils for multiple derivative orders - these can be written as a matrix S.
All components that multiply u(0) are factored out, turns out to only involve the first colum of S, s̄0. The rest of S is denoted S`. the coeff of u(0) is s̄0⋅ᾱ[3:end] + α[2].
the remaining components turn out to be ᾱ[3:end]⋅(S`ū`) or equivalantly (transpose(ᾱ[3:end])*S`)⋅ū`. Rearranging, a stencil q_a to be dotted with ū` upon extension can readily be found, along with a constant component q_b
"""
struct GeneralBC{T} <:AffineBC{T}
    a_l::Vector{T}
    b_l::T
    a_r::Vector{T}
    b_r::T
    function GeneralBC(αl::AbstractVector{T}, αr::AbstractVector{T}, dx::AbstractVector{T}, order = 1) where {T}
        dx_l, dx_r = dx
        nl = length(αl)
        nr = length(αr)
        S_l = zeros(T, (nl-2, order+nl-2))
        S_r = zeros(T, (nr-2, order+nr-2))

        for i in 1:(nl-2)
            S_l[i,:] = [transpose(calculate_weights(i, one(T), Array(one(T):convert(T, order+i)))) transpose(zeros(T, Int(nl-2-i)))] #am unsure if the length of the dummy_x is correct here
        end

        for i in 1:(nr-2)
            S_r[i,:] = [transpose(calculate_weights(i, convert(T, order+i), Array(one(T):convert(T, order+i)))) transpose(zeros(T, Int(nr-2-i)))]
        end
        s0_l = S_l[:,1] ; Sl = S_l[:,2:end]
        s0_r = S_r[:,1] ; Sr = S_r[:,2:end]

        denoml = αl[2] .+ αl[3:end] ⋅ s0_l
        denomr = αr[2] .+ αr[3:end] ⋅ s0_r

        a_l = -transpose(transpose(αl[3:end]) * Sl) ./denoml
        a_r = -transpose(transpose(αr[3:end]) * Sr) ./denomr

        b_l = -αl[1]/denoml
        b_r = -αr[1]/denomr
        new{T}(a_l,b_l,reverse!(a_r),b_r)
    end
end


"""
Quick and dirty way to allow mixed boundary types on each end of an array - may be cleaner and more versatile to split up left and right boundaries going forward
MixedBC(lowerBC, upperBC) is the interface.
"""

struct MixedBC{T, R <: SingleLayerBC{T}, S <: SingleLayerBC{T}} <: SingleLayerBC{T}
    lower::R
    upper::S
    MixedBC(Qlower,Qupper) = new{Union{gettype(Qlower), gettype(Qupper)}, typeof(Qlower), typeof(Qupper)}(Qlower, Qupper)
end

function Base.:*(Q::MixedBC, u::AbstractVector)
    lower = Q.lower*u
    upper = Q.upper*u
    return BoundaryPaddedVector(lower.l, upper.r, u)
end

#implement Neumann and Dirichlet as special cases of RobinBC
NeumannBC(α::AbstractVector{T}, dx::AbstractVector{T}, order = 1) where T = RobinBC([zero(T), one(T), α[1]], [zero(T), one(T), α[2]], dx, order)
DirichletBC(α::AbstractVector{T}, dx::AbstractVector{T}, order = 1) where T = RobinBC([one(T), zero(T), α[1]], [one(T), zero(T), α[2]], dx, order)
#specialized constructors for Neumann0 and Dirichlet0
Dirichlet0BC(dx::AbstractVector{T}, order = 1) where T = DirichletBC([zero(T), zero(T)], dx, order = 1)
Neumann0BC(dx::AbstractVector{T}, order = 1) where T = NeumannBC([zero(T), zero(T)], dx, order = 1)

# other acceptable argument signatures
RobinBC(al::T, bl::T, cl::T, dx_l::T, ar::T, br::T, cr::T, dx_r::T, order = 1) where T = RobinBC([al,bl, cl], [ar, br, cr], [dx_l, dx_r], order)
"""
Allows seperate domains governed by seperate equations to be bridged together, should be used as one end of a MixedBC as it will extend both boundaries with the same value
"""
struct BridgeBC{T,I,N} <: SingleLayerBC{T}
    from::SubArray{T,0,Array{T,N},NTuple{N,I},true}
end
function BridgeBC(u::AbstractArray{T,N}, inds) where {T, N}
    @assert length(inds) == N-1
    @assert mapreduce(x -> typeof(x) <: Integer, (&), inds)
    BridgeBC{T, N, eltype(inds)}(view(u, inds...))
end

Base.:*(Q::BridgeBC{T,I,N}, u::AbstractVector{T}) where {T, I, N} = BoundaryPaddedVector{T, typeof(u)}(Q.from, Q.from, u)

"""
A vector type that extends a vector u with ghost points at either end
"""

struct BoundaryPaddedVector{T,T2 <: AbstractVector{T}} <: AbstractBoundaryPaddedArray{T, 1}
    l::T
    r::T
    u::T2
end


Base.:*(Q::AffineBC, u::AbstractVector) = BoundaryPaddedVector(Q.a_l ⋅ u[1:length(Q.a_l)] + Q.b_l, Q.a_r ⋅ u[(end-length(Q.a_r)+1):end] + Q.b_r, u)
Base.:*(Q::PeriodicBC, u::AbstractVector) = BoundaryPaddedVector(u[end], u[1], u)
Base.size(Q::SingleLayerBC) = (Inf, Inf) #Is this nessecary?
Base.length(Q::BoundaryPaddedVector) = length(Q.u) + 2
Base.size(Q::BoundaryPaddedVector) = (length(Q),)
Base.lastindex(Q::BoundaryPaddedVector) = Base.length(Q)
gettype(Q::AbstractBC{T}) where T = T

function Base.getindex(Q::BoundaryPaddedVector,i)
    if i == 1
        return Q.l
    elseif i == length(Q)
        return Q.r
    else
        return Q.u[i-1]
    end
end

function LinearAlgebra.Array(Q::AffineBC{T}, N::Int) where {T}
    Q_L = [transpose(Q.a_l) transpose(zeros(T, N-length(Q.a_l))); Diagonal(ones(T,N)); transpose(zeros(T, N-length(Q.a_r))) transpose(Q.a_r)]
    Q_b = [Q.b_l; zeros(T,N); Q.b_r]
    return (Array(Q_L), Q_b)
end

function SparseArrays.SparseMatrixCSC(Q::AffineBC{T}, N::Int) where {T}
    Q_L = [transpose(Q.a_l) transpose(zeros(T, N-length(Q.a_l))); Diagonal(ones(T,N)); transpose(zeros(T, N-length(Q.a_r))) transpose(Q.a_r)]
    Q_b = [Q.b_l; zeros(T,N); Q.b_r]
    return (Q_L, Q_b)
end

function SparseArrays.sparse(Q::AffineBC{T}, N::Int) where {T}
    SparseMatrixCSC(Q,N)
end

LinearAlgebra.Array(Q::PeriodicBC{T}, N::Int) where T = Array([transpose(zeros(T, N-1)) one(T); Diagonal(ones(T,N)); one(T) transpose(zeros(T, N-1))])
SparseArrays.SparseMatrixCSC(Q::PeriodicBC{T}, N::Int) where T = [transpose(zeros(T, N-1)) one(T); Diagonal(ones(T,N)); one(T) transpose(zeros(T, N-1))]
SparseArrays.sparse(Q::PeriodicBC{T}, N::Int) where T = SparseMatrixCSC(Q,N)

function LinearAlgebra.Array(Q::BoundaryPaddedVector)
    return [Q.l; Q.u; Q.r]
end

function Base.convert(::Type{Array},A::SingleLayerBC{T}) where T
    Array(A)
end

function Base.convert(::Type{SparseMatrixCSC},A::SingleLayerBC{T}) where T
    SparseMatrixCSC(A)
end

function Base.convert(::Type{AbstractMatrix},A::SingleLayerBC{T}) where T
    SparseMatrixCSC(A)
end

#######################################################################
# Multidimensional
#######################################################################

#SingleLayerBCSubtypes = Union{vcat(InteractiveUtils.subtypes(SingleLayerBC{T}), InteractiveUtils.subtypes(AffineBC{T}))...} where T

# A union type to allow dispatch for MultiDimBC to work correctly
#UnionSingleLayerBCArray{T,N} = Union{[Array{B,N} for B in InteractiveUtils.subtypes(SingleLayerBCSubtypes{T})]..., [Array{MixedBC{T, R, S}, N} for R in InteractiveUtils.subtypes(SingleLayerBCSubtypes{T}), S in InteractiveUtils.subtypes(SingleLayerBCSubtypes{T})]..., Array{SingleLayerBC{T}, N}}


"""
slicemul is the only limitation on the BCs here being used up to arbitrary dimension, an N dimensional implementation is sorely needed
"""
@inline function slicemul(A::Array{SingleLayerBC{T},1}, u::AbstractArray{T, 2}, dim::Integer) where T
    s = size(u)
    if dim == 1
        lower = zeros(T, s[2])
        upper = deepcopy(lower)
        for i in 1:(s[2])
            tmp = A[i]*u[:,i]
            lower[i], upper[i] = (tmp.l, tmp.r)
        end
    elseif dim == 2
        lower = zeros(T, s[1])
        upper = deepcopy(lower)
        for i in 1:(s[1])
            tmp = A[i]*u[i,:]
            lower[i], upper[i] = (tmp.l, tmp.r)
        end
    elseif dim == 3
        throw("The 3 dimensional Method should be being called, not this one. Check dispatch.")
    else
        throw("Dim greater than 3 not supported!")
    end
    return lower, upper
end


@inline function slicemul(A::Array{SingleLayerBC{T},2}, u::AbstractArray{T, 3}, dim::Integer) where {T}
    s = size(u)
    if dim == 1
        lower = zeros(T, s[2], s[3])
        upper = deepcopy(lower)
        for j in 1:s[3]
            for i in 1:s[2]
                tmp = A[i,j]*u[:,i,j]
                lower[i,j], upper[i,j] = (tmp.l, tmp.r)
            end
        end
    elseif dim == 2
        lower = zeros(T, s[1], s[3])
        upper = deepcopy(lower)
        for j in 1:s[3]
            for i in 1:s[1]
                tmp = A[i,j]*u[i,:,j]
                lower[i,j], upper[i,j] = (tmp.l, tmp.r)
            end
        end
    elseif dim == 3
        lower = zeros(T, s[1], s[2])
        upper = deepcopy(lower)
        for j in 1:s[2]
            for i in 1:s[1]
                tmp = A[i,j]*u[i,j,:]
                lower[i,j], upper[i,j] = (tmp.l, tmp.r)
            end
        end
    else
        throw("Dim greater than 3 not supported!")
    end
    return lower, upper
end

"""
A multiple dimensional BC, supporting arbitrary BCs at each boundary point.
To construct an arbitrary BC, pass an Array of BCs with dimension one less than that of your domain u - denoted N,
with a size of size(u)[setdiff(1:N, dim)], where dim is the dimension orthogonal to the boundary that you want to extend.
It is also possible to call MultiDimBC(YourBC, size(u), dim) to use YourBC for the whole boundary orthogonal to that dimension.
Further, it is possible to call Qx, Qy, Qz... = MultiDimBC(YourBC, size(u)) to use YourBC for the whole boundary for all dimensions.
"""
struct MultiDimensionalSingleLayerBC{T<:Number, D, N, M} <: MultiDimensionalBC{T, N}
    BC::Array{SingleLayerBC{T},M} #dimension M=N-1 array of BCs to extend dimension D
end

struct ComposedMultiDimBC{T,N,M} <: MultiDimensionalBC{T, N}
    BCs::Vector{Array{SingleLayerBC{T}, M}} # The typing here is a nightmare
end

MultiDimBC(BC::Array{SingleLayerBC{T},N}, dim::Integer) where {T,N} = MultiDimensionalSingleLayerBC{T, dim, N+1, N}(BC)
#s should be size of the domain
MultiDimBC(BC::SingleLayerBC{T}, s, dim::Integer) where {T} = MultiDimensionalSingleLayerBC{T, dim, length(s), length(s)-1}(fill(BC, s[setdiff(1:length(s), dim)]))
#Extra constructor to make a set of BC operators that extend an atomic BC Operator to the whole domain

MultiDimBC(BC::SingleLayerBC{T}, s) where T = Tuple([MultiDimensionalSingleLayerBC{T, dim, length(s), length(s)-1}(fill(BC, s[setdiff(1:length(s), dim)])) for dim in 1:length(s)])
PeriodicBC{T}(s) where T = MultiDimBC(PeriodicBC{T}(), s)


"""
Higher dimensional generalization of BoundaryPaddedVector, pads an array of dimension N with 2 Arrays of dimension N-1, stored in lower and upper along the dimension D

"""
struct BoundaryPaddedArray{T<:Number, D, N, M, V<:AbstractArray{T, N}, B<: AbstractArray{T, M}} <: AbstractBoundaryPaddedArray{T,N}
    lower::B #an array of dimension M = N-1, used to extend the lower index boundary
    upper::B #Ditto for the upper index boundary
    u::V
end

struct ComposedBoundaryPaddedArray{T<:Number, N, M, V<:AbstractArray{T, N}, B<: AbstractArray{T, M}} <: AbstractBoundaryPaddedArray{T, N}
    lower::Vector{B}
    upper::Vector{B}
    u::V
end


function compose(BCs...)
    T = gettype(BCs[1])
    N = ndims(BCs[1])
    Ds = getaxis.(BCs)
    (length(BCs) == N) || throw("There must be enough BCs to cover every dimension - check that the number of MultiDimBCs == N")
    for D in Ds
        length(setdiff(Ds, D)) == (N-1) || throw("There are multiple boundary conditions that extend along $D - make sure every dimension has a unique extension")
    end
    BCs = BCs[sortperm([Ds...])]

    ComposedMultiDimBC{T,N,N-1}([condition.BC for condition in BCs])
end

function compose(padded_arrays::BoundaryPaddedArray...)
    N = ndims(padded_arrays[1])
    Ds = getaxis.(padded_arrays)
    (length(padded_arrays) == N) || throw("The padded_arrays must cover every dimension - make sure that the number of padded_arrays is equal to ndims(u).")
    for D in Ds
        length(setdiff(Ds, D)) == (N-1) || throw("There are multiple Arrays that extend along $D - make sure every dimension has a unique extension")
    end
    reduce((|), fill(padded_arrays[1].u, (length(padded_arrays),)) .== getfield.(padded_arrays, :u)) || throw("The padded_arrays do not all extend the same u!")
    padded_arrays = padded_arrays[sortperm([Ds...])]
    lower = [padded_array.lower for padded_array in padded_arrays]
    upper = [padded_array.upper for padded_array in padded_arrays]

    ComposedBoundaryPaddedArray{gettype(padded_arrays[1]),N,N-1,typeof(padded_arrays[1].u),typeof(lower[1])}(lower, upper, padded_arrays[1].u)
end

function LinearAlgebra.Array(Q::ComposedBoundaryPaddedArray{T,N,M,V,B}) where {T,N,M,V,B}
    S = size(Q)
    out = zeros(T, S...)
    dimset = 1:N
    uview = out
    for dim in dimset
        ulowview = selectdim(out, dim, 1)
        uhighview = selectdim(out, dim, S[dim])
        uview = selectdim(uview, dim, 2:(S[dim]-1))
        for (index, otherdim) in enumerate(setdiff(dimset, dim))
            ulowview = selectdim(ulowview, index, 2:(S[otherdim]-1))
            uhighview = selectdim(uhighview, index, 2:(S[otherdim]-1))
        end
        ulowview .= Q.lower[dim]
        uhighview .= Q.upper[dim]
    end
    uview .= Q.u
    return out
end

function LinearAlgebra.Array(Q::BoundaryPaddedArray{T,D,N,M,V,B}) where {T,D,N,M,V,B}
    S = size(Q)
    out = zeros(T, S...)
    dim = D
    dimset = 1:N
    ulowview = selectdim(out, dim, 1)
    uhighview = selectdim(out, dim, S[dim])
    uview = selectdim(out, dim, 2:(S[dim]-1))
    ulowview .= Q.lower
    uhighview .= Q.upper
    uview .= Q.u
    return out
end

AbstractBoundaryPaddedMatrix{T} = AbstractBoundaryPaddedArray{T,2}
AbstractBoundaryPadded3Tensor{T} = AbstractBoundaryPaddedArray{T,3}

BoundaryPaddedMatrix{T, D, V, B} = BoundaryPaddedArray{T, D, 2, 1, V, B}
BoundaryPadded3Tensor{T, D, V, B} = BoundaryPaddedArray{T, D, 3, 2, V, B}

ComposedBoundaryPaddedMatrix{T,V,B} = ComposedBoundaryPaddedArray{T,2,1,V,B}
ComposedBoundaryPadded3Tensor{T,V,B} = ComposedBoundaryPaddedArray{T,3,2,V,B}


function Base.size(Q::BoundaryPaddedArray)
    S = [size(Q.u)...]
    S[getaxis(Q)] += 2
    return Tuple(S)
end

Base.length(Q::AbstractBoundaryPaddedArray) = reduce((*), size(Q))
Base.lastindex(Q::AbstractBoundaryPaddedArray) = Base.length(Q)
gettype(Q::AbstractBoundaryPaddedArray{T,N}) where {T,N} = T
Base.ndims(Q::AbstractBoundaryPaddedArray{T,N}) where {T,N} = N
getaxis(Q::BoundaryPaddedArray{T,D,N,M,V,B}) where {T,D,N,M,V,B} = D
getaxis(Q::MultiDimensionalSingleLayerBC{T, D, N, K}) where {T, D, N, K} = D
perpsize(A::AbstractArray{T,N}, dim::Integer) where {T,N} = size(A)[setdiff(1:N, dim)] #the size of A perpendicular to dim

Base.size(Q::ComposedBoundaryPaddedArray) = size(Q.u).+2

Base.ndims(Q::MultiDimensionalBC{T,N}) where {T,N} = N

decompose(A::ComposedBoundaryPaddedArray) = Tuple([BoundaryPaddedArray{gettype(A), ndims(A), ndims(A)-1, typeof(lower[1])}(A.lower[i], A.upper[i], A.u) for i in 1:ndims(A)])
decompose(Q::ComposedMultiDimBC{T,N,M}) where {T,N,M} = Tuple([MultiDimBC(Q.BC[i], i) for i in 1:N])

add_dim(A::AbstractArray, i) = reshape(A, size(A)...,i)
add_dim(i) = i

function experms(N::Integer, dim) # A function to correctly permute the dimensions of the padding arrays so that they can be concatanated with the rest of u in getindex(::BoundaryPaddedArray)
    if dim == N
        return Vector(1:N)
    elseif dim < N
        P = experms(N, dim+1)
        P[dim], P[dim+1] = P[dim+1], P[dim]
        return P
    else
        throw("Dim is greater than N!")
    end
end

function Base.getindex(Q::BoundaryPaddedArray{T,D,N,M,V,B}, _inds...) where {T,D,N,M,V,B} #supports range and colon indexing!
    inds = [_inds...]
    S = size(Q)
    dim = D
    otherinds = inds[setdiff(1:N, dim)]
    @assert length(S) == N
    if inds[dim] == 1
        return Q.lower[otherinds...]
    elseif inds[dim] == S[dim]
        return Q.upper[otherinds...]
    elseif typeof(inds[dim]) <: Integer
        inds[dim] = inds[dim] - 1
        return Q.u[inds...]
    elseif typeof(inds[dim]) == Colon
        if mapreduce(x -> typeof(x) != Colon, (|), otherinds)
            return vcat(Q.lower[otherinds...],  Q.u[inds...], Q.upper[otherinds...])
        else
            throw("A colon on the extended dim is as yet incompatible with additional colons")
            #=lower = Q.lower[otherinds...]
            upper = Q.upper[otherinds...]
            extradimslower = N - ndims(lower)
            extradimsupper = N - ndims(upper)
            lower = permutedims(add_dim(lower, extradimslower), experms(N, dim)) #Adding dimensions and permuting so that cat doesn't get confused
            upper = permutedims(add_dim(upper, extradimsupper), experms(N, dim))
            return cat(lower, Q.u[inds...],  upper; dims=dim)
            =#
        end
    elseif typeof(inds[dim]) <: AbstractArray
        throw("Range indexing not yet supported!")
        #=
        @assert reduce((|), 1 .<= inds[dim] .<= S[dim])
        inds[dim] = Array(inds[dim])
        if (1 ∈ inds[dim]) | (S[dim] ∈ inds[dim])
            inds[dim] .= inds[dim] .- 1
            lower = permutedims(add_dim(Q.lower[otherinds...]), experms(N,dim)) #Adding dimensions and permuting so that cat doesn't get confused
            upper = permutedims(add_dim(Q.upper[otherinds...]),  experms(N,dim))
            if 1 ∉ inds[dim]
                return cat(Q.u[inds...], upper; dims=dim)
            elseif S[dim] ∉ inds[dim]
                return cat(lower, Q.u[inds...]; dims=dim)
            else
                return cat(lower, Q.u[inds...],  upper; dims=dim)
            end
        end
        inds[dim] .= inds[dim] .- 1
        return Q.u[inds...]
        =#
    end
end

function Base.getindex(Q::ComposedBoundaryPaddedArray, inds...) #as yet no support for range indexing or colon indexing
    S = size(Q)
    T = gettype(Q)
    N = ndims(Q)
    @assert reduce((&), inds .< S)
    for (dim, index) in enumerate(inds)
        if index == 1
            _inds = inds[setdiff(1:N, dim)]
            if (1 ∈ _inds) | reduce((|), S[setdiff(1:N, dim)] .== _inds)
                return zero(T)
            else
                return Q.lower[dim][(_inds.-1)...]
            end
        elseif index == S[dim]
            _inds = inds[setdiff(1:N, dim)]
            if (1 ∈ _inds) | reduce((|), S[setdiff(1:N, dim)] .== _inds)
                return zero(T)
            else
                return Q.upper[dim][(_inds.-1)...]
            end
        end
     end
    return Q.u[(inds.-1)...]
end


function Base.:*(Q::MultiDimensionalSingleLayerBC{T, D, N, K}, u::AbstractArray{T, N}) where {T, D, N, K}
    lower, upper = slicemul(Q.BC, u, D)
    return BoundaryPaddedArray{T, D, N, K, typeof(u), typeof(lower)}(lower, upper, u)
end

function Base.:*(Q::ComposedMultiDimBC{T, N, K}, u::AbstractArray{T, N}) where {T, N, K}
    lower, upper = slicemul.(Q.BCs, fill(u, N), 1:N)
    return ComposedBoundaryPaddedArray{T, N, K, typeof(u), typeof(lower[1])}([lower...], [upper...], u)
end

function LinearAlgebra.mul!(u_temp::AbstractArray{T,N}, Q::MultiDimensionalSingleLayerBC{T, D, N, K}, u::AbstractArray{T, N}) where {T,D,N,K}
    u_temp = Array(Q*u)
end
