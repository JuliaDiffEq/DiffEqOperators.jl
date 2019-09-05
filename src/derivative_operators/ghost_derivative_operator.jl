struct GhostDerivativeOperator{T, E<:AbstractDiffEqLinearOperator{T}, F<:AbstractBC{T}} <: AbstractDiffEqLinearOperator{T}
    L :: E
    Q :: F
end

function Base.:*(L::AbstractDiffEqLinearOperator{T}, Q::AbstractBC{T}) where{T}
    return GhostDerivativeOperator{T, typeof(L), typeof(Q)}(L,Q)
end

function LinearAlgebra.mul!(x::AbstractArray{T,N}, A::GhostDerivativeOperator{T,E,F}, u::AbstractArray{T,N}) where {T,E,F,N}
    LinearAlgebra.mul!(x, A.L, u)
end

function *(A::GhostDerivativeOperator{T,E,F}, u::AbstractArray{T}) where {T,E,F}
    #TODO Implement a function domaincheck(L::AbstractDiffEqLinearOperator, u) to see if components of L along each dimension match the size of u
    x = similar(u)
    LinearAlgebra.mul!(x, A, A.Q*u)
    return x
end


function \(A::GhostDerivativeOperator{T,E,F}, u::AbstractArray{T,N}) where {T,E,F,N}
    #TODO implement check that A has compatible size with u
    s = size(u)
    x = zeros(T,prod(s))
    LinearAlgebra.ldiv!(x, A, u)
    return reshape(x, s)
end


function LinearAlgebra.ldiv!(x::AbstractVector{T}, A::GhostDerivativeOperator{T,E,F}, u::AbstractVector{T}) where {T,E,F}
    @assert length(x) == size(A.L,1)
    (AL,Ab) = sparse(A, size(A.L,1))
    LinearAlgebra.ldiv!(x, AL, u.-Ab)
end

function LinearAlgebra.ldiv!(x::AbstractVector{T}, A::GhostDerivativeOperator{T,E,F}, u::AbstractMatrix{T}) where {T,E,F} # Needs to be specifically defined to avoid ambiuguity with the fallback method in DiffEqBase
    s_ = prod(size(u))
    @assert length(x) == s_
    Al, Ab = sparse(A, size(u))
    LinearAlgebra.ldiv!(x, Al, reshape(u, s_).-Ab)
end

function LinearAlgebra.ldiv!(x::AbstractVector{T}, A::GhostDerivativeOperator{T,E,F}, u::AbstractArray{T,N}) where {T,E,F,N}
    s_ = prod(size(u))
    @assert length(x) == s_
    Al, Ab = sparse(A, size(u))
    LinearAlgebra.ldiv!(x, Al, reshape(u, s_).-Ab)
end


function \(A::GhostDerivativeOperator{T,E,F}, u::AbstractVector{T}) where {T,E,F}
    x = zeros(T,length(u))
    LinearAlgebra.ldiv!(x, A, u)
    return x
end

# update coefficients
function DiffEqBase.update_coefficients!(A::GhostDerivativeOperator{T,E,F},u,p,t) where {T,E,F}
    DiffEqBase.update_coefficients!(A.L,u,p,t)
end

function *(coeff_func::Function, A::GhostDerivativeOperator{T,N,Wind}) where {T,N,Wind}
    (coeff_func*A.L)*A.Q
end

# length and sizes
Base.ndims(A::GhostDerivativeOperator) = 2
Base.size(A::GhostDerivativeOperator) = (size(A.L, 2), size(A.L, 2))
Base.size(A::GhostDerivativeOperator,i::Integer) = size(A)[i]
Base.length(A::GhostDerivativeOperator) = reduce(*, size(A))
