struct CompositeVectorDerivativeOperator{T,N,M,O,S} <:AbstractDiffEqCompositeOperator{T} # If an element that ops returns is vector or tensor valued with dimension N, the result is flattened along the first N dimensions, with the other dimensions in ops coming after.
    ops::AbstractArray{O,N}
    reduction
end
struct VectorDerivativeOperator{T<:Real,D,C,Wind,T2,S1,S2<:SVector,T3,F} <: AbstractDerivativeOperator{T}
    op::DerivativeOperator{T<:Real,N,Wind,T2,S1,S2<:SVector,T3,F}
end

struct VectorDifference{D,C} end


function DelOperator(derivative_order::Int,
                        approximation_order::Int, Δ::NTuple{N,AbstractVector{T}},
                        s::NTuple{N,Int}, coeff_func=nothing) where {T<:Real}
    ops = permutedims([CenteredDifference{n}(derivative_order, approximation_order, Δ[n] s[n], coeff_func) for n in 1:N])
    reduction = nothing
    CompositeVectorDerivativeOperator{T,1,N,Union{typeof(ops)...},"Del"}(ops, reduction)
end
#fallback
LinearAlgebra.dot(V::CompositeVectorDerivativeOperator{T,N,1}, u::AbstractArray{T,N}) where {T<:AbstractVector,N}
    u_temp = similar(u)
    u_temp = V.ops.*u

    flatten(u_temp)
end
