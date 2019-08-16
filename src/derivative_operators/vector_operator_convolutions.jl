###############################################################################
# Vector operator Convolutions
###############################################################################

#referring to https://en.wikipedia.org/wiki/Curl_(mathematics)
# h[i] = √((Δᵤ[i]*x)^2 + (Δᵤ[i]*y)^2 + (Δᵤ[i]*z)^2)) where x,y,z are cartesian coords and u[1], u[2], u[3] is the orthogonal coord system in use
function convolve_interior!(x_temp::AbstractArray{SVector{3, T}, 3}, u::AbstractArray{SVector{3, T},3}, A::CurlOperator)
    s = size(x_temp)
    stencil = A.stencil_coefs
    coeff   = A.coefficients
    R = CartesianIndices((2+A.boundary_point_count) : (s[i]-A.boundary_point_count-3) for i in 1:3)
    ê = begin #create unit CartesianIndex for each dimension
        out = Vector{CartesianIndex{N}}(undef, 3)
        null = zeros(Int64, 3)
        for i in 1:3
            unit_i = copy(null)
            unit_i[i] = 1
            out[i] = CartesianIndex(Tuple(unit_i))
        end
        out
    end
    mid = div(A.stencil_length,2)
    for I in R
        x̄ = zeros(T,3)
        cur_stencil = eltype(stencil) <: AbstractArray ? stencil[I] : stencil
        cur_coeff   = typeof(coeff)   <: AbstractArray ? coeff[I] : true
        cur_stencil = use_winding(A) && cur_coeff < 0 ? reverse(cur_stencil) : cur_stencil
        for idx in (1-mid):(A.stencil_length-mid)
            x̄[1] += cur_coeff*cur_stencil[idx]*(A.h[I-ê[2]*idx] * u[I-ê[2]*idx][3] - A.h[I-ê[2]*idx] * u[I-ê[3]*idx][2])/(A.h[I-ê[2]*idx]*A.h[I-ê[3]*idx])
            x̄[2] += cur_coeff*cur_stencil[idx]*(A.h[I-ê[3]*idx] * u[I-ê[3]*idx][1] - A.h[I-ê[1]*idx] * u[I-ê[1]*idx][3])/(A.h[I-ê[3]*idx]*A.h[I-ê[1]*idx])
            x̄[3] += cur_coeff*cur_stencil[idx]*(A.h[I-ê[1]*idx] * u[I-ê[1]*idx][2] - A.h[I-ê[2]*idx] * u[I-ê[2]*idx][1])/(A.h[I-ê[1]*idx]*A.h[I-ê[2]*idx])
        end
        x_temp[i,j,k] = SVector(x̄)
    end
end

function convolve_interior!(x_temp::AbstractArray{SVector{N, T}, N}, u::AbstractArray{SVector{N, T},N}, A::DivOperator) where {T,N}
    s = size(x_temp)
    stencil = A.stencil_coefs
    coeff   = A.coefficients
    R = CartesianIndices((2+A.boundary_point_count) : (s[i]-A.boundary_point_count-3) for i in 1:N)
    ê = begin #create unit CartesianIndex for each dimension
        out = Vector{CartesianIndex{N}}(undef, N)
        null = zeros(Int64, N)
        for i in 1:N
            unit_i = copy(null)
            unit_i[i] = 1
            out[i] = CartesianIndex(Tuple(unit_i))
        end
        out
    end
    mid = div(A.stencil_length,2)
    for I in R
        x̄ = zero(T)
        cur_stencil = eltype(stencil) <: AbstractArray ? stencil[I] : stencil
        cur_coeff   = typeof(coeff)   <: AbstractArray ? coeff[I] : true
        cur_stencil = use_winding(A) && cur_coeff < 0 ? reverse(cur_stencil) : cur_stencil
        for dim in 1:N
            for idx in (1-mid):(A.stencil_length-mid)
                x̄ += cur_coeff * cur_stencil[idx] * x[I-idx*ê[dim]]
            end
        end
        x_temp[i,j,k] = x̄
    end
end
