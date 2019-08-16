# mul! done by convolutions
function LinearAlgebra.mul!(x_temp::AbstractVector{T}, A::DerivativeOperator, x::AbstractVector{T}; overwrite = true) where T<:Real
    convolve_BC_left!(x_temp, x, A, overwrite = overwrite)
    convolve_interior!(x_temp, x, A, overwrite = overwrite)
    convolve_BC_right!(x_temp, x, A, overwrite = overwrite)
end

################################################

# Against a standard vector, assume already padded and just apply the stencil
function convolve_interior!(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::DerivativeOperator; overwrite = true) where {T<:Real}
    @assert length(x_temp)+2 == length(x)
    stencil = A.stencil_coefs
    coeff   = A.coefficients
    mid = div(A.stencil_length,2)
    for i in (1+A.boundary_point_count) : (length(x_temp)-A.boundary_point_count)
        xtempi = zero(T)
        cur_stencil = eltype(stencil) <: AbstractVector ? stencil[i-A.boundary_point_count] : stencil
        cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[i] : coeff isa Number ? coeff : true
        cur_stencil = use_winding(A) && cur_coeff < 0 ? reverse(cur_stencil) : cur_stencil
        for idx in 1:A.stencil_length
            xtempi += cur_coeff * cur_stencil[idx] * x[i - mid + idx]
        end
        x_temp[i] = xtempi + !overwrite*x_temp[i]
    end
end

function convolve_BC_left!(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::DerivativeOperator; overwrite = true) where {T<:Real}
    stencil = A.low_boundary_coefs
    coeff   = A.coefficients
    for i in 1 : A.boundary_point_count
        cur_stencil = stencil[i]
        cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[i] : coeff isa Number ? coeff : true
        cur_stencil = use_winding(A) && cur_coeff < 0 ? reverse(cur_stencil) : cur_stencil
        xtempi = cur_coeff*stencil[i][1]*x[1]
        for idx in 2:A.boundary_stencil_length
            xtempi += cur_coeff * cur_stencil[idx] * x[idx]
        end
        x_temp[i] = xtempi + !overwrite*x_temp[i]
    end
end

function convolve_BC_right!(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::DerivativeOperator; overwrite = true) where {T<:Real}
    stencil = A.high_boundary_coefs
    coeff   = A.coefficients
    for i in 1 : A.boundary_point_count
        cur_stencil = stencil[i]
        cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[i] : coeff isa Number ? coeff : true
        cur_stencil = use_winding(A) && cur_coeff < 0 ? reverse(cur_stencil) : cur_stencil
        xtempi = cur_coeff*stencil[i][end]*x[end]
        for idx in (A.boundary_stencil_length-1):-1:1
            xtempi += cur_coeff * cur_stencil[end-idx] * x[end-idx]
        end
        x_temp[end-A.boundary_point_count+i] = xtempi + !overwrite*x_temp[end-A.boundary_point_count+i]
    end
end

###########################################

# Against A BC-padded vector, specialize the computation to explicitly use the left, right, and middle parts
function convolve_interior!(x_temp::AbstractVector{T}, _x::BoundaryPaddedVector, A::DerivativeOperator; overwrite = true) where {T<:Real}
    stencil = A.stencil_coefs
    coeff   = A.coefficients
    x = _x.u
    mid = div(A.stencil_length,2) + 1
    # Just do the middle parts
    for i in (2+A.boundary_point_count) : (length(x_temp)-A.boundary_point_count)-1
        xtempi = zero(T)
        cur_stencil = eltype(stencil) <: AbstractVector ? stencil[i-A.boundary_point_count] : stencil
        cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[i-A.boundary_point_count] : coeff isa Number ? coeff : true
        cur_stencil = use_winding(A) && cur_coeff < 0 ? reverse(cur_stencil) : cur_stencil
        @inbounds for idx in 1:A.stencil_length
            xtempi += cur_coeff * cur_stencil[idx] * x[(i-1) - (mid-idx) + 1]
        end
        x_temp[i] = xtempi + !overwrite*x_temp[i]
    end
end

function convolve_BC_left!(x_temp::AbstractVector{T}, _x::BoundaryPaddedVector, A::DerivativeOperator; overwrite = true) where {T<:Real}
    stencil = A.low_boundary_coefs
    coeff   = A.coefficients
    for i in 1 : A.boundary_point_count
        cur_stencil = stencil[i]
        cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[i] : coeff isa Number ? coeff : true
        xtempi = cur_coeff*cur_stencil[1]*_x.l
        cur_stencil = use_winding(A) && cur_coeff < 0 ? reverse(cur_stencil) : cur_stencil
        @inbounds for idx in 2:A.boundary_stencil_length
            xtempi += cur_coeff * cur_stencil[idx] * _x.u[idx-1]
        end
        x_temp[i] = xtempi + !overwrite*x_temp[i]
    end
    # need to account for x.l in first interior
    mid = div(A.stencil_length,2) + 1
    x = _x.u
    i = 1 + A.boundary_point_count
    xtempi = zero(T)
    cur_stencil = eltype(A.stencil_coefs) <: AbstractVector ? A.stencil_coefs[i-A.boundary_point_count] : A.stencil_coefs
    cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[i-A.boundary_point_count] : coeff isa Number ? coeff : true
    cur_stencil = use_winding(A) && cur_coeff < 0 ? reverse(cur_stencil) : cur_stencil
    xtempi = cur_coeff*cur_stencil[1]*_x.l
    @inbounds for idx in 2:A.stencil_length
        xtempi += cur_coeff * cur_stencil[idx] * x[(i-1) - (mid-idx) + 1]
    end
    x_temp[i] = xtempi + !overwrite*x_temp[i]
end

function convolve_BC_right!(x_temp::AbstractVector{T}, _x::BoundaryPaddedVector, A::DerivativeOperator; overwrite = true) where {T<:Real}
    stencil = A.high_boundary_coefs
    coeff   = A.coefficients
    bc_start = length(_x.u) - A.boundary_point_count
    # need to account for _x.r in last interior convolution
    mid = div(A.stencil_length,2) + 1
    x = _x.u
    i = length(x_temp)-A.boundary_point_count
    xtempi = zero(T)
    cur_stencil = eltype(A.stencil_coefs) <: AbstractVector ? A.stencil_coefs[i-A.boundary_point_count] : A.stencil_coefs
    cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[i-A.boundary_point_count] : coeff isa Number ? coeff : true
    cur_stencil = use_winding(A) && cur_coeff < 0 ? reverse(cur_stencil) : cur_stencil
    xtempi = cur_coeff*cur_stencil[end]*_x.r
    @inbounds for idx in 1:A.stencil_length-1
        xtempi += cur_coeff * cur_stencil[idx] * x[(i-1) - (mid-idx) + 1]
    end
    x_temp[i] = xtempi + !overwrite*x_temp[i]
    for i in 1 : A.boundary_point_count
        cur_stencil = stencil[i]
        cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[bc_start + i] : coeff isa Number ? coeff : true
        xtempi = cur_coeff*cur_stencil[end]*_x.r
        cur_stencil = use_winding(A) && cur_coeff < 0 ? reverse(cur_stencil) : cur_stencil
        @inbounds for idx in A.stencil_length:-1:1
            xtempi += cur_coeff * cur_stencil[end-idx] * _x.u[end-idx+1]
        end
        x_temp[bc_start + i] = xtempi + !overwrite*x_temp[bc_start + i]
    end
end

###########################################

function convolve_interior_add_range!(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::DerivativeOperator, offset::Int) where {T<:Real}
    @assert length(x_temp)+2 == length(x)
    stencil = A.stencil_coefs
    coeff   = A.coefficients
    mid = div(A.stencil_length,2)
    for i in [(1+A.boundary_point_count):(A.boundary_point_count+offset); (length(x_temp)-A.boundary_point_count-offset+1):(length(x_temp)-A.boundary_point_count)]
        xtempi = zero(T)
        cur_stencil = eltype(stencil) <: AbstractVector ? stencil[i] : stencil
        cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[i] : coeff isa Number ? coeff : true
        cur_stencil = use_winding(A) && cur_coeff < 0 ? reverse(cur_stencil) : cur_stencil
        for idx in 1:A.stencil_length
            xtempi += cur_coeff * cur_stencil[idx] * x[i - mid + idx]
        end
        x_temp[i] += xtempi
    end
end

################################################################################
# Stub for curl operator convolution draft while its still in my head
################################################################################

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
