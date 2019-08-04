# mul! done by convolutions
function LinearAlgebra.mul!(x_temp::AbstractVector{T}, A::DerivativeOperator, x::AbstractVector{T}) where T<:Real
    convolve_BC_left!(x_temp, x, A)
    convolve_interior!(x_temp, x, A)
    convolve_BC_right!(x_temp, x, A)
end

################################################

# Against a standard vector, assume already padded and just apply the stencil
function convolve_interior!(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::DerivativeOperator) where {T<:Real}
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
            x_idx = use_winding(A) && cur_coeff < 0 ? x[i + mid - idx] : x[i - mid + idx]
            xtempi += cur_coeff * cur_stencil[idx] * x_idx
            # @show i, idx, cur_stencil[idx], i-mid+idx, x[i-mid+idx]
        end
        x_temp[i] = xtempi
    end
end

function convolve_BC_left!(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::DerivativeOperator) where {T<:Real}
    stencil = A.low_boundary_coefs
    coeff   = A.coefficients

    _bpc = A.boundary_point_count
    flag = false
    if stencil == []
        _bpc = A.boundary_point_count + 1
        flag = true
    end

    for i in 1 : _bpc
        if flag == true
            cur_stencil = A.stencil_coefs
            slen = length(A.stencil_coefs)
        else
            cur_stencil = stencil[i]
            slen = length(cur_stencil)
        end

        cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[i] : coeff isa Number ? coeff : true
        cur_stencil = use_winding(A) && cur_coeff < 0 ? reverse(cur_stencil) : cur_stencil
        xtempi = cur_coeff*cur_stencil[1]*x[1]
        for idx in 2:slen
            xtempi += cur_coeff * cur_stencil[idx] * x[idx]
        end
        x_temp[i] = xtempi
    end
end

function convolve_BC_right!(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::DerivativeOperator) where {T<:Real}
    stencil = A.high_boundary_coefs
    coeff   = A.coefficients
    N       = length(x)
    L       = A.boundary_stencil_length

    _bpc = A.boundary_point_count
    flag = false

    if stencil == []
        _bpc = 1
        flag = true
    end

    for i in 1 : _bpc
        # @show _bpc
        if flag == true
            cur_stencil = A.stencil_coefs
            slen = length(A.stencil_coefs)
            L = A.stencil_length
        else
            cur_stencil = stencil[i]
            slen = length(cur_stencil)
        end

        cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[i] : coeff isa Number ? coeff : true
        cur_stencil = use_winding(A) && cur_coeff < 0 ? reverse(cur_stencil) : cur_stencil
        cs = cur_stencil*(A.dx^A.derivative_order)
        xtempi = zero(T)
        for idx in 1:slen
            # @show idx, N-L+idx, cs[idx], x[N-L+idx]
            xtempi += cur_coeff * cur_stencil[idx] * x[N-L+idx]
        end
        # @show xtempi*(1/(A.dx^A.derivative_order))
        x_temp[end-_bpc+i] = xtempi
    end
end

###########################################

# Against A BC-padded vector, specialize the computation to explicitly use the left, right, and middle parts
function convolve_interior!(x_temp::AbstractVector{T}, _x::BoundaryPaddedVector, A::DerivativeOperator) where {T<:Real}
    @assert length(x_temp) == length(_x.u)
    stencil = A.stencil_coefs
    coeff   = A.coefficients

    mid = div(A.stencil_length,2)

    x = _x.u
    i = 1+A.boundary_point_count
    idx = 1
    xtempi = zero(T)
    cur_stencil = eltype(stencil) <: AbstractVector ? stencil[i-A.boundary_point_count] : stencil
    cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[i-A.boundary_point_count] : coeff isa Number ? coeff : true
    cur_stencil = use_winding(A) && cur_coeff < 0 ? reverse(cur_stencil) : cur_stencil
    x_idx = _x.l
    xtempi += cur_coeff * cur_stencil[idx] * x_idx
    @inbounds for idx in 2:A.stencil_length
        # @show i, idx, cur_stencil[idx], i-mid+idx, x[i-mid+idx]
        x_idx = use_winding(A) && cur_coeff < 0 ? x[i + mid - idx - 1] : x[i - mid + idx - 1]
        xtempi += cur_coeff * cur_stencil[idx] * x_idx
    end
    x_temp[i] = xtempi


    # Just do the middle parts
    for i in (2+A.boundary_point_count) : (length(x_temp)-A.boundary_point_count-1)
        xtempi = zero(T)
        cur_stencil = eltype(stencil) <: AbstractVector ? stencil[i-A.boundary_point_count] : stencil
        cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[i-A.boundary_point_count] : coeff isa Number ? coeff : true
        cur_stencil = use_winding(A) && cur_coeff < 0 ? reverse(cur_stencil) : cur_stencil
        @inbounds for idx in 1:A.stencil_length
            # @show i, idx, cur_stencil[idx], i-mid+idx, x[i-mid+idx]
            x_idx = use_winding(A) && cur_coeff < 0 ? x[i + mid - idx - 1] : x[i - mid + idx - 1]
            xtempi += cur_coeff * cur_stencil[idx] * x_idx
        end
        x_temp[i] = xtempi
    end

    i = length(x_temp)-A.boundary_point_count
    xtempi = zero(T)
    cur_stencil = eltype(stencil) <: AbstractVector ? stencil[i-A.boundary_point_count] : stencil
    cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[i-A.boundary_point_count] : coeff isa Number ? coeff : true
    cur_stencil = use_winding(A) && cur_coeff < 0 ? reverse(cur_stencil) : cur_stencil
    @inbounds for idx in 1:A.stencil_length-1
        # @show i, idx, cur_stencil[idx], i-mid+idx, x[i-mid+idx]
        x_idx = use_winding(A) && cur_coeff < 0 ? x[i + mid - idx - 1] : x[i - mid + idx - 1]
        xtempi += cur_coeff * cur_stencil[idx] * x_idx
    end
    x_idx = _x.r
    idx = A.stencil_length
    xtempi += cur_coeff * cur_stencil[idx] * x_idx

    x_temp[i] = xtempi

end

function convolve_BC_left!(x_temp::AbstractVector{T}, _x::BoundaryPaddedVector, A::DerivativeOperator) where {T<:Real}
    stencil = A.low_boundary_coefs
    coeff   = A.coefficients

    _bpc = A.boundary_point_count
    flag = false

    if stencil == []
        _bpc = 1
        flag = true
    end

    for i in 1 : _bpc
        if flag == true
            cur_stencil = A.stencil_coefs
            slen = length(A.stencil_coefs)
        else
            cur_stencil = stencil[i]
            slen = length(cur_stencil)
        end

        cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[i] : coeff isa Number ? coeff : true
        cur_stencil = use_winding(A) && cur_coeff < 0 ? reverse(cur_stencil) : cur_stencil

        # need to account for x.l in first interior
        xtempi = cur_coeff*cur_stencil[1]*_x.l
        @inbounds for idx in 2:slen
            xtempi += cur_coeff * cur_stencil[idx] * _x.u[idx-1]
        end
        x_temp[i] = xtempi
    end

    # DO WE REALLY NEED IT?
    # # need to account for x.l in first interior
    # mid = div(A.stencil_length,2) + 1
    # x = _x.u
    # i = 1 + A.boundary_point_count
    # xtempi = zero(T)
    # cur_stencil = eltype(A.stencil_coefs) <: AbstractVector ? A.stencil_coefs[i-A.boundary_point_count] : A.stencil_coefs
    # cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[i-A.boundary_point_count] : coeff isa Number ? coeff : true
    # cur_stencil = use_winding(A) && cur_coeff < 0 ? reverse(cur_stencil) : cur_stencil
    # xtempi = cur_coeff*cur_stencil[1]*_x.l
    # @inbounds for idx in 2:A.stencil_length
    #     xtempi += cur_coeff * cur_stencil[idx] * x[(i-1) - (mid-idx) + 1]
    # end
    # x_temp[i] = xtempi

end

function convolve_BC_right!(x_temp::AbstractVector{T}, _x::BoundaryPaddedVector, A::DerivativeOperator) where {T<:Real}
    stencil = A.high_boundary_coefs
    coeff   = A.coefficients
    N = length(_x.u)
    bpc = A.boundary_point_count
    # need to account for _x.r in last interior convolution
    mid = div(A.stencil_length,2) + 1
    x = _x.u
    i = length(x_temp)-A.boundary_point_count
    L = A.boundary_stencil_length


    _bpc = A.boundary_point_count
    flag = false

    if stencil == []
        _bpc = 1
        L = A.stencil_length
        flag = true
    end

    bc_start = N - _bpc + 1

    for i in 1 : _bpc
        if flag == true
            cur_stencil = A.stencil_coefs
            slen = length(A.stencil_coefs)
        else
            cur_stencil = stencil[i]
            slen = length(cur_stencil)
        end

        cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[bc_start + i] : coeff isa Number ? coeff : true
        xtempi = cur_coeff*cur_stencil[end]*_x.r
        cur_stencil = use_winding(A) && cur_coeff < 0 ? reverse(cur_stencil) : cur_stencil

        @inbounds for idx in slen-1:-1:1
            # @show idx, N-L+idx, cs[idx], x[N-L+idx]
            xtempi += cur_coeff * cur_stencil[end-idx] * _x.u[end-idx+1]
        end
        # @inbounds for idx in 1:A.boundary_stencil_length-1
        #     xtempi += cur_coeff * cur_stencil[idx] * _x.u[end-L+idx+1]
        # end

        x_temp[end-_bpc+i] = xtempi
    end
end