module EGGO

import IMAS
import Interpolations
using Flux
using Random
using Flux.Losses
using Statistics
using RegularizedLeastSquares
using PolygonOps
import VacuumFields
using LinearAlgebra
using StatsBase

include("structures.jl")
include("io.jl")
include("free_eggo.jl")

function fit_ppffp(pp::Vector{T}, ffp::Vector{T}, basis_functions_1d::BasisFunctions1D{Float64}) where {T<:Real}
    return fit_ppffp(pp, ffp, basis_functions_1d, length(basis_functions_1d.pp[:, 1]), length(basis_functions_1d.ffp[:, 1]))
end

function fit_ppffp(pp::Vector{T}, ffp::Vector{T}, basis_functions_1d::BasisFunctions1D{Float64}, pp_index::Integer, ffp_index::Integer) where {T<:Real}
    S = ADMM(transpose(basis_functions_1d.pp[1:pp_index, :]); reg=L2Regularization(1.0))
    xp = solve!(S, pp)
    Sf = ADMM(transpose(basis_functions_1d.ffp[1:ffp_index, :]); reg=L2Regularization(1.0))
    xf = solve!(Sf, ffp)
    return xp, xf
end

function get_ΨaxisΨbndffppp(psirz::Matrix{T},
    green::GreenFunctionTables{Float64},
    basis_functions::BasisFunctions{Float64},
    basis_functions_1d::BasisFunctions1D{Float64},
    bf1d_itp::BasisFunctions1Dinterp,
    wall::Wall,
    pp_fit::Vector{T},
    ffp_fit::Vector{T},
    Ip_target=0.0) where {T<:Real}

    r = range(green.rgrid[1], green.rgrid[end], length(green.rgrid))
    z = range(green.zgrid[1], green.zgrid[end], length(green.zgrid))

    rguess = green.rgrid[(green.nw+1)÷2]
    zguess = green.zgrid[(green.nw+1)÷2]

    PSI_itp = Interpolations.cubic_spline_interpolation((r, z), psirz; extrapolation_bc=Interpolations.Line())
    Raxis, Zaxis = IMAS.find_magnetic_axis(r, z, PSI_itp, 1; rguess=rguess, zguess=zguess)
    Ψaxis = PSI_itp(Raxis, Zaxis)
    axis2bnd = :increasing
    Ψbnd =
        IMAS.find_psi_boundary(
            r,
            z,
            psirz,
            Ψaxis,
            axis2bnd,
            Raxis,
            Zaxis,
            wall[:rlim],
            wall[:zlim];
            PSI_interpolant=PSI_itp,
            raise_error_on_not_open=false,
            raise_error_on_not_closed=false
        ).last_closed

    dpsi = (Ψbnd - Ψaxis) / (green.nw - 1)
    psi1d = range(Ψaxis, Ψbnd, green.nw)
    dpsi = psi1d[2] .- psi1d[1]

    lcfs = IMAS.trace_simple_surfaces(psi1d[end-1:end], green.rgrid, green.zgrid, psirz, PSI_itp, Raxis, Zaxis, wall.rlim, wall.zlim)[end]
    Rb, Zb = lcfs.r, lcfs.z
    if Ip_target > 0.0
        is_inside = get_isinside(lcfs.r, lcfs.z, green)
        psin_rz = (psirz .- Ψaxis) ./ (Ψbnd - Ψaxis)
        Jt_pp, Jt_ffp = get_Jt_fb(pp_fit, ffp_fit, psin_rz, basis_functions_1d, bf1d_itp, green, is_inside)

        dR = green.rgrid[2] - green.rgrid[1]
        dZ = green.zgrid[2] - green.zgrid[1]
        Ic = sum(Jt_ffp) * dZ * dR + sum(Jt_pp) * dZ * dR
        If_c = sum(Jt_ffp) * dZ * dR
        ffp_scale = 1 + (Ip_target - Ic) / If_c
    else
        ffp_scale = 1.0
    end
    # pp' and ff' that were actually used in EGGO
    pp = zero(basis_functions_1d.pp[1, :])
    for k in eachindex(pp_fit)
        pp .+= pp_fit[k] .* basis_functions_1d.pp[k, :]
    end
    ffp = zero(basis_functions_1d.ffp[1, :])
    for k in eachindex(ffp_fit)
        ffp .+= ffp_fit[k] .* basis_functions_1d.ffp[k, :]
    end

    return Ψaxis, Raxis, Zaxis, Ψbnd, ffp, pp
end

function fill_eqt(eqt::IMAS.equilibrium__time_slice, psirz, green, wall, pp, ffp, Btcenter, Rcenter, pend, Ψbnd, Ψaxis, Raxis, Zaxis)
    eqt.global_quantities.vacuum_toroidal_field.r0 = Rcenter
    eqt.global_quantities.vacuum_toroidal_field.b0 = Btcenter
    eqt.global_quantities.magnetic_axis.r = Raxis
    eqt.global_quantities.magnetic_axis.z = Zaxis
    eqt.global_quantities.psi_boundary = Ψbnd * 2π
    eqt.global_quantities.psi_axis = Ψaxis * 2π

    eqt1d = eqt.profiles_1d
    eqt1d.psi = range(Ψaxis, Ψbnd, green.nw) * 2π
    eqt1d.dpressure_dpsi = pp / 2π
    eqt1d.f_df_dpsi = ffp / 2π

    fend = Btcenter * Rcenter
    f2 = 2 * IMAS.cumtrapz(eqt1d.psi, eqt1d.f_df_dpsi)
    f2 .= f2 .+ -f2[end] .+ fend^2
    eqt1d.f = sign(fend) .* sqrt.(abs.(f2))

    eqt1d.pressure = IMAS.cumtrapz(eqt1d.psi, eqt1d.dpressure_dpsi)
    eqt1d.pressure .+= pend .- eqt1d.pressure[end]

    eq2d = resize!(eqt.profiles_2d, 1)[1]
    eq2d.grid_type.index = 1
    eq2d.grid.dim1 = green.rgrid
    eq2d.grid.dim2 = green.zgrid
    eq2d.psi = psirz * 2π

    return eqt
end

function minmax_normalize(x)
    min_x = minimum(x; dims=2)
    max_x = maximum(x; dims=2)
    x_norm = (x .- min_x) ./ (max_x .- min_x .+ eps())  # Add eps() to avoid division by zero
    return x_norm, min_x, max_x
end

function minmax_normalize(x, min_x, max_x)
    x_norm = (x .- min_x) ./ (max_x .- min_x .+ eps())  # Add eps() to avoid division by zero
    return x_norm
end

function minmax_unnormalize(x_norm, min_x, max_x)
    return x_norm .* (max_x .- min_x) .+ min_x
end

function predict_model_from_boundary(
    Rb::Vector{T},
    Zb::Vector{T},
    pp::Vector{T},
    ffp::Vector{T},
    NNmodel::Dict{Symbol,<:Any},
    green::GreenFunctionTables{Float64},
    basis_functions::BasisFunctions{Float64},
    basis_functions_1d::BasisFunctions1D{Float64},
    coils::Union{Vector{<:VacuumFields.AbstractCoil},Nothing},
    Ip_target::Float64=0.0,
    use_vacuumfield_green::Bool=false
) where {T<:Real}

    pp_fit, ffp_fit = fit_ppffp(pp, ffp, basis_functions_1d)
    return predict_model_from_boundary(Rb, Zb, pp_fit, ffp_fit, NNmodel, green, basis_functions, coils, Ip_target, use_vacuumfield_green)
end

function predict_model_from_boundary(
    Rb::Vector{T},
    Zb::Vector{T},
    pp_fit::Vector{T},
    ffp_fit::Vector{T},
    NNmodel::Dict{Symbol,<:Any},
    green::GreenFunctionTables{Float64},
    basis_functions::BasisFunctions{Float64},
    coils::Union{Vector{<:VacuumFields.AbstractCoil},Nothing},
    Ip_target::Float64=0.0,
    use_vacuumfield_green::Bool=false
) where {T<:Real}

    bound_mxh = IMAS.MXH(Rb, Zb, 4)

    xunnorm = vcat(
        bound_mxh.R0,
        bound_mxh.Z0,
        bound_mxh.ϵ,
        bound_mxh.κ,
        bound_mxh.tilt,
        bound_mxh.δ,
        bound_mxh.ζ,
        bound_mxh.𝚶,
        bound_mxh.twist,
        bound_mxh.c,
        bound_mxh.s,
        pp_fit,
        ffp_fit)

    model = NNmodel[:model]

    x_min = NNmodel[:x_min]
    x_max = NNmodel[:x_max]
    y_min = NNmodel[:y_min]
    y_max = NNmodel[:y_max]

    x = minmax_normalize(xunnorm, x_min, x_max)
    y = model(x)

    x = minmax_unnormalize(x, x_min, x_max) # Convert back to original scale
    y = minmax_unnormalize(y, y_min, y_max)  # Convert back to original scale

    Jt, psirz, Ip = predict_model(y, green, basis_functions, Ip_target)
    psirz .+= calculate_psiext(Rb, Zb, psirz, green, coils, use_vacuumfield_green)

    return Jt, psirz, Ip
end

function calculate_psiext(Rb_target::Vector{T},
    Zb_target::Vector{T},
    psipla::Matrix{T},
    green::GreenFunctionTables{Float64},
    coils::Union{Vector{<:VacuumFields.AbstractCoil},Nothing},
    use_vacuumfield_green::Bool=true) where {T<:Real}

    r = range(green.rgrid[1], green.rgrid[end], length(green.rgrid))
    z = range(green.zgrid[1], green.zgrid[end], length(green.zgrid))
    Ψpl_itp = Interpolations.cubic_spline_interpolation((r, z), psipla; extrapolation_bc=Interpolations.Line())
    psiext = zeros(length(r), length(z))
    if use_vacuumfield_green
        iso_cps = VacuumFields.IsoControlPoints(Rb_target, Zb_target)
        fixed = Int[] # Integer vector denoting of fixed coils
        dΨpl_dR = (x, y) -> Interpolations.gradient(Ψpl_itp, x, y)[1]
        dΨpl_dZ = (x, y) -> Interpolations.gradient(Ψpl_itp, x, y)[2]
        fixed_coils = coils[fixed]
        active_coils = isempty(fixed_coils) ? coils : coils[setdiff(eachindex(coils), fixed)]
        flux_cps = VacuumFields.FluxControlPoint{Real}[]
        saddle_cps = VacuumFields.SaddleControlPoint{Real}[]

        fcurrt_vf = VacuumFields.find_coil_currents!(active_coils, Ψpl_itp, dΨpl_dR, dΨpl_dZ; iso_cps, flux_cps, saddle_cps, fixed_coils, λ_regularize=-1.0)[1]

        VacuumFields.flux_on_grid!(psiext, green.ggridfc_vf, r, z, coils)
    else
        ecurrt_vf, fcurrt_vf = predict_coil_currents(Rb_target, Zb_target, green, Ψpl_itp)
        psiext = calculate_psiext(fcurrt_vf, ecurrt_vf, green)
    end

    return psiext
end

function predict_model_from_coils(
    pp::Vector{T},
    ffp::Vector{T},
    ecurrt::Vector{Float64},
    fcurrt::Vector{Float64},
    NNmodel::Dict{Symbol,<:Any},
    green::GreenFunctionTables{Float64},
    basis_functions::BasisFunctions{Float64},
    basis_functions_1d::BasisFunctions1D{Float64},
    Ip_target::Float64=0.0,
    use_vacuumfield_green::Bool=false
) where {T<:Real}

    pp_fit, ffp_fit = fit_ppffp(pp, ffp, basis_functions_1d)

    return predict_model_from_coils(pp_fit, ffp_fit, ecurrt, fcurrt, NNmodel, green, basis_functions, Ip_target, use_vacuumfield_green)
end

function predict_model_from_coils(
    pp_fit::Vector{T},
    ffp_fit::Vector{T},
    ecurrt::Vector{Float64},
    fcurrt::Vector{Float64},
    NNmodel::Dict{Symbol,<:Any},
    green::GreenFunctionTables{Float64}, basis_functions::BasisFunctions{Float64},
    Ip_target::Float64=0.0,
    use_vacuumfield_green::Bool=false
) where {T<:Real}

    xunnorm = vcat(
        pp_fit,
        ffp_fit,
        ecurrt,
        fcurrt
    )

    model = NNmodel[:model]
    x_min = NNmodel[:x_min]
    x_max = NNmodel[:x_max]
    y_min = NNmodel[:y_min]
    y_max = NNmodel[:y_max]

    x = minmax_normalize(xunnorm, x_min, x_max)
    y = model(x)
    x = minmax_unnormalize(x, x_min, x_max)
    y = minmax_unnormalize(y, y_min, y_max)  # Convert back to original scale

    nfsum = green.nfsum
    nesum = green.nesum

    fcurrt = @views x[end-nfsum+1:end]
    ecurrt = @views x[end-nfsum-nesum+1:end-nfsum]
    Jt, psirz, Ip = predict_model(y, green, basis_functions, Ip_target)
    psirz .+= calculate_psiext(fcurrt, ecurrt, green)
    return Jt, psirz, Ip
end

function calculate_psiext(fcurrt::AbstractVector{T}, ecurrt::AbstractVector{T}, green::GreenFunctionTables{Float64}) where {T<:Real}
    # Direct matrix-vector multiplications (16641,18) * (18,) + (16641,6) * (6,) = (16641,)
    psiext_flat = green.ggridfc * fcurrt + green.gridec * ecurrt

    # Reshape to target dimensions and negate
    return -transpose(reshape(psiext_flat, green.nh, green.nw))
end

function predict_model(y::Matrix{T},
    green::GreenFunctionTables{Float64},
    basis_functions::BasisFunctions{Float64},
    Ip_target::T
) where {T<:Real}

    nw = green.nw
    nh = green.nh
    npca = length(basis_functions.Ip)

    Ip = dot(@views(y[1:npca]), basis_functions.Ip)
    if Ip_target !== 0.0
        y .*= Ip_target / Ip
        Ip = Ip_target
    end

    psipla = zeros(T, (nw, nh))
    for ipca in 1:npca
        @views psipla .+= y[ipca] .* transpose(basis_functions.psi[:, :, ipca])
    end

    Jt = zeros(T, (nw, nh))
    for ipca in 1:npca
        @views Jt .+= y[ipca] .* transpose(basis_functions.Jt[:, :, ipca])
    end

    return Jt, Matrix(transpose(psipla)), Ip
end

function get_isinside(Rb, Zb, green)
    is_inside = zeros(green.nw, green.nh)
    _bnd = [[Rb[k], Zb[k]] for k in eachindex(Rb)]
    push!(_bnd, [Rb[1], Zb[1]])
    for (i, r) in enumerate(green.rgrid)
        for (j, z) in enumerate(green.zgrid)
            is_inside[j, i] = inpolygon((r, z), _bnd) == 1
        end
    end
    return is_inside
end

function get_Jt_fb(pp_fit::Vector{T},
    ffp_fit::Vector{T},
    psin_rz::Matrix{T},
    basis_functions_1d::BasisFunctions1D{Float64},
    bf1d_itp::BasisFunctions1Dinterp,
    green::GreenFunctionTables{Float64},
    is_inside::Matrix{T}) where {T<:Real}

    # Pre-extract values to avoid repeated dictionary lookups
    rgrid = green.rgrid
    zgrid = green.zgrid
    nh = green.nh
    nw = green.nw

    npp = length(pp_fit)
    nffp = length(ffp_fit)

    # Pre-allocate output arrays
    Jt_pp = zeros(T, nh, nw)
    Jt_ffp = zeros(T, nh, nw)

    Threads.@threads for j in 1:nw
        z = zgrid[j]
        @inbounds for i in 1:nh
            r = rgrid[i]
            if is_inside[i, j] != 0  # Skip if outside
                psin_val = psin_rz[j, i]

                # Vectorized computation for pp
                for ib in 1:npp
                    Jt_pp[i, j] -= pp_fit[ib] * bf1d_itp.pp[ib](psin_val) * r * is_inside[i, j]
                end

                # Vectorized computation for ffp
                r_inv = inv(r)
                for ib in 1:nffp
                    Jt_ffp[i, j] -= ffp_fit[ib] * bf1d_itp.ffp[ib](psin_val) * r_inv * is_inside[i, j] / IMAS.mks.μ_0
                end
            end
        end
    end

    return Jt_pp, Jt_ffp
end


function fit_pp_ffp(pp_fit::Vector{T},
    ffp_fit::Vector{T},
    psin_rz::Matrix{T},
    basis_functions_1d::BasisFunctions1D{Float64},
    bf1d_itp::BasisFunctions1Dinterp,
    green::GreenFunctionTables{Float64},
    is_inside::Matrix{T}) where {T<:Real}

    # Pre-extract values to avoid repeated dictionary lookups
    rgrid = green.rgrid
    zgrid = green.zgrid
    nh = green.nh
    nw = green.nw

    npp = length(pp_fit)
    nffp = length(ffp_fit)

    # Pre-allocate output arrays
    Jt_pp = zeros(T, nh, nw)
    Jt_ffp = zeros(T, nh, nw)

    Threads.@threads for j in 1:nw
        z = zgrid[j]
        @inbounds for i in 1:nh
            r = rgrid[i]
            if is_inside[i, j] != 0  # Skip if outside
                psin_val = psin_rz[j, i]

                # Vectorized computation for pp
                for ib in 1:npp
                    Jt_pp[i, j] -= pp_fit[ib] * bf1d_itp.pp[ib](psin_val) * r * is_inside[i, j]
                end

                # Vectorized computation for ffp
                r_inv = inv(r)
                for ib in 1:nffp
                    Jt_ffp[i, j] -= ffp_fit[ib] * bf1d_itp.ffp[ib](psin_val) * r_inv * is_inside[i, j] / IMAS.mks.μ_0
                end
            end
        end
    end

    return Jt_pp, Jt_ffp
end



function calc_pffprime2(psinrz::Matrix{Float64}, Jt::Matrix{Float64}, rgrid::Vector{Float64}, is_inside::Matrix{Float64},bf1d_itp:: BasisFunctions1Dinterp)

    npp = length( bf1d_itp.pp)
    nffp = length( bf1d_itp.ffp)
    n = length(psinrz)
    A = zeros(npp + nffp, n)
    nw = nh = 129
    nffp = 6
    npp = 6
    # Construct X matrix
    Threads.@threads for j in 1:nw
        @inbounds for i in 1:nh
            r = rgrid[j]
            ij = j*nw +i
            if is_inside[i, j] != 0  # Skip if outside
                psin_val = psinrz[j, i]

                # Vectorized computation for pp
                for ib in 1:npp
                    A[ib, ij] = bf1d_itp.pp[ib](psin_val) /r#/ r^2  #/ r#* r^2 
                end

                for ib in 1:nffp
                    A[ib+npp, ij] = bf1d_itp.ffp[ib](psin_val) 
                end
            end
        end
    end
    #return A
    b = vec(Jt)
    # Solve least squares: x = argmin ||X'x - b||
    x = reg_solve(A', b, 1.0)

    return x[1:npp],x[npp+1:end]
end


"""
function predict_coil_currents(Rb::Vector{T},Zb::Vector{T},green::GreenFunctionTables{Float64},
psipla::Interpolations.AbstractInterpolation) where {T<:Real}
    nesum = green.nesum
    nfsum = green.nfsum
    n = length(Rb)
    npoints = n * (n - 1) ÷ 2

    A = zeros(nfsum+nesum, npoints)
    j = 1
    @time for (j1,(R1,Z1)) in enumerate(zip(Rb,Zb))
        for (R2,Z2) in zip(Rb[j1+1:end],Zb[j1+1:end])
            for i in 1:nesum
                A[i, j] = green.gridec_itp[i](R1,Z1) -(green.gridec_itp[i](R2,Z2))
            end
            j+=1
        end
    end
    j = 1
    @time for (j1,(R1,Z1)) in enumerate(zip(Rb,Zb))
        for (R2,Z2) in zip(Rb[j1+1:end],Zb[j1+1:end])
            for i in 1:nfsum
                A[nesum+i, j] = green.ggridfc_itp[i](R1,Z1) - (green.ggridfc_itp[i](R2,Z2))
            end
            j+=1
        end
    end

    b= zeros(npoints)
    j = 1
    @time for (j1,(R1,Z1)) in enumerate(zip(Rb,Zb))
        for (R2,Z2) in zip(Rb[j1+1:end],Zb[j1+1:end])
            b[j] = (psipla(R1,Z1)- psipla(R2,Z2))
            j+=1
        end 
    end

    @time x = reg_solve(A', b, 1e-16)

    return x[1:nesum],x[nesum+1:end]
end
"""

function predict_coil_currents(Rb::Vector{T}, Zb::Vector{T}, green::GreenFunctionTables{Float64}, psipla::Interpolations.AbstractInterpolation) where {T<:Real}
    nesum = green.nesum
    nfsum = green.nfsum
    n = length(Rb)
    npoints = n * (n - 1) ÷ 2

    # Pre-allocate all arrays
    A = zeros(T, nfsum + nesum, npoints)
    b = zeros(T, npoints)

    # Pre-extract interpolation functions to avoid dictionary lookups
    gridec_itp = green.gridec_itp
    ggridfc_itp = green.ggridfc_itp

    j = 1
    @inbounds for j1 in 1:n-1
        R1, Z1 = Rb[j1], Zb[j1]
        gridec_vals_1 = [gridec_itp[i](R1, Z1) for i in 1:nesum]
        ggridfc_vals_1 = [ggridfc_itp[i](R1, Z1) for i in 1:nfsum]
        psipla_1 = psipla(R1, Z1)

        for j2 in j1+1:n
            R2, Z2 = Rb[j2], Zb[j2]
            psipla_2 = psipla(R2, Z2)

            for i in 1:nesum
                A[i, j] = gridec_vals_1[i] - gridec_itp[i](R2, Z2)
            end

            for i in 1:nfsum
                A[nesum+i, j] = ggridfc_vals_1[i] - ggridfc_itp[i](R2, Z2)
            end

            b[j] = psipla_1 - psipla_2

            j += 1
        end
    end

    # Solve the system
    x = reg_solve(A', b, 1e-16)

    return x[1:nesum], x[nesum+1:end]
end

function reg_solve(A::AbstractArray{T}, b::AbstractArray{T}, λ::T) where {T<:Real}
    return (A' * A + λ * I) \ A' * b
end

export get_greens_function_tables, get_basis_functions, get_model, get_basis_functions_1d, predict_model

const document = Dict()
document[Symbol(@__MODULE__)] = [name for name in Base.names(@__MODULE__; all=false, imported=false) if name != Symbol(@__MODULE__)]

end
