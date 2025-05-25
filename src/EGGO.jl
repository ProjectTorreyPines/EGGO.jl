module EGGO

import IMAS
import Interpolations
using Flux
using Random
using Flux.Losses
using Statistics
using RegularizedLeastSquares
using PolygonOps
using LinearAlgebra

include("io.jl")

function fit_ppffp(pp, ffp, basis_functions_1d)
    S = ADMM(transpose(basis_functions_1d[:pp]); reg=L1Regularization(1.0))
    xp = solve!(S, pp)

    Sf = ADMM(transpose(basis_functions_1d[:ffp]); reg=L1Regularization(1.0))
    xf = solve!(Sf, ffp)

    return xp, xf
end

function fill_eqt(eqt::IMAS.equilibrium__time_slice, psirz, green, wall, pp, ffp, Btcenter, Rcenter, pend)
    r = range(green[:rgrid][1], green[:rgrid][end], length(green[:rgrid]))
    z = range(green[:zgrid][1], green[:zgrid][end], length(green[:zgrid]))

    RR = green[:RR]
    ZZ = green[:ZZ]
    ind = argmin(psirz)

    PSI_itp = Interpolations.cubic_spline_interpolation((r, z), psirz; extrapolation_bc=Interpolations.Line())
    Raxis, Zaxis = IMAS.find_magnetic_axis(r, z, PSI_itp, 1; rguess=RR[ind], zguess=ZZ[ind])
    Ψaxis = PSI_itp(Raxis, Zaxis)
    axis2bnd = :increasing
    Ψbnd =
        IMAS.find_psi_boundary(r, z, psirz, Ψaxis, axis2bnd, Raxis, Zaxis, wall[:rlim], wall[:zlim];
            PSI_interpolant=PSI_itp, raise_error_on_not_open=false, raise_error_on_not_closed=false).last_closed

    dpsi = (Ψbnd - Ψaxis) / (green[:nw] - 1)
    psi1d = range(Ψaxis, Ψbnd, green[:nw])
    dpsi = psi1d[2] .- psi1d[1]

    eqt.global_quantities.vacuum_toroidal_field.r0 = Rcenter
    eqt.global_quantities.vacuum_toroidal_field.b0 = Btcenter
    eqt.global_quantities.magnetic_axis.z = Zaxis
    eqt.global_quantities.magnetic_axis.r = Raxis
    eqt.global_quantities.magnetic_axis.z = Zaxis
    eqt.global_quantities.psi_boundary = Ψbnd
    eqt.global_quantities.psi_axis = Ψaxis

    eqt1d = eqt.profiles_1d
    eqt1d.psi = psi1d * 2π
    eqt1d.dpressure_dpsi = pp / 2π
    eqt1d.f_df_dpsi = ffp / 2π

    fend = Btcenter * Rcenter
    f2 = 2 * IMAS.cumtrapz(eqt1d.psi, eqt1d.f_df_dpsi)
    f2 .= f2 .- f2[end] .+ fend^2
    eqt1d.f = sign(fend) .* sqrt.(f2)

    eqt1d.pressure = IMAS.cumtrapz(eqt1d.psi, eqt1d.dpressure_dpsi)
    eqt1d.pressure .+= pend .- eqt1d.pressure[end]

    eq2d = resize!(eqt.profiles_2d, 1)[1]
    eq2d.grid_type.index = 1
    eq2d.grid.dim1 = collect(r)
    eq2d.grid.dim2 = collect(z)
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

function predict_model(
    Rb::Vector{T},
    Zb::Vector{T},
    pp::Vector{T},
    ffp::Vector{T},
    ecurrt::Vector{Float64},
    NNmodel::Dict{Symbol,Any},
    green::Dict{Symbol,Any},
    basis_functions::Dict{Symbol,Any},
    basis_functions_1d::Dict{Symbol,Any},
    Ip_target::Float64=0.0
) where {T<:Real}
    bound_mxh = IMAS.MXH(Rb, Zb, 4)
    pp_fit, ffp_fit = fit_ppffp(pp, ffp, basis_functions_1d)

    return predict_model(bound_mxh, pp_fit, ffp_fit, ecurrt, NNmodel, green, basis_functions, Ip_target)
end

function predict_model(
    bound_mxh::IMAS.MXH,
    pp_fit::Vector{T},
    ffp_fit::Vector{T},
    ecurrt::Vector{Float64},
    NNmodel::Dict{Symbol,Any},
    green::Dict{Symbol,Any},
    basis_functions::Dict{Symbol,Any},
    Ip_target::Float64=0.0
) where {T<:Real}
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
        ffp_fit,
        ecurrt
    )

    model = NNmodel[:model]

    x_min = NNmodel[:x_min]
    x_max = NNmodel[:x_max]
    y_min = NNmodel[:y_min]
    y_max = NNmodel[:y_max]

    x = minmax_normalize(xunnorm, x_min, x_max)
    y = model(x)
    y = minmax_unnormalize(y, y_min, y_max)  # Convert back to original scale

    return predict_model(x, y, green, basis_functions, Ip_target)
end

function predict_model(x::Matrix{T}, y::Matrix{T}, green, basis_functions, Ip_target) where {T<:Real}
    nfsum = green[:nfsum]
    nesum = green[:nesum]
    nw = green[:nw]
    nh = green[:nh]

    npca = length(basis_functions[:Ip])

    fcurrt = @views y[npca+1:npca+nfsum]
    ecurrt = @views x[end-5:end]
    psiext_1d = green[:ggridfc] * fcurrt
    mul!(psiext_1d, green[:gridec][:, :, 1], ecurrt, 1.0, 1.0)
    psiext = reshape(psiext_1d, nh, nw)

    Ip = dot(@views(y[1:npca]), basis_functions[:Ip])
    if Ip_target !== 0.0
        y .*= Ip_target / Ip
        Ip = Ip_target
    end

    psipla = zeros(T, (nw, nh))
    for ipca in 1:npca
        @views psipla .+= y[ipca] .* transpose(basis_functions[:psi][:, :, ipca])
    end
    psirz = -1.0 * (psiext - psipla)

    Jt = zeros(T, (nw, nh))
    for ipca in 1:npca
        @views Jt .+= y[ipca] .* transpose(basis_functions[:Jt][:, :, ipca])
    end

    return Jt, Matrix(transpose(psirz)), Ip, fcurrt
end

function get_isinside(Rb, Zb, Ψ, Ψbnd, green)
    is_inside = zeros(green[:nw], green[:nh])
    _bnd = [[Rb[k], Zb[k]] for k in eachindex(Rb)]
    push!(_bnd, [Rb[1], Zb[1]])
    for (i, r) in enumerate(green[:rgrid])
        for (j, z) in enumerate(green[:zgrid])
            is_inside[j, i] = inpolygon((r, z), _bnd) == 1
        end
    end
    return is_inside
end

function get_Jt_fb(pp_fit, ffp_fit, psin_rz, basis_functions_1d, bf1d_itp, green, is_inside)
    bf2d_ppffp = Dict{Symbol,Any}()
    bf2d_ppffp = Dict{Symbol,Any}()

    npp = size(basis_functions_1d[:pp])[1]
    nffp = size(basis_functions_1d[:ffp])[1]
    bf2d_ppffp[:pp] = zeros(npp, green[:nh] * green[:nw])
    bf2d_ppffp[:ffp] = zeros(nffp, green[:nh] * green[:nw])
    Jt_fb = zeros(green[:nh], green[:nw])
    for (j, z) in enumerate(green[:zgrid])
        for (i, r) in enumerate(green[:rgrid])
            for ib in 1:npp
                Jt_fb[i, j] -= pp_fit[ib] .* bf1d_itp[:pp][ib](psin_rz[j, i]) * green[:rgrid][i] * is_inside[i, j]
            end
            for ib in 1:nffp
                Jt_fb[i, j] -= ffp_fit[ib] .* bf1d_itp[:ffp][ib](psin_rz[j, i]) / green[:rgrid][i] * is_inside[i, j] / (4 * pi * 1e-7)
            end
        end
    end
    return Jt_fb
end

export get_greens_function_tables, get_basis_functions, get_model, get_basis_functions_1d, predict_model

const document = Dict()
document[Symbol(@__MODULE__)] = [name for name in Base.names(@__MODULE__; all=false, imported=false) if name != Symbol(@__MODULE__)]

end