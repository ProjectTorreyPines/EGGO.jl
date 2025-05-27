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

include("io.jl")

function fit_ppffp(pp, ffp, basis_functions_1d)
    return fit_ppffp(pp, ffp, basis_functions_1d,length(basis_functions_1d[:pp][:,1]),length(basis_functions_1d[:ffp][:,1]))
end


function fit_ppffp(pp, ffp, basis_functions_1d,pp_index,ffp_index)
    S = ADMM(transpose(basis_functions_1d[:pp][1:pp_index,:]); reg=L2Regularization(1.0))
    xp = solve!(S, pp)
    Sf = ADMM(transpose(basis_functions_1d[:ffp][1:ffp_index,:]); reg=L2Regularization(1.0))
    xf = solve!(Sf, ffp)
    return xp, xf
end

function fill_eqt(eqt::IMAS.equilibrium__time_slice, psirz, green, wall, pp, ffp, Btcenter, Rcenter, pend)
    r = range(green[:rgrid][1], green[:rgrid][end], length(green[:rgrid]))
    z = range(green[:zgrid][1], green[:zgrid][end], length(green[:zgrid]))


    rguess = green[:RR][(green[:nw] + 1) ÷ 2,(green[:nh] + 1) ÷ 2]
    zguess = green[:ZZ][(green[:nw] + 1) ÷ 2,(green[:nh] + 1) ÷ 2]
    
    PSI_itp = Interpolations.cubic_spline_interpolation((r, z), psirz; extrapolation_bc=Interpolations.Line())
    Raxis, Zaxis = IMAS.find_magnetic_axis(r, z, PSI_itp, 1; rguess=rguess, zguess=zguess)
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
    f2 .= f2 .+ - f2[end] .+ fend^2
    eqt1d.f = sign(fend) .* sqrt.(abs.(f2))

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

function predict_mode_from_coils(
    pp::Vector{T},
    ffp::Vector{T},
    ecurrt::Vector{Float64},
    fcurrt::Vector{Float64},
    NNmodel::Dict{Symbol,Any},
    green::Dict{Symbol,Any},
    basis_functions::Dict{Symbol,Any},
    basis_functions_1d::Dict{Symbol,Any},
    Ip_target::Float64=0.0,
    model_name::Symbol=:model_efit01
) where {T<:Real}
    
    pp_fit, ffp_fit = fit_ppffp(pp, ffp, basis_functions_1d)

    return predict_mode_from_coils(pp_fit, ffp_fit, ecurrt,fcurrt, NNmodel, green, basis_functions, Ip_target)
end

function predict_model_from_boundary(
    Rb::Vector{T},
    Zb::Vector{T},
    pp::Vector{T},
    ffp::Vector{T},
    NNmodel::Dict{Symbol,Any},
    green::Dict{Symbol,Any},
    basis_functions::Dict{Symbol,Any},
    basis_functions_1d::Dict{Symbol,Any},
    Ip_target::Float64=0.0,
    model_name::Symbol=:model_efit01
) where {T<:Real}
    
    pp_fit, ffp_fit = fit_ppffp(pp, ffp, basis_functions_1d)
    @show(pp_fit,ffp_fit)
    return predict_model_from_boundary(Rb, Zb, pp_fit, ffp_fit, NNmodel, green, basis_functions, Ip_target)
end

function predict_model_from_boundary(
    Rb::Vector{T},
    Zb::Vector{T},
    pp_fit::Vector{T},
    ffp_fit::Vector{T},
    NNmodel::Dict{Symbol,Any},
    green::Dict{Symbol,Any},
    basis_functions::Dict{Symbol,Any},
    coils::IMAS.pf_active__coil{Real},
    Ip_target::Float64=0.0,
    model_name::Symbol=:model_efit01
) where {T<:Real}
    bound_mxh =  IMAS.MXH(Rb, Zb, 4)
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
        ffp_fit    )

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
    #r = range(green[:rgrid][1], green[:rgrid][end], length(green[:rgrid]))
    #z = range(green[:zgrid][1], green[:zgrid][end], length(green[:zgrid]))
    #Ψpl_itp = Interpolations.cubic_spline_interpolation((r, z), psirz; extrapolation_bc=Interpolations.Line())     
    #ecurrt,fcurrt = predict_coil_currents(Rb,Zb,green,Ψpl_itp)
    psirz .+= calculate_psiext(Rb,Zb,psirz,green,coils)#calculate_psiext(fcurrt,ecurrt,green)
    return Jt,psirz,Ip
end

function calculate_psiext(Rb_target,Zb_target,psirz,green,coils)

    r = range(green[:rgrid][1], green[:rgrid][end], length(green[:rgrid]))
    z = range(green[:zgrid][1], green[:zgrid][end], length(green[:zgrid]))
    Ψpl_itp = Interpolations.cubic_spline_interpolation((r, z), psirz; extrapolation_bc=Interpolations.Line())   

    iso_cps = VacuumFields.IsoControlPoints(Rb_target, Zb_target)
    
    fixed = Int[] # Integer vector denoting of fixed coils
    Ψpl_itp = Interpolations.cubic_spline_interpolation((r, z), psipla; extrapolation_bc=Interpolations.Line()) 
    dΨpl_dR = (x, y) -> Interpolations.gradient(Ψpl_itp, x, y)[1]
    dΨpl_dZ = (x, y) -> Interpolations.gradient(Ψpl_itp, x, y)[2]
    coils = VacuumFields.IMAS_pf_active__coils(dd; green_model=:quad, zero_currents=false)
    fixed_coils = coils[fixed]
    active_coils = isempty(fixed_coils) ? coils : coils[setdiff(eachindex(coils), fixed)]
    flux_cps = VacuumFields.FluxControlPoint{Real}[]
    saddle_cps  = VacuumFields.SaddleControlPoint{Real}[]
    
    @time fcurrt_vf = VacuumFields.find_coil_currents!(active_coils, Ψpl_itp, dΨpl_dR, dΨpl_dZ; iso_cps, flux_cps, saddle_cps, fixed_coils, λ_regularize=1.0)[1]

    psiext = zeros(length(rgrid),length(zgrid))
    for (iR,R) in enumerate(dd.equilibrium.time_slice[1].profiles_2d[1].grid.dim1)
        for (iZ,Z) in enumerate(dd.equilibrium.time_slice[1].profiles_2d[1].grid.dim2)
            psiext[iR,iZ] = sum(VacuumFields.ψ.(dd.pf_active.coil,R,Z))
        end
    end
    return psiext
end

function predict_model_from_coils(
    pp_fit::Vector{T},
    ffp_fit::Vector{T},
    ecurrt::Vector{Float64},
    fcurrt::Vector{Float64},
    NNmodel::Dict{Symbol,Any},
    green::Dict{Symbol,Any},
    basis_functions::Dict{Symbol,Any},
    Ip_target::Float64=0.0,
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

    nfsum = green[:nfsum]
    nesum = green[:nesum]
    npca = length(basis_functions[:Ip])

    fcurrt = @views x[end-nfsum+1:end]
    ecurrt = @views x[end-nfsum-nesum+1:end-nfsum]
    Jt, psirz, Ip = predict_model(y, green, basis_functions, Ip_target)
    psirz .+= calculate_psiext(fcurrt,ecurrt,green)

    return  Jt, psirz, Ip
end

function calculate_psiext(fcurrt::Vector{T},ecurrt::Vector{T},green)where {T<:Real}
    psiext_1d = sum(green[:ggridfc] .* reshape(fcurrt, 1, green[:nfsum]); dims=2)
    psiext_1d .+= @views sum(green[:gridec] .* reshape(ecurrt, 1, green[:nesum]); dims=2)[:, :, 1]
    return -transpose(reshape(psiext_1d, green[:nh], green[:nw]))
end

function predict_model(y::Matrix{T},
    green::Dict{Symbol,Any},
    basis_functions::Dict{Symbol,Any},
    Ip_target::T,
) where {T<:Real}

    nfsum = green[:nfsum]
    nesum = green[:nesum]
    nw = green[:nw]
    nh = green[:nh]
    npca = length(basis_functions[:Ip])

    psipla = zeros(T, (nw, nh))

    fcurrt = @views y[npca+1:npca+nfsum]
    ecurrt = @views x[end-5:end]
    psiext_1d = green[:ggridfc] * fcurrt
    mul!(psiext_1d, green[:gridec][:, :, 1], ecurrt, 1.0, 1.0)
    psiext = reshape(psiext_1d, nh, nw)

    Ip = dot(@views(y[1:npca]), basis_functions[:Ip])
    if Ip_target !== 0.0
        y .*= Ip/Ip_target 
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

    return Jt, Matrix(transpose(psipla)), Ip
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

function predict_coil_currents(Rb,Zb,green::Dict{Symbol,Any},psipla)

    nesum = green[:nesum]
    nfsum = green[:nfsum]

    A = zeros(nfsum+nesum, length(Rb)^2)
    R1  = Rb[1]
    Z1 = Zb[1]
    for i in 1:nesum
        for (j1,(R1,Z1)) in enumerate(zip(Rb,Zb))
            for (j2, (R2,Z2)) in enumerate(zip(Rb[2:end],Zb[2:end]))
                A[i, (j1-1)*length(Rb)+j2] = green[:gridec_itp][i](R1,Z1) -(green[:gridec_itp][i](R2,Z2))
            end
        end
    end

    for i in 1:nfsum
        for (j1,(R1,Z1)) in enumerate(zip(Rb,Zb))
            for (j2, (R2,Z2)) in enumerate(zip(Rb[1:end],Zb[1:end]))
                A[nesum+i, (j1-1)*length(Rb)+j2] = green[:ggridfc_itp][i](R1,Z1) - (green[:ggridfc_itp][i](R2,Z2))
            end
        end
    end
    b= zeros(length(Rb)^2)
    for (j1,(R1,Z1)) in enumerate(zip(Rb,Zb))
        for (j2, (R2,Z2)) in enumerate(zip(Rb[1:end],Zb[1:end]))
            b[(j1-1)*length(Rb)+j2] = (psipla(R1,Z1)- psipla(R2,Z2))
        end 
    end

   x = transpose(A)\b
    #x = reg_solve(A,b,λ)
    ecurrt = x[1:nesum]
    fcurrt = x[nesum+1:end]
    return ecurrt,fcurrt
end

function reg_solve(A, b, λ)
    return (A' * A + λ ) \ A' * b
end

export get_greens_function_tables, get_basis_functions, get_model, get_basis_functions_1d, predict_model

const document = Dict()
document[Symbol(@__MODULE__)] = [name for name in Base.names(@__MODULE__; all=false, imported=false) if name != Symbol(@__MODULE__)]

end
