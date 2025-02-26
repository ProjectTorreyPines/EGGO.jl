module EGGO

import IMAS
import Interpolations
import BSON
using Flux
using Random
using Flux.Losses
using Statistics
using RegularizedLeastSquares
using PolygonOps



mutable struct Canvas{T<:Real}
    Rs::StepRangeLen{T, Base.TwicePrecision{T}, Base.TwicePrecision{T}, Int}
    Zs::StepRangeLen{T, Base.TwicePrecision{T}, Base.TwicePrecision{T}, Int}
    Ψ::Matrix{T}
    Ip::T
    ecoil_currents::Vector{T}
    fcoil_currents::Vector{T}
    Rw::Vector{T}
    Zw::Vector{T}
    pp::Vector{T}
    ffp::Vector{T}
    pres::Vector{T}
    fpol::Vector{T}
    Raxis::T
    Zaxis::T
    Ψaxis::T
    Ψbnd::T
    _Rb_target::Vector{T}
    _Zb_target::Vector{T}
    _surfaces::Vector{IMAS.SimpleSurface{T}}
    _Vp::Vector{T}
    _gm1::Vector{T}
    _gm9::Vector{T}
    _ffp_target::Vector{T}
    _pp_target::Vector{T}
end


function fit_ppffp(pp,ffp,basis_functions_1d)
    npp = size(basis_functions_1d[:pp])[1]
    nffp = size(basis_functions_1d[:ffp])[1]

    S = ADMM(transpose(basis_functions_1d[:pp]), reg = L1Regularization(1.))
    xp = solve!(S, pp)

    Sf = ADMM(transpose(basis_functions_1d[:ffp]), reg = L1Regularization(1.))
    xf = solve!(Sf, ffp)
    return xp,xf
end



function get_surfaces(dd, psirz, Ip,fcurrt,green,wall,Rb_target,Zb_target,pp_target,ffp_target,ecurrt_target, Btcenter,Rcenter,pend)
    r = range(green[:rgrid][1],green[:rgrid][end],length(green[:rgrid]))
    z = range(green[:zgrid][1],green[:zgrid][end],length(green[:zgrid]))

    rwall = Float64.(wall[:rlim])
    zwall = Float64.(wall[:zlim])
    PSI_itp = Interpolations.cubic_spline_interpolation((r, z), psirz; extrapolation_bc=Interpolations.Line())

    RR  = hcat(green[:RR]...) |> x -> reshape(x, :,green[:nw]);
    ZZ  = hcat(green[:ZZ]...) |> x -> reshape(x, :,green[:nw]);
    ind = argmin(psirz) 
    
    Raxis, Zaxis = IMAS.find_magnetic_axis(r, z, PSI_itp, 1; rguess=RR[ind], zguess=ZZ[ind])
    Ψaxis = PSI_itp(Raxis,Zaxis)
    axis2bnd=:increasing
    empty_r = zeros(1)[1:0]
    empty_z = zeros(1)[1:0]
    Ψbnd = IMAS.find_psi_boundary(r, z, psirz, Ψaxis, axis2bnd, Raxis, Zaxis, rwall, zwall, empty_r,  empty_z;
                                      PSI_interpolant=PSI_itp, raise_error_on_not_open=false, raise_error_on_not_closed=false).last_closed
    
    dpsi = (Ψbnd-Ψaxis)/(green[:nw]-1)
    psi1d = collect(Ψaxis:dpsi:Ψbnd)
    surfaces = IMAS.trace_simple_surfaces(psi1d,
        r,
        z,
        psirz,
        PSI_itp,
        Raxis,
        Zaxis,
        rwall,
        zwall
    );
    
    nsurf = length(surfaces)
    ffp = zeros(nsurf)
    pp = zeros(nsurf)
    Vp = zeros(nsurf)
    gm1 = zeros(nsurf)
    gm9 = zeros(nsurf)
    for (k, surface) in enumerate(surfaces)
        sign_dpsi = -1
        Vp[k] = sign_dpsi * surface.int_fluxexpansion_dl
    
        # gm1 = <1/R^2>
        f1 = (j, xx) ->  surface.fluxexpansion[j] / surface.r[j] ^ 2
        gm1[k] = IMAS.flux_surface_avg(f1, surface)
    
        # gm9 = <1/R>
        f9 = (j, xx) ->  surface.fluxexpansion[j] / surface.r[j]
        gm9[k] = IMAS.flux_surface_avg(f9, surface)
    end

    eq = dd.equilibrium
    eqt = resize!(dd.equilibrium.time_slice)
    eqt1d = eqt.profiles_1d
    eq2d = resize!(eqt.profiles_2d, 1)[1]
    
    eqt.global_quantities.magnetic_axis.r = Raxis
    eqt.global_quantities.magnetic_axis.z = Zaxis
    eqt.global_quantities.psi_boundary = Ψbnd
    eqt.global_quantities.psi_axis = Ψaxis
    eqt1d.psi =  psi1d * (2π)
    
    Npsi = length(eqt1d.psi)
    eqt1d.dpressure_dpsi = pp_target / (2π)
    eqt1d.f_df_dpsi = ffp_target / (2π)
    
    eq.vacuum_toroidal_field.b0 = Btcenter * ones(1)
    eq.vacuum_toroidal_field.r0 = Rcenter 
    
    fend = eqt.global_quantities.vacuum_toroidal_field.b0 * eqt.global_quantities.vacuum_toroidal_field.r0
    f2 = 2 * IMAS.cumtrapz(eqt1d.psi, eqt1d.f_df_dpsi)
    f2 .= f2 .- f2[end] .+ fend^2
    eqt1d.f = sign(fend) .* sqrt.(f2)
    
    eqt1d.pressure = IMAS.cumtrapz(eqt1d.psi, eqt1d.dpressure_dpsi)
    eqt1d.pressure .+= pend .- eqt1d.pressure[end]
    eqt1d.gm1 = gm1 
    eqt1d.gm9 = gm9
    eqt1d.dvolume_dpsi = Vp / (2π)
    eqt1d.q = eqt1d.dvolume_dpsi .* eqt1d.f .* eqt1d.gm1 / (2π)

    eq2d.grid_type.index = 1
    eq2d.grid.dim1 = collect(r)
    eq2d.grid.dim2 = collect(z)
    eq2d.psi = psirz * (2π)

    return dd
end

function minmax_normalize(x)
    min_x = minimum(x, dims=2)
    max_x = maximum(x, dims=2)
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

function get_basis_functions(model_name, green)
    if model_name == :d3d_efit01
        filename=dirname(@__DIR__) * "/models/basis_functions.bson"
    end
    nw = green[:nw]
    nh = green[:nh]
    nsilop = green[:nsilop]
    magpr2 = green[:magpr2]
    basis_functions = BSON.load(filename)
    basis_functions[:psi_loop] = hcat(basis_functions[:psi_loop]...) |> x -> reshape(x, nsilop,:)
    basis_functions[:bp_probe] = hcat(basis_functions[:bp_probe]...) |> x -> reshape(x, magpr2,:)
    basis_functions[:Ip] = hcat(basis_functions[:Ip]...) |> x -> reshape(x, 1,:)
    basis_functions[:psi] = collect(Base.Iterators.flatten(collect(Base.Iterators.flatten(basis_functions[:psi]))));
    basis_functions[:psi] = reshape(Float64.(basis_functions[:psi]), nw,nh,32)
    basis_functions[:Jt] = collect(Base.Iterators.flatten(collect(Base.Iterators.flatten(basis_functions[:Jt]))));
    basis_functions[:Jt] = reshape(Float64.(basis_functions[:Jt]), nw,nh,32)

    return basis_functions
end #get_basis_functions

function get_basis_functions_1d(model_name)
    if model_name == :d3d_efit01
        filename=dirname(@__DIR__) * "/models/basis_functions_1d.bson"
    end

    # Create interpolation of these functions on import
    basis_functions_1d = BSON.load(filename)[:basis_functions_1d]
    bf1d_itp = Dict()
    bf1d_itp[:pp] = Vector()
    for i in 1:size(basis_functions_1d[:pp])[1]
        push!(bf1d_itp[:pp], IMAS.interp1d(basis_functions_1d[:psi],basis_functions_1d[:pp][i,:]))
    end
    bf1d_itp[:ffp] = Vector()
    for i in 1:size(basis_functions_1d[:ffp])[1]
        push!(bf1d_itp[:ffp], IMAS.interp1d(basis_functions_1d[:psi],basis_functions_1d[:ffp][i,:]))
    end

    return basis_functions_1d, bf1d_itp
end #get_basis_functions_1d

function get_greens_function_tables(model_name)
    if model_name == :d3d_efit01
        filename=dirname(@__DIR__) * "/models/green.bson"
    end
    green = BSON.load(filename)

    green[:rsilfc] = hcat(green[:rsilfc]...) |> x -> reshape(x, :,green[:nfsum])
    green[:rsilec] = hcat(green[:rsilec]...) |> x -> reshape(x, :,green[:nesum])
    green[:rmp2fc] = hcat(green[:rmp2fc]...) |> x -> reshape(x, :,green[:nfsum])
    green[:rmp2ec] = hcat(green[:rmp2ec]...) |> x -> reshape(x, :,green[:nesum])
    green[:ggridfc] = hcat(green[:ggridfc]...) |> x -> reshape(x, :,green[:nfsum])
    green[:gridec] = hcat(green[:gridec]...) |> x -> reshape(x, :,green[:nesum])
    return green
end #get_greens_function_tables

function get_wall(model_name)
    if model_name == :d3d_efit01
        filename=dirname(@__DIR__) * "/models/wall.bson"
    end

    wall =  BSON.load(filename)
    wall[:rlim] = Float64.(wall[:rlim])
    wall[:zlim] = Float64.(wall[:zlim])
    return wall
end #get_wall

function get_model(model_name)
    if model_name == :d3d_efit01
        filename=dirname(@__DIR__) * "/models/model_efit01.bson"
    end
    return BSON.load(filename)[:NNmodel]
end #get_model
Vector{Float64}

function predict_model(Rb::Vector{Float64},Zb::Vector{Float64},pp::Vector{Float64},ffp::Vector{Float64},ecurrt::Vector{Float64}
                    , NNmodel::Dict, green::Dict, basis_functions::Dict,basis_functions_1d::Dict,Ip_target=nothing)

    bound_mxh = IMAS.MXH(Rb,Zb,4)
    pp_fit,ffp_fit = fit_ppffp(pp,ffp,basis_functions_1d)

    predict_model(bound_mxh,pp_fit,ffp_fit,ecurrt,NNmodel, green, basis_functions,Ip_target)
end #predict_model

function predict_model(bound_mxh::IMAS.MXH,pp_fit::Vector{Float64},ffp_fit::Vector{Float64},ecurrt::Vector{Float64},
                       NNmodel::Dict, green::Dict, basis_functions::Dict,Ip_target=nothing)
    xunnorm = vcat(bound_mxh.R0,bound_mxh.Z0,bound_mxh.ϵ,bound_mxh.κ,bound_mxh.tilt,bound_mxh.δ,bound_mxh.ζ,bound_mxh.𝚶,bound_mxh.twist,
    bound_mxh.c,bound_mxh.s, pp_fit, ffp_fit, ecurrt)
    #xunnorm = reshape(xunnorm,28,1)

    println(size(xunnorm))
    model = NNmodel[:model]
    x_min = NNmodel[:x_min]
    x_max = NNmodel[:x_max]
    y_min = NNmodel[:y_min]
    y_max = NNmodel[:y_max]
    #print(x_min)
    x = minmax_normalize(xunnorm,x_min,x_max)
    y = model(x)

    y = minmax_unnormalize(y, y_min, y_max)  # Convert back to original scale
    predict_model(x,y, green, basis_functions,Ip_target)
end #predict_model

function predict_model(x,y, green, basis_functions,basis, Ip_target=nothing)
    nfsum = green[:nfsum]
    nesum = green[:nesum]
    nw = green[:nw]
    nh = green[:nh]
    nbbbs = 9+4*2

    npca = length(basis_functions[:Ip])

    fcurrt = y[npca+1:npca+nfsum]
    ecurrt = x[end-5:end]
    psiext_1d = sum(green[:ggridfc] .* reshape(fcurrt,1,nfsum), dims=2)
    psiext_1d .+= sum(green[:gridec] .* reshape(ecurrt,1,nesum), dims=2)[ :, :,1]
    psiext = reshape(psiext_1d,nh,nw)
    psipla = zeros(nw,nh)
    
    Ip =0
    for ipca in 1:npca
        Ip += y[ipca]*basis_functions[:Ip][ipca]
    end

    if Ip_target != nothing
        y .*= Ip_target/Ip
        Ip = Ip_target
    end
    for ipca in 1:npca
        psipla .+= y[ipca].*transpose(basis_functions[:psi][:,:,ipca])
    end

    psi = -1.0 * (psiext-psipla)

    Ip =0
    for ipca in 1:npca
        Ip += y[ipca]*basis_functions[:Ip][ipca]
    end

    Jt = zeros(nw,nh)
    for ipca in 1:npca
        Jt .+= y[ipca].*transpose(basis_functions[:Jt][:,:,ipca])
    end
    return Jt,psi, Ip,fcurrt
end #predict_model

function get_flux_surfaces(psi,Ip,fcurrt,green,wall,Rb_target,Zb_target,pp_target,ffp_target,ecurrt_target);#(green)

    r = range(green[:rgrid][1],green[:rgrid][end],length(green[:rgrid]))
    z = range(green[:zgrid][1],green[:zgrid][end],length(green[:zgrid]))
    rwall = Float64.(wall[:rlim])
    zwall = Float64.(wall[:zlim])
    PSI_itp = Interpolations.cubic_spline_interpolation((r, z), psi; extrapolation_bc=Interpolations.Line())

    RR  = hcat(green[:RR]...) |> x -> reshape(x, :,green[:nw]);
    ZZ  = hcat(green[:ZZ]...) |> x -> reshape(x, :,green[:nw]);
    ind = argmin(psi) 
    Raxis, Zaxis = IMAS.find_magnetic_axis(r, z, PSI_itp, 1; rguess=RR[ind], zguess=ZZ[ind])
    psiaxis = PSI_itp(Raxis,Zaxis)
    axis2bnd=:increasing
    empty_r = zeros(1)[1:0]
    empty_z = zeros(1)[1:0]
    Ψbnd = IMAS.find_psi_boundary(r, z, psi, psiaxis, axis2bnd, Raxis, Zaxis, rwall, zwall, empty_r,  empty_z;
                                      PSI_interpolant=PSI_itp, raise_error_on_not_open=false, raise_error_on_not_closed=false).last_closed
    
    dpsi = (Ψbnd-psiaxis)/(green[:nw]-1)
    surfaces = IMAS.trace_simple_surfaces(collect(psiaxis:dpsi:Ψbnd),
        r,
        z,
        psi,
        PSI_itp,
        Raxis,
        Zaxis,
        rwall,
        zwall
    );
    
    nsurf = length(surfaces)
    ffp = zeros(nsurf)
    pp = zeros(nsurf)
    Vp = zeros(nsurf)
    gm1 = zeros(nsurf)
    gm9 = zeros(nsurf)
    for (k, surface) in enumerate(surfaces)
        tmp = (j, xx) ->  surface.fluxexpansion[j]*FFP_itp(surface.r[j],surface.z[j]) 
        ffp[k] =  IMAS.flux_surface_avg(tmp, surface)
        
        tmp = (j, xx) ->  surface.fluxexpansion[j]*PP_itp(surface.r[j],surface.z[j]) 
        pp[k] =  IMAS.flux_surface_avg(tmp, surface)    # Vp = dvolume_dpsi
        sign_dpsi = -1
        Vp[k] = sign_dpsi * surface.int_fluxexpansion_dl
    
        # gm1 = <1/R^2>
        f1 = (j, xx) ->  surface.fluxexpansion[j] / surface.r[j] ^ 2
        gm1[k] = IMAS.flux_surface_avg(f1, surface)
    
        # gm9 = <1/R>
        f9 = (j, xx) ->  surface.fluxexpansion[j] / surface.r[j]
        gm9[k] = IMAS.flux_surface_avg(f9, surface)
    end

end


function in_core(r::Real, z::Real, psin::Real, psib::Real, bnd)
    #ellipse::Union{Nothing,AbstractVector{<:Real}}=nothing)

    # Check psinorm value
    #psin > psib && return false

    # Check outside bounding box
    #!in_plasma_bb(r, z, canvas) && return false

    #in_ellipse(r, z, ellipse) && return true

    # Finally make sure it's in the boundary
    return inpolygon((r, z), bnd) == 1
end

function get_isinside(Rb,Zb,Ψ, Ψbnd, green)
    is_inside = zeros(green[:nw],green[:nh])

    _bnd = [[Rb[k], Zb[k]] for k in eachindex(Rb)]
    push!(_bnd,[Rb[1],Zb[1]])
    for (i, r) in enumerate(green[:rgrid])
        for (j, z) in enumerate(green[:zgrid])
            is_inside[j,i] = in_core(r, z, Ψ[i,j],Ψbnd,_bnd)
        end
    end
    return is_inside
end

function get_Jt_fb(pp_fit,ffp_fit,psin_rz,basis_functions_1d,bf1d_itp,green,is_inside)
    bf2d_ppffp = Dict()
    bf2d_ppffp[:pp] = Vector()
    bf2d_ppffp = Dict()
    
    npp = size(basis_functions_1d[:pp])[1]
    nffp = size(basis_functions_1d[:ffp])[1]
    bf2d_ppffp[:pp] = zeros(npp,green[:nh]*green[:nw])
    bf2d_ppffp[:ffp] = zeros(nffp,green[:nh]*green[:nw])
    Jt_fb = zeros(green[:nh],green[:nw])
    for (j, z) in enumerate(green[:zgrid])
        for (i, r) in enumerate(green[:rgrid])
            for ib in 1:npp
                Jt_fb[i,j] -= pp_fit[ib].* bf1d_itp[:pp][ib](psin_rz[j,i])*green[:rgrid][i] * is_inside[i,j] 
            end
            for ib in 1:nffp
                Jt_fb[i,j] -= ffp_fit[ib] .* bf1d_itp[:ffp][ib](psin_rz[j,i])/green[:rgrid][i] * is_inside[i,j] /(4*pi*1e-7)
            end
        end
    end
    return Jt_fb
end


end