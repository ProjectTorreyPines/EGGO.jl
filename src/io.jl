import Interpolations
import BSON
import HDF5
using IMAS


"""
    get_basis_functions(model_name::Symbol)

Load basis functions for plasma current, magnetic probes, and flux loops
associated with a given equilibrium model.
"""
function get_basis_functions(model_name)
    if model_name == :d3d_efit01 || model_name == :d3d_efit01_coils
        filename = dirname(@__DIR__) * "/models/basis_functions.h5"
    elseif model_name == :d3d_efit01efit02cake02 || model_name == :d3d_efit01efit02cake02_coils

        filename = dirname(@__DIR__) * "/models/basis_functions_efit01efit02cake02.h5"
    elseif model_name == :d3d_cakenn_free
        filename = dirname(@__DIR__) * "/models/basis_functions_cakenn.h5"
    end
    raw = read_hdf5_auto(filename)
    return BasisFunctions(raw[:Ip],
        raw[:psi_loop],
        raw[:bp_probe],
        raw[:Jt],
        raw[:psi]
    )
end

"""
    get_basis_functions_1d(model_name::Symbol)

Load 1D basis functions (and their interpolants) for a given equilibrium model.
"""
function get_basis_functions_1d(model_name::Symbol)
    if model_name == :d3d_efit01 || model_name == :d3d_efit01_coils
        filename = dirname(@__DIR__) * "/models/basis_functions_1d.h5"
    elseif model_name == :d3d_efit01efit02cake02 || model_name == :d3d_efit01efit02cake02_coils
        filename = dirname(@__DIR__) * "/models/basis_functions_1d_efit01efit02cake02.h5"
    elseif model_name == :d3d_cakenn_free
        filename = dirname(@__DIR__) * "/models/basis_functions_1d_cakenn.h5"
    end

    raw = read_hdf5_auto(filename)
    #newne = zeros(9, 129)

    #newne[1, :] = 0.5 .- 0.5 .* tanh.((raw[:psi] .- 0.975) ./ 0.025)
    #newne[2, :] = 0.5 .- 0.5 .* tanh.((raw[:psi] .- 0.95) ./ 0.025)
    #newne[3, :] = 0.5 .- 0.5 .* tanh.((raw[:psi] .- 0.925) ./ 0.025)
    #newne[4, :] = 0.5 .- 0.5 .* tanh.((raw[:psi] .- 0.90) ./ 0.025)
    #newne[5, :] = 0.5 .- 0.5 .* tanh.((raw[:psi] .- 0.875) ./ 0.025)
    #newne[6, :] = 1 .- raw[:psi]
    #newne[7, :] = 1 .- raw[:psi].^2
    #newne[8, :] = 1 .- raw[:psi].^3
    #newne[9, :] = 1 .- raw[:psi].^4
    # Create interpolation of these functions on import
    raw_itp = Dict{Symbol,Any}()
    raw_itp[:pp] = [
        IMAS.interp1d(raw[:psi], raw[:pp][i, :])
        for i in 1:size(raw[:pp])[1]
    ]
    raw_itp[:ffp] = [
        IMAS.interp1d(raw[:psi], raw[:ffp][i, :])
        for i in 1:size(raw[:ffp])[1]
    ]

    if model_name == :d3d_cakenn_free
        raw_itp[:ne] = [
            IMAS.interp1d(raw[:psi], raw[:ne][i, :])
            for i in 1:size(raw[:ne])[1]
        ]
        raw_itp[:Te] = [
            IMAS.interp1d(raw[:psi], raw[:Te][i, :])
            for i in 1:size(raw[:Te])[1]
        ]
        raw_itp[:nc] = [
            IMAS.interp1d(raw[:psi], raw[:nc][i, :])
            for i in 1:size(raw[:nc])[1]
        ]
        raw_itp[:Ti] = [
            IMAS.interp1d(raw[:psi], raw[:Ti][i, :])
            for i in 1:size(raw[:Ti])[1]
        ]
        raw_itp[:Vt] = [
            IMAS.interp1d(raw[:psi], raw[:Vt][i, :])
            for i in 1:size(raw[:Vt])[1]
        ]
    else
        raw[:ne] = missing
        raw[:Te] = missing
        raw[:nc] = missing
        raw[:Ti] = missing
        raw[:Vt] = missing
        raw_itp[:ne] = missing
        raw_itp[:Te] = missing
        raw_itp[:nc] = missing
        raw_itp[:Ti] = missing
        raw_itp[:Vt] = missing
    end

    return BasisFunctions1D(raw[:psi], raw[:pp], raw[:ffp], raw[:ne], raw[:Te], raw[:nc],raw[:Ti],raw[:Vt]),
    BasisFunctions1Dinterp(raw_itp[:pp], raw_itp[:ffp], raw_itp[:ne], raw_itp[:Te], raw_itp[:nc],raw_itp[:Ti],raw_itp[:Vt])
end



function get_greens_function_tables(model_name)
    if (
        model_name == :d3d_efit01 || model_name == :d3d_efit01_coils || model_name == :d3d_efit01efit02cake02 || model_name == :d3d_efit01efit02cake02_coils ||
        model_name == :d3d_cakenn_free
    )
        filename = dirname(@__DIR__) * "/models/green.h5"
    end

    raw = read_hdf5_auto(filename)

    r = range(raw[:rgrid][1], raw[:rgrid][end]; length=length(raw[:rgrid]))
    z = range(raw[:zgrid][1], raw[:zgrid][end]; length=length(raw[:zgrid]))

    ggridfc_itp = Vector{Any}()
    gridec_itp = Vector{Any}()

    for i in 1:size(raw[:ggridfc], 2)
        push!(ggridfc_itp, Interpolations.cubic_spline_interpolation((r, z),
            transpose(reshape(raw[:ggridfc][:, i], raw[:nw], raw[:nh]));
            extrapolation_bc=Interpolations.Line()))
    end

    for i in 1:size(raw[:gridec], 2)
        push!(gridec_itp, Interpolations.cubic_spline_interpolation((r, z),
            transpose(reshape(raw[:gridec][:, i], raw[:nw], raw[:nh]));
            extrapolation_bc=Interpolations.Line()))
    end

    return GreenFunctionTables(
        raw[:nw],
        raw[:nh],
        raw[:nfsum],
        raw[:nvsum],
        raw[:ngam],
        raw[:nesum],
        raw[:ivesel],
        raw[:imse],
        raw[:iecoil],
        raw[:nsilop],
        raw[:magpr2],
        raw[:rgrid],
        raw[:zgrid],
        raw[:rsilec],
        raw[:gsilvs],
        raw[:rsilfc],
        raw[:gsilvs],
        raw[:rmp2fc],
        raw[:rmp2ec],
        raw[:ggridfc],
        raw[:gridec],
        ggridfc_itp,
        gridec_itp,
        missing
    )
end


"""
    get_wall(model_name::Symbol)

Load the machine wall definition associated with a given model.
"""
function get_wall(model_name::Symbol)
    if model_name == :d3d_efit01 || model_name == :d3d_efit01_coils || model_name == :d3d_efit01efit02cake02 || model_name == :d3d_efit01efit02cake02_coils ||
       model_name == :d3d_cakenn_free
        filename = dirname(@__DIR__) * "/models/wall.h5"
    end
    raw = read_hdf5_auto(filename)
    return Wall(raw[:rlim], raw[:zlim])
end


"""
    read_hdf5_auto(filename::String) 

Read an HDF5 file into a dictionary with symbol keys.
"""
function read_hdf5_auto(filename::String)
    data_dict = Dict{Symbol,Any}()
    HDF5.h5open(filename, "r") do file
        for key in keys(file)
            dataset = file[key]
            dims = size(dataset)
            data_dict[Symbol(key)] = dataset[fill(:, length(dims))...]
        end
    end
    return data_dict
end


"""
    get_model(model_name::Symbol)

Load a pretrained neural network equilibrium model by name.
"""
function get_model(model_name::Symbol)
    if model_name == :d3d_efit01
        filename = dirname(@__DIR__) * "/models/model_efit01.bson"
    elseif model_name == :d3d_efit01efit02cake02
        filename = dirname(@__DIR__) * "/models/model_efit01efit02cake02.bson"
    elseif model_name == :d3d_efit01efit02cake02_coils
        filename = dirname(@__DIR__) * "/models/model_efit01efit02cake02_coils.bson"
    elseif model_name == :d3d_efit01_coils
        filename = dirname(@__DIR__) * "/models/model_efit01_coils.bson"
    elseif model_name == :d3d_cakenn_free
        filename = dirname(@__DIR__) * "/models/model_cakenn_free.bson"
    end
    NNmodel_raw = Dict{Symbol,Any}()
    for (field, value) in BSON.load(filename, @__MODULE__)[:NNmodel]
        NNmodel_raw[field] = value
    end
    NNmodel_raw[:model] = Flux.fmap(Flux.f64, NNmodel_raw[:model]) # map to 64 bits
    return NeuralNetModel(NNmodel_raw[:model], NNmodel_raw[:y_min], NNmodel_raw[:y_max],
        NNmodel_raw[:x_min], NNmodel_raw[:x_max])
end

function get_model1d(model_name::Symbol)

    if model_name == :d3d_cakenn_free
        filename = dirname(@__DIR__) * "/models/model1d_cakenn_free.bson"
    end
    NNmodel_raw = Dict{Symbol,Any}()
    for (field, value) in BSON.load(filename, @__MODULE__)[:NNmodel]
        NNmodel_raw[field] = value
    end
    NNmodel_raw[:model] = Flux.fmap(Flux.f64, NNmodel_raw[:model]) # map to 64 bits
    return NeuralNetModel(NNmodel_raw[:model], NNmodel_raw[:y_min], NNmodel_raw[:y_max],
        NNmodel_raw[:x_min], NNmodel_raw[:x_max])
end

function predict_from_dd(dd, t,green,NNmodel1d)
    dd.global_time = t
    
    nfsum = green.nfsum
    nesum = green.nesum
    nsilop = green.nsilop
    magpr2 = green.magpr2
    nw = green.nw
    nh = green.nh
    
    expsi = zeros(nsilop)
    fwtsi = ones(nsilop)
    for (ifloop,floop) in enumerate(dd.magnetics.flux_loop)
        if floop.flux.validity == -2
            fwtsi[ifloop] = 0.0
        end
        expsi[ifloop]= -1*(@ddtime floop.flux.data)/2/pi
    end
    expsi[2:end] .-= expsi[1]

    siref = deepcopy(expsi[1])

    expmp2 = zeros(magpr2)
    fwtmp2 = ones(magpr2)
    for (ibprobe,bprobe) in enumerate(dd.magnetics.b_field_pol_probe)
        if bprobe.field.validity == -2
            fwtmp2[ibprobe] = 0.0
        end
        expmp2[ibprobe] =  float(@ddtime bprobe.field.data)
    end

    currents = []
    turns = []
    for coil in dd.pf_active.coil
        push!(currents,(@ddtime coil.current.data))
        push!(turns,(@ddtime coil.element[1].turns_with_sign))
    end

    #siref = expsi[1]
    #expsi[1] = 0.0
    ecurrt = currents[1:6] .* turns[1:6]
    fcurrt = currents[7:end] .* turns[7:end]
    Ip = dd.equilibrium.time_slice[].global_quantities.ip

    shot = dd.dataset_description.data_entry.pulse
    y_psi,XNN = EGGO.predict_psipla_free(shot,expsi,fwtsi,expmp2,fwtmp2,fcurrt,ecurrt,Ip,NNmodel,green,basis_functions)
    #@time y_ne, y_Te, y_nc = EGGO.predict_kinetic(y_psi, r_tom,z_tom,ne_tom,Te_tom,r_cer,z_cer,nc_cer,fcurrt,ecurrt,green,wall,basis_functions,bf1d_itp)

    y= y_psi
    psipla = zeros(Float64, 129, 129)
    Jt = zeros(Float64, 129, 129)
    Ip1 = zeros(Float64,32)
    # Vectorized operations for basis function combination
    @views for ipca in 1:32
        psipla .+= y_psi[ipca] .* basis_functions.psi[:, :, ipca]
        Jt .+= y_psi[ipca] .* basis_functions.Jt[:, :, ipca]    
    end
    psiext = EGGO.calculate_psiext(fcurrt, ecurrt, green)
    psi = psipla .+ psiext

    # Create ranges once
    r = range(green.rgrid[1], green.rgrid[end]; length=length(green.rgrid))
    z = range(green.zgrid[1], green.zgrid[end]; length=length(green.zgrid))
    rwall = Float64.(wall.rlim)
    zwall = Float64.(wall.zlim)
    PSI_itp = Interpolations.scale(Interpolations.interpolate(psi, Interpolations.BSpline(Interpolations.Cubic(Interpolations.Line(Interpolations.OnGrid())))), r, z)
    Raxis, Zaxis = IMAS.find_magnetic_axis(r, z, PSI_itp, 1; rguess=r[65], zguess=z[65])
    psiaxis = PSI_itp(Raxis, Zaxis)
    axis2bnd = :increasing
    empty_r = Float64[]
    empty_z = Float64[]
    psibnd = IMAS.find_psi_boundary(r, z, psi, psiaxis, axis2bnd, Raxis, Zaxis, rwall, zwall, empty_r, empty_z;
    PSI_interpolant = PSI_itp, raise_error_on_not_open=false, raise_error_on_not_closed=false).last_closed
    
    model = NNmodel1D.model
    x_min = NNmodel1D.x_min
    x_max = NNmodel1D.x_max
    y_min = NNmodel1D.y_min
    y_max = NNmodel1D.y_max
    XNN[end-24] *= 1e6
    x = EGGO.minmax_normalize(XNN, x_min, x_max)
    y1d = model(x)
    x = EGGO.minmax_unnormalize(x, x_min, x_max)
    y1d = EGGO.minmax_unnormalize(y1d, y_min, y_max)  # Convert back to original scale
    npca1d = 7
    ne_fit = y1d[1:npca1d,1]
    Te_fit = y1d[npca1d+1:2*npca1d,1]
    nc_fit = y1d[2*npca1d+1:3*npca1d,1]
    Ti_fit = y1d[3*npca1d+1:4*npca1d,1]
    Vt_fit = y1d[4*npca1d+1:5*npca1d,1]

    pp_fit = deepcopy(y_psi[32+1:32+npca1d])
    ffp_fit = deepcopy(y_psi[32+npca1d+1:32+npca1d*2])

    pp = zero(basis_functions_1d.pp[1, :])
    for k in eachindex(pp_fit)
        pp .+= pp_fit[k] .* basis_functions_1d.pp[k, :]
    end
    
    ffp = zero(basis_functions_1d.ffp[1, :])
    for k in eachindex(ffp_fit)
        ffp .+= ffp_fit[k] .* basis_functions_1d.ffp[k, :]
    end
    
    ne = zero(basis_functions_1d.ne[1, :])
    for k in eachindex(ne_fit)
        ne .+= ne_fit[k] .* basis_functions_1d.ne[k, :]
    end     
    Te = zero(basis_functions_1d.ne[1, :])
    for k in eachindex(Te_fit)
        Te .+= Te_fit[k] .* basis_functions_1d.Te[k, :]
    end 
    nc = zero(basis_functions_1d.nc[1, :])
    for k in eachindex(nc_fit)
        nc .+= nc_fit[k] .* basis_functions_1d.nc[k, :]
    end 

    Ti = zero(basis_functions_1d.Ti[1, :])
    for k in eachindex(Ti_fit)
        Ti .+= Ti_fit[k] .* basis_functions_1d.Ti[k, :]
    end 

    Vt = zero(basis_functions_1d.Vt[1, :])
    for k in eachindex(Vt_fit)
        Vt .+= Vt_fit[k] .* basis_functions_1d.Vt[k, :]
    end 


    psi1d = LinRange(psiaxis,psibnd,129)
    pres = IMAS.cumtrapz(psi1d,pp)
    pres .-= pres[end]
    return psi,pp_fit,ffp_fit,ne,Te,nc,Ti,Vt
end