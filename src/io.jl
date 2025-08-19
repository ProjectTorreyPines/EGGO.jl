import Interpolations
import BSON
import HDF5


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
    else
        raw[:ne] = missing
        raw[:Te] = missing
        raw[:nc] = missing
        raw_itp[:ne] = missing
        raw_itp[:Te] = missing
        raw_itp[:nc] = missing
    end

    return BasisFunctions1D(raw[:psi], raw[:pp], raw[:ffp], raw[:ne], raw[:Te], raw[:nc]),
    BasisFunctions1Dinterp(raw_itp[:pp], raw_itp[:ffp], raw_itp[:ne], raw_itp[:Te], raw_itp[:nc])
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
    return NeuralNetModel(NNmodel_raw[:model],NNmodel_raw[:y_min],NNmodel_raw[:y_max],
                          NNmodel_raw[:x_min],NNmodel_raw[:x_max])
end