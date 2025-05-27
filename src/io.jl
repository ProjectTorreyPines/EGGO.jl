import Interpolations
import BSON
import HDF5

function get_basis_functions(model_name)
    if model_name == :d3d_efit01 || model_name == :d3d_efit01_coils
        filename = dirname(@__DIR__) * "/models/basis_functions.h5"
    elseif model_name == :d3d_efit01efit02cake02
        filename = dirname(@__DIR__) * "/models/basis_functions_efit01efit02cake02.h5"
    end
    basis_functions = read_hdf5_auto(filename)
    return basis_functions
end

function get_basis_functions_1d(model_name)
    if model_name == :d3d_efit01 || model_name == :d3d_efit01_coils
        filename = dirname(@__DIR__) * "/models/basis_functions_1d.h5"
    elseif model_name == :d3d_efit01efit02cake02
        filename = dirname(@__DIR__) * "/models/basis_functions_1d_efit01efit02cake02.h5"
    end
    basis_functions_1d = read_hdf5_auto(filename)

    # Create interpolation of these functions on import
    bf1d_itp = Dict{Symbol,Any}()
    bf1d_itp[:pp] = [
        IMAS.interp1d(basis_functions_1d[:psi], basis_functions_1d[:pp][i, :])
        for i in 1:size(basis_functions_1d[:pp])[1]
    ]
    bf1d_itp[:ffp] = [
        IMAS.interp1d(basis_functions_1d[:psi], basis_functions_1d[:ffp][i, :])
        for i in 1:size(basis_functions_1d[:ffp])[1]
    ]

    return basis_functions_1d, bf1d_itp
end

function get_greens_function_tables(model_name)
    if model_name == :d3d_efit01 || model_name == :d3d_efit01_coils || model_name == :d3d_efit01efit02cake02
        filename = dirname(@__DIR__) * "/models/green.h5"
    end
    green = read_hdf5_auto(filename)

    r = range(green[:rgrid][1],green[:rgrid][end],length(green[:rgrid]))
    z = range(green[:zgrid][1],green[:zgrid][end],length(green[:zgrid]))
    green[:ggridfc_itp]=Vector()
    green[:gridec_itp] =Vector()
    
    for i in 1:size(green[:ggridfc])[2]
        push!(green[:ggridfc_itp], Interpolations.cubic_spline_interpolation((r, z), transpose(reshape(green[:ggridfc][:,i],green[:nw],green[:nh])); extrapolation_bc=Interpolations.Line()))
    end
    
    for i in 1:size(green[:gridec])[2]
        push!(green[:gridec_itp], Interpolations.cubic_spline_interpolation((r, z), transpose(reshape(green[:gridec][:,i],green[:nw],green[:nh])); extrapolation_bc=Interpolations.Line()))
    end
    
    return green
end

function get_wall(model_name)
    if model_name == :d3d_efit01 || model_name == :d3d_efit01_coils || model_name == :d3d_efit01efit02cake02
        filename = dirname(@__DIR__) * "/models/wall.h5"
    end
    wall = read_hdf5_auto(filename)
    return wall
end

function read_hdf5_auto(filename)
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

function get_model(model_name)
    if model_name == :d3d_efit01
        filename = dirname(@__DIR__) * "/models/model_efit01.bson"
    elseif model_name == :d3d_efit01efit02cake02
        filename = dirname(@__DIR__) * "/models/model_efit01efit02cake02.bson"
    elseif model_name == :d3d_efit01_coils
        filename = dirname(@__DIR__) * "/models/model_efit01_coils.bson"
    end
    NNmodel = Dict{Symbol,Any}()
    for (field, value) in BSON.load(filename, @__MODULE__)[:NNmodel]
        NNmodel[field] = value
    end
    NNmodel[:model] = Flux.fmap(Flux.f64, NNmodel[:model]) # map to 64 bits
    return NNmodel
end