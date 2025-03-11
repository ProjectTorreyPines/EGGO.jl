# import Pkg
# Pkg.add("BSON")
# using BSON
# Pkg.add("HDF5")
using EGGO
using HDF5
using EGGO.BSON

model_name = :d3d_efit01
basis_functions_1d, bf1d_itp = EGGO.get_basis_functions_1d(model_name)
green = EGGO.get_greens_function_tables(model_name)
basis_functions = EGGO.get_basis_functions(model_name, green)
wall = EGGO.get_wall(model_name)

@show "basis_functions_1d"
HDF5.h5open(joinpath(@__DIR__, "basis_functions_1d.h5"), "w") do fid
    for (field, value) in basis_functions_1d
        @show field, typeof(value)
        fid[string(field)] = value
    end
end
BSON.bson(joinpath(@__DIR__, "basis_functions_1d.bson"), basis_functions_1d)

@show "basis_functions"
HDF5.h5open(joinpath(@__DIR__, "basis_functions.h5"), "w") do fid
    for (field, value) in basis_functions
        @show field, typeof(value)
        fid[string(field)] = value
    end
end
BSON.bson(joinpath(@__DIR__, "basis_functions.bson"), basis_functions)

@show "green"
HDF5.h5open(joinpath(@__DIR__, "green.h5"), "w") do fid
    for (field, value) in collect(green)
        if field in (:rgrid, :zgrid, :RR, :ZZ, :nw, :nh, :nsilop, :magpr2, :rsilfc, :rsilec, :nesum, :nfsum, :rmp2ec, :rmp2fc, :ggridfc, :gridec)
            @show field, typeof(value)
            fid[string(field)] = value
        else
            pop!(green, field)
        end
    end
end
BSON.bson(joinpath(@__DIR__, "green.bson"), green)

@show "wall"
HDF5.h5open(joinpath(@__DIR__, "wall.h5"), "w") do fid
    for (field, value) in wall
        @show field, typeof(value)
        fid[string(field)] = value
    end
end
BSON.bson(joinpath(@__DIR__, "wall.bson"), wall)
