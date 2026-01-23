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

"""
    NormalizedModel{M,T}

A Flux layer that wraps a neural network model with input/output normalization.
This makes the model self-contained - users don't need to manually normalize inputs
or unnormalize outputs.

# Fields
- `model::M`: The underlying Flux model (e.g., Chain)
- `x_min::Matrix{T}`: Minimum values for input normalization
- `x_max::Matrix{T}`: Maximum values for input normalization
- `y_min::Matrix{T}`: Minimum values for output unnormalization
- `y_max::Matrix{T}`: Maximum values for output unnormalization

# Example
```julia
normalized_model = NormalizedModel(model, x_min, x_max, y_min, y_max)
y = normalized_model(x)  # Automatically normalizes x, runs model, unnormalizes y
```
"""
struct NormalizedModel{M,T}
    model::M
    x_min::Matrix{T}
    x_max::Matrix{T}
    y_min::Matrix{T}
    y_max::Matrix{T}
end

"""
    (m::NormalizedModel)(x)

Forward pass through the normalized model. Automatically handles normalization
of inputs and unnormalization of outputs.
"""
function (m::NormalizedModel)(x)
    x_norm = minmax_normalize(x, m.x_min, m.x_max)
    y_norm = m.model(x_norm)
    y = minmax_unnormalize(y_norm, m.y_min, m.y_max)
    return y
end

# Register with Flux as a layer - automatically handles parameters and pretty-printing
Flux.@layer NormalizedModel

"""
    to_normalized_model(nn_model::NeuralNetModel)

Convert a legacy NeuralNetModel to a self-contained NormalizedModel that includes
preprocessing (normalization) as part of the model itself.

# Example
```julia
# Load your existing model
nn_model = EGGO.get_model(:d3d_cakenn_free)

# Convert to normalized model
normalized = EGGO.to_normalized_model(nn_model)

# Now you can use it directly without manual normalization
y = normalized(x)  # That's it!
```
"""
function to_normalized_model(nn_model::NeuralNetModel)
    return NormalizedModel(
        nn_model.model,
        nn_model.x_min,
        nn_model.x_max,
        nn_model.y_min,
        nn_model.y_max
    )
end

"""
    RegularizedLeastSquaresLayer

ONNX-compatible layer for regularized least squares.
Precomputes the pseudoinverse for fixed A matrix and regularization.

NO MASKING - assumes all measurements are always used. For bad/missing measurements,
users should set them to 0 in the input.

W = (A[:, 1:npca]' * A[:, 1:npca] + λI)^(-1) * A[:, 1:npca]'
y_lsq = W * X

# Fields
- `W::Matrix`: Precomputed pseudoinverse (npca × n_measurements)
- `A::Matrix`: Original A matrix (n_measurements × n_basis)
- `npca::Int`: Number of PCA components
"""
struct RegularizedLeastSquaresLayer{T<:Real}
    W::Matrix{T}       # Precomputed (A' * A + λI)^(-1) * A'
    A::Matrix{T}       # Original A matrix
    npca::Int
end

function RegularizedLeastSquaresLayer(A::Matrix{T}, npca::Int, lambda::T) where {T<:Real}
    A_pca = A[:, 1:npca]
    # Precompute pseudoinverse: (A' * A + λI)^(-1) * A' (ONNX-compatible!)
    W = (A_pca' * A_pca + lambda * I) \ A_pca'
    return RegularizedLeastSquaresLayer(W, A, npca)
end

# Forward pass - pure matrix multiplication (ONNX-compatible!)
function (layer::RegularizedLeastSquaresLayer)(X::AbstractVector)
    # Solve: y_lsq = W * X (just a matrix multiplication!)
    y_lsq = layer.W * X

    # Reconstruct: X_reconstructed = A[:, 1:npca] * y_lsq
    X_reconstructed = layer.A[:, 1:layer.npca] * y_lsq

    return X_reconstructed, y_lsq
end

Flux.@layer RegularizedLeastSquaresLayer

"""
    CompletePreprocessingModel{M,G,B}

A comprehensive Flux layer that encapsulates the entire prediction pipeline:
1. Raw diagnostic inputs (expsi, expmp2, fcurrt, ecurrt, Ip, fwtsi, fwtmp2)
2. Flux/probe preprocessing and masking
3. Regularized least squares solve (ONNX-compatible!)
4. NN prediction with normalization

This makes the model completely self-contained - users just provide raw diagnostics
and get predictions without knowing any preprocessing details.

ONNX-EXPORTABLE: All operations are pure tensor ops (matmul, add, multiply, etc.)

# Fields
- `nn_model::M`: The NormalizedModel (includes normalization)
- `green::G`: Green function tables (as constant weights)
- `basis_functions::B`: Basis functions for reconstruction (as constant weights)
- `lsq_layer`: Precomputed regularized least squares layer
"""
struct CompletePreprocessingModel{M,G,B,L}
    nn_model::M
    green::G
    basis_functions::B
    lsq_layer::L
end

"""
    (m::CompletePreprocessingModel)(expsi, expmp2, fcurrt, ecurrt, Ip)

Forward pass through the complete preprocessing model. Takes raw diagnostic data
and returns predictions.

ALL OPERATIONS ARE ONNX-COMPATIBLE - pure tensor operations (matmul, add, multiply, etc.)
NO MASKING - for bad/missing diagnostics, set them to 0 in the input.

# Arguments
- `expsi`: Experimental psi loop measurements (set bad measurements to 0)
- `expmp2`: Experimental magnetic probe measurements (set bad measurements to 0)
- `fcurrt`: F-coil currents
- `ecurrt`: E-coil currents
- `Ip`: Plasma current

# Returns
- Predictions from the neural network (Matrix)
"""
function (m::CompletePreprocessingModel)(expsi, expmp2, fcurrt, ecurrt, Ip)
    # Compute external flux contributions for loops (ONNX: matmul, add, sub)
    siref = expsi[1]
    cm1_flux = m.green.rsilfc * fcurrt
    cm2_flux = m.green.rsilec * ecurrt
    psiloop_ext = cm1_flux .+ cm2_flux .- siref
    psiloop_in = expsi .- psiloop_ext
    psiloop_in[1] -= siref

    # Compute external contributions for probes (ONNX: matmul, sub)
    cm1_probe = m.green.rmp2fc * fcurrt
    cm2_probe = m.green.rmp2ec * ecurrt
    bp_in = expmp2 .- cm1_probe .- cm2_probe

    # Build plasma X vector (ONNX: concatenate, divide)
    X = vcat(psiloop_in, bp_in, [Ip / 1e6])

    # Regularized least squares (ONNX: just matrix multiply!)
    X_reconstructed, y_lsq = m.lsq_layer(X)

    # Build input for NN (ONNX: concatenate)
    XNN = vcat(X_reconstructed, fcurrt, ecurrt)

    # Run through NN with built-in normalization (ONNX: matmul, relu, add, multiply)
    y = m.nn_model(XNN)

    # Correct Ip to match experimental Ip (ONNX: dot product, divide, multiply)
    IpNN = sum(m.basis_functions.Ip .* y[1:32])
    y = y .* (Ip / IpNN)

    return y
end

# Register with Flux
Flux.@layer CompletePreprocessingModel

"""
    to_complete_model(nn_model, green, basis_functions; npca=8, reg_lambda=1e-10)

Create a completely self-contained ONNX-EXPORTABLE model that includes ALL preprocessing,
from raw diagnostics to final predictions.

**IMPORTANT**: This model uses NO MASKING for ONNX compatibility.
For bad/missing diagnostic measurements, set them to 0 in the input.

All operations are pure tensor operations (matmul, add, multiply, etc.):
- Green table multiplications → precomputed constant matrices
- Regularized least squares → precomputed pseudoinverse (just a matmul!)
- Min-max normalization → built into the NormalizedModel
- Neural network → standard Flux layers

# Example
```julia
complete_model = EGGO.to_complete_model(
    NNmodel,
    green,
    basis_functions,
    npca=8,
    reg_lambda=1e-10
)

# Julia usage - just provide raw diagnostics (set bad ones to 0)
predictions = complete_model(expsi, expmp2, fcurrt, ecurrt, Ip)

# Export to ONNX for Python
using ONNX
ONNX.save("model.onnx", complete_model)

# Python usage
# import onnxruntime as ort
# model = ort.InferenceSession("model.onnx")
# predictions = model.run(None, {
#     'expsi': expsi,    # Set bad measurements to 0
#     'expmp2': expmp2,  # Set bad measurements to 0
#     'fcurrt': fcurrt,
#     'ecurrt': ecurrt,
#     'Ip': Ip
# })[0]
```
"""
function to_complete_model(nn_model, green, basis_functions; npca=8, reg_lambda=1e-10)
    # Convert to NormalizedModel if it isn't already
    if isa(nn_model, NeuralNetModel)
        nn_model = to_normalized_model(nn_model)
    end

    # Build the A matrix for least squares
    A = vcat(
        basis_functions.psi_loop,
        basis_functions.bp_probe,
        -reshape(basis_functions.Ip, 1, 32) ./ 1e6
    )

    # Create precomputed least squares layer
    lsq_layer = RegularizedLeastSquaresLayer(A, npca, reg_lambda)

    return CompletePreprocessingModel(
        nn_model,
        green,
        basis_functions,
        lsq_layer
    )
end


"""
    fit_ppffp(pp::Vector{T}, ffp::Vector{T}, basis_functions_1d::BasisFunctions1D{Float64}) 
    fit_ppffp(pp::Vector{T}, ffp::Vector{T}, basis_functions_1d::BasisFunctions1D{Float64}, pp_index::Integer, ffp_index::Integer)

Fit pressure (`pp`) and poloidal flux function derivative (`ffp`) profiles
to a set of 1D basis functions using ADMM with L2 regularization.
"""
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

"""
    get_ΨaxisΨbndffppp(psirz, green, basis_functions, basis_functions_1d, bf1d_itp, wall, pp_fit, ffp_fit; Ip_target=0.0)

Compute the magnetic axis, plasma boundary, and reconstructed pressure / `f f'` profiles
from a given equilibrium flux distribution.
"""

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
            wall.rlim,
            wall.zlim;
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


"""
    fill_eqt(eqt, psirz, green, wall, pp, ffp, Btcenter, Rcenter, pend, Ψbnd, Ψaxis, Raxis, Zaxis)
        -> IMAS.equilibrium__time_slice

Populate an IMAS equilibrium time slice with global quantities, 1D profiles, and 2D flux surfaces
from reconstructed equilibrium data.
"""

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

"""
    minmax_normalize(x) )

Apply row-wise min–max normalization to an array.
"""

function minmax_normalize(x)
    min_x = minimum(x; dims=2)
    max_x = maximum(x; dims=2)
    x_norm = (x .- min_x) ./ (max_x .- min_x .+ eps())  # Add eps() to avoid division by zero
    return x_norm, min_x, max_x
end

"""
    minmax_normalize(x, min_x, max_x)

Normalize an array using provided row-wise minimum and maximum values.
"""
function minmax_normalize(x, min_x, max_x)
    x_norm = (x .- min_x) ./ (max_x .- min_x .+ eps())  # Add eps() to avoid division by zero
    return x_norm
end

"""
    minmax_unnormalize(x_norm, min_x, max_x)

Reverse min–max normalization using provided row-wise minimum and maximum values.
"""
function minmax_unnormalize(x_norm, min_x, max_x)
    return x_norm .* (max_x .- min_x) .+ min_x
end


"""
    predict_NN(xunnorm, NNmodel:)

Predict output using a trained neural network model, handling input and output normalization.
"""
function predict_NN(xunnorm::AbstractArray{Float64}, NNmodel::NeuralNetModel)

    model = NNmodel.model

    x_min = NNmodel.x_min
    x_max = NNmodel.x_max
    y_min = NNmodel.y_min
    y_max = NNmodel.y_max

    x = minmax_normalize(xunnorm, x_min, x_max)
    y = model(x)

    x = minmax_unnormalize(x, x_min, x_max) # Convert back to original scale
    y = minmax_unnormalize(y, y_min, y_max)  # Convert back to original scale
    return y
end

"""
    predict_model_from_boundary(Rb, Zb, pp, ffp, NNmodel, green, basis_functions, basis_functions_1d, coils; Ip_target=0.0, use_vacuumfield_green=false)

    predict_model_from_boundary(Rb, Zb, pp_fit, ffp_fit, NNmodel, green, basis_functions, coils; Ip_target=0.0, use_vacuumfield_green=false)

Fit pressure and f f' profiles and predict equilibrium using a neural network model from plasma boundary points.
"""
function predict_model_from_boundary(
    Rb::Vector{T},
    Zb::Vector{T},
    pp::Vector{T},
    ffp::Vector{T},
    NNmodel::NeuralNetModel{Float64},
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
    NNmodel::NeuralNetModel{Float64},
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

    y = predict_NN(xunnorm, NNmodel)

    Jt, psirz, Ip = predict_model(y, green, basis_functions, Ip_target)
    psirz .+= calculate_psiext(Rb, Zb, psirz, green, coils, use_vacuumfield_green)

    return Jt, psirz, Ip
end


"""
    calculate_psiext(Rb_target, Zb_target, psipla, green, coils; use_vacuumfield_green=true)

    calculate_psiext(fcurrt, ecurrt, green)

Compute the external poloidal flux (`Ψ_ext`) on the grid given target boundary points
and an existing plasma flux distribution, optionally using vacuum-field Green's functions.
"""
function calculate_psiext(
    Rb_target::Vector{T},
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
        iso_cps = VacuumFields.IsoControlPoints(Rb_target, Zb_target; weight=1.0)
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

function calculate_psiext(fcurrt::AbstractVector{T}, ecurrt::AbstractVector{T}, green::GreenFunctionTables{Float64}) where {T<:Real}
    # Direct matrix-vector multiplications (16641,18) * (18,) + (16641,6) * (6,) = (16641,)
    psiext_flat = green.ggridfc * fcurrt + green.gridec * ecurrt

    # Reshape to target dimensions and negate
    return -transpose(reshape(psiext_flat, green.nh, green.nw))
end

"""
    predict_model_from_coils(pp, ffp, ecurrt, fcurrt, NNmodel, green, basis_functions, basis_functions_1d; Ip_target=0.0, use_vacuumfield_green=false)
    predict_model_from_coils(pp_fit,ffp_fit, ecurrt,fcurrt,NNmodel,green,basis_functions; Ip_target=0.0, use_vacuumfield_green=false)

Fit pressure and f f' profiles and predict equilibrium using a neural network model from coil currents.
"""
function predict_model_from_coils(
    pp::Vector{T},
    ffp::Vector{T},
    ecurrt::Vector{Float64},
    fcurrt::Vector{Float64},
    NNmodel::NeuralNetModel{Float64},
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
    NNmodel::NeuralNetModel{Float64},
    green::GreenFunctionTables{Float64},
    basis_functions::BasisFunctions{Float64},
    Ip_target::Float64=0.0,
    use_vacuumfield_green::Bool=false
) where {T<:Real}

    xunnorm = vcat(
        pp_fit,
        ffp_fit,
        ecurrt,
        fcurrt
    )

    y = predict_NN(xunnorm, NNmodel)

    Jt, psirz, Ip = predict_model(y, green, basis_functions, Ip_target)
    psirz .+= calculate_psiext(fcurrt, ecurrt, green)
    return Jt, psirz, Ip
end

"""
    predict_model(y, green, basis_functions, Ip_target)

Predict the toroidal current density and poloidal flux from neural network outputs and basis functions.
"""

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

"""
    get_isinside(Rb, Zb, green)

Compute a mask indicating which points on the `(R, Z)` grid lie inside a closed boundary.
"""
function get_isinside(Rb, Zb, green)
    is_inside = zeros(green.nw, green.nh)
    _bnd = [[Rb[k], Zb[k]] for k in eachindex(Rb)]
    push!(_bnd, [Rb[1], Zb[1]])
    for (i, r) in enumerate(green.rgrid)
        for (j, z) in enumerate(green.zgrid)
            is_inside[i, j] = inpolygon((r, z), _bnd) == 1
        end
    end
    return is_inside
end


"""
    get_Jt_fb(pp_fit, ffp_fit, psin_rz, basis_functions_1d, bf1d_itp, green, is_inside)

Compute the toroidal current contributions from pressure (`pp`) and poloidal current (`ffp`) profiles
on a `(R, Z)` grid inside the plasma boundary.
"""
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
                psin_val = psin_rz[i, j]

                # Vectorized computation for pp
                for ib in 1:npp
                    Jt_pp[i, j] -= pp_fit[ib] * bf1d_itp.pp[ib](psin_val) * r
                end

                # Vectorized computation for ffp
                r_inv = inv(r)
                for ib in 1:nffp
                    Jt_ffp[i, j] -= ffp_fit[ib] * bf1d_itp.ffp[ib](psin_val) * r_inv / IMAS.mks.μ_0
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


"""
    calc_pffprime2(psinrz, Jt, rgrid, is_inside, bf1d_itp)

Compute the coefficients of pressure (`pp`) and poloidal current (`ff'`) basis functions
from a toroidal current distribution.
"""
function calc_pffprime2(psinrz::Matrix{Float64}, Jt::Matrix{Float64}, rgrid::Vector{Float64}, is_inside::Matrix{Float64}, bf1d_itp::BasisFunctions1Dinterp)

    npp = length(bf1d_itp.pp)
    nffp = length(bf1d_itp.ffp)
    n = length(psinrz)
    A = zeros(npp + nffp, n)
    nw = nh = 129
    nffp = 17
    npp = 17
    # Construct A matrix
    Threads.@threads for j in 1:nw
        @inbounds for i in 1:nh
            r = rgrid[i]
            ij = j * nw + i
            if is_inside[i, j] != 0  # Skip if outside
                psin_val = psinrz[i, j]

                # Vectorized computation for pp
                for ib in 1:npp
                    A[ib, ij] = bf1d_itp.pp[ib](psin_val) * r
                end

                for ib in 1:nffp
                    A[ib+npp, ij] = bf1d_itp.ffp[ib](psin_val) / r
                end
            end
        end
    end

    b = vec(Jt)
    # Solve least squares: x = argmin ||X'x - b||
    x = reg_solve(A', b, 1.0)

    return x[1:npp], x[npp+1:end]
end

"""
    predict_coil_currents(Rb, Zb, green, psipla)

Estimate coil currents required to reproduce a given plasma flux distribution.
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

"""
    reg_solve(A, b, λ)

Solve a linear system with L2 (Tikhonov) regularization.
"""
function reg_solve(A::AbstractArray{T}, b::AbstractArray{T}, λ::T) where {T<:Real}
    return (A' * A + λ * I) \ A' * b
end

export get_greens_function_tables, get_basis_functions, get_model, get_basis_functions_1d, predict_model
export NormalizedModel, to_normalized_model, minmax_normalize, minmax_unnormalize
export CompletePreprocessingModel, to_complete_model, RegularizedLeastSquaresLayer

const document = Dict()
document[Symbol(@__MODULE__)] = [name for name in Base.names(@__MODULE__; all=false, imported=false) if name != Symbol(@__MODULE__)]

end
