using StatsBase
using IMAS

"""
    predict_psipla_free(shot, expsi, fwtsi, expmp2, fwtmp2, fcurrt, ecurrt, ip, NNmodel, green, basis_functions)

Predict the plasma poloidal flux distribution using a neural network in reconstruction mode
"""
function predict_psipla_free(shot::Int,
    expsi::Vector{T},
    fwtsi::Vector{T},
    expmp2::Vector{T},
    fwtmp2::Vector{T},
    fcurrt::Vector{T},
    ecurrt::Vector{T},
    ip::T,
    NNmodel::NeuralNetModel{T},
    NNmodel1D::NeuralNetModel{T},
    green::GreenFunctionTables{T},
    basis_functions::BasisFunctions{T}) where {T<:Float64}

    nfsum = green.nfsum
    nesum = green.nesum
    nsilop = green.nsilop
    magpr2 = green.magpr2

    siref = copy(expsi[1])    # Clone the first column to `siref`

    cm1_flux = sum(green.rsilfc .* reshape(fcurrt, 1, nfsum); dims=2)[:, 1]
    cm2_flux = sum(green.rsilec .* reshape(ecurrt, 1, nesum); dims=2)[:, 1]
    cm3_flux = 0.0
    psiloop_ext = cm1_flux .+ cm2_flux .+ cm3_flux .- siref
    psiloop_in = expsi .- psiloop_ext
    psiloop_in[1] -= siref
    # Compute external contributions for probes

    mask_mpi = ones(Float64, 76)

    if shot ≥ 156001 && shot < 197555
        mask_mpi[[61, 69, 70, 72]] .= 0.0
    elseif shot ≥ 124400 && shot < 156001
        mask_mpi[[10, 13, 61, 68, 69, 70, 72, 74, 75, 76]] .= 0.0
    elseif shot < 124400
        mask_mpi[[2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 61, 68, 69, 70, 72, 74, 75, 76]] .= 0.0
    end

    mask_mpi = fwtmp2 .* mask_mpi

    cm1_probe = sum(green.rmp2fc .* reshape(fcurrt, 1, nfsum); dims=2)[:, 1]
    cm2_probe = sum(green.rmp2ec .* reshape(ecurrt, 1, nesum); dims=2)[:, 1]
    cm3_probe = 0.0

    bp_in = expmp2 .- cm1_probe .- cm2_probe .- cm3_probe

    # Build plasma X matrix and diagnostic matrix Xdia
    X = zeros(Float64, magpr2 + nsilop + 1, 1)
    # Fill X matrix

    X[1:nsilop, 1] = psiloop_in
    X[nsilop+1:nsilop+magpr2, 1] = bp_in
    X[end, 1] = ip ./ 1e6

    # Build matrices for regression

    A = vcat(basis_functions.psi_loop, basis_functions.bp_probe, -reshape(basis_functions.Ip, 1, 32) ./ 1e6)
    mask_psi = fwtsi .* expsi .!= 0.0
    mask_bp = mask_mpi .* expmp2 .!= 0.0
    mask_ip = trues(1)  # Always true
    mask = vcat(mask_psi, mask_bp, mask_ip)

    npca = 8
    y_lsq = reg_solve(A[mask, 1:npca], X[mask, 1], 1e-10)
    X[.!mask, 1] .= 0.0
    for ipca in 1:npca
        X[.!mask, 1] += A[.!mask, ipca] * y_lsq[ipca]
    end

    XNN = vcat(X, fcurrt, ecurrt)

    y = predict_NN(XNN, NNmodel)

    # Correct Ip to match experimental Ip
    IpNN = sum(basis_functions.Ip .* y[1:32, 1])
    y .*= ip / IpNN
    #y_lsq.*= ip / IpNN
    #y = vcat(-1y_lsq,zeros(24),y[33:end])

    model1d = NNmodel1D.model
    x_min = NNmodel1D.x_min
    x_max = NNmodel1D.x_max
    y_min = NNmodel1D.y_min
    y_max = NNmodel1D.y_max
    XNN[end-24] *= 1e6
    x = EGGO.minmax_normalize(XNN, x_min, x_max)
    y1d = model1d(x)
    x = EGGO.minmax_unnormalize(x, x_min, x_max)
    y1d = EGGO.minmax_unnormalize(y1d, y_min, y_max)  # Convert back to original scale

    return y, y1d
end

"""
    predict_kinetic(y, r_tom, z_tom, ne_tom, Te_tom, r_cer, z_cer, nc_cer, fcurrt, ecurrt, green, wall, basis_functions, bf1d_itp)

Predict kinetic profile coefficients (electron density, electron temperature, and impurity density) from NN equilibrium outputs and diagnostics.
"""
function predict_kinetic(y::Vector{T},
    r_tom::Vector{T},
    z_tom::Vector{T},
    ne_tom::Vector{T},
    Te_tom::Vector{T},
    r_cer::Vector{T},
    z_cer::Vector{T},
    nc_cer::Vector{T},
    fcurrt::Vector{T},
    ecurrt::Vector{T},
    green::GreenFunctionTables{Float64},
    wall::Wall,
    basis_functions::BasisFunctions{Float64},
    bf1d_itp::BasisFunctions1Dinterp) where {T<:Real}

    # Pre-allocate with correct size and use view for efficiency
    psipla = zeros(T, 129, 129)
    Ip1 = zero(T)

    # Vectorized operations for basis function combination
    @views for ipca in 1:32
        psipla .+= y[ipca] .* basis_functions.psi[:, :, ipca]
        Ip1 += basis_functions.Ip[ipca] * y[ipca]
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

    # Pre-allocate and vectorize psi calculations
    psi_tom = Vector{T}(undef, length(r_tom))
    @inbounds for i in eachindex(r_tom, z_tom)
        psi_tom[i] = PSI_itp(r_tom[i], z_tom[i])
    end

    # Optimized separatrix calculation
    T_sep = 80.0
    tolerance = 40.0

    # Use broadcasting for condition check
    close_to_sep = abs.(Te_tom .- T_sep) .< tolerance
    indices = findall(close_to_sep)

    psi_sep = if !isempty(indices)
        # Vectorized weight calculation
        Te_subset = @view Te_tom[indices]
        weights = @. exp(-((Te_subset - T_sep)^2) / (0.5 * tolerance^2))
        sum(@view(psi_tom[indices]) .* weights) / sum(weights)  # Manual weighted mean for efficiency
    else
        axis2bnd = :increasing
        empty_r = T[]
        empty_z = T[]
        IMAS.find_psi_boundary(r, z, psi, psiaxis, axis2bnd, Raxis, Zaxis, rwall, zwall, empty_r, empty_z;
            PSI_interpolant=PSI_itp, raise_error_on_not_open=false, raise_error_on_not_closed=false).last_closed
    end

    # Combined matrix construction for Thomson data
    n_tom = length(psi_tom)
    npca1d_tom = min(6, n_tom)

    # Pre-allocate matrices
    Ane = zeros(T, n_tom, npca1d_tom)
    ATe = zeros(T, n_tom, npca1d_tom)

    # Combined loop with bounds checking
    psi_norm_factor = inv(psi_sep - psiaxis)
    @inbounds for i in 1:n_tom
        psi_norm = (psi_tom[i] - psiaxis) * psi_norm_factor

        # Fill both matrices in single loop
        if ne_tom[i] > 0
            for j in 1:npca1d_tom
                Ane[i, j] = bf1d_itp.ne[j](psi_norm)
            end
        end

        if Te_tom[i] > 0
            for j in 1:npca1d_tom
                ATe[i, j] = bf1d_itp.Te[j](psi_norm)
            end
        end
    end

    # Solve for Thomson coefficients
    c_ne = EGGO.reg_solve(Ane, ne_tom, 1.0e6)
    c_Te = EGGO.reg_solve(ATe, Te_tom, 1.0e6)

    # CER data processing
    n_cer = length(r_cer)
    psi_cer = Vector{T}(undef, n_cer)
    @inbounds for i in 1:n_cer
        psi_cer[i] = PSI_itp(r_cer[i], z_cer[i])
    end

    npca1d_cer = min(9, n_cer)
    Anc = zeros(T, n_cer, npca1d_cer)

    @inbounds for i in 1:n_cer
        if i <= length(ne_tom) && ne_tom[i] > 0  # Guard against index bounds
            psi_norm = (psi_cer[i] - psiaxis) * psi_norm_factor
            for j in 1:npca1d_cer
                Anc[i, j] = bf1d_itp.nc[j](psi_norm)
            end
        end
    end

    c_nc = EGGO.reg_solve(Anc, nc_cer, 1.0)

    return c_ne, c_Te, c_nc
end

function fill_cp1d(cp1d, ne, Te, nc, Ti, Vt, rho, psi)
    cp1d.grid.rho_tor_norm = rho
    cp1d.grid.psi = psi
    cp1d.electrons.density_thermal = ne
    cp1d.electrons.temperature = Te
    cp1d.electrons.density_thermal = ne
    resize!(cp1d.ion, 2)
    resize!(cp1d.ion[1].element, 1)
    cp1d.ion[1].element[1].a = 2.0
    cp1d.ion[1].element[1].z_n = 1.0
    cp1d.ion[1].label = "D"

    resize!(cp1d.ion[2].element, 1)
    cp1d.ion[2].element[1].a = 12.0
    cp1d.ion[2].element[1].z_n = 6.0
    cp1d.ion[2].label = "C"

    cp1d.ion[1].temperature = Ti
    cp1d.ion[2].temperature = Ti
    cp1d.ion[1].density_thermal = ne - 6.0 * nc
    cp1d.ion[2].density_thermal = nc
    return cp1d.ion[2].rotation_frequency_tor = Vt
end

function predict_from_dd(dd::IMAS.dd{Float64}, t::Float64,
    NNmodel::NeuralNetModel,
    NNmodel1D::NeuralNetModel,
    green::GreenFunctionTables{Float64},
    wall::Wall,
    basis_functions::BasisFunctions{Float64},
    basis_functions_1d::BasisFunctions1D{Float64})
    dd.global_time = t

    nfsum = green.nfsum
    nesum = green.nesum
    nsilop = green.nsilop
    magpr2 = green.magpr2
    nw = green.nw
    nh = green.nh

    expsi = zeros(nsilop)
    fwtsi = ones(nsilop)
    for (ifloop, floop) in enumerate(dd.magnetics.flux_loop)
        if floop.flux.validity == -2
            fwtsi[ifloop] = 0.0
        end
        expsi[ifloop] = -1 * (@ddtime floop.flux.data) / 2 / pi
    end
    expsi[2:end] .-= expsi[1]

    siref = deepcopy(expsi[1])

    expmp2 = zeros(magpr2)
    fwtmp2 = ones(magpr2)
    for (ibprobe, bprobe) in enumerate(dd.magnetics.b_field_pol_probe)
        if bprobe.field.validity == -2
            fwtmp2[ibprobe] = 0.0
        end
        expmp2[ibprobe] = float(@ddtime bprobe.field.data)
    end

    currents = []
    turns = []
    for coil in dd.pf_active.coil
        push!(currents, (@ddtime coil.current.data))
        push!(turns, (@ddtime coil.element[1].turns_with_sign))
    end

    ecurrt = currents[1:6] .* turns[1:6]
    fcurrt = currents[7:end] .* turns[7:end]
    Ip = @ddtime dd.pulse_schedule.flux_control.i_plasma.reference#dd.pulse_schedule.dd.equilibrium.time_slice[].global_quantities.ip

    shot = dd.dataset_description.data_entry.pulse
    y_psi, y1d = EGGO.predict_psipla_free(shot, expsi, fwtsi, expmp2, fwtmp2, fcurrt, ecurrt, Ip, NNmodel, NNmodel1D, green, basis_functions)
    return y_psi, y1d, fcurrt, ecurrt

end

function fit_profiles(y_psi::Matrix{T},
    y1d::Matrix{T},
    fcurrt::Vector{T},
    ecurrt::Vector{T},
    green::GreenFunctionTables{T},
    basis_functions::BasisFunctions{T},
    basis_functions_1d::BasisFunctions1D{T},
    wall::Wall) where {T<:Float64}
    psipla = zeros(Float64, 129, 129)
    Jt = zeros(Float64, 129, 129)
    Ip1 = zeros(Float64, 32)
    # Vectorized operations for basis function combination
    @views for ipca in 1:32
        psipla .+= y_psi[ipca] .* basis_functions.psi[:, :, ipca]
        Jt .+= y_psi[ipca] .* basis_functions.Jt[:, :, ipca]
    end
    psiext = EGGO.calculate_psiext(fcurrt, ecurrt, green)
    psi = psipla .+ psiext

    npca1d = length(basis_functions_1d.ne[:, 1])
    ne_fit = y1d[1:npca1d, 1]
    Te_fit = y1d[npca1d+1:2*npca1d, 1]
    nc_fit = y1d[2*npca1d+1:3*npca1d, 1]
    Ti_fit = y1d[3*npca1d+1:4*npca1d, 1]
    Vt_fit = y1d[4*npca1d+1:5*npca1d, 1]

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

    return psi, pp_fit, ffp_fit, ne, Te, nc, Ti, Vt
end

function calculate_boundary(y::Matrix{T},
    fcurrt::Vector{T},
    ecurrt::Vector{T},
    green::GreenFunctionTables{T},
    basis_functions::BasisFunctions{T},
    wall::Wall
) where {T<:Float64}

    nw = green.nw
    nh = green.nh
    npca = length(basis_functions.Ip)

    psipla = zeros(T, (nw, nh))
    for ipca in 1:npca
        @views psipla .+= y[ipca] .* (basis_functions.psi[:, :, ipca])
    end

    psiext = calculate_psiext(fcurrt, ecurrt, green)

    psirz = psipla .+ psiext

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

    return Rb, Zb
end