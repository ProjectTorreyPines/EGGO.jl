using StatsBase

function predict_psipla_free(shot::Int,
    expsi::Vector{T},
    fwtsi::Vector{T},
    expmp2::Vector{T},
    fwtmp2::Vector{T},
    fcurrt::Vector{T},
    ecurrt::Vector{T},
    ip::T,
    NNmodel::NeuralNetModel{T},
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
    IpNN = sum(basis_functions.Ip .* y[1:32,1])
    y .*= ip / IpNN

    return y[:, 1], XNN#-1*y_lsq
end

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