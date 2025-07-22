using StatsBase

function predict_psipla_free(shot::Int,
    expsi::Vector{T},
    fwtsi::Vector{T},
    expmp2::Vector{T},
    fwtmp2::Vector{T},
    fcurrt::Vector{T},
    ecurrt::Vector{T},
    ip::T,
    NNmodel::Dict{Symbol,<:Any},
    green::GreenFunctionTables{Float64},
    basis_functions::Dict{Symbol,<:Any}) where {T<:Real}
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

    A = vcat(basis_functions[:psi_loop], basis_functions[:bp_probe], -reshape(basis_functions[:Ip], 1, 32) ./ 1e6)

    mask_psi = fwtsi .* expsi .!= 0.0
    mask_bp = mask_mpi .* expmp2 .!= 0.0
    mask_ip = trues(1)  # Always true
    mask = vcat(mask_psi, mask_bp, mask_ip)

    npca = 8
    y_lsq = reg_solve(A[mask, 1:npca], X[mask, 1], 1e-10)
    X[.!mask, 1] .=0.0
    for ipca in 1:npca
        X[.!mask, 1] += A[.!mask, ipca] * y_lsq[ipca]
    end

    model = NNmodel[:model]
    x_min = NNmodel[:x_min]
    x_max = NNmodel[:x_max]
    y_min = NNmodel[:y_min]
    y_max = NNmodel[:y_max]

    XNN = vcat(X, fcurrt, ecurrt)
    x = EGGO.minmax_normalize(XNN, x_min, x_max)
    y = model(x)
    x = EGGO.EGGO.minmax_unnormalize(x, x_min, x_max)
    y = EGGO.minmax_unnormalize(y, y_min, y_max)  # Convert back to original scale

    # Correct Ip to match experimental Ip
    IpNN = sum(basis_functions[:Ip].*y)
    y .*=  ip /IpNN 

    return y[:,1], XNN#-1*y_lsq
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
     wall::Dict{Symbol,<:Any}, 
     basis_functions::Dict{Symbol,<:Any}, 
     bf1d_itp::Dict{Symbol,<:Any}) where {T<:Real}
    psipla = zeros(129, 129)
    Ip1 = 0.0

    for ipca in 1:32
        psipla .+= y[ipca] .* basis_functions[:psi][:, :, ipca]
        Ip1 += basis_functions[:Ip][ipca] * y[ipca]
    end

    psiext = EGGO.calculate_psiext(fcurrt, ecurrt, green)
    psi = psipla .+ psiext

    r = range(green.rgrid[1], green.rgrid[end], length(green.rgrid))
    z = range(green.zgrid[1], green.zgrid[end], length(green.zgrid))
    rwall = Float64.(wall[:rlim])
    zwall = Float64.(wall[:zlim])
    PSI_itp = Interpolations.scale(Interpolations.interpolate(psi, Interpolations.BSpline(Interpolations.Cubic(Interpolations.Line(Interpolations.OnGrid())))), r, z)

    Raxis, Zaxis = IMAS.find_magnetic_axis(r, z, PSI_itp, 1; rguess=r[65], zguess=z[65])
    psiaxis = PSI_itp(Raxis, Zaxis)

    psi_tom = []
    for (r, z) in zip(r_tom, z_tom)
        push!(psi_tom, PSI_itp(r, z))
    end

    # Here we use Gaussian-like weighting to align separatrix
    T_sep = 80.0
    tolerance = 40.0
    indices = findall(abs.(Te_tom .- T_sep) .< tolerance)

    if length(indices) > 0
        weights = [exp(-((Te - T_sep)^2) / (2 * (tolerance / 2)^2)) for Te in Te_tom[indices]]
        psi_sep = mean(psi_tom[indices], Weights(weights))
    else
        axis2bnd = :increasing
        empty_r = zeros(1)[1:0]
        empty_z = zeros(1)[1:0]
        psi_sep =
            IMAS.find_psi_boundary(r, z, psi, psiaxis, axis2bnd, Raxis, Zaxis, rwall, zwall, empty_r, empty_z;
                PSI_interpolant=PSI_itp, raise_error_on_not_open=false, raise_error_on_not_closed=false).last_closed
    end

    npca1d = min(9, length(psi_tom))

    Ane = zeros(length(psi_tom), npca1d)
    ATe = zeros(length(psi_tom), npca1d)

    for (i, ps) in enumerate(psi_tom)
        for j in 1:npca1d
            if ne_tom[i] > 0
                Ane[i, j] = bf1d_itp[:ne][j]((ps - psiaxis) / (psi_sep - psiaxis))
            end
            if Te_tom[i] > 0
                ATe[i, j] = bf1d_itp[:Te][j]((ps - psiaxis) / (psi_sep - psiaxis))
            end
        end
    end

    c_ne = EGGO.reg_solve(Ane, ne_tom, 1.0)
    c_Te = EGGO.reg_solve(ATe, Te_tom, 1.0)

    psi_cer = []
    for (r, z) in zip(r_cer, z_cer)
        push!(psi_cer, PSI_itp(r, z))
    end

    npca1d = min(9, length(psi_cer))

    Anc = zeros(length(psi_cer), npca1d)
    for (i, ps) in enumerate(psi_cer)
        for j in 1:npca1d
            if ne_tom[i] > 0
                Anc[i, j] = bf1d_itp[:nc][j]((ps - psiaxis) / (psi_sep - psiaxis))
            end

        end
    end
    
    c_nc = EGGO.reg_solve(Anc, nc_cer, 1.0)

    return c_ne, c_Te, c_nc
end