struct BasisFunctions{T<:Float64}
    Ip::Vector{T}
    psi_loop::Matrix{T}
    bp_probe::Matrix{T}
    Jt::Array{T,3}
    psi::Array{T,3}
end

struct BasisFunctions1D{T<:Float64}
    psi::Vector{T}
    pp::Matrix{T}
    ffp::Matrix{T}
    ne::Union{Matrix{T},Missing}
    Te::Union{Matrix{T},Missing}
    nc::Union{Matrix{T},Missing}
end

struct BasisFunctions1Dinterp
    pp::Vector{Any}
    ffp::Vector{Any}
    ne::Union{Vector{Any},Missing}
    Te::Union{Vector{Any},Missing}
    nc::Union{Vector{Any},Missing}
end

struct GreenFunctionTables{T<:Float64}
    nw::Int
    nh::Int
    nfsum::Int
    nvsum::Int
    ngam::Int
    nesum::Int
    ivesel::Int
    imse::Int
    iecoil::Int
    nsilop::Int
    magpr2::Int
    rgrid::Vector{T}
    zgrid::Vector{T}
    rsilec::Matrix{T}
    gmp2vs::Matrix{T}
    rsilfc::Matrix{T}
    gsilvs::Matrix{T}
    rmp2fc::Matrix{T}
    rmp2ec::Matrix{T}
    ggridfc::Matrix{T}
    gridec::Matrix{T}
    ggridfc_itp::Vector{Any}
    gridec_itp::Vector{Any}
end

struct Wall
    rlim::Vector{Float64}
    zlim::Vector{Float64}
end
