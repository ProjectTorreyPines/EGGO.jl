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
end #where {T<:Real}