
# Freeze-frame structures
abstract type FreezeFrameOutput{T} <: AbstractOutput{T} end

"""

Holds the state of the system (and derivatives) at the
  pre-specified times.
"""
struct FreezeFrame{T<:AbstractFloat} <: FreezeFrameOutput{T}
    tt::Matrix{T}
    dtdq0::Array{T,4}
    dtdelements::Array{T,4}
    count::Vector{Int64}
    ntt::Int64
    ti::Int64
    occs::Vector{Int64}
    dtdq::Array{T,3}
    gsave::Vector{T}
    s_prior::State{T}
    s_transit::State{T}
    tout::Vector{T}
    snapshots::Vector{State{T}}
end

"""
Initialize the FreezeFrame structure
"""
function FreezeFrame(tmax,ic::ElementsIC{T},ti::Int64=1,toutput::Vector{T}) where T<:AbstractFloat
    n = ic.nbody
    ind = isfinite.(tmax./ic.elements[:,2])
    ntt = maximum(ceil.(Int64,abs.(tmax./ic.elements[ind,2])).+3)
    tt = zeros(T,n,ntt)
    dtdq0 = zeros(T,n,ntt,7,n)
    dtdelements = zeros(T,n,ntt,7,n)
    count = zeros(Int64,n)
    occs = setdiff(collect(1:n),ti)
    dtdq = zeros(T,1,7,n)
    gsave = zeros(T,n)
    s_prior = State(ic)
    s_transit = State(ic)
    tout = toutput
    snapshots = Array{State,1}(undef,length(tout))
    return FreezeFrame(tt,dtdq0,dtdelements,count,ntt,ti,occs,dtdq,gsave,s_prior,s_transit,tout,snapshots)
end

