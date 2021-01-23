# TTV structures
"""

Holds the transit times and derivatives.
"""
struct TransitTiming{T<:AbstractFloat} <: AbstractOutput{T}
    tt::Matrix{T}
    dtdq0::Array{T,4}
    dtdelements::Array{T,4}
    count::Vector{Int64}
    ntt::Int64
    ti::Int64
    occs::Vector{Int64}
end

function TransitTiming(tmax,ic::ElementsIC{T},ti::Int64=1) where T<:AbstractFloat
    n = ic.nbody
    ind = isfinite.(tmax./ic.elements[:,2])
    ntt = maximum(ceil.(Int64,abs.(tmax./ic.elements[ind,2])).+3)
    tt = zeros(T,n,ntt)
    dtdq0 = zeros(T,n,ntt,7,n)
    dtdelements = zeros(T,n,ntt,7,n)
    count = zeros(Int64,n)
    occs = setdiff(collect(1:n),ti)
    return TransitTiming(tt,dtdq0,dtdelements,count,ntt,ti,occs)
end

"""

Structure for transit timing, impact parameter, and sky velocity
"""
struct TransitParameters{T<:AbstractFloat} <: AbstractOutput{T}
    ttbv::Array{T,3}
    dtbvdq0::Array{T,5}
    dtbvdelements::Array{T,5}
    count::Vector{Int64}
    ntt::Int64
    ti::Int64
    occs::Vector{Int64}
end

function TransitParameters(tmax,ic::ElementsIC{T},ti::Int64=1) where T<:AbstractFloat
    n = ic.nbody
    ind = isfinite.(tmax./ic.elements[:,2])
    ntt = maximum(ceil.(Int64,abs.(tmax./ic.elements[ind,2])).+3)
    ttbv = zeros(T,3,n,ntt)
    dtbvdq0 = zeros(T,3,n,ntt,7,n)
    dtbvdelements = zeros(T,3,n,ntt,7,n)
    count = zeros(Int64,n)
    occs = setdiff(collect(1:n),ti)
    return TransitParameters(ttbv,dtbvdq0,dtbvdelements,count,ntt,ti,occs)
end

function Base.iterate(tt::AbstractOutput,state=1)
    fields = setdiff(fieldnames(typeof(tt)),[:times,:ti,:occs])
    if state > length(fields)
        return nothing
    end
    return (getfield(tt,fields[state]), state+1)
end

function zero_out!(tt::AbstractOutput{T}) where T
    for i in tt
        if ~(typeof(i) <: Integer)
            i .= zero(T)
        end
    end
end

"""

Integrator method for outputing `TransitTiming` or `TransitParameters`.
"""
function (i::Integrator)(s::State{T},tt::AbstractOutput;grad::Bool=true) where T<:AbstractFloat
    #s2 = zero(T) # For compensated summation

    # Preallocate struct of arrays for derivatives (and pair)
    pair = zeros(Bool,s.n,s.n)

    # Run integrator and calculate transit times, with derivatives.
    rstar::T = 1e12 # Need to pass this in.
    calc_tt!(s,i,tt,rstar,pair;grad=grad)
    if grad
        calc_dtdelements!(s,tt)
    end
    return
end

"""

Integrator method for outputing `TransitParameters`.
"""
function (i::Integrator)(s::State{T},ttbv::TransitParameters;grad::Bool=true) where T<:AbstractFloat
    #s2 = zero(T) # For compensated summation

    # Preallocate struct of arrays for derivatives (and pair)
    pair = zeros(Bool,s.n,s.n)

    # Run integrator and calculate transit times, with derivatives.
    rstar::T = 1e12 # Need to pass this in.
    calc_tt!(s,i,ttbv,rstar,pair;grad=grad)
    if grad
        calc_dtdelements!(s,ttbv)
    end
    return
end

# Includes for source
files = ["ttv.jl","ttv_no_grad.jl"]
include.(files)