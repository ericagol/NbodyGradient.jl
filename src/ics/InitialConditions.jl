abstract type AbstractInitialConditions end
"""

Holds orbital elements of a single body.
"""
struct Elements{T<:AbstractFloat} <: AbstractInitialConditions
    m::T
    P::T
    t0::T
    ecosϖ::T
    esinϖ::T
    I::T
    Ω::T
    a::T
    e::T
    ϖ::T
end

"""Main `Elements` constructor.`"""
function Elements(m::T,P::T,t0::T,ecosϖ::T,esinϖ::T,I::T,Ω::T) where T<:Real 
    e = sqrt(ecosϖ^2 + esinϖ^2)
    ϖ = atan(esinϖ,ecosϖ)
    Elements(m,P,t0,ecosϖ,esinϖ,I,Ω,0.0,e,ϖ)
end

function Base.show(io::IO, ::MIME"text/plain" ,elems::Elements{T}) where T <: Real
    fields = fieldnames(typeof(elems))
    vals = Dict([fn => getfield(elems,fn) for fn in fields])
    println(io, "Elements{$T}")
    for key in keys(vals)
        println(io,key,": ",vals[key])
    end
    if elems.a == 0.0
        println(io, "Semi-major axis: undefined")
    end
    return
end

"""Allow keywargs"""
Elements(;m::T=0.0,P::T=0.0,t0::T=0.0,ecosϖ::T=0.0,esinϖ::T=0.0,I::T=0.0,Ω::T=0.0) where T<:Real = Elements(m,P,t0,ecosϖ,esinϖ,I,Ω)

abstract type InitialConditions{T} end
"""

Holds relevant initial conditions arrays. Uses orbital elements.
"""
struct ElementsIC{T<:AbstractFloat,V<:Vector{T},M<:Matrix{T}} <: InitialConditions{T}
    elements::M
    H::Vector{Int64}
    ϵ::M
    amat::M
    nbody::Int64
    m::V
    t0::T
    der::Bool

    function ElementsIC(t0::T,H::Union{Array{Int64,1},Int64},elems::Union{String,Array{T,2}};der::Bool=true) where T <: AbstractFloat
        # Check if only number of bodies was passed. Assumes fully nested.
        if typeof(H) == Int64; H::Vector{Int64} = [H,ones(H-1)...]; end
        nbody = H[1]
        ϵ = convert(Array{T},hierarchy(H))
        if typeof(elems) == String
            elements = convert(Array{T},readdlm(elems,',',comments=true)[1:nbody,:])
        else
            elements = elems
        end
        m = elements[1:nbody,1]
        amat = amatrix(ϵ,m)
        return new{T,Vector{T},Matrix{T}}(elements,H,ϵ,amat,nbody,m,t0,der);
    end
end

"""

Collects `Elements` and produces an `ElementsIC` struct.
"""
function ElementsIC(t0::T,H::Union{Int64,Vector{Int64}},elems::Elements{T}...) where T <: AbstractFloat
    if H isa Int64; H = [H,ones(Int64,H-1)...]; end
    elements = zeros(T,H[1],7)
    fields = setdiff(fieldnames(Elements),[:a,:e,:ϖ])
    for i in eachindex(elems)
        elements[i,:] .= [getfield(elems[i],f) for f in fields]
    end
    return ElementsIC(t0,H,elements) 
end

"""Allows for array of `Elements` argument."""
ElementsIC(t0::T,H::Union{Int64,Vector{Int64}},elems::Array{Elements{T},1}) where T<:AbstractFloat = ElementsIC(t0,H,elems...)

"""Shows the elements array."""
Base.show(io::IO,::MIME"text/plain",ic::ElementsIC{T}) where {T} = begin
println(io,"ElementsIC{$T}\nOrbital Elements: "); show(io,"text/plain",ic.elements); end;

# Include ics source files
const ics = ["kepler","kepler_init","setup_hierarchy","init_nbody"]
for i in ics; include("$(i).jl"); end
