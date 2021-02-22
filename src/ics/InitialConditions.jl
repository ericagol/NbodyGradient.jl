abstract type AbstractInitialConditions end

"""
    Elements{T<:AbstractFloat} <: AbstractInitialConditions

Orbital elements of a binary, and mass of a 'outer' body. See [Examples](@ref) for units and conventions.

# Fields
- `m::T` : Mass of outer body.
- `P::T` : Period [Days].
- `t0::T` : Initial time of transit [Days].
- `ecosϖ` : Eccentricity vector x-component (eccentricity times cosine of the longitude of periastron)
- `esinϖ` : Eccentricity vector y-component (eccentricity times sine of the longitude of periastron)
- `I::T` : Inclination, as measured from sky-plane [Radians].
- `Ω::T` : Longitude of ascending node, as measured from +x-axis [Radians].
- `a::T` : Orbital semi-major axis [AU].
- `e::T` : Eccentricity.
- `ϖ::T` : Longitude of periastron [Radians].
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

"""
    Elements(m,P,t0,ecosϖ,esinϖ,I,Ω)

Main [`Elements`](@ref) constructor. May use keyword arguments, see [Examples](@ref).
"""
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

"""Allows keyword arguments"""
Elements(;m::T=0.0,P::T=0.0,t0::T=0.0,ecosϖ::T=0.0,esinϖ::T=0.0,I::T=0.0,Ω::T=0.0) where T<:Real = Elements(m,P,t0,ecosϖ,esinϖ,I,Ω)

"""Abstract type for initial conditions specifications."""
abstract type InitialConditions{T} end

"""
    ElementsIC{T<:AbstractFloat} <: InitialConditions{T}

Initial conditions, specified by a hierarchy vector and orbital elements.

# Fields
- `elements::Matrix{T}` : Masses and orbital elements.
- `H::Vector{Int64}` : Hierarchy vector.
- `ϵ::Matrix{T}` : Matrix of Jacobi coordinates
- `amat::Matrix{T}` : A matrix from [Hamers and Portegies Zwart 2016](https://doi.org/10.1093/mnras/stw784).
- `nbody::Int64` : Number of bodies.
- `m::Vector{T}` : Masses of bodies.
- `t0::T` : Initial time [Days].
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
    ElementsIC(t0,H,elems...)

Collects `Elements` and produces an `ElementsIC` struct.

# Arguments
- `t0::T` : Initial time [Days].
- `H::Union{Int64,Vector{Int64}}` : Hierarchy specification. See below and [Examples](@ref)
- `elems::Elements{T}...` : A sequence of `Elements{T}`. Elements should be passed in the order they appear in the hierarchy (left to right).
### Hierarchy
`H::Int64`: The system will be given by a 'fully-nested' Keplerian.

`H = 4`
```raw
3        ____|____
        |         |
2    ___|___      d
    |       |
1 __|__     c
 |     |
 a     b
```
`H::Vector{Int64}`: The first elements is the number of bodies. Each subsequent is the number of binaries on a level of the hierarchy.

`H = [4,2,1]`. Two binaries on level 1, one on level 2.
```raw
2    ____|____
    |         |
1 __|__     __|__
 |     |   |     |
 a     b   c     d
```
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
