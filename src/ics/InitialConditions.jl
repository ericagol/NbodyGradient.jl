abstract type AbstractInitialConditions end

"""
    Elements{T<:AbstractFloat} <: AbstractInitialConditions

Orbital elements of a binary, and mass of a 'outer' body. See [Tutorials](@ref) for units and conventions.

# Fields
- `m::T` : Mass of outer body.
- `R::T` : Radius [Jupiter Radii].
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
    R::T
end

"""
    Elements(m,R,P,t0,ecosϖ,esinϖ,I,Ω)

Main [`Elements`](@ref) constructor. May use keyword arguments, see [Tutorials](@ref).
"""
function Elements(m::T,P::T,t0::T,ecosϖ::T,esinϖ::T,I::T,Ω::T,R::T) where T<:Real
    e = sqrt(ecosϖ^2 + esinϖ^2)
    ϖ = atan(esinϖ,ecosϖ)
    Elements(m,P,t0,ecosϖ,esinϖ,I,Ω,0.0,e,ϖ,R)
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
Elements(;m::T=0.0,P::T=0.0,t0::T=0.0,ecosϖ::T=0.0,esinϖ::T=0.0,I::T=0.0,Ω::T=0.0,R::T=0.0) where T<:Real = Elements(m,P,t0,ecosϖ,esinϖ,I,Ω,R)

"""Abstract type for initial conditions specifications."""
abstract type InitialConditions{T} end

"""
    ElementsIC{T<:AbstractFloat} <: InitialConditions{T}

Initial conditions, specified by a hierarchy vector and orbital elements.

# Fields
- `elements::Matrix{T}` : Masses and orbital elements.
- `ϵ::Matrix{T}` : Matrix of Jacobi coordinates
- `amat::Matrix{T}` : 'A' matrix from [Hamers and Portegies Zwart 2016](https://doi.org/10.1093/mnras/stw784).
- `nbody::Int64` : Number of bodies.
- `m::Vector{T}` : Masses of bodies.
- `t0::T` : Initial time [Days].
"""
struct ElementsIC{T<:AbstractFloat} <: InitialConditions{T}
    elements::Matrix{T}
    ϵ::Matrix{T}
    amat::Matrix{T}
    nbody::Int64
    m::Vector{T}
    R::Vector{T}
    t0::T
    der::Bool

    function ElementsIC(t0::T, H::Matrix{T}, elements::Matrix{T}; der::Bool=true) where T<:AbstractFloat
        nbody = size(H)[1]
        m = elements[1:nbody, 1]
        R = elements[1:nbody, 8]
        A = amatrix(H, m)
        # B = amatrix(A, R)
        return new{T}(elements, H, A, nbody, m, R, t0, der)
    end
end
ElementsIC(t0::T, H::Matrix{<:Real}, elements::Matrix{T}) where T<:Real = ElementsIC(t0, T.(H), elements)

"""
    ElementsIC(t0,H,elems)

Collects `Elements` and produces an `ElementsIC` struct.

# Arguments
- `t0::T` : Initial time [Days].
- `H` : Hierarchy specification.
- `elems` : The orbital elements and masses of the system.

------------
There are a number of way to specify the initial conditions. Below we've described the arguments `ElementsIC` takes. Any combination of `H` and `elems` may be used. For a concrete example see [Tutorials](@ref).

# Elements
- `elems...` : A sequence of `Elements{T}`. Elements should be passed in the order they appear in the hierarchy (left to right).
- `elems::Vector{Elements}` : A vector of `Elements`. As above, Elements should be in order.
- `elems::Matrix{T}` : An matrix containing the masses and orbital elements.
- `elems::String` : Name of a file containing the masses and orbital elements.

Each method is simply populating the `ElementsIC.elements` field, which is a `Matrix{T}`.

# Hierarchy
- Number of bodies: `H::Int64`: The system will be given by a 'fully-nested' Keplerian.

`H = 4` corresponds to:
```raw
3        ____|____
        |         |
2    ___|___      d
    |       |
1 __|__     c
 |     |
 a     b
```
- Hierarchy Vector: `H::Vector{Int64}`: The first elements is the number of bodies. Each subsequent is the number of binaries on a level of the hierarchy.

`H = [4,2,1]`. Two binaries on level 1, one on level 2.
```raw
2    ____|____
    |         |
1 __|__     __|__
 |     |   |     |
 a     b   c     d
```
- Full Hierarchy Matrix: `H::Matrix{<:Real}`: Provide the hierarchy matrix, directly.

`H = [-1 1 0 0; 0 0 -1 1; -1 -1 1 1; -1 -1 -1 -1]`. Produces the same system as `H = [4,2,1]`.
"""
function ElementsIC(t0::T, H::Matrix{<:Real}, elems::Elements{T}...) where T<:AbstractFloat
    elements = zeros(T,size(H)[1],8)
    fields = [:m, :P, :t0, :ecosϖ, :esinϖ, :I, :Ω, :R]
    for i in eachindex(elems)
        elements[i,:] .= [getfield(elems[i],f) for f in fields]
    end
    return ElementsIC(t0,T.(H),elements)
end

"""Allows for vector of `Elements` argument."""
ElementsIC(t0::T,H::Matrix{<:Real},elems::Vector{Elements{T}}) where T<:AbstractFloat = ElementsIC(t0,T.(H),elems...)

"""Allow user to pass a file containing orbital elements."""
function ElementsIC(t0::T,H::Matrix{<:Real},elems::String) where T<:AbstractFloat
    elements = T.(readdlm(elems, ',', comments=true)[1:size(H)[1],:])
    return ElementsIC(t0, T.(H), elements)
end

## Let user pass hierarchy vector; dispatch on different elements ##
ElementsIC(t0::T,H::Vector{Int64},elems::Elements{T}...) where T <: AbstractFloat = ElementsIC(t0,hierarchy(H),elems...)
ElementsIC(t0::T,H::Vector{Int64},elems::Vector{Elements{T}}) where T<:AbstractFloat = ElementsIC(t0,hierarchy(H),elems...)
ElementsIC(t0::T,H::Vector{Int64},elems::Matrix{T}) where T<:AbstractFloat = ElementsIC(t0,hierarchy(H),elems)
ElementsIC(t0::T,H::Vector{Int64},elems::String) where T<:AbstractFloat = ElementsIC(t0,hierarchy(H),elems)

## Let user specify only the number of bodies; the hierarchy is filled in as fully-nested; dispatch on different elements ##
ElementsIC(t0::T,H::Int64,elems::Elements{T}...) where T<:AbstractFloat = ElementsIC(t0,[H, ones(Int64,H-1)...],elems...)
ElementsIC(t0::T,H::Int64,elems::Vector{Elements{T}}) where T<:AbstractFloat = ElementsIC(t0,[H, ones(Int64,H-1)...],elems...)
ElementsIC(t0::T,H::Int64,elems::Matrix{T}) where T<:AbstractFloat = ElementsIC(t0,[H, ones(Int64,H-1)...],elems)
ElementsIC(t0::T,H::Int64,elems::String) where T<:AbstractFloat = ElementsIC(t0,[H, ones(Int64, H-1)...],elems)

"""Shows the elements array."""
Base.show(io::IO,::MIME"text/plain",ic::ElementsIC{T}) where {T} = begin
println(io,"ElementsIC{$T}\nOrbital Elements: "); show(io,"text/plain",ic.elements); end;

"""
    CartesianIC{T<:AbstractFloat} <: InitialConditions{T}

Initial conditions, specified by the Cartesian coordinates and masses of each body.

# Fields
- `x::Matrix{T}` : Positions of each body [dimension, body].
- `v::Matrix{T}` : Velocities of each body [dimension, body].
- `m::Vector{T}` : masses of each body.
- `nbody::Int64` : Number of bodies in system.
- `t0::T` : Initial time.
"""
struct CartesianIC{T<:AbstractFloat} <: InitialConditions{T}
    x::Matrix{T}
    v::Matrix{T}
    m::Vector{T}
    nbody::Int64
    t0::T
end

"""Allow user to pass matrix of row-vectors for `CartesianIC`."""
function CartesianIC(t0::T, N::Int64, coords::Matrix{T}) where T<:AbstractFloat
    m = coords[1:N,1]
    x = permutedims(coords[1:N,2:4])
    v = permutedims(coords[1:N,5:7])
    return CartesianIC(x,v,m,N,t0)
end

"""Allow input of Cartesian coordinate file."""
function CartesianIC(t0::T, N::Int64, coordinateFile::String) where T <: AbstractFloat
    coords = readdlm(coordinateFile,',')
    m = coords[1:N,1]
    x = permutedims(coords[1:N,2:4])
    v = permutedims(coords[1:N,5:7])
    return CartesianIC(x,v,m,N,t0)
end

# Include ics source files
const ics = ["kepler","kepler_init","setup_hierarchy","init_nbody"]
for i in ics; include("$(i).jl"); end
