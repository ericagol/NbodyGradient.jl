abstract type AbstractInitialConditions end

"""
    Elements{T<:AbstractFloat} <: AbstractInitialConditions

Orbital elements of a binary, and mass of a 'outer' body. See [Tutorials](@ref) for units and conventions.

# Fields
- `m::T` : Mass of outer body.
- `P::T` : Period [Days].
- `t0::T` : Initial time of transit [Days].
- `ecosω::T` : Eccentricity vector x-component (eccentricity times cosine of the argument of periastron)
- `esinω::T` : Eccentricity vector y-component (eccentricity times sine of the argument of periastron)
- `I::T` : Inclination, as measured from sky-plane [Radians].
- `Ω::T` : Longitude of ascending node, as measured from +x-axis [Radians].
- `a::T` : Orbital semi-major axis [AU].
- `e::T` : Eccentricity.
- `ω::T` : Argument of periastron [Radians].
- `tp::T` : Time of periastron passage [Days].
"""
struct Elements{T<:AbstractFloat} <: AbstractInitialConditions
    m::T
    P::T
    t0::T
    ecosω::T
    esinω::T
    I::T
    Ω::T
    a::T
    e::T
    ω::T
    tp::T
end

"""
    Elements(m,P,t0,ecosω,esinω,I,Ω)

Main [`Elements`](@ref) constructor. May use keyword arguments, see [Tutorials](@ref).
"""
function Elements(m::T,P::T,t0::T,ecosω::T,esinω::T,I::T,Ω::T) where T<:Real
    e = sqrt(ecosω^2 + esinω^2)
    ω = atan(esinω,ecosω)
    Elements(m,P,t0,ecosω,esinω,I,Ω,0.0,e,ω,0.0)
end

function Base.show(io::IO, ::MIME"text/plain", elems::Elements{T}) where T <: Real
    fields = fieldnames(typeof(elems))
    vals = [fn => getfield(elems,fn) for fn in fields]
    println(io, "Elements{$T}")
    for pair in vals
        println(io,first(pair),": ",last(pair))
    end
    if elems.a == 0.0
        println(io, "Orbital semi-major axis: undefined")
    end
    return
end

iszeroall(x...) = all(iszero.(x))
isnanall(x...) = all(isnan.(x))
replacenan(x) = isnan(x) ? zero(x) : x

"""Allows keyword arguments"""
function Elements(;m::Real,P::Real=NaN,t0::Real=NaN,ecosω::Real=NaN,esinω::Real=NaN,I::Real=NaN,Ω::Real=NaN,a::Real=NaN,ω::Real=NaN,e::Real=NaN,tp::Real=NaN)
    # Promote everything
    m,P,t0,ecosω,esinω,I,Ω,a,ω,e,tp = promote(m,P,t0,ecosω,esinω,I,Ω,a,ω,e,tp)
    T = typeof(P)

    # Assume if only mass is set, we want the elements for the star (or central body in binary)
    if isnanall(P,t0,ecosω,esinω,I,Ω,a,ω,e,tp); return Elements(m, zeros(T, length(fieldnames(Elements))-1)...); end
    if isnanall(P,a); throw(ArgumentError("Must specify either P or a.")); end
    if P > zero(T) && a > zero(T); throw(ArgumentError("Only one of P (period) or a (semi-major axis) can be defined")); end
    if P < zero(T); throw(ArgumentError("Period must be positive")); end
    if a < zero(T); throw(ArgumentError("Orbital semi-major axis must be positive")); end
    if !(zero(T) <= e < one(T)) && !isnan(e); throw(ArgumentError("Eccentricity must be in [0,1), e=$(e)")); end
    if !(isnan(ecosω) && isnan(esinω)) && !isnan(e); error("Must specify either e and ecosω/esinω, but not both."); end
    if !(isnan(tp) || isnan(t0)); error("Must specify either initial transit time or time of periastron passage, but not both."); end

    ω = replacenan(ω)
    if isnanall(e,ecosω,esinω); e = ecosω = esinω = zero(T); end
    if isnan(e)
        ecosω = replacenan(ecosω)
        esinω = replacenan(esinω)
        e = sqrt(ecosω*ecosω + esinω*esinω)
        @assert zero(T) <= e < one(T) "Eccentricity must be in [0,1)"
        ω = atan(esinω, ecosω)
    else
        esinω, ecosω = e.*sincos(ω)
        esinω = abs(esinω) < eps(T) ? zero(T) : esinω
        ecosω = abs(ecosω) < eps(T) ? zero(T) : ecosω
    end

    t0 = replacenan(t0)
    n = 2π/P
    if isnan(tp) && !iszero(e)
        tp = (t0 - sqrt(1.0-e*e)*ecosω/(n*(1.0-esinω)) - (2.0/n)*atan(sqrt(1.0-e)*(esinω+ecosω+e), sqrt(1.0+e)*(esinω-ecosω-e))) % P
    elseif isnan(tp)
        θ = π/2 + ω
        tp = θ / n
    end

    a = replacenan(a)
    I = replacenan(I)
    Ω = replacenan(Ω)

    return Elements(m,P,t0,ecosω,esinω,I,Ω,a,e,ω,tp)
end

# Handle the deprecated fields
function Base.getproperty(obj::Elements, sym::Symbol)
    if (sym === :ecosϖ)
        Base.depwarn("ecosϖ (\\varpi) will be removed, use ecosω (\\omega).", :getproperty, force=true)
        return obj.ecosω
    end
    if (sym === :esinϖ)
        Base.depwarn("esinϖ (\\varpi) will be removed, use esinω (\\omega).", :getproperty, force=true)
        return obj.esinω
    end
    if (sym === :ϖ)
        Base.depwarn("ϖ (\\varpi) will be removed, use ω (\\omega).", :getproperty, force=true)
        return obj.ω
    end
    return getfield(obj, sym)
end

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
    t0::T
    der::Bool

    function ElementsIC(t0::T, H::Matrix{T}, elements::Matrix{T}; der::Bool=true) where T<:AbstractFloat
        nbody = size(H)[1]
        m = elements[1:nbody, 1]
        A = amatrix(H, m)

        # Allow user to input more orbital elements than used, but only use N
        if size(elements) != (nbody, 7)
            elements = copy(elements[1:nbody,:])
        end
        return new{T}(copy(elements), copy(H), A, nbody, m, t0, der)
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
    elements = zeros(T,size(H)[1],7)
    fields = [:m, :P, :t0, :ecosω, :esinω, :I, :Ω]
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
const ics = ["kepler","kepler_init","setup_hierarchy","init_nbody","defaults"]
for i in ics; include("$(i).jl"); end
