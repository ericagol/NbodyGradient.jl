# Collection of functions to convert cartesian coordinate to orbital elements
using LinearAlgebra, NbodyGradient
import NbodyGradient: InitialConditions

## Utils ##

const GNEWT = 39.4845 / (365.242 * 365.242) # AU^3 Msol^-1 Day^-2

# h vector
function hvec(x::V, v::V) where {V<:Vector{<:Real}}
    hx, hy, hz = cross(x, v)
    hz >= 0.0 ? hy *= -1 : hx *= -1
    return [hx, hy, hz]
end

# R dot
Rdotmag(R::T, V::T, x::Vector{T}, v::Vector{T}, h::T) where {T<:AbstractFloat} = sign(dot(x, v)) * sqrt(V^2 - (h / R)^2)

## Elements ##

# Semi-major axis
calc_a(R::T, V::T, Gm::T) where {T<:AbstractFloat} = 1.0 / ((2.0 / R) - (V * V) / (Gm))

# Eccentricity
calc_e(h::T, Gm::T, a::T) where {T<:AbstractFloat} = sqrt(1.0 - (h * h / (Gm * a)))

# Inclination
calc_I(hz::T, h::T) where {T<:AbstractFloat} = acos(hz / h)

# Long. of Ascending Node
function calc_Ω(h::T, hx::T, hy::T, sI::T) where {T<:AbstractFloat}
    sΩ = hx / (h * sI)
    cΩ = hy / (h * sI)
    return atan(sΩ, cΩ)
end

# Long. of Pericenter
function calc_ϖ(x::Vector{T}, R::T, Ṙ::T, Ω::T, I::T, e::T, h::T, a::T) where {T<:AbstractFloat}
    X, _, Z = x
    sΩ, cΩ = sincos(Ω)
    sI, cI = sincos(I)

    # ω + f
    swpf = Z / (R * sI)
    cwpf = ((X / R) + sΩ * swpf * cI) / cΩ
    wpf = atan(swpf, cwpf)

    # true anomaly, f
    sf = a * Ṙ * (1.0 - e^2) / (h * e)
    cf = (a * (1.0 - e^2) / R - 1.0) / e
    f = atan(sf, cf)

    # ϖ: Ω + ω
    return Ω + (wpf - f - π)
end

# Time of pericenter passage
function calc_t0(R::T, a::T, e::T, Gm::T, t::T) where {T<:AbstractFloat}
    E = acos((1.0 - (R / a)) / e)
    return t - (E - e * sin(E)) / sqrt(Gm / (a * a * a))
end

# Period
calc_P(a::T, Gm::T) where {T<:AbstractFloat} = (2.0 * π) * sqrt(a * a * a / Gm)

# Get relative hierarchy positions 
function calc_X(s::State{T}, ic::InitialConditions{T}) where {T<:AbstractFloat}
    X = zeros(3, s.n - 1)
    V = zeros(3, s.n - 1)
    for i in 1:s.n-1
        for k in 1:s.n
            X[:, i] .+= ic.amat[i, k] .* s.x[:, k]
            V[:, i] .+= ic.amat[i, k] .* s.v[:, k]
        end
    end
    return X, V
end

function gmass(m::Vector{T}, ic::InitialConditions{T}) where {T<:AbstractFloat}
    N = length(m)
    M = zeros(N - 1)
    for i in 1:N-1
        for j in 1:N
            M[i] += abs(ic.ϵ[i, j]) * m[j]
        end
    end
    return GNEWT .* M
end

"""

Converts cartesian coordinates to orbital elements. (t0 : time of transit)
"""
function obtain_orbital_elements(s::State{T}, ic::InitialConditions{T}) where {T<:AbstractFloat}
    X::Matrix{T}, Xdot::Matrix{T} = calc_X(s, ic)
    Gm = gmass(s.m, ic)
    R = [norm(X[:, i]) for i in 1:s.n-1]
    V = [norm(Xdot[:, i]) for i in 1:s.n-1]
    h_vec::Matrix{T} = hcat([hvec(X[:, i], Xdot[:, i]) for i in 1:s.n-1]...)
    hx, hy, hz = h_vec[1, :], h_vec[2, :], h_vec[3, :]
    h = norm.([hx[i], hy[i], hz[i]] for i in 1:s.n-1)
    Rdot = [Rdotmag(R[i], V[i], X[:, i], Xdot[:, i], h[i]) for i in 1:s.n-1]

    a = calc_a.(R, V, Gm)
    e = calc_e.(h, Gm, a)
    I = calc_I.(hz, h)
    Ω = calc_Ω.(h, hx, hy, sin.(I))
    ϖ = [calc_ϖ(X[:, i], R[i], Rdot[i], Ω[i], I[i], e[i], h[i], a[i]) for i in 1:s.n-1]
    #t0 = calc_t0.(R,a,e,Gm,s.t) # Our elements use the initial transit time
    t0 = ic.elements[2:end, 3]
    P = calc_P.(a, Gm)
    elements = zeros(size(ic.elements))
    elements[1] = ic.elements[1]
    elements[2:end, 1:end-1] = hcat(s.m[2:end], P, t0, e .* cos.(ϖ), e .* sin.(ϖ), I, Ω)
    elements[:, end] = ic.elements[:, end] # Assume radius does not change
    return elements
end
