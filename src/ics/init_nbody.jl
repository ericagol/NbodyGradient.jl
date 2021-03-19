"""
    init_nbody(ic,t0)

Converts initial orbital elements into Cartesian coordinates.

# Arguments
- `ic::ElementsIC{T<:Real}`: Initial conditions of the system. See [`InitialConditions`](@ref).
# Outputs
- `x::Array{<:Real,2}`: Cartesian positions of each body.
- `v::Array{<:Real,2}`: Cartesian velocites of each body.
- `jac_init::Array{<:Real,2}`: Derivatives of A-matrix and x,v with respect to the masses of each object.
"""
function init_nbody(ic::ElementsIC{T}) where T <: AbstractFloat

    r, rdot, jac_init = kepcalc(ic)

    Ainv = inv(ic.amat)

    # Cartesian coordinates
    x = zeros(T,NDIM,ic.nbody)
    x = permutedims(*(Ainv,r))

    v = zeros(T,NDIM,ic.nbody)
    v = permutedims(*(Ainv,rdot))

    return x,v,jac_init
end

"""Return the cartesian coordinates."""
function init_nbody(ic::CartesianIC{T}) where T <: AbstractFloat
    x = copy(ic.x)
    v = copy(ic.v)
    n = size(x)[2]
    jac_init = Matrix{T}(I,7*n,7*n)
    return x,v,jac_init
end

"""
    kepcalc(ic,t0)

Computes Kepler's problem for each pair of bodies in the system.

# Arguments
- `ic::ElementsIC{T<:Real}`: Initial conditions structure.
# Outputs
- `rkepler::Array{T<:Real,2}`: Matrix of initial position vectors for each keplerian.
- `rdotkepler::Array{T<:Real,2}`: Matrix of initial velocity vectors for each keplerian.
- `jac_init::Array{T<:Real,2}`: Derivatives of the A-matrix and cartesian positions and velocities with respect to the masses of each object.
"""
function kepcalc(ic::ElementsIC{T}) where T<:AbstractFloat
    n = ic.nbody
    rkepler = zeros(T,n,NDIM)
    rdotkepler = zeros(T,n,NDIM)
    if ic.der
        jac_kepler = zeros(T,6*n,7*n)
        jac_21 = zeros(T,7,7)
    end
    # Compute Kepler's problem for each binary
    i = 1; b = 0
    while i < ic.nbody
        ind = Bool.(abs.(ic.ϵ[i,:]))
        μ = sum(ic.m[ind])

        # Check for a new binary.
        if first(ind) == zero(T)
            b += 1
        end

        # Solve Kepler's problem
        if ic.der
            r,rdot = kepler_init(ic.t0,μ,ic.elements[i+1+b,2:7],jac_21)
        else
            r,rdot = kepler_init(ic.t0,μ,ic.elements[i+1+b,2:7])
        end
        rkepler[i,:] .= r
        rdotkepler[i,:] .= rdot

        if ic.der
            for j=1:6, k=1:6
                jac_kepler[(i-1)*6+j,i*7+k] = jac_21[j,k]
            end
            for j in 1:n
                if ic.ϵ[i,j] != 0
                    for k = 1:6
                        jac_kepler[(i-1)*6+k,j*7] = jac_21[k,7]
                    end
                end
            end
        end
        # Check if last binary was a new branch.
        if b > 0
            b -= 2
        elseif b < 0
            b = 0
            #i += 1
        end
        i += 1
    end
    if ic.der
        jac_init = d_dm(ic,rkepler,rdotkepler,jac_kepler)
        return rkepler,rdotkepler,jac_init
    else
        return rkepler,rdotkepler,zeros(T,0,0)
    end
end

"""
    d_dm(ic,rkepler,rdotkepler,jac_kepler)

Computes derivatives of A-matrix, position, and velocity with respect to the masses of each object.

# Arguments
- `ic::IC`: Initial conditions structure.
- `rkepler::Array{<:Real,2}`: Position vectors for each Keplerian.
- `rdotkepler::Array{<:Real,2}`: Velocity vectors for each Keplerian.
- `jac_kepler::Array{<:Real,2}`: Keplerian Jacobian matrix.
# Outputs
- `jac_init::Array{<:Real,2}`: Derivatives of the A-matrix and cartesian positions and velocities with respect to the masses of each object.
"""
function d_dm(ic::ElementsIC{T},rkepler::Array{T,2},rdotkepler::Array{T,2},jac_kepler::Array{T,2}) where T <: AbstractFloat

    N = ic.nbody
    m = ic.m
    ϵ = ic.ϵ
    jac_init = zeros(T,7*N,7*N)
    dAdm = zeros(T,N,N,N)
    dxdm = zeros(T,NDIM,N)
    dvdm = zeros(T,NDIM,N)

    # Differentiate A matrix wrt the mass of each body
    for k in 1:N, i in 1:N, j in 1:N
        dAdm[i,j,k] = ((δ_(k,j)*ϵ[i,j])/Σm(m,i,j,ϵ)) -
        ((δ_(ϵ[i,j],ϵ[i,k]))*ϵ[i,j]*m[j]/(Σm(m,i,j,ϵ)^2))
    end

    # Calculate inverse of dAdm
    Ainv = inv(ic.amat)
    dAinvdm = zeros(T,N,N,N)
    for k in 1:N
        dAinvdm[:,:,k] .= -Ainv * dAdm[:,:,k] * Ainv
    end

    # Fill in jac_init array
    for i in 1:N
        for k in 1:N
            for j in 1:3, l in 1:7*N
                jac_init[(i-1)*7+j,l] += Ainv[i,k]*jac_kepler[(k-1)*6+j,l]
                jac_init[(i-1)*7+3+j,l] += Ainv[i,k]*jac_kepler[(k-1)*6+3+j,l]
            end
        end

        # Derivatives of cartesian coordinates wrt masses
        for k in 1:N
            dxdm = transpose(dAinvdm[:,:,k]*rkepler)
            dvdm = transpose(dAinvdm[:,:,k]*rdotkepler)
            jac_init[(i-1)*7+1:(i-1)*7+3,k*7] += dxdm[1:3,i]
            jac_init[(i-1)*7+4:(i-1)*7+6,k*7] += dvdm[1:3,i]
        end
        jac_init[i*7,i*7] = 1.0
    end
    return jac_init
end

"""
    amatrix(elements,ϵ,m)

Creates the A matrix presented in Hamers & Portegies Zwart 2016 (HPZ16).

# Arguments
- `elements::Array{<:Real,2}`: Array of masses and orbital elements. See [`IC`](@ref)
- `ϵ::Array{<:Real,2}`: Epsilon matrix. See [`IC.elements`]
- `m::Array{<:Real,2}`: Array of masses of each body.
# Outputs
- `A::Array{<:Real,2}`: A matrix.
"""
function amatrix(ϵ::Array{T,2},m::Array{T,1}) where T<:AbstractFloat
    A = zeros(T,size(ϵ)) # Empty A matrix
    N = length(ϵ[:,1]) # Number of bodies in system

    for i in 1:N, j in 1:N
        A[i,j] = (ϵ[i,j]*m[j])/(Σm(m,i,j,ϵ))
    end
    return A
end

function amatrix(ic::ElementsIC{T}) where T <: Real
    ic.amat .= amatrix(ic.ϵ,ic.m)
end

"""
    Σm(masses,i,j,ϵ)

Sums masses in current Keplerian.

# Arguments
- `masses::Array{<:Real,2}`: Array of masses in system.
- `i::Integer`: Summation index.
- `j::Integer`: Summation index.
- `ϵ::Array{<:Real,2}`: Epsilon matrix.
# Outputs
- `m<:Real`: Sum of the masses.
"""
function Σm(masses::Array{T,1},i::Integer,j::Integer,
            ϵ::Array{T,2}) where T <: AbstractFloat
    m = 0.0
    for l in 1:length(masses)
        m += masses[l]*δ_(ϵ[i,j],ϵ[i,l])
    end
    return m
end

"""
    δ_(i,j)

An NxN Kronecker Delta function.

# Arguements
- `i<:Real`: First arguement.
- `j<:Real`: Second arguement.
# Outputs
- `::Bool`
"""
function δ_(i::T,j::T) where T <: Real
    if i == j
        return 1.0
    else
        return 0.0
    end
end

