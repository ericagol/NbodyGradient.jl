"""
    init_nbody(init,t0)

Converts initial orbital elements into Cartesian coordinates.

# Arguments
- `init::ElementsIC{T<:Real}`: Initial conditions of the system. See [`InitialConditions`](@ref).
# Outputs
- `x::Array{<:Real,2}`: Cartesian positions of each body.
- `v::Array{<:Real,2}`: Cartesian velocites of each body.
- `jac_init::Array{<:Real,2}`: Derivatives of A-matrix and x,v with respect to the masses of each object.
"""
function init_nbody(init::ElementsIC{T}) where T <: AbstractFloat

    r, rdot, jac_init = kepcalc(init)

    Ainv = inv(init.amat)

    # Cartesian coordinates
    x = zeros(T,NDIM,init.nbody)
    x = Array(transpose(*(Ainv,r)))
    
    v = zeros(T,NDIM,init.nbody)
    v = Array(transpose(*(Ainv,rdot)))

    return x,v,jac_init
end

"""
    kepcalc(init,t0)

Computes Kepler's problem for each pair of bodies in the system.

# Arguments
- `init::ElementsIC{T<:Real}`: Initial conditions structure.
# Outputs
- `rkepler::Array{T<:Real,2}`: Matrix of initial position vectors for each keplerian.
- `rdotkepler::Array{T<:Real,2}`: Matrix of initial velocity vectors for each keplerian.
- `jac_init::Array{T<:Real,2}`: Derivatives of the A-matrix and cartesian positions and velocities with respect to the masses of each object.
"""
function kepcalc(init::ElementsIC{T}) where T <: AbstractFloat

    # Kepler position/velocity arrays (body,pos/vel)
    rkepler = zeros(T,init.nbody,NDIM)
    rdotkepler = zeros(T,init.nbody,NDIM)
    if init.der
        jac_kepler = zeros(T,6*init.nbody,7*init.nbody)
    end
    # Compute Kepler's problem
    for i in 1:init.nbody

        ind1 = findall(isequal(-1.0),init.ϵ[i,:])
        ind2 = findall(isequal(1.0),init.ϵ[i,:])
        m1 = sum(init.m[ind1])
        m2 = sum(init.m[ind2])

        if i < init.nbody
            if init.der
                jac_21 = zeros(T,7,7)
                r, rdot = kepler_init(init.t0,m1+m2,init.elements[i+1,2:7],jac_21)
            else
                r, rdot = kepler_init(init.t0,m1+m2,init.elements[i+1,2:7])
            end
            for j = 1:NDIM
                rkepler[i,j] = r[j]
                rdotkepler[i,j] = rdot[j]
            end
            if init.der
                # Save Keplerian Jacobian. First, positions/velocity vs. elements
                for j=1:6, k=1:6
                    jac_kepler[(i-1)*6+j,i*7+k] = jac_21[j,k]
                end
                # Then mass derivatives
        		for j=1:init.nbody
        	    	if init.ϵ[i,j] != 0
        	        	for k = 1:6
        		    	jac_kepler[(i-1)*6+k,j*7] = jac_21[k,7]
        				end
        	    	end
        		end
            end
        end
    end
    if init.der
        jac_init = d_dm(init,rkepler,rdotkepler,jac_kepler)
        return rkepler,rdotkepler,jac_init
    else
        return rkepler,rdotkepler,zeros(T,0,0)
    end
end

"""
    d_dm(init,rkepler,rdotkepler,jac_kepler)

Computes derivatives of A-matrix, position, and velocity with respect to the masses of each object.

# Arguments
- `init::IC`: Initial conditions structure.
- `rkepler::Array{<:Real,2}`: Position vectors for each Keplerian.
- `rdotkepler::Array{<:Real,2}`: Velocity vectors for each Keplerian.
- `jac_kepler::Array{<:Real,2}`: Keplerian Jacobian matrix.
# Outputs
- `jac_init::Array{<:Real,2}`: Derivatives of the A-matrix and cartesian positions and velocities with respect to the masses of each object.
"""
function d_dm(init::ElementsIC{T},rkepler::Array{T,2},rdotkepler::Array{T,2},jac_kepler::Array{T,2}) where T <: AbstractFloat

    N = init.nbody
    m = init.m
    ϵ = init.ϵ
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
    Ainv = inv(init.amat)
    dAinvdm = zeros(T,N,N,N)
    for k in 1:N
        dAinvdm[:,:,k] = -Ainv*dAdm[:,:,k]*Ainv
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

function amatrix(init::ElementsIC{T}) where T <: Real
    init.amat = amatrix(init.ϵ,init.m)
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

