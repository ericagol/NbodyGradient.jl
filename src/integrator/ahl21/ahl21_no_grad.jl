# The AHL21 integrator WITHOUT derivatives.
"""

Carries out AHL21 mapping with compensated summation, WITHOUT derivatives
"""
function ahl21!(s::State{T},h::T) where T<:AbstractFloat
    h2 = 0.5*h;
    h6 = h/6
    kickfast!(s,h6)
    drift!(s,h2)
    drift_kepler!(s,h2)
    phic!(s,h)
    phisalpha!(s,h,T(2.0))
    kepler_drift!(s,h2)
    drift!(s,h2)
    kickfast!(s,h6)
    return
end

"""

Drifts all particles with compensated summation.
"""
function drift!(s::State{T},h::T) where {T <: Real}
    @inbounds for i=1:s.n, j=1:NDIM
        s.x[j,i],s.xerror[j,i] = comp_sum(s.x[j,i],s.xerror[j,i],h*s.v[j,i])
    end
    return
end

"""

Computes "fast" kicks for pairs of bodies in lieu of -drift+Kepler with compensated summation
"""
function kickfast!(s::State{T},h::T) where {T <: Real}
    s.rij .= zero(T)
    @inbounds for i=1:s.n-1
        for j = i+1:s.n
            if s.pair[i,j]
                for k=1:3
                    s.rij[k] = s.x[k,i] - s.x[k,j]
                    #r2 += s.rij[k]*s.rij[k]
                end
                r2inv = 1.0/dot_fast(s.rij)
                r3inv = r2inv*sqrt(r2inv)
                fac2 = h*GNEWT*r3inv
                for k=1:3
                    fac = fac2*s.rij[k]
                    s.v[k,i],s.verror[k,i] = comp_sum(s.v[k,i],s.verror[k,i],-s.m[j]*fac)
                    s.v[k,j],s.verror[k,j] = comp_sum(s.v[k,j],s.verror[k,j], s.m[i]*fac)
                end
            end
        end
    end
    return
end

"""

Computes correction for pairs which are kicked.
"""
function phic!(s::State{T},h::T) where {T <: Real}
    s.a .= zero(T)
    s.rij .= zero(T)
    s.aij .= zero(T)
    @inbounds for i=1:s.n-1, j = i+1:s.n
        if s.pair[i,j] # kick group
            for k=1:3
                s.rij[k] = s.x[k,i] - s.x[k,j]
                #r2 += s.rij[k]^2
            end
            r2inv = inv(dot_fast(s.rij))
            r3inv = r2inv*sqrt(r2inv)
            for k=1:3
                fac = GNEWT*s.rij[k]*r3inv
                facv = fac*2*h/3
                s.v[k,i],s.verror[k,i] = comp_sum(s.v[k,i],s.verror[k,i],-s.m[j]*facv)
                s.v[k,j],s.verror[k,j] = comp_sum(s.v[k,j],s.verror[k,j],s.m[i]*facv)
                s.a[k,i] -= s.m[j]*fac
                s.a[k,j] += s.m[i]*fac
            end
        end
    end
    coeff = h^3/36*GNEWT
    @inbounds for i=1:s.n-1 ,j=i+1:s.n
        if s.pair[i,j] # kick group
            for k=1:3
                s.aij[k] = s.a[k,i] - s.a[k,j]
                s.rij[k] = s.x[k,i] - s.x[k,j]
            end
            r2 = dot_fast(s.rij)
            r1 = sqrt(r2)
            ardot = dot_fast(s.aij,s.rij)
            fac1 = coeff/(r2*r2*r1)
            fac2 = 3*ardot
            for k=1:3
                fac = fac1*(s.rij[k]*fac2-r2*s.aij[k])
                s.v[k,i],s.verror[k,i] = comp_sum(s.v[k,i],s.verror[k,i],s.m[j]*fac)
                s.v[k,j],s.verror[k,j] = comp_sum(s.v[k,j],s.verror[k,j],-s.m[i]*fac)
            end
        end
    end
    return
end

"""

Computes the 4th-order correction with compensated summation.
"""
function phisalpha!(s::State{T},h::T,alpha::T) where {T <: Real}
    s.a .= zero(T)
    s.rij .= zero(T)
    s.aij .= zero(T)
    coeff = alpha*h^3/96*2*GNEWT

    fac = zero(T); fac1 = zero(T); fac2 = zero(T); r1 = zero(T); r2 = zero(T); r3 = zero(T)
    @inbounds for i=1:s.n-1
        for j = i+1:s.n
            if ~s.pair[i,j] # correction for Kepler pairs
                for k=1:3
                    s.rij[k] = s.x[k,i] - s.x[k,j]
                end
                r2 = dot_fast(s.rij)
                r3 = r2*sqrt(r2)
                fac2 = GNEWT/r3
                for k=1:3
                    fac = fac2*s.rij[k]
                    s.a[k,i] -= s.m[j]*fac
                    s.a[k,j] += s.m[i]*fac
                end
            end
        end
    end
    # Next, compute \tilde g_i acceleration vector (this is rewritten
    # slightly to avoid reference to \tilde a_i):
    @inbounds for i=1:s.n-1
        for j=i+1:s.n
            if ~s.pair[i,j] # correction for Kepler pairs
                for k=1:3
                    s.aij[k] = s.a[k,i] - s.a[k,j]
                    s.rij[k] = s.x[k,i] - s.x[k,j]
                end
                r2 = dot_fast(s.rij)
                r1 = sqrt(r2)
                ardot = dot_fast(s.aij,s.rij)
                # fac1 = coeff/r1^5
                fac1 = coeff/(r2*r2*r1)
                fac2 = (2*GNEWT*(s.m[i]+s.m[j])/r1 + 3*ardot)
                for k=1:3
                    fac = fac1*(s.rij[k]*fac2-r2*s.aij[k])
                    s.v[k,i],s.verror[k,i] = comp_sum(s.v[k,i],s.verror[k,i], s.m[j]*fac)
                    s.v[k,j],s.verror[k,j] = comp_sum(s.v[k,j],s.verror[k,j],-s.m[i]*fac)
                end
            end
        end
    end
    return
end

"""

Take a combined -drift+kepler step for each pair of bodies.
"""
function drift_kepler!(s::State{T}, h::T) where T<:Real
    n = s.n
    @inbounds for i in 1:n-1, j in i+1:n
        if ~s.pair[i,j]
            kepler_driftij_gamma!(s,i,j,h,true)
        end
    end
    return
end

"""

Take a combined kepler-drift step for each pair of bodies
"""
function kepler_drift!(s::State{T}, h::T) where T<:Real
    n = s.n
    @inbounds for i in n-1:-1:1, j in n:-1:i+1
        if ~s.pair[i,j]
            kepler_driftij_gamma!(s,i,j,h,false)
        end
    end
    return
end

"""

Carries out a Kepler step and reverse drift for bodies i & j with compensated summation.
"""
function kepler_driftij_gamma!(s::State{T},i::Int64,j::Int64,h::T,drift_first::Bool) where {T <: Real}
    @inbounds for k=1:NDIM
        s.x0[k] = s.x[k,i] - s.x[k,j] # x0 = positions of body i relative to j
        s.v0[k] = s.v[k,i] - s.v[k,j] # v0 = velocities of body i relative to j
    end
    gm = GNEWT*(s.m[i]+s.m[j])
    if gm == 0; return; end
    # Compute differences in x & v over time step:
    jac_delxv_gamma!(s,gm,h,drift_first,grad=false)
    mijinv =1.0/(s.m[i] + s.m[j])
    mi = s.m[i]*mijinv # Normalize the masses
    mj = s.m[j]*mijinv
    @inbounds for k=1:3
        # Add kepler-drift differences, weighted by masses, to start of step:
        s.x[k,i],s.xerror[k,i] = comp_sum(s.x[k,i],s.xerror[k,i], mj*s.delxv[k])
        s.x[k,j],s.xerror[k,j] = comp_sum(s.x[k,j],s.xerror[k,j],-mi*s.delxv[k])
        s.v[k,i],s.verror[k,i] = comp_sum(s.v[k,i],s.verror[k,i], mj*s.delxv[3+k])
        s.v[k,j],s.verror[k,j] = comp_sum(s.v[k,j],s.verror[k,j],-mi*s.delxv[3+k])
    end
    return
end
