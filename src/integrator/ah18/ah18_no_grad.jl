# The AH18 integrator WITHOUT derivatives.

"""

Carries out AH18 mapping with compensated summation.
"""
function ah18!(x::Array{T,2},v::Array{T,2},xerror::Array{T,2},verror::Array{T,2},h::T,m::Array{T,1},n::Int64,pair::Array{Bool,2}) where {T <: Real}
    h2 = 0.5*h
    drift!(x,v,xerror,verror,h2,n)
    kickfast!(x,v,xerror,verror,h/6,m,n,pair)
    @inbounds for i=1:n-1
        for j=i+1:n
            if ~pair[i,j]
                kepler_driftij_gamma!(m,x,v,xerror,verror,i,j,h2,true)
            end
        end
    end
    phic!(x,v,xerror,verror,h,m,n,pair)
    phisalpha!(x,v,xerror,verror,h,m,convert(T,2),n,pair)
    for i=n-1:-1:1
        for j=n:-1:i+1
            if ~pair[i,j]
                kepler_driftij_gamma!(m,x,v,xerror,verror,i,j,h2,false)
            end
        end
    end
    kickfast!(x,v,xerror,verror,h/6,m,n,pair)
    drift!(x,v,xerror,verror,h2,n)
    return
end

"""

Drifts all particles with compensated summation.
"""
function drift!(x::Array{T,2},v::Array{T,2},xerror::Array{T,2},verror::Array{T,2},h::T,n::Int64) where {T <: Real}
    @inbounds for i=1:n, j=1:NDIM
        x[j,i],xerror[j,i] = comp_sum(x[j,i],xerror[j,i],h*v[j,i])
    end
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
function kickfast!(x::Array{T,2},v::Array{T,2},xerror::Array{T,2},verror::Array{T,2},h::T,m::Array{T,1},n::Int64,pair::Array{Bool,2}) where {T <: Real}
    rij = zeros(T,3)
    @inbounds for i=1:n-1
        for j = i+1:n
            if pair[i,j]
                r2 = zero(T)
                for k=1:3
                    rij[k] = x[k,i] - x[k,j]
                    r2 += rij[k]*rij[k]
                end
                r3_inv = 1.0/(r2*sqrt(r2))
                for k=1:3
                    fac = h*GNEWT*rij[k]*r3_inv
                    v[k,i],verror[k,i] = comp_sum(v[k,i],verror[k,i],-m[j]*fac)
                    v[k,j],verror[k,j] = comp_sum(v[k,j],verror[k,j], m[i]*fac)
                end
            end
        end
    end
    return
end

function kickfast!(s::State{T},h::T,pair::Array{Bool,2}) where {T <: Real}
    s.rij .= zero(T)
    @inbounds for i=1:s.n-1
        for j = i+1:s.n
            if pair[i,j]
                r2 = zero(T)
                for k=1:3
                    rij[k] = s.x[k,i] - s.x[k,j]
                    r2 += rij[k]*rij[k]
                end
                r3_inv = 1.0/(r2*sqrt(r2))
                for k=1:3
                    fac = h*GNEWT*rij[k]*r3_inv
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
function phic!(x::Array{T,2},v::Array{T,2},xerror::Array{T,2},verror::Array{T,2},h::T,m::Array{T,1},n::Int64,pair::Array{Bool,2}) where {T <: Real}
    a = zeros(T,3,n)
    rij = zeros(T,3)
    aij = zeros(T,3)
    @inbounds for i=1:n-1, j = i+1:n
        if pair[i,j] # kick group
            r2 = 0.0
            for k=1:3
                rij[k] = x[k,i] - x[k,j]
                r2 += rij[k]^2
            end
            r3_inv = 1.0/(r2*sqrt(r2))
            for k=1:3
                fac = GNEWT*rij[k]*r3_inv
                facv = fac*2*h/3
                v[k,i],verror[k,i] = comp_sum(v[k,i],verror[k,i],-m[j]*facv)
                v[k,j],verror[k,j] = comp_sum(v[k,j],verror[k,j],m[i]*facv)
                a[k,i] -= m[j]*fac
                a[k,j] += m[i]*fac
            end
        end
    end
    coeff = h^3/36*GNEWT
    @inbounds for i=1:n-1 ,j=i+1:n
        if pair[i,j] # kick group
            for k=1:3
                aij[k] = a[k,i] - a[k,j]
                rij[k] = x[k,i] - x[k,j]
            end
            r2 = dot(rij,rij)
            r5inv = 1.0/(r2^2*sqrt(r2))
            ardot = dot(aij,rij)
            for k=1:3
                fac = coeff*r5inv*(rij[k]*3*ardot-r2*aij[k])
                v[k,i],verror[k,i] = comp_sum(v[k,i],verror[k,i],m[j]*fac)
                v[k,j],verror[k,j] = comp_sum(v[k,j],verror[k,j],-m[i]*fac)
            end
        end
    end
    return
end

function phic!(s::State{T},h::T,pair::Array{Bool,2}) where {T <: Real}
    s.a .= zero(T)
    s.rij .= zero(T)
    s.aij .= zero(T)
    @inbounds for i=1:s.n-1, j = i+1:s.n
        if pair[i,j] # kick group
            r2 = zero(T)
            for k=1:3
                s.rij[k] = s.x[k,i] - s.x[k,j]
                r2 += s.rij[k]^2
            end
            r3_inv = 1.0/(r2*sqrt(r2))
            for k=1:3
                fac = GNEWT*s.rij[k]*r3_inv
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
        if pair[i,j] # kick group
            for k=1:3
                s.aij[k] = s.a[k,i] - s.a[k,j]
                s.rij[k] = s.x[k,i] - s.x[k,j]
            end
            r2 = dot(s.rij,s.rij)
            r5inv = 1.0/(r2^2*sqrt(r2))
            ardot = dot(s.aij,s.rij)
            for k=1:3
                fac = coeff*r5inv*(s.rij[k]*3*ardot-r2*s.aij[k])
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
function phisalpha!(x::Array{T,2},v::Array{T,2},xerror::Array{T,2},verror::Array{T,2},h::T,m::Array{T,1},alpha::T,n::Int64,pair::Array{Bool,2}) where {T <: Real}
    a = zeros(T,3,n)
    rij = zeros(T,3)
    aij = zeros(T,3)
    coeff = alpha*h^3/96*2*GNEWT

    fac = zero(T); fac1 = zero(T); fac2 = zero(T); r1 = zero(T); r2 = zero(T); r3 = zero(T)
    @inbounds for i=1:n-1
        for j = i+1:n
            if ~pair[i,j] # correction for Kepler pairs
                for k=1:3
                    rij[k] = x[k,i] - x[k,j]
                end
                r2 = rij[1]*rij[1]+rij[2]*rij[2]+rij[3]*rij[3]
                r3 = r2*sqrt(r2)
                for k=1:3
                    fac = GNEWT*rij[k]/r3
                    a[k,i] -= m[j]*fac
                    a[k,j] += m[i]*fac
                end
            end
        end
    end
    # Next, compute \tilde g_i acceleration vector (this is rewritten
    # slightly to avoid reference to \tilde a_i):
    @inbounds for i=1:n-1
        for j=i+1:n
            if ~pair[i,j] # correction for Kepler pairs
                for k=1:3
                    aij[k] = a[k,i] - a[k,j]
                    rij[k] = x[k,i] - x[k,j]
                end
                r2 = rij[1]*rij[1]+rij[2]*rij[2]+rij[3]*rij[3]
                r1 = sqrt(r2)
                ardot = aij[1]*rij[1]+aij[2]*rij[2]+aij[3]*rij[3]
                fac1 = coeff/r1^5
                fac2 = (2*GNEWT*(m[i]+m[j])/r1 + 3*ardot)
                for k=1:3
                    fac = fac1*(rij[k]*fac2- r2*aij[k])
                    v[k,i],verror[k,i] = comp_sum(v[k,i],verror[k,i], m[j]*fac)
                    v[k,j],verror[k,j] = comp_sum(v[k,j],verror[k,j],-m[i]*fac)
                end
            end
        end
    end
    return
end

function phisalpha!(s::State{T},h::T,alpha::T,pair::Array{Bool,2}) where {T <: Real}
    s.a .= zero(T)
    s.rij .= zero(T)
    s.aij .= zero(T)
    coeff = alpha*h^3/96*2*GNEWT

    fac = zero(T); fac1 = zero(T); fac2 = zero(T); r1 = zero(T); r2 = zero(T); r3 = zero(T)
    @inbounds for i=1:s.n-1
        for j = i+1:s.n
            if ~pair[i,j] # correction for Kepler pairs
                for k=1:3
                    s.rij[k] = s.x[k,i] - s.x[k,j]
                end
                r2 = s.rij[1]*s.rij[1]+s.rij[2]*s.rij[2]+s.rij[3]*s.rij[3]
                r3 = r2*sqrt(r2)
                for k=1:3
                    fac = GNEWT*s.rij[k]/r3
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
            if ~pair[i,j] # correction for Kepler pairs
                for k=1:3
                    s.aij[k] = s.a[k,i] - s.a[k,j]
                    s.rij[k] = s.x[k,i] - s.x[k,j]
                end
                r2 = s.rij[1]*s.rij[1]+s.rij[2]*s.rij[2]+s.rij[3]*s.rij[3]
                r1 = sqrt(r2)
                ardot = s.aij[1]*s.rij[1]+s.aij[2]*s.rij[2]+s.aij[3]*s.rij[3]
                fac1 = coeff/r1^5
                fac2 = (2*GNEWT*(s.m[i]+s.m[j])/r1 + 3*ardot)
                for k=1:3
                    fac = fac1*(s.rij[k]*fac2- r2*s.aij[k])
                    s.v[k,i],s.verror[k,i] = comp_sum(s.v[k,i],s.verror[k,i], s.m[j]*fac)
                    s.v[k,j],s.verror[k,j] = comp_sum(s.v[k,j],s.verror[k,j],-s.m[i]*fac)
                end
            end
        end
    end
    return
end

"""

Carries out a Kepler step and reverse drift for bodies i & j with compensated summation.
"""
function kepler_driftij_gamma!(m::Array{T,1},x::Array{T,2},v::Array{T,2},xerror::Array{T,2},verror::Array{T,2},i::Int64,j::Int64,h::T,drift_first::Bool) where {T <: Real}
    x0 = zeros(T,NDIM) # x0 = positions of body i relative to j
    v0 = zeros(T,NDIM) # v0 = velocities of body i relative to j
    @inbounds for k=1:NDIM
        x0[k] = x[k,i] - x[k,j]
        v0[k] = v[k,i] - v[k,j]
    end
    gm = GNEWT*(m[i]+m[j])
    if gm == 0
        #  Do nothing
        #  for k=1:3
        #    x[k,i] += h*v[k,i]
        #    x[k,j] += h*v[k,j]
        #  end
    else
        # Compute differences in x & v over time step:
        delxv = jac_delxv_gamma!(x0,v0,gm,h,drift_first)
        mijinv =1.0/(m[i] + m[j])
        mi = m[i]*mijinv # Normalize the masses
        mj = m[j]*mijinv
        @inbounds for k=1:3
            # Add kepler-drift differences, weighted by masses, to start of step:
            x[k,i],xerror[k,i] = comp_sum(x[k,i],xerror[k,i], mj*delxv[k])
            x[k,j],xerror[k,j] = comp_sum(x[k,j],xerror[k,j],-mi*delxv[k])
            v[k,i],verror[k,i] = comp_sum(v[k,i],verror[k,i], mj*delxv[3+k])
            v[k,j],verror[k,j] = comp_sum(v[k,j],verror[k,j],-mi*delxv[3+k])
        end
    end
    return
end

function kepler_driftij_gamma!(s::State{T},i::Int64,j::Int64,h::T,drift_first::Bool) where {T <: Real}
    s.x0 .= zero(T) # x0 = positions of body i relative to j  
    s.v0 .= zero(T) # v0 = velocities of body i relative to j  
    @inbounds for k=1:NDIM
        s.x0[k] = s.x[k,i] - s.x[k,j]
        s.v0[k] = s.v[k,i] - s.v[k,j]
    end
    gm = GNEWT*(s.m[i]+s.m[j])
    if gm == 0
        #  Do nothing
        #  for k=1:3
        #    x[k,i] += h*v[k,i]
        #    x[k,j] += h*v[k,j]
        #  end
    else
        # Compute differences in x & v over time step:
        delxv = jac_delxv_gamma!(s,gm,h,drift_first)
        mijinv =1.0/(s.m[i] + s.m[j])
        mi = s.m[i]*mijinv # Normalize the masses
        mj = s.m[j]*mijinv
        @inbounds for k=1:3
            # Add kepler-drift differences, weighted by masses, to start of step:
            s.x[k,i],s.xerror[k,i] = comp_sum(s.x[k,i],s.xerror[k,i], mj*delxv[k])
            s.x[k,j],s.xerror[k,j] = comp_sum(s.x[k,j],s.xerror[k,j],-mi*delxv[k])
            s.v[k,i],s.verror[k,i] = comp_sum(s.v[k,i],s.verror[k,i], mj*delxv[3+k])
            s.v[k,j],s.verror[k,j] = comp_sum(s.v[k,j],s.verror[k,j],-mi*delxv[3+k])
        end
    end
    return
end

"""

Computes Jacobian of delx and delv with respect to x0, v0, k, and h.
"""
function jac_delxv_gamma!(x0::Array{T,1},v0::Array{T,1},k::T,h::T,drift_first::Bool;grad::Bool=false,auto::Bool=false,dlnq::T=convert(T,0.0),debug=false) where {T <: Real}
    # Using autodiff, computes Jacobian of delx & delv with respect to x0, v0, k & h.

    # Autodiff requires a single-vector input, so create an array to hold the independent variables:
    input = zeros(T,8)
    input[1:3]=x0; input[4:6]=v0; input[7]=k; input[8]=h
    if grad
        if debug 
            # Also output gamma, r, fm1, dfdt, gmh, dgdtm1, and for debugging:
            delxv_jac = zeros(T,12,8)
        else
            # Output \delta x & \delta v only:
            delxv_jac = zeros(T,6,8)
        end
    end

    # Create a closure so that the function knows value of drift_first:

    function delx_delv(input::Array{T2,1}) where {T2 <: Real} # input = x0,v0,k,h,drift_first
        # Compute delx and delv from h, s, k, beta0, x0 and v0:
        x0 = input[1:3]; v0 = input[4:6]; k = input[7]; h = input[8]
        # Compute r0:
        drift_first ?  r0 = norm(x0-h*v0) : r0 = norm(x0)
        # And its inverse:
        r0inv = inv(r0)
        # Compute beta_0:
        beta0 = 2k*r0inv-dot(v0,v0)
        beta0inv = inv(beta0)
        signb = sign(beta0)
        sqb = sqrt(signb*beta0)
        zeta = k-r0*beta0
        gamma_guess = zero(T2)
        # Compute \eta_0 = x_0 . v_0:
        drift_first ?  eta = dot(x0-h*v0,v0) : eta = dot(x0,v0)
        if zeta != zero(T2)
            # Make sure we have a cubic in gamma (and don't divide by zero):
            gamma_guess = cubic1(3eta*sqb/zeta,6r0*signb*beta0/zeta,-6h*signb*beta0*sqb/zeta)
        else
            # Check that we have a quadratic in gamma (and don't divide by zero):
            if eta != zero(T2)
                reta = r0/eta
                disc = reta^2+2h/eta
                disc > zero(T2) ?  gamma_guess = sqb*(-reta+sqrt(disc)) : gamma_guess = h*r0inv*sqb
            else
                gamma_guess = h*r0inv*sqb
            end
        end
        gamma  = copy(gamma_guess)
        # Make sure prior two steps differ:
        gamma1 = 2*copy(gamma)
        gamma2 = 3*copy(gamma)
        iter = 0
        ITMAX = 20
        # Compute coefficients: (8/28/19 notes)
        #  c1 = k; c2 = -zeta; c3 = -eta*sqb; c4 = sqb*(eta-h*beta0); c5 = eta*signb*sqb
        c1 = k; c2 = -2zeta; c3 = 2*eta*signb*sqb; c4 = -sqb*h*beta0; c5 = 2eta*signb*sqb
        # Solve for gamma:
        while true 
            gamma2 = gamma1
            gamma1 = gamma
            xx = 0.5*gamma
            #    xx = gamma
            if beta0 > 0 
                sx = sin(xx); cx = cos(xx) 
            else 
                sx = sinh(xx); cx = exp(-xx)+sx
            end
            #    gamma -= (c1*gamma+c2*sx+c3*cx+c4)/(c2*cx+c5*sx+c1)
            gamma -= (k*gamma+c2*sx*cx+c3*sx^2+c4)/(2signb*zeta*sx^2+c5*sx*cx+r0*beta0)
            iter +=1 
            if iter >= ITMAX || gamma == gamma2 || gamma == gamma1
                break
            end
        end
        #  if typeof(gamma) ==  Float64
        #    println("s: ",gamma/sqb)
        #  end
        # Set up a single output array for delx and delv:
        if debug
            delxv = zeros(T2,12)
        else
            delxv = zeros(T2,6)
        end
        # Since we updated gamma, need to recompute:
        xx = 0.5*gamma
        if beta0 > 0 
            sx = sin(xx); cx = cos(xx) 
        else
            sx = sinh(xx); cx = exp(-xx)+sx
        end
        # Now, compute final values.  Compute Wisdom/Hernandez G_i^\beta(s) functions:
        g1bs = 2sx*cx/sqb
        g2bs = 2signb*sx^2*beta0inv
        g0bs = one(T2)-beta0*g2bs
        g3bs = G3(gamma,beta0)
        h1 = zero(T2); h2 = zero(T2)
        #  if typeof(g1bs) == Float64
        #    println("g1: ",g1bs," g2: ",g2bs," g3: ",g3bs)
        #  end
        # Compute r from equation (35):
        r = r0*g0bs+eta*g1bs+k*g2bs
        #  if typeof(r) == Float64
        #    println("r: ",r)
        #  end
        rinv = inv(r)
        dfdt = -k*g1bs*rinv*r0inv # [x]
        if drift_first
            # Drift backwards before Kepler step: (1/22/2018)
            fm1 = -k*r0inv*g2bs # [x]
            # This is given in 2/7/2018 notes: g-h*f
            #    gmh = k*r0inv*(r0*(g1bs*g2bs-g3bs)+eta*g2bs^2+k*g3bs*g2bs)  # [x]
            #    println("Old gmh: ",gmh," new gmh: ",k*r0inv*(h*g2bs-r0*g3bs))  # [x]
            gmh = k*r0inv*(h*g2bs-r0*g3bs)  # [x]
        else
            # Drift backwards after Kepler step: (1/24/2018)
            # The following line is f-1-h fdot:
            h1= H1(gamma,beta0); h2= H2(gamma,beta0)
            #    fm1 =  k*rinv*(g2bs-k*r0inv*H1(gamma,beta0))  # [x]
            fm1 =  k*rinv*(g2bs-k*r0inv*h1)  # [x]
            # This is g-h*dgdt
            #    gmh = k*rinv*(r0*H2(gamma,beta0)+eta*H1(gamma,beta0)) # [x]
            gmh = k*rinv*(r0*h2+eta*h1) # [x]
        end
        # Compute velocity component functions:
        if drift_first
            # This is gdot - h fdot - 1:
            #    dgdtm1 = k*r0inv*rinv*(r0*g0bs*g2bs+eta*g1bs*g2bs+k*g1bs*g3bs) # [x]
            #    println("gdot-h fdot-1: ",dgdtm1," alternative expression: ",k*r0inv*rinv*(h*g1bs-r0*g2bs))
            dgdtm1 = k*r0inv*rinv*(h*g1bs-r0*g2bs)
        else
            # This is gdot - 1:
            dgdtm1 = -k*rinv*g2bs # [x]
        end
        #  if typeof(fm1) == Float64
        #    println("fm1: ",fm1," dfdt: ",dfdt," gmh: ",gmh," dgdt-1: ",dgdtm1)
        #  end
        @inbounds for j=1:3
            # Compute difference vectors (finish - start) of step:
            delxv[  j] = fm1*x0[j]+gmh*v0[j]        # position x_ij(t+h)-x_ij(t) - h*v_ij(t) or -h*v_ij(t+h)
        end
        @inbounds for j=1:3
            delxv[3+j] = dfdt*x0[j]+dgdtm1*v0[j]    # velocity v_ij(t+h)-v_ij(t)
        end
        if debug
            delxv[7] = gamma
            delxv[8] = r
            delxv[9] = fm1
            delxv[10] = dfdt
            delxv[11] = gmh
            delxv[12] = dgdtm1
        end
        if grad == true && auto == false && dlnq == 0.0
            # Compute gradient analytically:
            jac_mass = zeros(T,6)
            compute_jacobian_gamma!(gamma,g0bs,g1bs,g2bs,g3bs,h1,h2,dfdt,fm1,gmh,dgdtm1,r0,r,r0inv,rinv,k,h,beta0,beta0inv,eta,sqb,zeta,x0,v0,delxv_jac,jac_mass,drift_first,debug)
        end
        return delxv
    end

    # Use autodiff to compute Jacobian:
    if grad
        if auto
            #    delxv_jac = ForwardDiff.jacobian(delx_delv,input)
            if debug
                delxv = zeros(T,12)
            else
                delxv = zeros(T,6)
            end
            out = DiffResults.JacobianResult(delxv,input)
            ForwardDiff.jacobian!(out,delx_delv,input)
            delxv_jac = DiffResults.jacobian(out)
            delxv = DiffResults.value(out)
        elseif dlnq != 0.0
            # Use finite differences to compute Jacobian:
            if debug
                delxv_jac = zeros(T,12,8)
            else
                delxv_jac = zeros(T,6,8)
            end
            delxv = delx_delv(input)
            @inbounds for j=1:8
                # Difference the jth component:
                inputp = copy(input); dp = dlnq*inputp[j]; inputp[j] += dp
                delxvp = delx_delv(inputp)
                inputm = copy(input); inputm[j] -= dp
                delxvm = delx_delv(inputm)
                delxv_jac[:,j] = (delxvp-delxvm)/(2*dp)
            end
        else
            # If grad = true and dlnq = 0.0, then the above routine will compute Jacobian analytically.
            delxv = delx_delv(input)
        end
        # Return Jacobian:
        return  delxv::Array{T,1},delxv_jac::Array{T,2}
    else
        return delx_delv(input)::Array{T,1}
    end
end

function jac_delxv_gamma!(s::State{T},k0::T,h0::T,drift_first::Bool;grad::Bool=false,auto::Bool=false,dlnq::T=convert(T,0.0),debug=false) where {T <: Real}
    # Using autodiff, computes Jacobian of delx & delv with respect to x0, v0, k & h.

    # Autodiff requires a single-vector input, so create an array to hold the independent variables:
    @views s.input[1:3] = s.x0[:];
    @views s.input[4:6] = s.v0[:];
    s.input[7]=k0; s.input[8]=h0
    if grad
        if debug 
            # Also output gamma, r, fm1, dfdt, gmh, dgdtm1, and for debugging:
            delxv_jac = zeros(T,12,8)
        else
            # Output \delta x & \delta v only:
            delxv_jac = zeros(T,6,8)
        end
    end

    # Create a closure so that the function knows value of drift_first:

    function delx_delv(input::Array{T2,1}) where {T2 <: Real} # input = x0,v0,k,h,drift_first
        # Compute delx and delv from h, s, k, beta0, x0 and v0:
        @views s.x0[:] = copy(input[1:3]);
        @views s.v0[:] = copy(input[4:6]);
        k::T2 = input[7]; h::T2 = input[8]
        # Compute r0:
        r0 = zero(T2)
        s.rtmp[:] .= s.x0.-h.*s.v0
        drift_first ?  r0 = norm(s.rtmp) : r0 = norm(s.x0)
        # And its inverse:
        r0inv::T2 = inv(r0)
        # Compute beta_0:
        beta0::T2 = 2k*r0inv-dot(s.v0,s.v0)
        beta0inv::T2 = inv(beta0)
        signb::T2 = sign(beta0)
        sqb::T2 = sqrt(signb*beta0)
        zeta::T2 = k-r0*beta0
        gamma_guess = zero(T2)
        # Compute \eta_0 = x_0 . v_0:
        eta = zero(T2)
        #drift_first ?  eta = dot(s.x0 .- h .* s.v0, s.v0) : eta = dot(s.x0,s.v0)
        if drift_first
            eta = s.rtmp[1]*s.v0[1]+s.rtmp[2]*s.v0[2]+s.rtmp[3]*s.v0[3]
        else
            eta = s.x0[1]*s.v0[1]+s.x0[2]*s.v0[2]+s.x0[3]*s.v0[3]
        end
        if zeta != zero(T2)
            # Make sure we have a cubic in gamma (and don't divide by zero):
            gamma_guess = cubic1(3eta*sqb/zeta,6r0*signb*beta0/zeta,-6h*signb*beta0*sqb/zeta)
        else
            # Check that we have a quadratic in gamma (and don't divide by zero):
            if eta != zero(T2)
                reta = r0/eta
                disc = reta^2+2h/eta
                disc > zero(T2) ?  gamma_guess = sqb*(-reta+sqrt(disc)) : gamma_guess = h*r0inv*sqb
            else
                gamma_guess = h*r0inv*sqb
            end
        end
        gamma  = copy(gamma_guess)
        # Make sure prior two steps differ:
        gamma1::T2 = 2*copy(gamma)
        gamma2::T2 = 3*copy(gamma)
        iter = 0
        ITMAX = 20
        # Compute coefficients: (8/28/19 notes)
        #  c1 = k; c2 = -zeta; c3 = -eta*sqb; c4 = sqb*(eta-h*beta0); c5 = eta*signb*sqb
        c1 = k; c2 = -2zeta; c3 = 2*eta*signb*sqb; c4 = -sqb*h*beta0; c5 = 2eta*signb*sqb
        # Solve for gamma:
        while true 
            gamma2 = gamma1
            gamma1 = gamma
            xx = 0.5*gamma
            #    xx = gamma
            if beta0 > 0 
                sx,cx = sincos(xx);
            else 
                sx = sinh(xx); cx = exp(-xx)+sx
            end
            #    gamma -= (c1*gamma+c2*sx+c3*cx+c4)/(c2*cx+c5*sx+c1)
            gamma -= (k*gamma+c2*sx*cx+c3*sx^2+c4)/(2signb*zeta*sx^2+c5*sx*cx+r0*beta0)
            iter +=1 
            if iter >= ITMAX || gamma == gamma2 || gamma == gamma1
                break
            end
        end
        #  if typeof(gamma) ==  Float64
        #    println("s: ",gamma/sqb)
        #  end
        # Set up a single output array for delx and delv:
        if debug
            s.delxv = zeros(T2,12)
        else
            s.delxv .= zero(T2)
        end
        # Since we updated gamma, need to recompute:
        xx = 0.5*gamma
        if beta0 > 0 
            sx,cs = sincos(xx)
        else
            sx = sinh(xx); cx = exp(-xx)+sx
        end
        # Now, compute final values.  Compute Wisdom/Hernandez G_i^\beta(s) functions:
        g1bs = 2sx*cx/sqb
        g2bs = 2signb*sx^2*beta0inv
        g0bs = one(T2)-beta0*g2bs
        g3bs = G3(gamma,beta0)
        h1 = zero(T2); h2 = zero(T2)
        #  if typeof(g1bs) == Float64
        #    println("g1: ",g1bs," g2: ",g2bs," g3: ",g3bs)
        #  end
        # Compute r from equation (35):
        r = r0*g0bs+eta*g1bs+k*g2bs
        #  if typeof(r) == Float64
        #    println("r: ",r)
        #  end
        rinv = inv(r)
        dfdt = -k*g1bs*rinv*r0inv # [x]
        if drift_first
            # Drift backwards before Kepler step: (1/22/2018)
            fm1 = -k*r0inv*g2bs # [x]
            # This is given in 2/7/2018 notes: g-h*f
            #    gmh = k*r0inv*(r0*(g1bs*g2bs-g3bs)+eta*g2bs^2+k*g3bs*g2bs)  # [x]
            #    println("Old gmh: ",gmh," new gmh: ",k*r0inv*(h*g2bs-r0*g3bs))  # [x]
            gmh = k*r0inv*(h*g2bs-r0*g3bs)  # [x]
        else
            # Drift backwards after Kepler step: (1/24/2018)
            # The following line is f-1-h fdot:
            h1= H1(gamma,beta0); h2= H2(gamma,beta0)
            #    fm1 =  k*rinv*(g2bs-k*r0inv*H1(gamma,beta0))  # [x]
            fm1 =  k*rinv*(g2bs-k*r0inv*h1)  # [x]
            # This is g-h*dgdt
            #    gmh = k*rinv*(r0*H2(gamma,beta0)+eta*H1(gamma,beta0)) # [x]
            gmh = k*rinv*(r0*h2+eta*h1) # [x]
        end
        # Compute velocity component functions:
        if drift_first
            # This is gdot - h fdot - 1:
            #    dgdtm1 = k*r0inv*rinv*(r0*g0bs*g2bs+eta*g1bs*g2bs+k*g1bs*g3bs) # [x]
            #    println("gdot-h fdot-1: ",dgdtm1," alternative expression: ",k*r0inv*rinv*(h*g1bs-r0*g2bs))
            dgdtm1 = k*r0inv*rinv*(h*g1bs-r0*g2bs)
        else
            # This is gdot - 1:
            dgdtm1 = -k*rinv*g2bs # [x]
        end
        #  if typeof(fm1) == Float64
        #    println("fm1: ",fm1," dfdt: ",dfdt," gmh: ",gmh," dgdt-1: ",dgdtm1)
        #  end
        @inbounds for j=1:3
            # Compute difference vectors (finish - start) of step:
            s.delxv[  j] = fm1*s.x0[j]+gmh*s.v0[j]        # position x_ij(t+h)-x_ij(t) - h*v_ij(t) or -h*v_ij(t+h)
        end
        @inbounds for j=1:3
            s.delxv[3+j] = dfdt*s.x0[j]+dgdtm1*s.v0[j]    # velocity v_ij(t+h)-v_ij(t)
        end
        if debug
            s.delxv[7] = gamma
            s.delxv[8] = r
            s.delxv[9] = fm1
            s.delxv[10] = dfdt
            s.delxv[11] = gmh
            s.delxv[12] = dgdtm1
        end
        if grad == true && auto == false && dlnq == 0.0
            # Compute gradient analytically:
            jac_mass = zeros(T,6)
            compute_jacobian_gamma!(gamma,g0bs,g1bs,g2bs,g3bs,h1,h2,dfdt,fm1,gmh,dgdtm1,r0,r,r0inv,rinv,k,h,beta0,beta0inv,eta,sqb,zeta,x0,v0,delxv_jac,jac_mass,drift_first,debug)
        end
        return s.delxv
    end

    # Use autodiff to compute Jacobian:
    if grad
        if auto
            #    delxv_jac = ForwardDiff.jacobian(delx_delv,input)
            if debug
                delxv = zeros(T,12)
            else
                delxv = zeros(T,6)
            end
            out = DiffResults.JacobianResult(delxv,input)
            ForwardDiff.jacobian!(out,delx_delv,input)
            delxv_jac = DiffResults.jacobian(out)
            delxv = DiffResults.value(out)
        elseif dlnq != 0.0
            # Use finite differences to compute Jacobian:
            if debug
                delxv_jac = zeros(T,12,8)
            else
                delxv_jac = zeros(T,6,8)
            end
            delxv = delx_delv(input)
            @inbounds for j=1:8
                # Difference the jth component:
                inputp = copy(input); dp = dlnq*inputp[j]; inputp[j] += dp
                delxvp = delx_delv(inputp)
                inputm = copy(input); inputm[j] -= dp
                delxvm = delx_delv(inputm)
                delxv_jac[:,j] = (delxvp-delxvm)/(2*dp)
            end
        else
            # If grad = true and dlnq = 0.0, then the above routine will compute Jacobian analytically.
            delxv = delx_delv(input)
        end
        # Return Jacobian:
        return  delxv::Array{T,1},delxv_jac::Array{T,2}
    else
        return delx_delv(s.input)::Array{T,1}
    end
end