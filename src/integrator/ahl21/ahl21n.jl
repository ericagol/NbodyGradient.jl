function ahl21n!(s::State{T}, h::T) where T <: AbstractFloat
    n = s.n
    @inbounds for i in 1:n-1
        for j in i+1:n
            if ~s.pair[i,j]
                keplerij_gamma!(s, i, j, h)
            end
        end
    end
    kickfast!(s, h)
    return
end

function keplerij_gamma!(s::State{T},i::Int64,j::Int64,h::T) where {T <: Real}
    #s.x0 .= zero(T) # x0 = positions of body i relative to j
    #s.v0 .= zero(T) # v0 = velocities of body i relative to j
    @inbounds for k=1:NDIM
        s.x0[k] = s.x[k,i] - s.x[k,j]
        s.v0[k] = s.v[k,i] - s.v[k,j]
    end
    gm = GNEWT*(s.m[i]+s.m[j])
    if gm == 0; return; end

    # Compute differences in x & v over time step:
    kepler_step!(s,gm,h,grad=false)
    mijinv = 1.0/(s.m[i] + s.m[j])
    mi = s.m[i]*mijinv # Normalize the masses
    mj = s.m[j]*mijinv

    # COM drift
    @inbounds for k in 1:3
        s.rtmp[k] = (mi*s.v[k,i] + mj*s.v[k,j]) * h
    end
    @inbounds for k=1:3
        kp3 = 3+k
        # Advance center of mass
        s.x[k,i],s.xerror[k,i] = comp_sum(s.x[k,i],s.xerror[k,i], mj*s.delxv[k])
        s.x[k,i],s.xerror[k,i] = comp_sum(s.x[k,i],s.xerror[k,i], s.rtmp[k])
        s.x[k,j],s.xerror[k,j] = comp_sum(s.x[k,j],s.xerror[k,j],-mi*s.delxv[k])
        s.x[k,j],s.xerror[k,j] = comp_sum(s.x[k,j],s.xerror[k,j], s.rtmp[k])

        s.v[k,i],s.verror[k,i] = comp_sum(s.v[k,i],s.verror[k,i], mj*s.delxv[kp3])
        s.v[k,j],s.verror[k,j] = comp_sum(s.v[k,j],s.verror[k,j],-mi*s.delxv[kp3])
    end
    return
end

function kepler_step!(s::State{T},k::T,h::T;grad::Bool=true) where {T <: Real}
    # Compute r0:
    r0 = norm(s.x0)
    # And its inverse:
    r0inv::T = inv(r0)
    # Compute beta_0:
    beta0::T = 2k*r0inv-dot_fast(s.v0,s.v0)
    beta0inv::T = inv(beta0)
    signb::T = sign(beta0)
    sqb::T = sqrt(signb*beta0)
    zeta::T = k-r0*beta0
    gamma_guess = zero(T)
    eta::T = dot_fast(s.x0,s.v0)
    if zeta != zero(T)
        # Make sure we have a cubic in gamma (and don't divide by zero):
        zinv = 6/zeta
        gamma_guess = cubic1(0.5*eta*sqb*zinv,r0*signb*beta0*zinv,-h*signb*beta0*sqb*zinv)
    else
        # Check that we have a quadratic in gamma (and don't divide by zero):
        if eta != zero(T)
            reta = r0/eta
            disc = reta^2+2h/eta
            disc > zero(T) ?  gamma_guess = sqb*(-reta+sqrt(disc)) : gamma_guess = h*r0inv*sqb
        else
            gamma_guess = h*r0inv*sqb
        end
    end
    gamma  = copy(gamma_guess)
    # Make sure prior two steps differ:
    gamma1::T = 2*copy(gamma)
    gamma2::T = 3*copy(gamma)
    iter = 0
    ITMAX = 20
    c1 = k; c2 = -2zeta; c3 = 2*eta*signb*sqb; c4 = -sqb*h*beta0; c5 = 2eta*signb*sqb
    # Solve for gamma:
    #    while true
    while iter < ITMAX
        gamma2 = gamma1
        gamma1 = gamma
        xx = 0.5*gamma
        if beta0 > 0
            sx,cx = sincos(xx);
        else
            sx = sinh(xx); cx = exp(-xx)+sx
        end
        gamma -= (k*gamma+c2*sx*cx+c3*sx^2+c4)/(2signb*zeta*sx^2+c5*sx*cx+r0*beta0)
        iter +=1
        if gamma == gamma2 || gamma == gamma1
            break
        end
    end
    # Set up a single output array for delx and delv:
    s.delxv .= zero(T)
    # Since we updated gamma, need to recompute:
    xx = 0.5*gamma
    if beta0 > 0
        sx,cs = sincos(xx)
    else
        sx = sinh(xx); cx = exp(-xx)+sx
    end
    # Now, compute final values.  Compute Wisdom/Hernandez G_i^\beta(s) functions:
    g1bs = 2sx*cx/sqb
    g2bs = 2signb*sx*sx*beta0inv
    g0bs = one(T)-beta0*g2bs
    # g3bs = G3(gamma,beta0,sqb)
    # h1 = zero(T); h2 = zero(T)
    # Compute r from equation (35):
    r = r0*g0bs+eta*g1bs+k*g2bs
    rinv = inv(r)
    f = one(T) - k*r0inv*g2bs
    dfdt = -k*g1bs*rinv*r0inv # [x]
    g = r0*g1bs + eta*g2bs
    dgdt = rinv*(r0*g0bs + eta*g1bs)
    @inbounds for j=1:3
        # Compute difference vectors (finish - start) of step:
        s.delxv[  j] = f*s.x0[j]+g*s.v0[j]        # position x_ij(t+h)-x_ij(t) - h*v_ij(t) or -h*v_ij(t+h)
    end
    @inbounds for j=1:3
        s.delxv[3+j] = dfdt*s.x0[j]+dgdt*s.v0[j]    # velocity v_ij(t+h)-v_ij(t)
    end
    #if grad
    #    return gamma,g0bs,g1bs,g2bs,g3bs,h1,h2,dfdt,fm1,gmh,dgdtm1,r0,r,r0inv,rinv,k,h,beta0,beta0inv,eta,sqb,zeta
    #end
end