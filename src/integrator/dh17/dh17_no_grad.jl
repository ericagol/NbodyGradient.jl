# The DH17 integrator WITHOUT derivatives.
"""
Carries out the DH17 mapping with compensated summation.
"""
function dh17!(x::Array{T,2},v::Array{T,2},xerror::Array{T,2},verror::Array{T,2},h::T,m::Array{T,1},n::Int64,pair::Array{Bool,2}) where {T <: Real}
    alpha = convert(T,alpha0)
    h2 = 0.5*h
    # alpha = 0. is similar in precision to alpha=0.25
    kickfast!(x,v,xerror,verror,h/6,m,n,pair)
    if alpha != 0.0
        phisalpha!(x,v,xerror,verror,h,m,alpha,n,pair)
    end
    #mbig = big.(m); h2big = big(h2)
    #xbig = big.(x); vbig = big.(v)
    drift!(x,v,xerror,verror,h2,n)
    #drift!(xbig,vbig,h2big,n)
    #xcompare = convert(Array{T,2},xbig); vcompare = convert(Array{T,2},vbig)
    #x .= xcompare; v .= vcompare
    #hbig = big(h)
    @inbounds for i=1:n-1
        @inbounds for j=i+1:n
            if ~pair[i,j]
                #      xbig = big.(x); vbig = big.(v)
                driftij!(x,v,xerror,verror,i,j,-h2)
                #      driftij!(xbig,vbig,i,j,-h2big)
                #      xcompare = convert(Array{T,2},xbig); vcompare = convert(Array{T,2},vbig)
                #      x .= xcompare; v .= vcompare
                #      xbig = big.(x); vbig = big.(v)
                #      keplerij!(mbig,xbig,vbig,i,j,h2big)
                #      xcompare = convert(Array{T,2},xbig); vcompare = convert(Array{T,2},vbig)
                keplerij!(m,x,v,xerror,verror,i,j,h2)
                #      x .= xcompare; v .= vcompare
            end
        end
    end
    # kick and correction for pairs which are kicked:
    phic!(x,v,xerror,verror,h,m,n,pair)
    if alpha != 1.0
        #  xbig = big.(x); vbig = big.(v)
        phisalpha!(x,v,xerror,verror,h,m,2.0*(1.0-alpha),n,pair)
        #  phisalpha!(xbig,vbig,hbig,mbig,2*(1-big(alpha)),n,pair)
        #  xcompare = convert(Array{T,2},xbig); vcompare = convert(Array{T,2},vbig)
        #  x .= xcompare; v .= vcompare
    end
    @inbounds for i=n-1:-1:1
        @inbounds for j=n:-1:i+1
            if ~pair[i,j]
                #      xbig = big.(x); vbig = big.(v)
                #      keplerij!(mbig,xbig,vbig,i,j,h2big)
                #x = convert(Array{T,2},xbig); v = convert(Array{T,2},vbig)
                #      xcompare = convert(Array{T,2},xbig); vcompare = convert(Array{T,2},vbig)
                keplerij!(m,x,v,xerror,verror,i,j,h2)
                #      x .= xcompare; v .= vcompare
                #      xbig = big.(x); vbig = big.(v)
                driftij!(x,v,xerror,verror,i,j,-h2)
                #      driftij!(xbig,vbig,i,j,-h2big)
                #      xcompare = convert(Array{T,2},xbig); vcompare = convert(Array{T,2},vbig)
                #      x .= xcompare; v .= vcompare
            end
        end
    end
    #xbig = big.(x); vbig = big.(v)
    drift!(x,v,xerror,verror,h2,n)
    #drift!(xbig,vbig,h2big,n)
    #xcompare = convert(Array{T,2},xbig); vcompare = convert(Array{T,2},vbig)
    #x .= xcompare; v .= vcompare
    if alpha != 0.0
        phisalpha!(x,v,xerror,verror,h,m,alpha,n,pair)
    end
    kickfast!(x,v,xerror,verror,h/6,m,n,pair)
    return
end

"""

Drifts bodies i & j with compensated summation:
"""
function driftij!(x::Array{T,2},v::Array{T,2},xerror::Array{T,2},verror::Array{T,2},i::Int64,j::Int64,h::T) where {T <: Real}
    for k=1:NDIM
        x[k,i],xerror[k,i] = comp_sum(x[k,i],xerror[k,i],h*v[k,i])
        x[k,j],xerror[k,j] = comp_sum(x[k,j],xerror[k,j],h*v[k,j])
    end
    return
end

"""

Carries out a Kepler step for bodies i & j with compensated summation:
"""
function keplerij!(m::Array{T,1},x::Array{T,2},v::Array{T,2},xerror::Array{T,2},verror::Array{T,2},i::Int64,j::Int64,h::T) where {T <: Real}
    # The state vector has: 1 time; 2-4 position; 5-7 velocity; 8 r0; 9 dr0dt; 10 beta; 11 s; 12 ds
    # Initial state:
    state0 = zeros(T,12)
    # Final state (after a step):
    state = zeros(T,12)
    delx = zeros(T,NDIM)
    delv = zeros(T,NDIM)
    #println("Masses: ",i," ",j)
    for k=1:NDIM
        state0[1+k     ] = x[k,i] - x[k,j]
        state0[1+k+NDIM] = v[k,i] - v[k,j]
    end
    gm = GNEWT*(m[i]+m[j])
    if gm == 0
        for k=1:NDIM
            #x[k,i] += h*v[k,i]
            x[k,i],xerror[k,i] = comp_sum(x[k,i],xerror[k,i],h*v[k,i])
            #x[k,j] += h*v[k,j]
            x[k,j],xerror[k,j] = comp_sum(x[k,j],xerror[k,j],h*v[k,j])
        end
    else
        # predicted value of s
        kepler_step!(gm, h, state0, state)
        for k=1:NDIM
            delx[k] = state[1+k] - state0[1+k]
            delv[k] = state[1+NDIM+k] - state0[1+NDIM+k]
        end
        # Advance center of mass:
        # Compute COM coords:
        mijinv =1.0/(m[i] + m[j])
        vcm = zeros(T,NDIM)
        for k=1:NDIM
            vcm[k] = (m[i]*v[k,i] + m[j]*v[k,j])*mijinv
        end
        centerm!(m,mijinv,x,v,xerror,verror,vcm,delx,delv,i,j,h)
    end
    return
end

"""

Advances the center of mass of a binary (any pair of bodies) with compensated summation:
"""
function centerm!(m::Array{T,1},mijinv::T,x::Array{T,2},v::Array{T,2},xerror::Array{T,2},verror::Array{T,2},vcm::Array{T,1},delx::Array{T,1},delv::Array{T,1},i::Int64,j::Int64,h::T) where {T <: Real}
    for k=1:NDIM
        #x[k,i] +=  m[j]*mijinv*delx[k] + h*vcm[k]
        x[k,i],xerror[k,i] =  comp_sum(x[k,i],xerror[k,i],m[j]*mijinv*delx[k])
        x[k,i],xerror[k,i] =  comp_sum(x[k,i],xerror[k,i],h*vcm[k])
        #x[k,j] += -m[i]*mijinv*delx[k] + h*vcm[k]
        x[k,j],xerror[k,j] = comp_sum(x[k,j],xerror[k,j],-m[i]*mijinv*delx[k])
        x[k,j],xerror[k,j] = comp_sum(x[k,j],xerror[k,j],h*vcm[k])
        #v[k,i] +=  m[j]*mijinv*delv[k]
        v[k,i],verror[k,i] =  comp_sum(v[k,i],verror[k,i],m[j]*mijinv*delv[k])
        #v[k,j] += -m[i]*mijinv*delv[k]
        v[k,j],verror[k,j] = comp_sum(v[k,j],verror[k,j],-m[i]*mijinv*delv[k])
    end
    return
end

"""

Takes a single kepler step, calling Wisdom & Hernandez solver
"""
function kepler_step!(gm::T,h::T,state0::Array{T,1},state::Array{T,1}) where {T <: Real}
    # compute beta, r0,  get x/v from state vector & call correct subroutine
    x0 = zeros(eltype(state0),3)
    v0 = zeros(eltype(state0),3)
    for k=1:3
        x0[k]=state0[k+1]
        v0[k]=state0[k+4]
    end
    #  x0=state0[2:4]
    r0 = sqrt(x0[1]*x0[1]+x0[2]*x0[2]+x0[3]*x0[3])
    #  v0 = state0[5:7]
    beta0 = 2*gm/r0-(v0[1]*v0[1]+v0[2]*v0[2]+v0[3]*v0[3])
    s0=state0[11]
    iter = kep_ell_hyp!(x0,v0,r0,gm,h,beta0,s0,state)
    #  if beta0 > zero
    #    iter = kep_elliptic!(x0,v0,r0,gm,h,beta0,s0,state)
    #  else
    #    iter = kep_hyperbolic!(x0,v0,r0,gm,h,beta0,s0,state)
    #  end
    return
end

"""

Solves equation (35) from Wisdom & Hernandez for the elliptic case.
"""
function kep_ell_hyp!(x0::Array{T,1},v0::Array{T,1},r0::T,k::T,h::T,
                      beta0::T,s0::T,state::Array{T,1}) where {T <: Real}
    # Now, solve for s in elliptical Kepler case:
    f = zero(T); g=zero(T); dfdt=zero(T); dgdt=zero(T); cx=zero(T);sx=zero(T);g1bs=zero(T);g2bs=zero(T)
    s=zero(T); ds = zero(T); r = zero(T);rinv=zero(T); iter=0
    if beta0 > zero(T) || beta0 < zero(T)
        s,f,g,dfdt,dgdt,cx,sx,g1bs,g2bs,r,rinv,ds,iter = solve_kepler!(h,k,x0,v0,beta0,r0,
                                                                       s0,state)
    else
        println("Not elliptic or hyperbolic ",beta0," x0 ",x0)
        r= zero(T); fill!(state,zero(T)); rinv=zero(T); s=zero(T); ds=zero(T); iter = 0
    end
    state[8]= r
    state[9] = (state[2]*state[5]+state[3]*state[6]+state[4]*state[7])*rinv
    # recompute beta:
    # beta is element 10 of state:
    state[10] = 2.0*k*rinv-(state[5]*state[5]+state[6]*state[6]+state[7]*state[7])
    # s is element 11 of state:
    state[11] = s
    # ds is element 12 of state:
    state[12] = ds
    return iter
end

"""

Solves elliptic Kepler's equation for both elliptic and hyperbolic cases.
"""
function solve_kepler!(h::T,k::T,x0::Array{T,1},v0::Array{T,1},beta0::T,
                       r0::T,s0::T,state::Array{T,1}) where {T <: Real}
    # Initial guess (if s0 = 0):
    r0inv = inv(r0)
    beta0inv = inv(beta0)
    signb = sign(beta0)
    sqb = sqrt(signb*beta0)
    zeta = k-r0*beta0
    eta = dot(x0,v0)
    sguess = 0.0
    if s0 == zero(T)
        # Use cubic estimate:
        if zeta != zero(T)
            sguess = cubic1(3eta/zeta,6r0/zeta,-6h/zeta)
        else
            if eta != zero(T)
                reta = r0/eta
                disc = reta^2+2h/eta
                if disc > zero(T)
                    sguess =-reta+sqrt(disc)
                else
                    sguess = h*r0inv
                end
            else
                sguess = h*r0inv
            end
        end
        s = copy(sguess)
    else
        s = copy(s0)
    end
    s0 = copy(s)
    y = zero(T); yp = one(T)
    iter = 0
    ds = Inf
    KEPLER_TOL = sqrt(eps(h))
    ITMAX = 20
    while iter == 0 || (abs(ds) > KEPLER_TOL && iter < ITMAX)
        xx = sqb*s
        if beta0 > 0
            sx = sin(xx); cx = cos(xx)
        else
            cx = cosh(xx); sx = exp(xx)-cx
        end
        sx *= sqb
        # Third derivative:
        yppp = zeta*cx - signb*eta*sx
        # First derivative:
        yp = (-yppp+ k)*beta0inv
        # Second derivative:
        ypp = signb*zeta*beta0inv*sx + eta*cx
        y  = (-ypp + eta +k*s)*beta0inv - h  # eqn 35
        # Now, compute fourth-order estimate:
        ds = calc_ds_opt(y,yp,ypp,yppp)
        s += ds
        iter +=1
    end
    if iter == ITMAX
        println("Reached max iterations in solve_kepler: h ",h," s0: ",s0," s: ",s," ds: ",ds)
    end
    #println("sguess: ",sguess," s: ",s," s-sguess: ",s-sguess," ds: ",ds," iter: ",iter)
    # Since we updated s, need to recompute:
    xx = 0.5*sqb*s
    if beta0 > 0
        sx = sin(xx); cx = cos(xx)
    else
        cx = cosh(xx); sx = exp(xx)-cx
    end
    # Now, compute final values:
    g1bs = 2.0*sx*cx/sqb
    g2bs = 2.0*signb*sx^2*beta0inv
    f = one(T) - k*r0inv*g2bs # eqn (25)
    g = r0*g1bs + eta*g2bs # eqn (27)
    for j=1:3
        # Position is components 2-4 of state:
        state[1+j] = x0[j]*f+v0[j]*g
    end
    r = sqrt(state[2]*state[2]+state[3]*state[3]+state[4]*state[4])
    rinv = inv(r)
    dfdt = -k*g1bs*rinv*r0inv
    dgdt = (r0-r0*beta0*g2bs+eta*g1bs)*rinv
    for j=1:3
        # Velocity is components 5-7 of state:
        state[4+j] = x0[j]*dfdt+v0[j]*dgdt
    end
    return s,f,g,dfdt,dgdt,cx,sx,g1bs,g2bs,r,rinv,ds,iter
end

"""

Computes quartic Newton's update to equation y=0 using first through 3rd derivatives. 
Uses techniques outlined in Murray & Dermott for Kepler solver.
"""
function calc_ds_opt(y::T,yp::T,ypp::T,yppp::T) where {T <: Real}
# Rearrange to reduce number of divisions:
num = y*yp
den1 = yp*yp-y*ypp*.5
den12 = den1*den1
den2 = yp*den12-num*.5*(ypp*den1-third*num*yppp)
return -y*den12/den2
end

