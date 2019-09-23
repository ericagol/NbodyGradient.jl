
# Wisdom & Hernandez version of Kepler solver, but with quartic convergence.

using ForwardDiff

include("g3.jl")

#function calc_ds_opt(y::T,yp::T,ypp::T,yppp::T) where {T <: Real}
## Computes quartic Newton's update to equation y=0 using first through 3rd derivatives.
## Uses techniques outlined in Murray & Dermott for Kepler solver.
## Rearrange to reduce number of divisions:
#num = y*yp
#den1 = yp*yp-y*ypp*.5
#den12 = den1*den1
#den2 = yp*den12-num*.5*(ypp*den1-third*num*yppp)
#return -y*den12/den2
#end

function solve_kepler_drift!(h::T,k::T,x0::Array{T,1},v0::Array{T,1},beta0::T,
   r0::T,s0::T,state::Array{T,1},drift_first::Bool) where {T <: Real}
# Solves elliptic Kepler's equation for both elliptic and hyperbolic cases,
# along with a drift before or after the kepler step.
# Initial guess (if s0 = 0):
r0inv = inv(r0)
beta0inv = inv(beta0)
signb = sign(beta0)
if s0 == zero(T)
  s = h*r0inv
else
  s = copy(s0)
end
s0 = copy(s)
sqb = sqrt(signb*beta0)
y = zero(T); yp = one(T)
iter = 0
ds = Inf
zeta = k-r0*beta0
if drift_first
  eta = dot(x0-h*v0,v0)
else
  eta = dot(x0,v0)
end
KEPLER_TOL = sqrt(eps(h))
while iter == 0 || (abs(ds) > KEPLER_TOL && iter < 10)
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
# Since we updated s, need to recompute:
xx = 0.5*sqb*s
if beta0 > 0
  sx = sin(xx); cx = cos(xx)
else
  cx = cosh(xx); sx = exp(xx)-cx
end
# Now, compute final values.  Compute Wisdom/Hernandez G_i^\beta(s) functions:
g1bs = 2.*sx*cx/sqb
g2bs = 2.*signb*sx^2*beta0inv
g0bs = 1.0-beta0*g2bs
# This should be computed to prevent roundoff error. [ ]
#g3bs = (1.0-g1bs)*beta0inv
g3bs = G3(s*sqb,beta0)
#if typeof(g1bs) == Float64
#  println("g1: ",g1bs," g2: ",g2bs," g3: ",g3bs)
#end
# Compute Gauss' Kepler functions:
f = one(T) - k*r0inv*g2bs # eqn (25)
g = r0*g1bs + eta*g2bs # eqn (27)
if drift_first
  r = norm(f*(x0-h*v0)+g*v0)
else
  r = norm(f*x0+g*v0)
end
#if typeof(r) == Float64
#  println("r: ",r)
#end
rinv = inv(r)
dfdt = -k*g1bs*rinv*r0inv
if drift_first
  # Drift backwards before Kepler step: (1/22/2018)
  fm1 = -k*r0inv*g2bs
  # This is given in 2/7/2018 notes:
  gmh = k*r0inv*(r0*(g1bs*g2bs-g3bs)+eta*g2bs^2+k*g3bs*g2bs)
else
  # Drift backwards after Kepler step: (1/24/2018)
  fm1 =  k*rinv*(g2bs-k*r0inv*H1(s*sqb,beta0))
  # This is g-h*dgdt
  gmh = k*rinv*(r0*H2(s*sqb,beta0)+eta*H1(s*sqb,beta0))
end
# Compute velocity component functions:
if drift_first
  # 2/1/18 notes:
  dgdtm1 = k*r0inv*rinv*(r0*g0bs*g2bs+eta*g1bs*g2bs+k*g1bs*g3bs)
else
  # 1/22/18 notes:
  dgdtm1 = -k*rinv*g2bs
end
#if typeof(fm1) == Float64
#  println("fm1: ",fm1," dfdt: ",dfdt," gmh: ",gmh," dgdt-1: ",dgdtm1)
#end
for j=1:3
# Compute difference vectors (finish - start) of step:
  state[1+j] = fm1*x0[j]+gmh*v0[j]        # position x_ij(t+h)-x_ij(t) - h*v_ij(t) or -h*v_ij(t+h)
  state[4+j] = dfdt*x0[j]+dgdtm1*v0[j]    # velocity v_ij(t+h)-v_ij(t)
end  
return s,f,g,dfdt,dgdtm1,cx,sx,g1bs,g2bs,g3bs,r,rinv,ds,iter
end

function jac_delxv(x0::Array{T,1},v0::Array{T,1},k::T,s::T,beta0::T,h::T,drift_first::Bool) where {T <: Real}
# Using autodiff, computes Jacobian of delx & delv with respect to x0, v0, k, s, beta0 & h.

# Autodiff requires a single-vector input, so create an array to hold the independent variables:
  input = zeros(T,10)
  input[1:3]=x0; input[4:6]=v0; input[7]=k; input[8]=s; input[9]=beta0; input[10]=h

# Create a closure so that the function knows value of drift_first:

  function delx_delv(input::Array{T2,1}) where {T2 <: Real} # input = x0,v0,k,s,beta0,h,drift_first
  # Compute delx and delv from h, s, k, beta0, x0 and v0:
  x0 = input[1:3]; v0 = input[4:6]; k = input[7]
  s = input[8]; beta0=input[9]; h = input[10]
  # Set up a single output array for delx and delv:
  delxv = zeros(T2,6)
  # Compute square root of beta0:
  signb = sign(beta0)
  sqb = sqrt(abs(beta0))
  beta0inv=inv(beta0)
  if drift_first
    r0 = norm(x0-h*v0)
    eta = dot(x0-h*v0,v0)
  else
    r0 = norm(x0)
    eta = dot(x0,v0)
  end
  r0inv = inv(r0)
  # Since we updated s, need to recompute:
  xx = 0.5*sqb*s
  if beta0 > 0
    sx = sin(xx); cx = cos(xx)
  else
    cx = cosh(xx); sx = exp(xx)-cx
  end
  # Now, compute final values.  Compute Wisdom/Hernandez G_i^\beta(s) functions:
  g1bs = 2.*sx*cx/sqb
  g2bs = 2.*signb*sx^2*beta0inv
  g0bs = 1.0-beta0*g2bs
  g3bs = G3(s*sqb,beta0)
  # Compute Gauss' Kepler functions:
  f = 1.0 - k*r0inv*g2bs # eqn (25)
  g = r0*g1bs + eta*g2bs # eqn (27)
  if drift_first
    r = norm(f*(x0-h*v0)+g*v0)
  else
    r = norm(f*x0+g*v0)
  end
  rinv = inv(r)
  dfdt = -k*g1bs*rinv*r0inv
  if drift_first
    # Drift backwards before Kepler step: (1/22/2018)
    fm1 = -k*r0inv*g2bs
    # This is given in 2/7/2018 notes:
    gmh = k*r0inv*(r0*(g1bs*g2bs-g3bs)+eta*g2bs^2+k*g3bs*g2bs)
  else
    # Drift backwards after Kepler step: (1/24/2018)
    fm1 =  k*rinv*(g2bs-k*r0inv*H1(s*sqb,beta0))
    # This is g-h*dgdt
    gmh = k*rinv*(r0*H2(s*sqb,beta0)+eta*H1(s*sqb,beta0))
  end
  # Compute velocity component functions:
  if drift_first
    dgdtm1 = k*r0inv*rinv*(r0*g0bs*g2bs+eta*g1bs*g2bs+k*g1bs*g3bs)
  else
    dgdtm1 = -k*rinv*g2bs
  end
  for j=1:3
  # Compute difference vectors (finish - start) of step:
    delxv[  j] = fm1*x0[j]+gmh*v0[j]        # position x_ij(t+h)-x_ij(t) - h*v_ij(t) or -h*v_ij(t+h)
    delxv[3+j] = dfdt*x0[j]+dgdtm1*v0[j]    # velocity v_ij(t+h)-v_ij(t)
  end
  return delxv
  end

# Use autodiff to compute Jacobian:
delxv_jac = ForwardDiff.jacobian(delx_delv,input)
# Return Jacobian:
return  delxv_jac
end

function kep_drift_ell_hyp!(x0::Array{T,1},v0::Array{T,1},k::T,h::T,
  s0::T,state::Array{T,1},drift_first::Bool) where {T <: Real}
if drift_first
  r0 = norm(x0-h*v0)
else
  r0 = norm(x0)
end
beta0 = 2*k/r0-dot(v0,v0)
# Solves equation (35) from Wisdom & Hernandez for the elliptic case.
# Now, solve for s in elliptical Kepler case:
f = zero(T); g=zero(T); dfdt=zero(T); dgdtm1=zero(T); cx=zero(T);sx=zero(T);g1bs=zero(T);g2bs=zero(T);g3bs=zero(T)
s=zero(T); ds = zero(T); r = zero(T);rinv=zero(T); iter=0
if beta0 > zero(T) || beta0 < zero(T)
   s,f,g,dfdt,dgdtm1,cx,sx,g1bs,g2bs,g3bs,r,rinv,ds,iter = solve_kepler_drift!(h,k,x0,v0,beta0,r0,
    s0,state,drift_first)
else
#  println("Not elliptic or hyperbolic ",beta0," x0 ",x0)
  r= zero(T); fill!(state,zero(T)); rinv=zero(T); s=zero(T); ds=zero(T); iter = 0
end
state[8]= r
# These need to be updated. [ ]
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

function kep_drift_ell_hyp!(x0::Array{T,1},v0::Array{T,1},k::T,h::T,
  s0::T,state::Array{T,1},jacobian::Array{T,2},drift_first::Bool) where {T <: Real}
if drift_first
  r0 = norm(x0-h*v0)
else
  r0 = norm(x0)
end
beta0 = 2*k/r0-dot(v0,v0)
# Computes the Jacobian as well
# Solves equation (35) from Wisdom & Hernandez for the elliptic case.
r0inv = inv(r0)
beta0inv = inv(beta0)
# Now, solve for s in elliptical Kepler case:
f = zero(T); g=zero(T); dfdt=zero(T); dgdtm1=zero(T); cx=zero(T);sx=zero(T);g1bs=zero(T);g2bs=zero(T);g3bs=zero(T)
s=zero(T); ds=zero(T); r = zero(T);rinv=zero(T); iter=0
if beta0 > zero(T) || beta0 < zero(T)
   s,f,g,dfdt,dgdtm1,cx,sx,g1bs,g2bs,g3bs,r,rinv,ds,iter = solve_kepler_drift!(h,k,x0,v0,beta0,r0,
    s0,state,drift_first)
# Compute the Jacobian.  jacobian[i,j] is derivative of final state variable q[i]
# with respect to initial state variable q0[j], where q = {x,v} & q0 = {x0,v0}.
  delxv_jac = jac_delxv(x0,v0,k,s,beta0,h,drift_first)
#  println("computed jacobian with autodiff")
  # Add in partial derivatives with respect to x0, v0 and k:
  jacobian[1:6,1:7] = delxv_jac[1:6,1:7]
# Add in s and beta0 derivatives:
  compute_jacobian_kep_drift!(h,k,x0,v0,beta0,s,f,g,dfdt,dgdtm1,cx,sx,g1bs,g2bs,r0,r,jacobian,delxv_jac,drift_first)
else
#  println("Not elliptic or hyperbolic ",beta0," x0 ",x0)
  r= zero(T); fill!(state,zero(T)); rinv=zero(T); s=zero(T); ds=zero(T); iter = 0
end
# recompute beta:
state[8]= r
# These need to be updated. [ ]
state[9] = (state[2]*state[5]+state[3]*state[6]+state[4]*state[7])*rinv
# beta is element 10 of state:
state[10] = 2.0*k*rinv-(state[5]*state[5]+state[6]*state[6]+state[7]*state[7])
# s is element 11 of state:
state[11] = s
# ds is element 12 of state:
state[12] = ds
return iter
end

function compute_jacobian_kep_drift!(h::T,k::T,x0::Array{T,1},v0::Array{T,1},beta0::T,s::T,
  f::T,g::T,dfdt::T,dgdtm1::T,cx::T,sx::T,g1::T,g2::T,r0::T,r::T,jacobian::Array{T,2},drift_first::Bool) where {T <: Real}
# This needs to be updated to incorporate backwards drifts. [ ]
# Compute the Jacobian.  jacobian[i,j] is derivative of final state variable q[i]
# with respect to initial state variable q0[j], where q = {x,v,k} & q0 = {x0,v0,k}.
# Now, compute the Jacobian: (9/18/2017 notes)
g0 = one(T)-beta0*g2
# Expand g3 as a series if s is small:
sqb = sqrt(abs(beta0))
g3bs = G3(s*sqb,beta0)
#  g3bs = (s-g1)/beta0
if drift_first
  eta = dot(x0-h*v0,v0) 
else
  eta = dot(x0,v0) 
end
absv0 = norm(v0)
#dsdbeta = (2h-r0*(s*g0+g1)+k/beta0*(s*g0-g1)-eta*s*g1)/(2beta0*r)
# New expression derived on 8/14/2019:
dsdbeta = (h+eta*g2+2*k*g3bs-s*r)/(2beta0*r)
dsdr0 = -(2k/r0^2*dsdbeta+g1/r)
dsda0 = -g2/r
dsdv0 = -2absv0*dsdbeta
dsdk = 2/r0*dsdbeta-g3bs/r
dbetadr0 = -2k/r0^2
dbetadv0 = -2absv0
dbetadk  = 2/r0
# "p" for partial derivative:
for i=1:3
  pxpr0[i] = k/r0^2*g2*x0[i]+g1*v0[i]
  pxpa0[i] = g2*v0[i]
  pxpk[i]  = -g2/r0*x0[i]
  pxps[i]  = -k/r0*g1*x0[i]+(r0*g0+eta*g1)*v0[i]
  pxpbeta[i] = -k/(2beta0*r0)*(s*g1-2g2)*x0[i]+1/(2beta0)*(s*r0*g0-r0*g1+s*eta*g1-2*eta*g2)*v0[i]
  prvpr0[i] = k*g1/r0^2*x0[i]+g0*v0[i]
  prvpa0[i] = g1*v0[i]
  prvpk[i] = -g1/r0*x0[i]
  prvps[i] = -k*g0/r0*x0[i]+(-beta0*r0*g1+eta*g0)*v0[i]
  prvpbeta[i] = -k/(2beta0*r0)*(s*g0-g1)*x0[i]+1/(2beta0)*(-s*r0*beta0*g1+eta*s*g0-eta*g1)*v0[i]
end
prpr0 = g0
prpa0 = g1
prpk  = g2
prps = (k-beta0*r0)*g1+eta*g0
prpbeta = 1/(2beta0)*(s*(k-beta0*r0)*g1+eta*s*g0-eta*g1-2k*g2)
for i=1:3
  dxdr0[i] = pxps[i]*dsdr0 + pxpbeta[i]*dbetadr0 + pxpr0[i]
  dxda0[i] = pxps[i]*dsda0 + pxpa0[i]
  dxdv0[i] = pxps[i]*dsdv0 + pxpbeta[i]*dbetadv0
  dxdk[i]  = pxps[i]*dsdk  + pxpbeta[i]*dbetadk + pxpk[i]
  drvdr0[i] = prvps[i]*dsdr0 + prvpbeta[i]*dbetadr0 + prvpr0[i]
  drvda0[i] = prvps[i]*dsda0 + prvpa0[i]
  drvdv0[i] = prvps[i]*dsdv0 + prvpbeta[i]*dbetadv0
  drvdk[i]  = prvps[i]*dsdk  + prvpbeta[i]*dbetadk +prvpk[i]
end
drdr0 = prpr0 + prps*dsdr0 + prpbeta*dbetadr0
drda0 = prpa0 + prps*dsda0
drdv0 = prps*dsdv0 + prpbeta*dbetadv0
drdk  = prpk + prps*dsdk + prpbeta*dbetadk
for i=1:3
  vtmp[i] = dfdt*x0[i]+(dgdtm1+1.0)*v0[i]
  dvdr0[i] = (drvdr0[i]-drdr0*vtmp[i])/r
  dvda0[i] = (drvda0[i]-drda0*vtmp[i])/r
  dvdv0[i] = (drvdv0[i]-drdv0*vtmp[i])/r
  dvdk[i]  = (drvdk[i] -drdk *vtmp[i])/r
end
# Now, compute Jacobian:
for i=1:3
  jacobian[  i,  i] = f
  jacobian[  i,3+i] = g
  jacobian[3+i,  i] = dfdt
  jacobian[3+i,3+i] = dgdt+1.0
  jacobian[  i,7] = dxdk[i]
  jacobian[3+i,7] = dvdk[i]
end
for j=1:3
  for i=1:3
    jacobian[  i,  j] += dxdr0[i]*x0[j]/r0 + dxda0[i]*v0[j]
    jacobian[  i,3+j] += dxdv0[i]*v0[j]/absv0 + dxda0[i]*x0[j]
    jacobian[3+i,  j] += dvdr0[i]*x0[j]/r0 + dvda0[i]*v0[j]
    jacobian[3+i,3+j] += dvdv0[i]*v0[j]/absv0 + dvda0[i]*x0[j]
  end
end
jacobian[7,7]=one(T)
return
end

function compute_jacobian_kep_drift!(h::T,k::T,x0::Array{T,1},v0::Array{T,1},beta0::T,s::T,
  f::T,g::T,dfdt::T,dgdtm1::T,cx::T,sx::T,g1::T,g2::T,r0::T,r::T,jacobian::Array{T,2},
  delxv_jac::Array{T,2},drift_first::Bool) where {T <: Real}
# This needs to be updated to incorporate backwards drifts. [ ]
# Compute the Jacobian.  jacobian[i,j] is derivative of final state variable q[i]
# with respect to initial state variable q0[j], where q = {x,v,k} & q0 = {x0,v0,k}.
# Now, compute the Jacobian: (9/18/2017 notes)
g0 = one(T)-beta0*g2
# Expand g3 as a series if s is small:
sqb = sqrt(abs(beta0))
g3bs = G3(s*sqb,beta0)
#  g3bs = (s-g1)/beta0
if drift_first
  eta = dot(x0-h*v0,v0) 
  r0 = norm(x0-h*v0)
else
  eta = dot(x0,v0)
  r0 = norm(x0)
end
absv0 = norm(v0)
r0inv = inv(r0)
#dsdbeta = (2h-r0*(s*g0+g1)+k/beta0*(s*g0-g1)-eta*s*g1)/(2beta0*r)
# New expression derived on 8/14/2019:
dsdbeta = (h+eta*g2+2*k*g3bs-s*r)/(2beta0*r)
dsdr0 = -(2k*r0inv^2*dsdbeta+g1/r)
dsda0 = -g2/r
dsdv0 = -2absv0*dsdbeta
dsdk = 2*r0inv*dsdbeta-g3bs/r
dbetadr0 = -2k*r0inv^2
dbetadv0 = -2absv0
dbetadk  = 2r0inv
# Compute s & beta components of x0 & v0 derivatives:
for i=1:3
  dxdr0[i] = delxv_jac[  i,8]*dsdr0 + delxv_jac[  i,9]*dbetadr0
  dxda0[i] = delxv_jac[  i,8]*dsda0
  dxdv0[i] = delxv_jac[  i,8]*dsdv0 + delxv_jac[  i,9]*dbetadv0
  dxdk[i]  = delxv_jac[  i,8]*dsdk  + delxv_jac[  i,9]*dbetadk
  dvdr0[i] = delxv_jac[3+i,8]*dsdr0 + delxv_jac[3+i,9]*dbetadr0
  dvda0[i] = delxv_jac[3+i,8]*dsda0
  dvdv0[i] = delxv_jac[3+i,8]*dsdv0 + delxv_jac[3+i,9]*dbetadv0
  dvdk[i]  = delxv_jac[3+i,8]*dsdk  + delxv_jac[3+i,9]*dbetadk
end
# Now, compute Jacobian:
# Add in s & beta components of k derivative:
for i=1:3
  jacobian[  i,7] += dxdk[i]
  jacobian[3+i,7] += dvdk[i]
end
# Add in s & beta components of x0 & v0 derivatives:
for j=1:3
  for i=1:3
    if drift_first
      jacobian[  i,  j] +=    dxdr0[i]*(x0[j]-h*v0[j])*r0inv + dxda0[i]*v0[j]
      jacobian[  i,3+j] += -h*dxdr0[i]*(x0[j]-h*v0[j])*r0inv + dxdv0[i]*v0[j]/absv0 + dxda0[i]*(x0[j]-2*h*v0[j])
      jacobian[3+i,  j] +=    dvdr0[i]*(x0[j]-h*v0[j])*r0inv + dvda0[i]*v0[j]
      jacobian[3+i,3+j] += -h*dvdr0[i]*(x0[j]-h*v0[j])*r0inv + dvdv0[i]*v0[j]/absv0 + dvda0[i]*(x0[j]-2*h*v0[j])
    else
      jacobian[  i,  j] += dxdr0[i]*x0[j]*r0inv + dxda0[i]*v0[j]
      jacobian[  i,3+j] += dxdv0[i]*v0[j]/absv0 + dxda0[i]*x0[j]
      jacobian[3+i,  j] += dvdr0[i]*x0[j]*r0inv + dvda0[i]*v0[j]
      jacobian[3+i,3+j] += dvdv0[i]*v0[j]/absv0 + dvda0[i]*x0[j]
    end
  end
end
# Mass doesn't change:
jacobian[7,7]=one(T)
return
end
