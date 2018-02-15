# Wisdom & Hernandez version of Kepler solver, but with quartic convergence.

function cubic1(a::T, b::T, c::T) where {T <: Real}
a3 = a*third
Q = a3^2 - b*third
R = a3^3 + 0.5*(-a3*b + c)
if R^2 < Q^3
# println("Error in cubic solver ",R^2," ",Q^3)
 return -c/b
else
 A = -sign(R)*cbrt(abs(R) + sqrt(R*R - Q*Q*Q))
 if A == 0.0
   B = 0.0
  else
   B = Q/A
 end
 x1 = A + B - a3
 return x1
end
return
end

function calc_ds_opt(y::T,yp::T,ypp::T,yppp::T) where {T <: Real}
# Computes quartic Newton's update to equation y=0 using first through 3rd derivatives.
# Uses techniques outlined in Murray & Dermott for Kepler solver.
# Rearrange to reduce number of divisions:
num = y*yp
den1 = yp*yp-y*ypp*.5
den12 = den1*den1
den2 = yp*den12-num*.5*(ypp*den1-third*num*yppp)
return -y*den12/den2
end

function solve_kepler!(h::T,k::T,x0::Array{T,1},v0::Array{T,1},beta0::T,
   r0::T,s0::T,state::Array{T,1}) where {T <: Real}
zero = convert(typeof(h),0.0); one = convert(typeof(h),1.0)
# Solves elliptic Kepler's equation for both elliptic and hyperbolic cases.
# Initial guess (if s0 = 0):
r0inv = inv(r0)
beta0inv = inv(beta0)
signb = sign(beta0)
sqb = sqrt(signb*beta0)
zeta = k-r0*beta0
eta = dot(x0,v0)
if s0 == zero
  # Use cubic estimate:
  if zeta != zero
    s = cubic1(3eta/zeta,6r0/zeta,-6h/zeta)
  else
    s = h*r0inv
  end
else
  s = copy(s0)
end
s0 = copy(s)
y = zero; yp = one
iter = 0
ds = Inf
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
# Now, compute final values:
g1bs = 2.*sx*cx/sqb
g2bs = 2.*signb*sx^2*beta0inv
f = one - k*r0inv*g2bs # eqn (25)
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

function kep_ell_hyp!(x0::Array{T,1},v0::Array{T,1},r0::T,k::T,h::T,
  beta0::T,s0::T,state::Array{T,1}) where {T <: Real}
# Solves equation (35) from Wisdom & Hernandez for the elliptic case.
zero = convert(typeof(h),0.0); one = convert(typeof(h),1.0)
# Now, solve for s in elliptical Kepler case:
f = zero; g=zero; dfdt=zero; dgdt=zero; cx=zero;sx=zero;g1bs=zero;g2bs=zero
s=zero; ds = zero; r = zero;rinv=zero; iter=0
if beta0 > zero || beta0 < zero
   s,f,g,dfdt,dgdt,cx,sx,g1bs,g2bs,r,rinv,ds,iter = solve_kepler!(h,k,x0,v0,beta0,r0,
    s0,state)
else
  println("Not elliptic or hyperbolic ",beta0," x0 ",x0)
  r= zero; fill!(state,zero); rinv=zero; s=zero; ds=zero; iter = 0
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

function kep_ell_hyp!(x0::Array{T,1},v0::Array{T,1},r0::T,k::T,h::T,
  beta0::T,s0::T,state::Array{T,1},jacobian::Array{T,2}) where {T <: Real}
# Computes the Jacobian as well
# Solves equation (35) from Wisdom & Hernandez for the elliptic case.
zero = convert(typeof(h),0.0); one = convert(typeof(h),1.0)
r0inv = inv(r0)
beta0inv = inv(beta0)
# Now, solve for s in elliptical Kepler case:
f = zero; g=zero; dfdt=zero; dgdt=zero; cx=zero;sx=zero;g1bs=zero;g2bs=zero
s=zero; ds=zero; r = zero;rinv=zero; iter=0
if beta0 > zero || beta0 < zero
   s,f,g,dfdt,dgdt,cx,sx,g1bs,g2bs,r,rinv,ds,iter = solve_kepler!(h,k,x0,v0,beta0,r0,
    s0,state)
# Compute the Jacobian.  jacobian[i,j] is derivative of final state variable q[i]
# with respect to initial state variable q0[j], where q = {x,v} & q0 = {x0,v0}.
  fill!(jacobian,zero)
  compute_jacobian!(h,k,x0,v0,beta0,s,f,g,dfdt,dgdt,cx,sx,g1bs,g2bs,r0,r,jacobian)
else
  println("Not elliptic or hyperbolic ",beta0," x0 ",x0)
  r= zero; fill!(state,zero); rinv=zero; s=zero; ds=zero; iter = 0
end
# recompute beta:
state[8]= r
state[9] = (state[2]*state[5]+state[3]*state[6]+state[4]*state[7])*rinv
# beta is element 10 of state:
state[10] = 2.0*k*rinv-(state[5]*state[5]+state[6]*state[6]+state[7]*state[7])
# s is element 11 of state:
state[11] = s
# ds is element 12 of state:
state[12] = ds
return iter
end

function compute_jacobian!(h::T,k::T,x0::Array{T,1},v0::Array{T,1},beta0::T,s::T,
  f::T,g::T,dfdt::T,dgdt::T,cx::T,sx::T,g1::T,g2::T,r0::T,r::T,jacobian::Array{T,2}) where {T <: Real}
# Compute the Jacobian.  jacobian[i,j] is derivative of final state variable q[i]
# with respect to initial state variable q0[j], where q = {x,v,k} & q0 = {x0,v0,k}.
# Now, compute the Jacobian: (9/18/2017 notes)
zero = convert(typeof(h),0.0); one = convert(typeof(h),1.0)
g0 = one-beta0*g2
g3 = (s-g1)/beta0
eta = dot(x0,v0)  # unnecessary to divide by r0 for multiply for \dot\alpha_0
absv0 = sqrt(dot(v0,v0))
dsdbeta = (2h-r0*(s*g0+g1)+k/beta0*(s*g0-g1)-eta*s*g1)/(2beta0*r)
dsdr0 = -(2k/r0^2*dsdbeta+g1/r)
dsda0 = -g2/r
dsdv0 = -2absv0*dsdbeta
dsdk = 2/r0*dsdbeta-g3/r
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
  vtmp[i] = dfdt*x0[i]+dgdt*v0[i]
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
  jacobian[3+i,3+i] = dgdt
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
jacobian[7,7]=one
return
end
