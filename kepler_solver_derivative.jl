# Wisdom & Hernandez version of Kepler solver, but with quartic
# convergence.

function calc_ds_opt(y,yp,ypp,yppp)
# Computes quartic Newton's update to equation y=0 using first through 3rd derivatives.
# Uses techniques outlined in Murray & Dermott for Kepler solver.
# Rearrange to reduce number of divisions:
num = y*yp
den1 = yp*yp-y*ypp*.5
den12 = den1*den1
den2 = yp*den12-num*.5*(ypp*den1-third*num*yppp)
return -y*den12/den2
end

function kep_elliptic!(x0::Array{Float64,1},v0::Array{Float64,1},r0::Float64,dr0dt::Float64,k::Float64,h::Float64,beta0::Float64,s0::Float64,state::Array{Float64,1})
# Solves equation (35) from Wisdom & Hernandez for the elliptic case.

r0inv = inv(r0)
beta0inv = inv(beta0)
# Now, solve for s in elliptical Kepler case:
if beta0 > 0.0
# Initial guess (if s0 = 0):
  if s0 == 0.0
    s = h*r0inv
  else
    s = copy(s0)
  end
  s0 = copy(s)
  sqb = sqrt(beta0)
  y = 0.0; yp = 1.0
  iter = 0
  ds = Inf
  fac1 = k-r0*beta0
  fac2 = r0*dr0dt
  while iter == 0 || (abs(ds) > KEPLER_TOL && iter < 10)
    xx = sqb*s
    sx = sqb*sin(xx)
    cx = cos(xx)
# Third derivative:
    yppp = fac1*cx - fac2*sx
# Take derivative:
    yp = (-yppp+ k)*beta0inv
# Second derivative:
    ypp = fac1*beta0inv*sx + fac2*cx
    y  = (-ypp + fac2 +k*s)*beta0inv - h  # eqn 35
# Now, compute fourth-order estimate:
    ds = calc_ds_opt(y,yp,ypp,yppp)
    s += ds
    iter +=1
#    println(iter," ds: ",ds)
  end
#  if iter > 1
#    println(iter," ",s," ",s/s0-1," ds: ",ds)
#  end
# Since we updated s, need to recompute:
  xx = 0.5*sqb*s; sx = sin(xx) ; cx = cos(xx)
# Now, compute final values:
  g1bs = 2.*sx*cx/sqb
  g2bs = 2.*sx^2*beta0inv
  f = 1.0 - k*r0inv*g2bs # eqn (25)
  g = r0*g1bs + fac2*g2bs # eqn (27)
  for j=1:3
# Position is components 2-4 of state:
    state[1+j] = x0[j]*f+v0[j]*g
  end
  r = sqrt(state[2]*state[2]+state[3]*state[3]+state[4]*state[4])
  rinv = inv(r)
  dfdt = -k*g1bs*rinv*r0inv
  dgdt = r0*(1.0-beta0*g2bs+dr0dt*g1bs)*rinv
  for j=1:3
# Velocity is components 5-7 of state:
    state[4+j] = x0[j]*dfdt+v0[j]*dgdt
  end
else
  println("Not elliptic ",beta0," x0 ",x0)
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

function kep_elliptic!(x0::Array{Float64,1},v0::Array{Float64,1},r0::Float64,dr0dt::Float64,k::Float64,h::Float64,beta0::Float64,s0::Float64,state::Array{Float64,1},jacobian::Array{Float64,2})
# Computes the Jacobian as well
# Solves equation (35) from Wisdom & Hernandez for the elliptic case.

r0inv = inv(r0)
beta0inv = inv(beta0)
# Now, solve for s in elliptical Kepler case:
if beta0 > 0.0
# Initial guess (if s0 = 0):
  if s0 == 0.0
    s = h*r0inv
  else
    s = copy(s0)
  end
  s0 = copy(s)
  sqb = sqrt(beta0)
  y = 0.0; yp = 1.0
  iter = 0
  ds = Inf
  fac1 = k-r0*beta0
  fac2 = r0*dr0dt
  while iter == 0 || (abs(ds) > KEPLER_TOL && iter < 10)
    xx = sqb*s
    sx = sqb*sin(xx)
    cx = cos(xx)
# Third derivative:
    yppp = fac1*cx - fac2*sx
# Take derivative:
    yp = (-yppp+ k)*beta0inv
# Second derivative:
    ypp = fac1*beta0inv*sx + fac2*cx
    y  = (-ypp + fac2 +k*s)*beta0inv - h  # eqn 35
# Now, compute fourth-order estimate:
    ds = calc_ds_opt(y,yp,ypp,yppp)
    s += ds
    iter +=1
#    println(iter," ds: ",ds)
  end
#  if iter > 1
#    println(iter," ",s," ",s/s0-1," ds: ",ds)
#  end
# Since we updated s, need to recompute:
  xx = 0.5*sqb*s; sx = sin(xx) ; cx = cos(xx)
# Now, compute final values:
  g1bs = 2.*sx*cx/sqb
  g2bs = 2.*sx^2*beta0inv
  f = 1.0 - k*r0inv*g2bs # eqn (25)
  g = r0*g1bs + fac2*g2bs # eqn (27)
  for j=1:3
# Position is components 2-4 of state:
    state[1+j] = x0[j]*f+v0[j]*g
  end
  r = sqrt(state[2]*state[2]+state[3]*state[3]+state[4]*state[4])
  rinv = inv(r)
  dfdt = -k*g1bs*rinv*r0inv
  dgdt = r0*(1.0-beta0*g2bs+dr0dt*g1bs)*rinv
  for j=1:3
# Velocity is components 5-7 of state:
    state[4+j] = x0[j]*dfdt+v0[j]*dgdt
  end
# Now, compute the jacobian:
  fill!(jacobian,0.0)
  compute_jacobian!(h,k,x0,v0,beta0,s,f,g,dfdt,dgdt,cx,sx,g1bs,g2bs,r0,dr0dt,r,jacobian)
else
  println("Not elliptic ",beta0," x0 ",x0)
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
# Compute the Jacobian.  jacobian[i,j] is derivative of final state variable q[i]
# with respect to initial state variable q0[j], where q = {x,v} & q0 = {x0,v0}.

return iter
end

function kep_hyperbolic!(x0::Array{Float64,1},v0::Array{Float64,1},r0::Float64,dr0dt::Float64,k::Float64,h::Float64,beta0::Float64,s0::Float64,state::Array{Float64,1})
# Solves equation (35) from Wisdom & Hernandez for the hyperbolic case.

r0inv = inv(r0)
beta0inv = inv(beta0)
# Now, solve for s in hyperbolic Kepler case:
if beta0 < 0.0
# Initial guess (if s0 = 0):
  if s0 == 0.0
    s = h*r0inv
  else
    s = copy(s0)
  end
  s0 = copy(s)
  sqb = sqrt(-beta0)
  y = 0.0; yp = 1.0
  iter = 0
  ds = Inf
  fac1 = k-r0*beta0
  fac2 = r0*dr0dt
  while iter == 0 || (abs(ds) > KEPLER_TOL && iter < 10)
    xx = sqb*s; cx = cosh(xx); sx = sqb*(exp(xx)-cx)
# Third derivative:
    yppp = fac1*cx + fac2*sx
# Take derivative:
    yp = (-yppp+ k)*beta0inv
# Second derivative:
    ypp = -fac1*beta0inv*sx  + fac2*cx
    y  = (-ypp +fac2 +k*s)*beta0inv - h  # eqn 35
# Now, compute fourth-order estimate:
    ds = calc_ds_opt(y,yp,ypp,yppp)
    s += ds
    iter +=1
#    println(iter," ds: ",ds)
  end
#  if iter > 1
#    #println("iter: ",iter," ds/s: ",ds/s0)
#    println(iter," ",s," ",s/s0-1," ds: ",ds)
#  end
  xx = 0.5*sqb*s; cx = cosh(xx); sx = exp(xx)-cx
# Now, compute final values:
  g1bs = 2.0*sx*cx/sqb
  g2bs = -2.0*sx^2*beta0inv
  f = 1.0 - k*r0inv*g2bs # eqn (25)
  g = r0*g1bs + fac2*g2bs # eqn (27)
  for j=1:3
    state[1+j] = x0[j]*f+v0[j]*g
  end
  # r = norm(x)
  r = sqrt(state[2]*state[2]+state[3]*state[3]+state[4]*state[4])
  rinv = inv(r)
  dfdt = -k*g1bs*rinv*r0inv
  dgdt = r0*(1.0-beta0*g2bs+dr0dt*g1bs)*rinv
  for j=1:3
# Velocity is components 5-7 of state:
    state[4+j] = x0[j]*dfdt+v0[j]*dgdt
  end
else
  println("Not hyperbolic",beta0," x0 ",x0)
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

function kep_hyperbolic!(x0::Array{Float64,1},v0::Array{Float64,1},r0::Float64,dr0dt::Float64,k::Float64,h::Float64,beta0::Float64,s0::Float64,state::Array{Float64,1},jacobian::Array{Float64,2})
# Solves equation (35) from Wisdom & Hernandez for the hyperbolic case.

r0inv = inv(r0)
beta0inv = inv(beta0)
# Now, solve for s in hyperbolic Kepler case:
if beta0 < 0.0
# Initial guess (if s0 = 0):
  if s0 == 0.0
    s = h*r0inv
  else
    s = copy(s0)
  end
  s0 = copy(s)
  sqb = sqrt(-beta0)
  y = 0.0; yp = 1.0
  iter = 0
  ds = Inf
  fac1 = k-r0*beta0
  fac2 = r0*dr0dt
  while iter == 0 || (abs(ds) > KEPLER_TOL && iter < 10)
    xx = sqb*s; cx = cosh(xx); sx = sqb*(exp(xx)-cx)
# Third derivative:
    yppp = fac1*cx + fac2*sx
# Take derivative:
    yp = (-yppp+ k)*beta0inv
# Second derivative:
    ypp = -fac1*beta0inv*sx  + fac2*cx
    y  = (-ypp +fac2 +k*s)*beta0inv - h  # eqn 35
# Now, compute fourth-order estimate:
    ds = calc_ds_opt(y,yp,ypp,yppp)
    s += ds
    iter +=1
#    println(iter," ds: ",ds)
  end
#  if iter > 1
#    #println("iter: ",iter," ds/s: ",ds/s0)
#    println(iter," ",s," ",s/s0-1," ds: ",ds)
#  end
  xx = 0.5*sqb*s; cx = cosh(xx); sx = exp(xx)-cx
# Now, compute final values:
  g1bs = 2.0*sx*cx/sqb
  g2bs = -2.0*sx^2*beta0inv
  f = 1.0 - k*r0inv*g2bs # eqn (25)
  g = r0*g1bs + fac2*g2bs # eqn (27)
  for j=1:3
    state[1+j] = x0[j]*f+v0[j]*g
  end
  # r = norm(x)
  r = sqrt(state[2]*state[2]+state[3]*state[3]+state[4]*state[4])
  rinv = inv(r)
  dfdt = -k*g1bs*rinv*r0inv
  dgdt = r0*(1.0-beta0*g2bs+dr0dt*g1bs)*rinv
  for j=1:3
# Velocity is components 5-7 of state:
    state[4+j] = x0[j]*dfdt+v0[j]*dgdt
  end
# Now, compute the jacobian:
  fill!(jacobian,0.0)
  compute_jacobian!(h,k,x0,v0,beta0,s,f,g,dfdt,dgdt,cx,sx,g1bs,g2bs,r0,dr0dt,r,jacobian)
else
  println("Not hyperbolic",beta0," x0 ",x0)
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

function compute_jacobian!(h::Float64,k::Float64,x0::Array{Float64,1},v0::Array{Float64,1},beta0::Float64,s::Float64,f::Float64,g::Float64,dfdt::Float64,dgdt::Float64,cx::Float64,sx::Float64,g1::Float64,g2::Float64,r0::Float64,dr0dt::Float64,r::Float64,jacobian::Array{Float64,2})
# Compute the Jacobian.  jacobian[i,j] is derivative of final state variable q[i]
# with respect to initial state variable q0[j], where q = {x,v,k} & q0 = {x0,v0,k}.
# Now, compute the Jacobian: (9/18/2017 notes)
#g0 = cx^2-sx^2
g0 = 1.0-beta0*g2
g3 = (s-g1)/beta0
dotalpha0 = r0*dr0dt  # unnecessary to divide by r0 for dr0dt & multiply for \dot\alpha_0
absv0 = sqrt(dot(v0,v0))
dsdbeta = (2h-r0*(s*g0+g1)+k/beta0*(s*g0-g1)-dotalpha0*s*g1)/(2beta0*r)
dsdr0 = -(2k/r0^2*dsdbeta+g1/r)
dsda0 = -g2/r
dsdv0 = -2absv0*dsdbeta
dsdk = 2/r0*dsdbeta-g3/r
dbetadr0 = -2k/r0^2
dbetadv0 = -2absv0
dbetadk  = 2/r0
# "p" for partial derivative:
#pxpr0 = zeros(Float64,3); pxpa0=zeros(Float64,3); pxpk=zeros(Float64,3); pxps=zeros(Float64,3); pxpbeta=zeros(Float64,3)
#dxdr0 = zeros(Float64,3); dxda0=zeros(Float64,3); dxdk=zeros(Float64,3); dxdv0 =zeros(Float64,3)
#prvpr0 = zeros(Float64,3); prvpa0=zeros(Float64,3); prvpk=zeros(Float64,3); prvps=zeros(Float64,3); prvpbeta=zeros(Float64,3)
#drvdr0 = zeros(Float64,3); drvda0=zeros(Float64,3); drvdk=zeros(Float64,3); drvdv0=zeros(Float64,3)
#vtmp = zeros(Float64,3); dvdr0 = zeros(Float64,3); dvda0=zeros(Float64,3); dvdv0=zeros(Float64,3); dvdk=zeros(Float64,3)
for i=1:3
  pxpr0[i] = k/r0^2*g2*x0[i]+g1*v0[i]
  pxpa0[i] = g2*v0[i]
  pxpk[i]  = -g2/r0*x0[i]
  pxps[i]  = -k/r0*g1*x0[i]+(r0*g0+dotalpha0*g1)*v0[i]
  pxpbeta[i] = -k/(2beta0*r0)*(s*g1-2g2)*x0[i]+1/(2beta0)*(s*r0*g0-r0*g1+s*dotalpha0*g1-2*dotalpha0*g2)*v0[i]
  prvpr0[i] = k*g1/r0^2*x0[i]+g0*v0[i]
  prvpa0[i] = g1*v0[i]
  prvpk[i] = -g1/r0*x0[i]
  prvps[i] = -k*g0/r0*x0[i]+(-beta0*r0*g1+dotalpha0*g0)*v0[i]
  prvpbeta[i] = -k/(2beta0*r0)*(s*g0-g1)*x0[i]+1/(2beta0)*(-s*r0*beta0*g1+dotalpha0*s*g0-dotalpha0*g1)*v0[i]
end
prpr0 = g0
prpa0 = g1
prpk  = g2
prps = (k-beta0*r0)*g1+dotalpha0*g0
prpbeta = 1/(2beta0)*(s*(k-beta0*r0)*g1+dotalpha0*s*g0-dotalpha0*g1-2k*g2)
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
jacobian[7,7]=1.0
return
end
