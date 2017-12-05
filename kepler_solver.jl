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
if beta0 > 1e-15
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
  while iter == 0 || (abs(ds) > 1e-8 && iter < 10)
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
  end
#  if iter > 2
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

function kep_hyperbolic!(x0::Array{Float64,1},v0::Array{Float64,1},r0::Float64,dr0dt::Float64,k::Float64,h::Float64,beta0::Float64,s0::Float64,state::Array{Float64,1})
# Solves equation (35) from Wisdom & Hernandez for the hyperbolic case.

r0inv = inv(r0)
beta0inv = inv(beta0)
# Now, solve for s in hyperbolic Kepler case:
if beta0 < -1e-15
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
  while iter == 0 || (abs(ds) > 1e-8 && iter < 10)
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
  end
#  if iter > 2
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
