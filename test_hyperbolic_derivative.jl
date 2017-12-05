using PyPlot
#include("kepler_solver_derivative.jl")
include("ttv.jl")

# Define a constant of 1/3:
#const third = 1.0/3.0

function test_elliptic_derivative(dlnq)
# Call as: save,jac_num,jacobian=test_elliptic_derivative(1e-4)
# This routine runs a test of the kep_elliptic_jacobian function in kepler_solver.jl
# Define the central force constant in terms of AU and days:
k = (2.0*pi/365.25)^2
# Initial position at 1 AU:
x0 = [0.01,1.0,0.01]
#r0 = norm(x0)
# Circular velocity:
r0 = sqrt(x0[1]*x0[1]+x0[2]*x0[2]+x0[3]*x0[3])
vcirc = sqrt(k/r0)
# Define initial velocity at apastron:
v0 = [.9*vcirc,0.01*vcirc,0.01*vcirc]  # The eccentricity is about ~2(1-v0/vcirc).
dr0dt = (x0[1]*v0[1]+x0[2]*v0[2]+x0[3]*v0[3])/r0
h = 100.0 # 18-day timesteps

s0::Float64 = 0.0
x = zeros(Float64,3)
v = zeros(Float64,3)
s::Float64 = 0.0
t = 0.0
ssave = zeros(2)
nsteps = 1
xsave=zeros(Float64,12,nsteps)  # First is time (1,:); next three are position (2-4,:); next three velocity (5-7,:); then r (8,:), drdt (9,:); then beta (10,:); finally s & ds (11:12,:)
# Create a variable to store the state at the end of a step:
state=zeros(Float64,12)
state_plus=zeros(Float64,12)
state_minus=zeros(Float64,12)

xsave[1,1]=0.0
for jj=1:3
  xsave[jj+1,1]=x0[jj]
  xsave[jj+4,1]=v0[jj]
end
# Save beta:
xsave[8,1]=r0
xsave[9,1]=dr0dt
beta0 = 2.0*k/r0-dot(v0,v0)
xsave[10,1]=beta0
jacobian=zeros(Float64,7,7)
iter = kep_elliptic!(x0,v0,r0,dr0dt,k,h,beta0,s0,state,jacobian)
state_save = copy(state)
println("Initial conditions: ",x0,v0)
iter = kep_elliptic!(x0,v0,r0,dr0dt,k,h,beta0,s0,state)
println("Final state: ",state[2:7])

#read(STDIN,Char)
s = state[11]
# Now, compute the Jacobian numerically:
jac_num = zeros(Float64,7,7)
#dlnq = 1e-4
# jac_num[i,j]: derivative of (x_i,v_i,k) with respect to (x_{0,j},v_{0,j},k):
for j=1:3
#  x0 .= x0save
  x0save = copy(x0)
  dq = dlnq * x0[j]
  if x0[j] != 0.0
    x0[j] -=  dq
  else
    dq = dlnq
    x0[j] = -dq
  end
  # Recompute quantities:
  r0 = sqrt(x0[1]*x0[1]+x0[2]*x0[2]+x0[3]*x0[3])
  dr0dt = (x0[1]*v0[1]+x0[2]*v0[2]+x0[3]*v0[3])/r0
  beta0 = 2.0*k/r0-dot(v0,v0)
  iter = kep_elliptic!(x0,v0,r0,dr0dt,k,h,beta0,s0,state_minus)
  # Carry out double-sided derivative:
  if x0[j] != 0.0
    x0[j] +=  2.*dq
  else
    dq = dlnq
    x0[j] = dq
  end
  # Recompute quantities:
  r0 = sqrt(x0[1]*x0[1]+x0[2]*x0[2]+x0[3]*x0[3])
  dr0dt = (x0[1]*v0[1]+x0[2]*v0[2]+x0[3]*v0[3])/r0
  beta0 = 2.0*k/r0-dot(v0,v0)
  iter = kep_elliptic!(x0,v0,r0,dr0dt,k,h,beta0,s0,state_plus)
  for i=1:3
    jac_num[  i,  j] = .5*(state_plus[1+i]-state_minus[1+i])/dq
    jac_num[3+i,  j] = .5*(state_minus[4+i]-state_minus[4+i])/dq
  end
#  x0 .= x0save
  x0 = copy(x0save)
#  v0 .= v0save
  v0save = copy(v0)
  dq = dlnq * v0[j]
  if v0[j] != 0.0
    v0[j] -=  dq
  else
    dq = dlnq
    v0[j] = -dq
  end
  r0 = sqrt(x0[1]*x0[1]+x0[2]*x0[2]+x0[3]*x0[3])
  dr0dt = (x0[1]*v0[1]+x0[2]*v0[2]+x0[3]*v0[3])/r0
  beta0 = 2.0*k/r0-dot(v0,v0)
  iter = kep_elliptic!(x0,v0,r0,dr0dt,k,h,beta0,s0,state_minus)
  if v0[j] != 0.0
    v0[j] +=  2dq
  else
    dq = dlnq
    v0[j] = dq
  end
  r0 = sqrt(x0[1]*x0[1]+x0[2]*x0[2]+x0[3]*x0[3])
  dr0dt = (x0[1]*v0[1]+x0[2]*v0[2]+x0[3]*v0[3])/r0
  beta0 = 2.0*k/r0-dot(v0,v0)
  iter = kep_elliptic!(x0,v0,r0,dr0dt,k,h,beta0,s0,state_plus)
#  v0 .= v0save
  v0 = copy(v0save)
  for i=1:3
    jac_num[  i,3+j] = .5*(state_plus[1+i]-state_minus[1+i])/dq
    jac_num[3+i,3+j] = .5*(state_plus[4+i]-state_minus[4+i])/dq
  end
  # Now vary mass:
#  k = copy(k0)
  ksave = copy(k)
  dq = k*dlnq
  k -= dq
  r0 = sqrt(x0[1]*x0[1]+x0[2]*x0[2]+x0[3]*x0[3])
  dr0dt = (x0[1]*v0[1]+x0[2]*v0[2]+x0[3]*v0[3])/r0
  beta0 = 2.0*k/r0-dot(v0,v0)
  iter = kep_elliptic!(x0,v0,r0,dr0dt,k,h,beta0,s0,state_minus)
  k += 2*dq
  r0 = sqrt(x0[1]*x0[1]+x0[2]*x0[2]+x0[3]*x0[3])
  dr0dt = (x0[1]*v0[1]+x0[2]*v0[2]+x0[3]*v0[3])/r0
  beta0 = 2.0*k/r0-dot(v0,v0)
  iter = kep_elliptic!(x0,v0,r0,dr0dt,k,h,beta0,s0,state_plus)
  for i=1:3
    jac_num[  i,7] = .5*(state_plus[1+i]-state_minus[1+i])/dq
    jac_num[3+i,7] = .5*(state_plus[4+i]-state_minus[4+i])/dq
  end
  k = copy(ksave)
  jac_num[7,  7] = 1.0
end
return xsave,jac_num,jacobian
end

# First try:
xsave,jac_num1,jacobian=test_elliptic_derivative(1e-7)
# Second try:
xsave,jac_num2,jacobian=test_elliptic_derivative(1e-6)


println("Jac dlnq=1e-7 ")
#jac_num1
println("Jac dlnq=1e-6 ")
#jac_num2
#jacobian
#jac_num1-jac_num2
println(jac_num1./jacobian-1.)

# Next, try computing two-body Keplerian Jacobian:

n = 2
t0 = 7257.93115525
h  = 0.05
tmax = 600.0
dlnq = .3e-3

elements = readdlm("elements.txt",',')
elements[2,1] = 0.75
#elements[1,1] = 1e-5

m =zeros(n)
x0=zeros(NDIM,n)
v0=zeros(NDIM,n)

for k=1:n
  m[k] = elements[k,1]
end

x0,v0 = init_nbody(elements,t0,n)
# Tilt the orbits a bit:
x0[2,1] = 5e-1*sqrt(x0[1,1]^2+x0[3,1]^2)
x0[2,2] = -5e-1*sqrt(x0[1,2]^2+x0[3,2]^2)
v0[2,1] = 5e-1*sqrt(v0[1,1]^2+v0[3,1]^2)
v0[2,2] = -5e-1*sqrt(v0[1,2]^2+v0[3,2]^2)
# Reduce the masses to make it hyperbolic:
m *= 1e-1


jac_ij = zeros(14,14)
i=1 ; j=2
x = copy(x0) ; v=copy(v0)
# Predict values of s:
keplerij!(m,x,v,i,j,h,jac_ij)
x0 = copy(x) ; v0 = copy(v)
keplerij!(m,x,v,i,j,h,jac_ij)

# xtest = copy(x0) ; vtest=copy(v0)
# keplerij!(m,xtest,vtest,i,j,h,jac_ij)
# println("Test of jacobian vs. none: ",maximum(abs(x-xtest)),maximum(abs(v-vtest)))

# Now, compute the derivatives numerically:
jac_ij_num = zeros(14,14)
xsave = copy(x)
vsave = copy(v)
msave = copy(m)

for jj=1:3
  # Initial positions, velocities & masses:
  xm = copy(x0)
  vm = copy(v0)
  m = copy(msave)
  dq = dlnq * xm[jj,i]
  if xm[jj,i] != 0.0
    xm[jj,i] -=  dq
  else
    dq = dlnq
    xm[jj,i] = -dq
  end
  keplerij!(m,xm,vm,i,j,h)
  xp = copy(x0)
  vp = copy(v0)
  if xm[jj,i] != 0.0
    xp[jj,i] +=  dq
  else
    dq = dlnq
    xp[jj,i] = dq
  end
  keplerij!(m,xp,vp,i,j,h)
  # Now x & v are final positions & velocities after time step
  for k=1:3
    jac_ij_num[   k,  jj] = .5*(xp[k,i]-xm[k,i])/dq
    jac_ij_num[ 3+k,  jj] = .5*(vp[k,i]-vm[k,i])/dq
    jac_ij_num[ 7+k,  jj] = .5*(xp[k,j]-xm[k,j])/dq
    jac_ij_num[10+k,  jj] = .5*(vp[k,j]-vm[k,j])/dq
  end
  xm=copy(x0)
  vm=copy(v0)
  m=copy(msave)
  dq = dlnq * vm[jj,i]
  if vm[jj,i] != 0.0
    vm[jj,i] -=  dq
  else
    dq = dlnq
    vm[jj,i] = -dq
  end
  keplerij!(m,xm,vm,i,j,h)
  xp=copy(x0)
  vp=copy(v0)
  m=copy(msave)
  if vp[jj,i] != 0.0
    vp[jj,i] +=  dq
  else
    dq = dlnq
    vp[jj,i] = dq
  end
  keplerij!(m,xp,vp,i,j,h)
  for k=1:3
    jac_ij_num[   k,3+jj] = .5*(xp[k,i]-xm[k,i])/dq
    jac_ij_num[ 3+k,3+jj] = .5*(vp[k,i]-vm[k,i])/dq
    jac_ij_num[ 7+k,3+jj] = .5*(xp[k,j]-xm[k,j])/dq
    jac_ij_num[10+k,3+jj] = .5*(vp[k,j]-vm[k,j])/dq
  end
end
# Now vary mass of inner planet:
xm=copy(x0)
vm=copy(v0)
mm=copy(msave)
dq = mm[i]*dlnq
mm[i] -= dq
keplerij!(mm,xm,vm,i,j,h)
xp=copy(x0)
vp=copy(v0)
mp=copy(msave)
dq = mp[i]*dlnq
mp[i] += dq
keplerij!(mp,xp,vp,i,j,h)
for k=1:3
  jac_ij_num[   k,7] = .5*(xp[k,i]-xm[k,i])/dq
  jac_ij_num[ 3+k,7] = .5*(vp[k,i]-vm[k,i])/dq
  jac_ij_num[ 7+k,7] = .5*(xp[k,j]-xm[k,j])/dq
  jac_ij_num[10+k,7] = .5*(vp[k,j]-vm[k,j])/dq
end
# The mass doesn't change:
jac_ij_num[7,7] =  1.0
for jj=1:3
  # Now vary parameters of outer planet:
  xm = copy(x0)
  vm = copy(v0)
  m = copy(msave)
  dq = dlnq * xm[jj,j]
  if xm[jj,j] != 0.0
    xm[jj,j] -=  dq
  else
    dq = dlnq
    xm[jj,j] = -dq
  end
  keplerij!(m,xm,vm,i,j,h)
  xp = copy(x0)
  vp = copy(v0)
  if xp[jj,j] != 0.0
    xp[jj,j] +=  dq
  else
    dq = dlnq
    xp[jj,j] = dq
  end
  keplerij!(m,xp,vp,i,j,h)
  for k=1:3
    jac_ij_num[   k,7+jj] = .5*(xp[k,i]-xm[k,i])/dq
    jac_ij_num[ 3+k,7+jj] = .5*(vp[k,i]-vm[k,i])/dq
    jac_ij_num[ 7+k,7+jj] = .5*(xp[k,j]-xm[k,j])/dq
    jac_ij_num[10+k,7+jj] = .5*(vp[k,j]-vm[k,j])/dq
  end
  xm=copy(x0)
  vm=copy(v0)
  m=copy(msave)
  dq = dlnq * vm[jj,j]
  if vm[jj,j] != 0.0
    vm[jj,j] -=  dq
  else
    dq = dlnq
    vm[jj,j] = -dq
  end
  keplerij!(m,xm,vm,i,j,h)
  xp=copy(x0)
  vp=copy(v0)
  if vp[jj,j] != 0.0
    vp[jj,j] +=  dq
  else
    dq = dlnq
    vp[jj,j] = dq
  end
  keplerij!(m,xp,vp,i,j,h)
  for k=1:3
    jac_ij_num[   k,10+jj] = .5*(xp[k,i]-xm[k,i])/dq
    jac_ij_num[ 3+k,10+jj] = .5*(vp[k,i]-vm[k,i])/dq
    jac_ij_num[ 7+k,10+jj] = .5*(xp[k,j]-xm[k,j])/dq
    jac_ij_num[10+k,10+jj] = .5*(vp[k,j]-vm[k,j])/dq
  end
end
# Now vary mass of outer planet:
xm = copy(x0)
vm = copy(v0)
mm = copy(msave)
dq = mm[j]*dlnq
mm[j] -= dq
keplerij!(mm,xm,vm,i,j,h)
xp = copy(x0)
vp = copy(v0)
mp = copy(msave)
dq = mp[j]*dlnq
mp[j] += dq
keplerij!(mp,xp,vp,i,j,h)
for k=1:3
  jac_ij_num[   k,14] = .5*(xp[k,i]-xm[k,i])/dq
  jac_ij_num[ 3+k,14] = .5*(vp[k,i]-vm[k,i])/dq
  jac_ij_num[ 7+k,14] = .5*(xp[k,j]-xm[k,j])/dq
  jac_ij_num[10+k,14] = .5*(vp[k,j]-vm[k,j])/dq
end
# The mass doesn't change:
jac_ij_num[14,14] =  1.0

println(jac_ij./jac_ij_num)
emax = 0.0; imax = 0; jmax = 0
for i=1:14, j=1:14
  if jac_ij[i,j] != 0.0
    diff = abs(jac_ij_num[i,j]/jac_ij[i,j]-1.0)
    if  diff > emax
      emax = diff; imax = i; jmax = j
    end
  end
end
println("Maximum fractional error: ",emax," ",imax," ",jmax)
