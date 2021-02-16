# This code tests two functions: keplerij! and kep_elliptic!
#using PyPlot
#include("../src/kepler_solver_derivative.jl")
#include("../src/ttv.jl")

@testset "kep_elliptic" begin

# Define a constant of 1/3:
#const third = 1.0/3.0

function test_elliptic_derivative(dlnq::BigFloat)
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
h = 100.0 # 18-day timesteps
hbig = big(h)

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
state_diffbig=zeros(BigFloat,12)

xsave[1,1]=0.0
for jj=1:3
  xsave[jj+1,1]=x0[jj]
  xsave[jj+4,1]=v0[jj]
end
# Save beta:
xsave[8,1]=r0
beta0 = 2.0*k/r0-dot(v0,v0)
xsave[10,1]=beta0
jacobian=zeros(Float64,7,7)
#iter = kep_elliptic!(x0,v0,r0,k,h,beta0,s0,state,jacobian)
iter = kep_ell_hyp!(x0,v0,r0,k,h,beta0,s0,state,jacobian)
#println("Initial conditions: ",x0,v0)
#println("Final state: ",state)
#println("Differnce : ",state[2:4]-x0,state[5:7]-v0)
#read(STDIN,Char)

# Now, do finite differences at higher precision:
kbig = big(k); s0big = big(0.0);  statebig = big.(state)
x0big = big.(x0); v0big = big.(v0)
r0big = sqrt(x0big[1]*x0big[1]+x0big[2]*x0big[2]+x0big[3]*x0big[3])
beta0big = 2*kbig/r0big-dot(v0big,v0big)
#iter = kep_elliptic!(x0big,v0big,r0big,kbig,hbig,beta0big,s0big,statebig)
iter = kep_ell_hyp!(x0big,v0big,r0big,kbig,hbig,beta0big,s0big,statebig)
#println("Final state: ",statebig[2:7])

#read(STDIN,Char)
#s = statebig[11]
# Now, compute the Jacobian numerically:
#jac_num = zeros(Float64,7,7)
jac_num = zeros(BigFloat,7,7)
#dlnq = 1e-4
# jac_num[i,j]: derivative of (x_i,v_i,k) with respect to (x_{0,j},v_{0,j},k):
for j=1:3
  x0save = copy(x0big)
  dq = dlnq * x0big[j]
  if x0big[j] != 0.0
    x0big[j] +=  dq
  else
    dq = dlnq
    x0big[j] = dq
  end
  # Recompute quantities:
  r0big = sqrt(x0big[1]*x0big[1]+x0big[2]*x0big[2]+x0big[3]*x0big[3])
  beta0big = 2*kbig/r0big-dot(v0big,v0big)
#  iter = kep_elliptic!(x0big,v0big,r0big,kbig,hbig,beta0big,s0big,state_diffbig)
  iter = kep_ell_hyp!(x0big,v0big,r0big,kbig,hbig,beta0big,s0big,state_diffbig)
  x0big = copy(x0save)
  for i=1:3
    jac_num[  i,  j] = (state_diffbig[1+i]-statebig[1+i])/dq
    jac_num[3+i,  j] = (state_diffbig[4+i]-statebig[4+i])/dq
  end
  v0save = copy(v0big)
  dq = dlnq * v0big[j]
  if v0big[j] != 0.0
    v0big[j] +=  dq
  else
    dq = dlnq
    v0big[j] = dq
  end
  r0big = sqrt(x0big[1]*x0big[1]+x0big[2]*x0big[2]+x0big[3]*x0big[3])
  beta0big = 2*kbig/r0big-dot(v0big,v0big)
#  iter = kep_elliptic!(x0big,v0big,r0big,kbig,hbig,beta0big,s0big,state_diffbig)
  iter = kep_ell_hyp!(x0big,v0big,r0big,kbig,hbig,beta0big,s0big,state_diffbig)
  v0big = copy(v0save)
  for i=1:3
    jac_num[  i,3+j] = (state_diffbig[1+i]-statebig[1+i])/dq
    jac_num[3+i,3+j] = (state_diffbig[4+i]-statebig[4+i])/dq
  end
  # Now vary mass:
  ksave = copy(kbig)
  dq = kbig*dlnq
  kbig += dq
  r0big = sqrt(x0big[1]*x0big[1]+x0big[2]*x0big[2]+x0big[3]*x0big[3])
  beta0big = 2*kbig/r0big-dot(v0big,v0big)
#  iter = kep_elliptic!(x0big,v0big,r0big,kbig,hbig,beta0big,s0big,state_diffbig)
  iter = kep_ell_hyp!(x0big,v0big,r0big,kbig,hbig,beta0big,s0big,state_diffbig)
  for i=1:3
    jac_num[  i,7] = (state_diffbig[1+i]-statebig[1+i])/dq
    jac_num[3+i,7] = (state_diffbig[4+i]-statebig[4+i])/dq
  end
  kbig = copy(ksave)
  jac_num[7,  7] = 1.0
end
return xsave,jac_num,jacobian
end

# First try:
const KEPLER_TOL = 1e-12
xsave,jac_num1,jacobian=test_elliptic_derivative(big(1e-15))
# Second try:
#xsave,jac_num2,jacobian=test_elliptic_derivative(1e-6)


#println("Jac dlnq=1e-7 ")
#jac_num1
#println("Jac dlnq=1e-6 ")
#jac_num2
#jacobian
#jac_num1-jac_num2
#println("Fraction errors on Jacobian: ",jac_num1./jacobian-1.0)
#println("jac_num1: ",jac_num1)
emax = 0.0; imax = 0; jmax = 0
for i=1:7, j=1:7
  if jacobian[i,j] != 0.0
    diff = abs(jac_num1[i,j]/jacobian[i,j]-1.0)
    if  diff > emax
      emax = diff; imax = i; jmax = j
    end
  end
end
println("Maximum fractional error: ",emax," ",imax," ",jmax)
println("Maximum error jacobian: ",maximum(abs.(jacobian-jac_num1)))

#@test isapprox(jacobian,jac_num1)
@test isapprox(jacobian,jac_num1;norm=maxabs)
end
