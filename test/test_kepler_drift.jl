# This code tests kep_drift_ell_hyp!

@testset "kepler_drift" begin


function test_kepler_drift(dlnq::BigFloat,drift_first::Bool)
# Call as: save,jac_num,jacobian=test_kepler_drift(1e-4,true)
# This routine runs a test of the kep_elliptic_jacobian function in kepler_solver.jl
# Define the central force constant in terms of AU and days:
k = (2.0*pi/365.25)^2
# Initial position at 1 AU:
x0 = [0.01,1.0,0.01]
# Circular velocity:
r0 = sqrt(x0[1]*x0[1]+x0[2]*x0[2]+x0[3]*x0[3])
vcirc = sqrt(k/r0)
# Define initial velocity at apastron:
v0 = [.9*vcirc,0.01*vcirc,0.01*vcirc]  # The eccentricity is about ~2(1-v0/vcirc).
h = 18.0 # 18-day timesteps
hbig = big(h)

s0::Float64 = 0.0
x = zeros(Float64,3)
v = zeros(Float64,3)
s::Float64 = 0.0
t = 0.0
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
iter = kep_drift_ell_hyp!(x0,v0,k,h,s0,state,jacobian,drift_first)

# Now, do finite differences at higher precision:
kbig = big(k); s0big = big(0.0);  statebig = big.(state)
x0big = big.(x0); v0big = big.(v0)
r0big = sqrt(x0big[1]*x0big[1]+x0big[2]*x0big[2]+x0big[3]*x0big[3])
beta0big = 2*kbig/r0big-dot(v0big,v0big)
jacobian_big =zeros(BigFloat,7,7)
iter = kep_drift_ell_hyp!(x0big,v0big,kbig,hbig,s0big,statebig,jacobian_big,drift_first)
jac_frac = jacobian./convert(Array{Float64,2},jacobian_big).-1.0
println("Fractional Jacobian difference: ",maxabs(jac_frac[.~isnan.(jac_frac)]))

# Now, compute the Jacobian numerically:
jac_num = zeros(BigFloat,7,7)
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
  iter = kep_drift_ell_hyp!(x0big,v0big,kbig,hbig,s0big,state_diffbig,drift_first)
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
  iter = kep_drift_ell_hyp!(x0big,v0big,kbig,hbig,s0big,state_diffbig,drift_first)
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
  iter = kep_drift_ell_hyp!(x0big,v0big,kbig,hbig,s0big,state_diffbig,drift_first)
  for i=1:3
    jac_num[  i,7] = (state_diffbig[1+i]-statebig[1+i])/dq
    jac_num[3+i,7] = (state_diffbig[4+i]-statebig[4+i])/dq
  end
  kbig = copy(ksave)
  jac_num[7,  7] = 1.0
end
println("Maximum jac_big-jac_num: ",maxabs(convert(Array{Float64,2},jacobian_big-jac_num)))
#println(convert(Array{Float64,2},jacobian_big))
#println(convert(Array{Float64,2},jacobian_big./jac_num.-1.0))
return xsave,jac_num,jacobian
end

# First try:
xsave,jac_num1,jacobian=test_kepler_drift(big(1e-20),true)
# Second try:
xsave,jac_num1,jacobian=test_kepler_drift(big(1e-20),false)

emax = 0.0; imax = 0; jmax = 0
for i=1:7, j=1:7
  if jacobian[i,j] != 0.0
    diff = abs(convert(Float64,jac_num1[i,j])/jacobian[i,j]-1.0)
    if  diff > emax
      emax = diff; imax = i; jmax = j
    end
  end
end
println("Maximum fractional error: ",emax," ",imax," ",jmax)
println("Maximum jacobian-jac_num: ",maxabs(jacobian-convert(Array{Float64,2},jac_num1)))

@test isapprox(jacobian,jac_num1;norm=maxabs)
end
