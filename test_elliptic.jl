using PyPlot
include("kepler_solver.jl")

# Define a constant of 1/3:
const third = 1.0/3.0

function test_elliptic()
# This routine runs a test of the kep_elliptic function in kepler_solver_elliptic.jl
# Define the central force constant in terms of AU and days:
k = (2.0*pi/365.25)^2
# Initial position at 1 AU:
x0 = [0.0,1.0,0.0]
#r0 = norm(x0)
r0 = sqrt(x0[1]*x0[1]+x0[2]*x0[2]+x0[3]*x0[3])
# Circular velocity:
vcirc = sqrt(k/r0)
# Define initial velocity at apastron:
v0 = [0.98*vcirc,0.0,0.0]  # The eccentricity is about ~2(1-v0/vcirc).
dr0dt = (x0[1]*v0[1]+x0[2]*v0[2]+x0[3]*v0[3])/r0
h = 18.0 # 18-day timesteps

s0::Float64 = 0.0
x = zeros(Float64,3)
v = zeros(Float64,3)
s::Float64 = 0.0
t = 0.0
ssave = zeros(2)
#nsteps = 10000000
nsteps = 1000000
#nsteps = 1000
xsave=zeros(Float64,12,nsteps)  # First is time (1,:); next three are position (2-4,:); next three velocity (5-7,:); then r (8,:), drdt (9,:); then beta (10,:); finally s & ds (11:12,:)
# Create a variable to store the state at the end of a step:
state=zeros(Float64,12)

xsave[1,1]=0.0
for j=1:3
  xsave[j+1,1]=x0[j]
  xsave[j+4,1]=v0[j]
end
# Save beta:
xsave[8,1]=r0
xsave[9,1]=dr0dt
beta0 = 2.0*k/r0-dot(v0,v0)
xsave[10,1]=beta0
#@inbounds for i=2:nsteps
for i=2:nsteps
#  x,v,r,drdt,s,beta,iter=kep_elliptic(x0,v0,r0,dr0dt,k,h,beta0,s0)
  iter = kep_elliptic!(x0,v0,r0,dr0dt,k,h,beta0,s0,state)
  s = state[11]
  ds = state[12]
  if iter > 2
   println("iter: ",iter," ds: ",ds)
  end
# Increment time by h:
  state[1] += h
  for j=1:12
    xsave[j,i]=state[j]
  end
  if i >= 4
# Quadratic prediction for s of next step:
    s0 = 3.0*s - 3.0*ssave[1] + ssave[2]
    ssave[2] = ssave[1]
    ssave[1] = s
  else
    s0 = s
    ssave[2]=ssave[1]
    ssave[1]=s
  end
# Now, proceed to next step:
  for j=1:3
    x0[j]=state[1+j]
    v0[j]=state[4+j]
  end
  r0=state[8]
  dr0dt=state[9]
  beta0=state[10]
#  println(i,x,v)
end
#plot(xsave[1],xsave[2])
return xsave
end
