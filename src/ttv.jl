# Translation of David Hernandez's nbody.c for integrating hiercharical
# system with BH15 integrator.  Please cite Hernandez & Bertschinger (2015)
# if using this in a paper.

if ~isdefined(:YEAR)
  const YEAR  = 365.242
  const GNEWT = 39.4845/YEAR^2
  const NDIM  = 3
#const TRANSIT_TOL = 1e-8
#  const TRANSIT_TOL = 10.*sqrt(eps(1.0))
#const TRANSIT_TOL = 10.*eps(1.0)
  const third = 1./3.
  const alpha0 = 0.0
end
include("kepler_step.jl")
include("kepler_drift_step.jl")
include("init_nbody.jl")

# These "constants" pre-allocate memory for matrices used in the derivative computation (to save time with allocation and garbage collection):
if ~isdefined(:pxpr0)
  const pxpr0 = zeros(Float64,3);const  pxpa0=zeros(Float64,3);const  pxpk=zeros(Float64,3);const  pxps=zeros(Float64,3);const  pxpbeta=zeros(Float64,3)
  const dxdr0 = zeros(Float64,3);const  dxda0=zeros(Float64,3);const  dxdk=zeros(Float64,3);const  dxdv0 =zeros(Float64,3)
  const prvpr0 = zeros(Float64,3);const  prvpa0=zeros(Float64,3);const  prvpk=zeros(Float64,3);const  prvps=zeros(Float64,3);const  prvpbeta=zeros(Float64,3)
  const drvdr0 = zeros(Float64,3);const  drvda0=zeros(Float64,3);const  drvdk=zeros(Float64,3);const  drvdv0=zeros(Float64,3)
  const vtmp = zeros(Float64,3);const  dvdr0 = zeros(Float64,3);const  dvda0=zeros(Float64,3);const  dvdv0=zeros(Float64,3);const  dvdk=zeros(Float64,3)
end

# Computes TTVs as a function of orbital elements, allowing for a single log perturbation of dlnq for body jq and element iq
#function ttv_elements!(n::Int64,t0::Float64,h::Float64,tmax::Float64,elements::Array{Float64,2},tt::Array{Float64,2},count::Array{Int64,1},dlnq::Float64,iq::Int64,jq::Int64)
function ttv_elements!(n::Int64,t0::T,h::T,tmax::T,elements::Array{T,2},tt::Array{T,2},count::Array{Int64,1},dlnq::T,iq::Int64,jq::Int64,rstar::T;fout="",iout=-1,pair = zeros(Bool,n,n)) where {T <: Real}
# 
# Input quantities:
# n     = number of bodies
# t0    = initial time of integration  [days]
# h     = time step [days]
# tmax  = duration of integration [days]
# elements[i,j] = 2D n x 7 array of the masses & orbital elements of the bodies (currently first body's orbital elements are ignored)
#            elements are ordered as: mass, period, t0, e*cos(omega), e*sin(omega), inclination, longitude of ascending node (Omega)
# tt    = pre-allocated array to hold transit times of size [n x max(ntt)] (currently only compute transits of star, so first row is zero) [days]
#         upon output, set to transit times of planets.
# count = pre-allocated array of the number of transits for each body upon output
#
# dlnq  = fractional variation in initial parameter jq of body iq for finite-difference calculation of
#         derivatives [this is only needed for testing derivative code, below].
#
# Example: see test_ttv_elements.jl in test/ directory
#
#fcons = open("fcons.txt","w");
# Set up mass, position & velocity arrays.  NDIM =3
m=zeros(eltype(elements),n)
x=zeros(eltype(elements),NDIM,n)
v=zeros(eltype(elements),NDIM,n)
# Fill the transit-timing array with zeros:
fill!(tt,0.0)
# Counter for transits of each planet:
fill!(count,0)
# Insert masses from the elements array:
for i=1:n
  m[i] = elements[i,1]
end
# Allow for perturbations to initial conditions: jq labels body; iq labels phase-space element (or mass)
# iq labels phase-space element (1-3: x; 4-6: v; 7: m)
dq = 0.0
if iq == 7 && dlnq != 0.0
  dq = m[jq]*dlnq
  m[jq] += dq
end
# Initialize the N-body problem using nested hierarchy of Keplerians:
x,v = init_nbody(elements,t0,n)
#elements_big=big.(elements); t0big = big(t0)
#xbig,vbig = init_nbody(elements_big,t0big,n)
#x = convert(Array{Float64,2},xbig); v = convert(Array{Float64,2},vbig)
# Perturb the initial condition by an amount dlnq (if it is non-zero):
if dlnq != 0.0 && iq > 0 && iq < 7
  if iq < 4
    if x[iq,jq] != 0
      dq = x[iq,jq]*dlnq
    else
      dq = dlnq
    end
    x[iq,jq] += dq
  else
  # Same for v
    if v[iq-3,jq] != 0
      dq = v[iq-3,jq]*dlnq
    else
      dq = dlnq
    end
    v[iq-3,jq] += dq
  end
end
ttv!(n,t0,h,tmax,m,x,v,tt,count,fout,iout,rstar,pair)
return dq
end

# Computes TTVs as a function of orbital elements, and computes Jacobian of transit times with respect to initial orbital elements.
# This version is used to test/debug findtransit2 by computing finite difference derivative of findtransit2.
function ttv_elements!(n::Int64,t0::Float64,h::Float64,tmax::Float64,elements::Array{Float64,2},tt::Array{Float64,2},count::Array{Int64,1},dtdq0::Array{Float64,4},dtdq0_num::Array{BigFloat,4},dlnq::BigFloat,rstar::Float64;pair=zeros(Bool,n,n))
# 
# Input quantities:
# n     = number of bodies
# t0    = initial time of integration  [days]
# h     = time step [days]
# tmax  = duration of integration [days]
# elements[i,j] = 2D n x 7 array of the masses & orbital elements of the bodies (currently first body's orbital elements are ignored)
#            elements are ordered as: mass, period, t0, e*cos(omega), e*sin(omega), inclination, longitude of ascending node (Omega)
# tt    = array of transit times of size [n x max(ntt)] (currently only compute transits of star, so first row is zero) [days]
# count = array of the number of transits for each body
# dtdq0 = derivative of transit times with respect to initial x,v,m [various units: day/length (3), day^2/length (3), day/mass]
#         4D array  [n x max(ntt) ] x [n x 7] - derivatives of transits of each planet with respect to initial positions/velocities
#             masses of *all* bodies.  Note: mass derivatives are *after* positions/velocities, even though they are at start
#             of the elements[i,j] array.
#
# Output quantity:
# dtdelements = 4D array  [n x max(ntt) ] x [n x 7] - derivatives of transits of each planet with respect to initial orbital
#             elements/masses of *all* bodies.  Note: mass derivatives are *after* elements, even though they are at start
#             of the elements[i,j] array
#
# Example: see test_ttv_elements.jl in test/ directory
#
# Define initial mass, position & velocity arrays:
m=zeros(Float64,n)
x=zeros(Float64,NDIM,n)
v=zeros(Float64,NDIM,n)
# Fill the transit-timing & jacobian arrays with zeros:
fill!(tt,0.0)
fill!(dtdq0,0.0)
fill!(dtdq0_num,0.0)
# Create an array for the derivatives with respect to the masses/orbital elements:
dtdelements = copy(dtdq0)
# Counter for transits of each planet:
fill!(count,0)
for i=1:n
  m[i] = elements[i,1]
end
# Initialize the N-body problem using nested hierarchy of Keplerians:
jac_init     = zeros(Float64,7*n,7*n)
x,v = init_nbody(elements,t0,n,jac_init)
#x,v = init_nbody(elements,t0,n)
ttv!(n,t0,h,tmax,m,x,v,tt,count,dtdq0,dtdq0_num,dlnq,rstar,pair)
# Need to apply initial jacobian TBD - convert from
# derivatives with respect to (x,v,m) to (elements,m):
ntt_max = size(tt)[2]
for i=1:n, j=1:count[i]
  if j <= ntt_max
  # Now, multiply by the initial Jacobian to convert time derivatives to orbital elements:
    for k=1:n, l=1:7
      dtdelements[i,j,l,k] = 0.0
      for p=1:n, q=1:7
        dtdelements[i,j,l,k] += dtdq0[i,j,q,p]*jac_init[(p-1)*7+q,(k-1)*7+l]
      end
    end
  end
end
return dtdelements
end

# Computes TTVs as a function of orbital elements, and computes Jacobian of transit times with respect to initial orbital elements.
function ttv_elements!(n::Int64,t0::Float64,h::Float64,tmax::Float64,elements::Array{Float64,2},tt::Array{Float64,2},count::Array{Int64,1},dtdq0::Array{Float64,4},rstar::Float64;pair=zeros(Bool,n,n))
# 
# Input quantities:
# n     = number of bodies
# t0    = initial time of integration  [days]
# h     = time step [days]
# tmax  = duration of integration [days]
# elements[i,j] = 2D n x 7 array of the masses & orbital elements of the bodies (currently first body's orbital elements are ignored)
#            elements are ordered as: mass, period, t0, e*cos(omega), e*sin(omega), inclination, longitude of ascending node (Omega)
# tt    = array of transit times of size [n x max(ntt)] (currently only compute transits of star, so first row is zero) [days]
# count = array of the number of transits for each body
# dtdq0 = derivative of transit times with respect to initial x,v,m [various units: day/length (3), day^2/length (3), day/mass]
#         4D array  [n x max(ntt) ] x [n x 7] - derivatives of transits of each planet with respect to initial positions/velocities
#             masses of *all* bodies.  Note: mass derivatives are *after* positions/velocities, even though they are at start
#             of the elements[i,j] array.
#
# Output quantity:
# dtdelements = 4D array  [n x max(ntt) ] x [n x 7] - derivatives of transits of each planet with respect to initial orbital
#             elements/masses of *all* bodies.  Note: mass derivatives are *after* elements, even though they are at start
#             of the elements[i,j] array
#
# Example: see test_ttv_elements.jl in test/ directory
#
# Define initial mass, position & velocity arrays:
m=zeros(Float64,n)
x=zeros(Float64,NDIM,n)
v=zeros(Float64,NDIM,n)
# Fill the transit-timing & jacobian arrays with zeros:
fill!(tt,0.0)
fill!(dtdq0,0.0)
# Create an array for the derivatives with respect to the masses/orbital elements:
dtdelements = copy(dtdq0)
# Counter for transits of each planet:
fill!(count,0)
for i=1:n
  m[i] = elements[i,1]
end
# Initialize the N-body problem using nested hierarchy of Keplerians:
jac_init     = zeros(Float64,7*n,7*n)
x,v = init_nbody(elements,t0,n,jac_init)
#x,v = init_nbody(elements,t0,n)
ttv!(n,t0,h,tmax,m,x,v,tt,count,dtdq0,rstar,pair)
# Need to apply initial jacobian TBD - convert from
# derivatives with respect to (x,v,m) to (elements,m):
ntt_max = size(tt)[2]
for i=1:n, j=1:count[i]
  if j <= ntt_max
  # Now, multiply by the initial Jacobian to convert time derivatives to orbital elements:
    for k=1:n, l=1:7
      dtdelements[i,j,l,k] = 0.0
      for p=1:n, q=1:7
        dtdelements[i,j,l,k] += dtdq0[i,j,q,p]*jac_init[(p-1)*7+q,(k-1)*7+l]
      end
     end
  end
end
return dtdelements
end

# Computes TTVs for initial x,v, as well as timing derivatives with respect to x,v,m (dtdq0).
function ttv!(n::Int64,t0::T,h::T,tmax::T,m::Array{T,1},
  x::Array{T,2},v::Array{T,2},tt::Array{T,2},count::Array{Int64,1},dtdq0::Array{T,4},rstar::T,pair::Array{Bool,2}) where {T <: Real}
xprior = copy(x)
vprior = copy(v)
xtransit = copy(x)
vtransit = copy(v)
# Set the time to the initial time:
t = t0
# Set step counter to zero:
istep = 0
# Jacobian for each step (7- 6 elements+mass, n_planets, 7 - 6 elements+mass, n planets):
jac_prior = zeros(typeof(h),7*n,7*n)
jac_transit = zeros(typeof(h),7*n,7*n)
# Initialize matrix for derivatives of transit times with respect to the initial x,v,m:
dtdq = zeros(typeof(h),7,n)
# Initialize the Jacobian to the identity matrix:
jac_step = eye(typeof(h),7*n)

# Save the g function, which computes the relative sky velocity dotted with relative position
# between the planets and star:
gsave = zeros(typeof(h),n)
for i=2:n
  # Compute the relative sky velocity dotted with position:
  gsave[i]= g!(i,1,x,v)
end
# Loop over time steps:
dt::Float64 = 0.0
gi = 0.0
ntt_max = size(tt)[2]
param_real = all(isfinite.(x)) && all(isfinite.(v)) && all(isfinite.(m))
while t < (t0+tmax) && param_real
  # Carry out a dh17 mapping step:
  dh17!(x,v,h,m,n,jac_step,pair)
  param_real = all(isfinite.(x)) && all(isfinite.(v)) && all(isfinite.(m)) && all(isfinite.(jac_step))
  # Check to see if a transit may have occured.  Sky is x-y plane; line of sight is z.
  # Star is body 1; planets are 2-nbody (note that this could be modified to see if
  # any body transits another body):
  for i=2:n
    # Compute the relative sky velocity dotted with position:
    gi = g!(i,1,x,v)
    ri = sqrt(x[1,i]^2+x[2,i]^2+x[3,i]^2)  # orbital distance
    # See if sign of g switches, and if planet is in front of star (by a good amount):
    # (I'm wondering if the direction condition means that z-coordinate is reversed? EA 12/11/2017)
    if gi > 0 && gsave[i] < 0 && x[3,i] > 0.25*ri && ri < rstar
      # A transit has occurred between the time steps - integrate dh17! between timesteps
      count[i] += 1
      if count[i] <= ntt_max
        dt0 = -gsave[i]*h/(gi-gsave[i])  # Starting estimate
        xtransit .= xprior; vtransit .= vprior; jac_transit .= jac_prior
#        dt = findtransit2!(1,i,n,h,dt0,m,xtransit,vtransit,jac_transit,dtdq,pair) # 20%
        dt = findtransit3!(1,i,n,h,dt0,m,xtransit,vtransit,jac_transit,dtdq,pair) # 20%
        tt[i,count[i]]=t+dt
        # Save for posterity:
        for k=1:7, p=1:n
          dtdq0[i,count[i],k,p] = dtdq[k,p]
        end
      end
    end
    gsave[i] = gi
  end
  # Save the current state as prior state:
  xprior .= x
  vprior .= v
  jac_prior .= jac_step
  # Increment time by the time step:
  t += h
  # Increment counter by one:
  istep +=1
end
return 
end

# Computes TTVs for initial x,v, as well as timing derivatives with respect to x,v,m (dtdq0).
function ttv!(n::Int64,t0::Float64,h::Float64,tmax::Float64,m::Array{Float64,1},x::Array{Float64,2},v::Array{Float64,2},tt::Array{Float64,2},count::Array{Int64,1},dtdq0::Array{Float64,4},rstar::Float64,pair::Array{Bool,2})
xprior = copy(x)
vprior = copy(v)
xtransit = copy(x)
vtransit = copy(v)
# Set the time to the initial time:
t = t0
# Set step counter to zero:
istep = 0
# Jacobian for each step (7- 6 elements+mass, n_planets, 7 - 6 elements+mass, n planets):
jac_prior = zeros(Float64,7*n,7*n)
jac_transit = zeros(Float64,7*n,7*n)
# Initialize matrix for derivatives of transit times with respect to the initial x,v,m:
dtdq = zeros(Float64,7,n)
# Initialize the Jacobian to the identity matrix:
jac_step = eye(Float64,7*n)

# Save the g function, which computes the relative sky velocity dotted with relative position
# between the planets and star:
gsave = zeros(Float64,n)
for i=2:n
  # Compute the relative sky velocity dotted with position:
  gsave[i]= g!(i,1,x,v)
end
# Loop over time steps:
dt::Float64 = 0.0
gi = 0.0
ntt_max = size(tt)[2]
param_real = all(isfinite.(x)) && all(isfinite.(v)) && all(isfinite.(m)) && all(isfinite.(jac_step))
while t < t0+tmax && param_real
  # Carry out a dh17 mapping step:
  dh17!(x,v,h,m,n,jac_step,pair)
  param_real = all(isfinite.(x)) && all(isfinite.(v)) && all(isfinite.(m)) && all(isfinite.(jac_step))
  # Check to see if a transit may have occured.  Sky is x-y plane; line of sight is z.
  # Star is body 1; planets are 2-nbody (note that this could be modified to see if
  # any body transits another body):
  for i=2:n
    # Compute the relative sky velocity dotted with position:
    gi = g!(i,1,x,v)
    ri = sqrt(x[1,i]^2+x[2,i]^2+x[3,i]^2)
    # See if sign of g switches, and if planet is in front of star (by a good amount):
    # (I'm wondering if the direction condition means that z-coordinate is reversed? EA 12/11/2017)
    if gi > 0 && gsave[i] < 0 && x[3,i] > 0.25*ri && ri < rstar
      # A transit has occurred between the time steps - integrate dh17! between timesteps
      count[i] += 1
      if count[i] <= ntt_max
        dt0 = -gsave[i]*h/(gi-gsave[i])  # Starting estimate
        xtransit .= xprior; vtransit .= vprior; jac_transit .= jac_prior
#        dt = findtransit2!(1,i,n,h,dt0,m,xtransit,vtransit,jac_transit,dtdq,pair) # 20%
        dt = findtransit3!(1,i,n,h,dt0,m,xtransit,vtransit,jac_transit,dtdq,pair) # 20%
        tt[i,count[i]]=t+dt
        # Save for posterity:
        for k=1:7, p=1:n
          dtdq0[i,count[i],k,p] = dtdq[k,p]
        end
      end
    end
    gsave[i] = gi
  end
  # Save the current state as prior state:
  xprior .= x
  vprior .= v
  jac_prior .= jac_step
  # Increment time by the time step:
  t += h
  # Increment counter by one:
  istep +=1
end
return
end

# Computes TTVs for initial x,v, as well as timing derivatives with respect to x,v,m (dtdq0).
function ttv!(n::Int64,t0::Float64,h::Float64,tmax::Float64,m::Array{Float64,1},x::Array{Float64,2},v::Array{Float64,2},tt::Array{Float64,2},count::Array{Int64,1},dtdq0::Array{Float64,4},rstar::Float64,pair::Array{Bool,2})
xprior = copy(x)
vprior = copy(v)
xtransit = copy(x)
vtransit = copy(v)
# Set the time to the initial time:
t = t0
# Set step counter to zero:
istep = 0
# Jacobian for each step (7- 6 elements+mass, n_planets, 7 - 6 elements+mass, n planets):
jac_prior = zeros(Float64,7*n,7*n)
jac_transit = zeros(Float64,7*n,7*n)
# Initialize matrix for derivatives of transit times with respect to the initial x,v,m:
dtdq = zeros(Float64,7,n)
# Initialize the Jacobian to the identity matrix:
jac_step = eye(Float64,7*n)

# Save the g function, which computes the relative sky velocity dotted with relative position
# between the planets and star:
gsave = zeros(Float64,n)
for i=2:n
  # Compute the relative sky velocity dotted with position:
  gsave[i]= g!(i,1,x,v)
end
# Loop over time steps:
dt::Float64 = 0.0
gi = 0.0
ntt_max = size(tt)[2]
param_real = all(isfinite.(x)) && all(isfinite.(v)) && all(isfinite.(m)) && all(isfinite.(jac_step))
while t < t0+tmax && param_real
  # Carry out a dh17 mapping step:
  dh17!(x,v,h,m,n,jac_step,pair)
  param_real = all(isfinite.(x)) && all(isfinite.(v)) && all(isfinite.(m)) && all(isfinite.(jac_step))
  # Check to see if a transit may have occured.  Sky is x-y plane; line of sight is z.
  # Star is body 1; planets are 2-nbody (note that this could be modified to see if
  # any body transits another body):
  for i=2:n
    # Compute the relative sky velocity dotted with position:
    gi = g!(i,1,x,v)
    ri = sqrt(x[1,i]^2+x[2,i]^2+x[3,i]^2)
    # See if sign of g switches, and if planet is in front of star (by a good amount):
    # (I'm wondering if the direction condition means that z-coordinate is reversed? EA 12/11/2017)
    if gi > 0 && gsave[i] < 0 && x[3,i] > 0.25*ri && ri < rstar
      # A transit has occurred between the time steps - integrate dh17! between timesteps
      count[i] += 1
      if count[i] <= ntt_max
        dt0 = -gsave[i]*h/(gi-gsave[i])  # Starting estimate
        xtransit .= xprior; vtransit .= vprior; jac_transit .= jac_prior
#        dt = findtransit2!(1,i,n,h,dt0,m,xtransit,vtransit,jac_transit,dtdq,pair) # 20%
      dt = findtransit3!(1,i,n,h,dt0,m,xtransit,vtransit,jac_transit,dtdq,pair) # 20%
        tt[i,count[i]]=t+dt
        # Save for posterity:
        for k=1:7, p=1:n
          dtdq0[i,count[i],k,p] = dtdq[k,p]
        end
      end
    end
    gsave[i] = gi
  end
  # Save the current state as prior state:
  xprior .= x
  vprior .= v
  jac_prior .= jac_step
  # Increment time by the time step:
  t += h
  # Increment counter by one:
  istep +=1
end
return
end

# Computes TTVs for initial x,v, as well as timing derivatives with respect to x,v,m (dtdq0).
# This version is used to test findtransit2 by computing finite difference derivative of findtransit2.
function ttv!(n::Int64,t0::Float64,h::Float64,tmax::Float64,m::Array{Float64,1},x::Array{Float64,2},v::Array{Float64,2},tt::Array{Float64,2},count::Array{Int64,1},dtdq0::Array{Float64,4},dtdq0_num::Array{BigFloat,4},dlnq::BigFloat,rstar::Float64,pair::Array{Bool,2})
xprior = copy(x)
vprior = copy(v)
#xtransit = big.(x); xtransit_plus = big.(x); xtransit_minus = big.(x)
#vtransit = big.(v); vtransit_plus = big.(v); vtransit_minus = big.(v)
xtransit = copy(x); xtransit_plus = big.(x); xtransit_minus = big.(x)
vtransit = copy(v); vtransit_plus = big.(v); vtransit_minus = big.(v)
m_plus = big.(m); m_minus = big.(m); hbig = big(h); dq = big(0.0)
if h == 0
  println("h is zero ",h)
end
# Set the time to the initial time:
t = t0
# Set step counter to zero:
istep = 0
# Initialize matrix for derivatives of transit times with respect to the initial x,v,m:
dtdqbig = zeros(BigFloat,7,n)
dtdq = zeros(Float64,7,n)
dtdq3 = zeros(Float64,7,n)
# Initialize the Jacobian to the identity matrix:
jac_prior = zeros(Float64,7*n,7*n)
jac_step = eye(Float64,7*n)

# Save the g function, which computes the relative sky velocity dotted with relative position
# between the planets and star:
gsave = zeros(Float64,n)
for i=2:n
  # Compute the relative sky velocity dotted with position:
  gsave[i]= g!(i,1,x,v)
end
# Loop over time steps:
dt::Float64 = 0.0
gi = 0.0
ntt_max = size(tt)[2]
param_real = all(isfinite.(x)) && all(isfinite.(v)) && all(isfinite.(m)) && all(isfinite.(jac_step))
while t < t0+tmax && param_real
  # Carry out a dh17 mapping step:
  dh17!(x,v,h,m,n,jac_step,pair)
  param_real = all(isfinite.(x)) && all(isfinite.(v)) && all(isfinite.(m)) && all(isfinite.(jac_step))
  # Check to see if a transit may have occured.  Sky is x-y plane; line of sight is z.
  # Star is body 1; planets are 2-nbody (note that this could be modified to see if
  # any body transits another body):
  for i=2:n
    # Compute the relative sky velocity dotted with position:
    gi = g!(i,1,x,v)
    ri = sqrt(x[1,i]^2+x[2,i]^2+x[3,i]^2)
    # See if sign of g switches, and if planet is in front of star (by a good amount):
    # (I'm wondering if the direction condition means that z-coordinate is reversed? EA 12/11/2017)
    if gi > 0 && gsave[i] < 0 && x[3,i] > 0.25*ri && ri < rstar
      # A transit has occurred between the time steps - integrate dh17! between timesteps
      count[i] += 1
      if count[i] <= ntt_max
#        dt0 = big(-gsave[i]*h/(gi-gsave[i]))  # Starting estimate
        dt0 = -gsave[i]*h/(gi-gsave[i])  # Starting estimate
#        xtransit .= big.(xprior); vtransit .= big.(vprior)
#        xtransit .= xprior; vtransit .= vprior
#        jac_transit = eye(jac_step)
#        dt,gdot = findtransit2!(1,i,n,h,dt0,m,xtransit,vtransit,jac_transit,dtdq,pair) # Just computing derivative since prior timestep, so start with identity matrix
#       dt = findtransit2!(1,i,n,h,dt0,m,xtransit,vtransit,jac_transit,dtdq,pair) # Just computing derivative since prior timestep, so start with identity matrix
#        jac_transit = eye(BigFloat,7*n)
#        hbig = big(h)
#        dtbig = findtransit2!(1,i,n,hbig,dt0,big.(m),xtransit,vtransit,jac_transit,dtdqbig,pair) # Just computing derivative since prior timestep, so start with identity matrix
#        dtdq = convert(Array{Float64,2},dtdqbig)
#        dt0 = -gsave[i]*h/(gi-gsave[i])  # Starting estimate
        # Now, recompute with findtransit3:
        xtransit .= xprior; vtransit .= vprior
        jac_transit = eye(jac_step)
        dt = findtransit3!(1,i,n,h,dt0,m,xtransit,vtransit,jac_transit,dtdq3,pair) # Just computing derivative since prior timestep, so start with identity matrix
        # Save for posterity:
        tt[i,count[i]]=t+dt
        for k=1:7, p=1:n
#          dtdq0[i,count[i],k,p] = dtdq[k,p]
          dtdq0[i,count[i],k,p] = dtdq3[k,p]
          # Compute numerical approximation of dtdq:
          dt_plus = big(dt)  # Starting estimate
#          dt_plus = dtbig  # Starting estimate
          xtransit_plus .= big.(xprior); vtransit_plus .= big.(vprior); m_plus .= big.(m)
          if k < 4; dq = dlnq*xtransit_plus[k,p]; xtransit_plus[k,p] += dq; elseif k < 7; dq =vtransit_plus[k-3,p]*dlnq; vtransit_plus[k-3,p] += dq; else; dq  = m_plus[p]*dlnq; m_plus[p] += dq; end
#          dt_plus = findtransit2!(1,i,n,hbig,dt_plus,m_plus,xtransit_plus,vtransit_plus,pair) # 20%
          dt_plus = findtransit3!(1,i,n,hbig,dt_plus,m_plus,xtransit_plus,vtransit_plus,pair) # 20%
          dt_minus= big(dt)  # Starting estimate
#          dt_minus= dtbig  # Starting estimate
          xtransit_minus .= big.(xprior); vtransit_minus .= big.(vprior); m_minus .= big.(m)
          if k < 4; dq = dlnq*xtransit_minus[k,p];xtransit_minus[k,p] -= dq; elseif k < 7; dq =vtransit_minus[k-3,p]*dlnq; vtransit_minus[k-3,p] -= dq; else; dq  = m_minus[p]*dlnq; m_minus[p] -= dq; end
          hbig = big(h)
#          dt_minus= findtransit2!(1,i,n,hbig,dt_minus,m_minus,xtransit_minus,vtransit_minus,pair) # 20%
          dt_minus= findtransit3!(1,i,n,hbig,dt_minus,m_minus,xtransit_minus,vtransit_minus,pair) # 20%
          # Compute finite-different derivative:
          dtdq0_num[i,count[i],k,p] = (dt_plus-dt_minus)/(2dq)
          if abs(dtdq0_num[i,count[i],k,p] - dtdq0[i,count[i],k,p]) > 1e-10
            # Compute gdot_num:
            dt_minus = big(dt)*(1-dlnq)  # Starting estimate
            xtransit_minus .= big.(xprior); vtransit_minus .= big.(vprior); m_minus .= big.(m)
            dh17!(xtransit_minus,vtransit_minus,dt_minus,m_minus,n,pair)
            # Compute time offset:
            gsky_minus = g!(i,1,xtransit_minus,vtransit_minus)
            dt_plus = big(dt)*(1+dlnq)  # Starting estimate
            xtransit_plus .= big.(xprior); vtransit_plus .= big.(vprior); m_plus .= big.(m)
            dh17!(xtransit_plus,vtransit_plus,dt_plus,m_plus,n,pair)
            # Compute time offset:
            gsky_plus = g!(i,1,xtransit_plus,vtransit_plus)
            gdot_num = convert(Float64,(gsky_plus-gsky_minus)/(2dlnq*big(dt)))
            println("i: ",i," count: ",count[i]," k: ",k," p: ",p," dt: ",dt," dq: ",dq," dtdq0: ",dtdq0[i,count[i],k,p]," dtdq0_num: ",convert(Float64,dtdq0_num[i,count[i],k,p])," ratio-1: ",dtdq0[i,count[i],k,p]/convert(Float64,dtdq0_num[i,count[i],k,p])-1.0," gdot: ",gdot," gdot_num: ",gdot_num," ratio-1: ",gdot/gdot_num-1.0)
            println("x0: ",xprior)
            println("v0: ",vprior)
            println("x: ",x)
            println("v: ",v)
#            read(STDIN,Char)
          end
        end
      end
    end
    gsave[i] = gi
  end
  # Save the current state as prior state:
  xprior .= x
  vprior .= v
  jac_prior .= jac_step
  # Increment time by the time step:
  t += h
  # Increment counter by one:
  istep +=1
end
return
end

# Computes TTVs as a function of initial x,v,m.
function ttv!(n::Int64,t0::T,h::T,tmax::T,m::Array{T,1},x::Array{T,2},v::Array{T,2},tt::Array{T,2},count::Array{Int64,1},fout::String,iout::Int64,rstar::T,pair::Array{Bool,2}) where {T <: Real}
# Make some copies to allocate space for saving prior step and computing coordinates at the times of transit.
xprior = copy(x)
vprior = copy(v)
xtransit = copy(x)
vtransit = copy(v)
# Set the time to the initial time:
t = t0
# Set step counter to zero:
istep = 0
# Jacobian for each step (7 elements+mass, n_planets, 7 elements+mass, n planets):
# Save the g function, which computes the relative sky velocity dotted with relative position
# between the planets and star:
gsave = zeros(typeof(h),n)
gi  = 0.0
dt::typeof(h) = 0.0
# Loop over time steps:
ntt_max = size(tt)[2]
param_real = all(isfinite.(x)) && all(isfinite.(v)) && all(isfinite.(m))
if fout != ""
  # Open file for output:
  file_handle =open(fout,"a")
end
while t < t0+tmax && param_real
  # Carry out a phi^2 mapping step:
#  phi2!(x,v,h,m,n)
  dh17!(x,v,h,m,n,pair)
  #xbig = big.(x); vbig = big.(v); hbig = big(h); mbig = big.(m)
  #dh17!(xbig,vbig,hbig,mbig,n,pair)
  #x = convert(Array{Float64,2},xbig); v = convert(Array{Float64,2},vbig)
  param_real = all(isfinite.(x)) && all(isfinite.(v)) && all(isfinite.(m))
  # Check to see if a transit may have occured.  Sky is x-y plane; line of sight is z.
  # Star is body 1; planets are 2-nbody:
  for i=2:n
    # Compute the relative sky velocity dotted with position:
    gi = g!(i,1,x,v)
    ri = sqrt(x[1,i]^2+x[2,i]^2+x[3,i]^2)
    # See if sign switches, and if planet is in front of star (by a good amount):
    if gi > 0 && gsave[i] < 0 && x[3,i] > 0.25*ri && ri < rstar
      # A transit has occurred between the time steps.
      # Approximate the planet-star motion as a Keplerian, weighting over timestep:
      count[i] += 1
#      tt[i,count[i]]=t+findtransit!(i,h,gi,gsave[i],m,xprior,vprior,x,v,pair)
      if count[i] <= ntt_max
        dt0 = -gsave[i]*h/(gi-gsave[i])
        xtransit .= xprior
        vtransit .= vprior
#        dt = findtransit2!(1,i,n,h,dt0,m,xtransit,vtransit,pair)
        #hbig = big(h); dt0big=big(dt0); mbig=big.(m); xtbig = big.(xtransit); vtbig = big.(vtransit)
        #dtbig = findtransit2!(1,i,n,hbig,dt0big,mbig,xtbig,vtbig,pair)
        #dt = convert(Float64,dtbig)
        dt = findtransit3!(1,i,n,h,dt0,m,xtransit,vtransit,pair)
        tt[i,count[i]]=t+dt
      end
#      tt[i,count[i]]=t+findtransit2!(1,i,n,h,gi,gsave[i],m,xprior,vprior,pair)
    end
    gsave[i] = gi
  end
  # Save the current state as prior state:
  xprior .=x
  vprior .=v
  if mod(istep,iout) == 0 && iout > 0
    # Write to file:
    writedlm(file_handle,[convert(Float64,t);convert(Array{Float64,1},reshape(x,3n));convert(Array{Float64,1},reshape(v,3n))]') # Transpose to write each line
  end
  # Increment time by the time step:
  t += h
  # Increment counter by one:
  istep +=1
end
if fout != ""
  # Close output file:
  close(file_handle)
end
return
end

# Advances the center of mass of a binary (any pair of bodies)
#function centerm!(m::Array{Float64,1},mijinv::Float64,x::Array{Float64,2},v::Array{Float64,2},vcm::Array{Float64,1},delx::Array{Float64,1},delv::Array{Float64,1},i::Int64,j::Int64,h::Float64)
function centerm!(m::Array{T,1},mijinv::T,x::Array{T,2},v::Array{T,2},vcm::Array{T,1},delx::Array{T,1},delv::Array{T,1},i::Int64,j::Int64,h::T) where {T <: Real}
for k=1:NDIM
  x[k,i] +=  m[j]*mijinv*delx[k] + h*vcm[k]
  x[k,j] += -m[i]*mijinv*delx[k] + h*vcm[k]
  v[k,i] +=  m[j]*mijinv*delv[k]
  v[k,j] += -m[i]*mijinv*delv[k]
end
return
end

# Drifts bodies i & j
#function driftij!(x::Array{Float64,2},v::Array{Float64,2},i::Int64,j::Int64,h::Float64)
function driftij!(x::Array{T,2},v::Array{T,2},i::Int64,j::Int64,h::T) where {T <: Real}
for k=1:NDIM
  x[k,i] += h*v[k,i]
  x[k,j] += h*v[k,j]
end
return
end

function driftij!(x::Array{T,2},v::Array{T,2},i::Int64,j::Int64,h::T,dqdt::Array{T,1},fac::T) where {T <: Real}
indi = (i-1)*7
indj = (j-1)*7
for k=1:NDIM
  x[k,i] += h*v[k,i]
  x[k,j] += h*v[k,j]
  # Time derivatives:
  dqdt[indi+k] += fac*v[k,i] + h*dqdt[indi+3+k]
  dqdt[indj+k] += fac*v[k,j] + h*dqdt[indj+3+k]
end
return
end

# Drifts bodies i & j and computes Jacobian:
function driftij!(x::Array{T,2},v::Array{T,2},i::Int64,j::Int64,h::T,jac_step::Array{T,2},nbody::Int64) where {T <: Real}
indi = (i-1)*7
indj = (j-1)*7
for k=1:NDIM
  x[k,i] += h*v[k,i]
  x[k,j] += h*v[k,j]
end
# Now for Jacobian:
for m=1:7*nbody, k=1:NDIM
  jac_step[indi+k,m] += h*jac_step[indi+3+k,m]
end
for m=1:7*nbody, k=1:NDIM
  jac_step[indj+k,m] += h*jac_step[indj+3+k,m]
end
return
end

# Carries out a Kepler step and reverse drift for bodies i & j
function kepler_driftij!(m::Array{T,1},x::Array{T,2},v::Array{T,2},i::Int64,j::Int64,h::T,drift_first::Bool) where {T <: Real}
# The state vector has: 1 time; 2-4 position; 5-7 velocity; 8 r0; 9 dr0dt; 10 beta; 11 s; 12 ds
# Initial state:
state0 = zeros(typeof(h),12)
# Final state (after a step):
state = zeros(typeof(h),12)
for k=1:NDIM
  state0[1+k] = x[k,i] - x[k,j]
  state0[4+k] = v[k,i] - v[k,j]
end
gm = GNEWT*(m[i]+m[j])
if gm == 0
#  Do nothing
#  for k=1:3
#    x[k,i] += h*v[k,i]
#    x[k,j] += h*v[k,j]
#  end
else
  # predicted value of s
  kepler_drift_step!(gm, h, state0, state,drift_first)
  mijinv =1.0/(m[i] + m[j])
  for k=1:3
    # Add kepler-drift differences, weighted by masses, to start of step:
    x[k,i] += m[j]*mijinv*state[1+k]
    x[k,j] -= m[i]*mijinv*state[1+k]
    v[k,i] += m[j]*mijinv*state[4+k]
    v[k,j] -= m[i]*mijinv*state[4+k]
  end
end
return
end

# Carries out a Kepler step and reverse drift for bodies i & j, and computes Jacobian:
function kepler_driftij!(m::Array{T,1},x::Array{T,2},v::Array{T,2},i::Int64,j::Int64,h::T,jac_ij::Array{T,2},drift_first::Bool) where {T <: Real}
# The state vector has: 1 time; 2-4 position; 5-7 velocity; 8 r0; 9 dr0dt; 10 beta; 11 s; 12 ds
# Initial state:
state0 = zeros(typeof(h),12)
# Final state (after a step):
state = zeros(typeof(h),12)
for k=1:NDIM
  state0[1+k] = x[k,i] - x[k,j]
  state0[4+k] = v[k,i] - v[k,j]
end
gm = GNEWT*(m[i]+m[j])
# jac_ij should be the Jacobian for going from (x_{0,i},v_{0,i},m_i) &  (x_{0,j},v_{0,j},m_j)
# to  (x_i,v_i,m_i) &  (x_j,v_j,m_j), a 14x14 matrix for the 3-dimensional case.
# Fill with zeros for now:
jac_ij .= eye(typeof(h),14)
if gm == 0
#  Do nothing
#  for k=1:3
#    x[k,i] += h*v[k,i]
#    x[k,j] += h*v[k,j]
#  end
else
  jac_kepler = zeros(typeof(h),7,7)
  # predicted value of s
  kepler_drift_step!(gm, h, state0, state,jac_kepler,drift_first)
  mijinv =1.0/(m[i] + m[j])
  mi = m[i]*mijinv # Normalize the masses
  mj = m[j]*mijinv
  for k=1:3
    # Add kepler-drift differences, weighted by masses, to start of step:
    x[k,i] += mj*state[1+k]
    v[k,i] += mj*state[4+k]
    x[k,j] -= mi*state[1+k]
    v[k,j] -= mi*state[4+k]
  end
  # Compute Jacobian:
  for l=1:6, k=1:6
# Compute derivatives of x_i,v_i with respect to initial conditions:
    jac_ij[  k,  l] += mj*jac_kepler[k,l]
    jac_ij[  k,7+l] -= mj*jac_kepler[k,l]
# Compute derivatives of x_j,v_j with respect to initial conditions:
    jac_ij[7+k,  l] -= mi*jac_kepler[k,l]
    jac_ij[7+k,7+l] += mi*jac_kepler[k,l]
  end
  for k=1:6
# Compute derivatives of x_i,v_i with respect to the masses:
    jac_ij[   k, 7] = -mj*state[1+k]*mijinv + GNEWT*mj*jac_kepler[  k,7]
    jac_ij[   k,14] =  mi*state[1+k]*mijinv + GNEWT*mj*jac_kepler[  k,7]
# Compute derivatives of x_j,v_j with respect to the masses:
    jac_ij[ 7+k, 7] = -mj*state[1+k]*mijinv - GNEWT*mi*jac_kepler[  k,7]
    jac_ij[ 7+k,14] =  mi*state[1+k]*mijinv - GNEWT*mi*jac_kepler[  k,7]
  end
end
return
end

# Carries out a Kepler step for bodies i & j
#function keplerij!(m::Array{Float64,1},x::Array{Float64,2},v::Array{Float64,2},i::Int64,j::Int64,h::Float64)
function keplerij!(m::Array{T,1},x::Array{T,2},v::Array{T,2},i::Int64,j::Int64,h::T) where {T <: Real}
# The state vector has: 1 time; 2-4 position; 5-7 velocity; 8 r0; 9 dr0dt; 10 beta; 11 s; 12 ds
# Initial state:
state0 = zeros(typeof(h),12)
# Final state (after a step):
state = zeros(typeof(h),12)
delx = zeros(typeof(h),NDIM)
delv = zeros(typeof(h),NDIM)
#println("Masses: ",i," ",j)
for k=1:NDIM
  state0[1+k     ] = x[k,i] - x[k,j]
  state0[1+k+NDIM] = v[k,i] - v[k,j]
end
gm = GNEWT*(m[i]+m[j])
if gm == 0
  for k=1:NDIM
    x[k,i] += h*v[k,i]
    x[k,j] += h*v[k,j]
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
  vcm = zeros(typeof(h),NDIM)
  for k=1:NDIM
    vcm[k] = (m[i]*v[k,i] + m[j]*v[k,j])*mijinv
  end
  centerm!(m,mijinv,x,v,vcm,delx,delv,i,j,h)
end
return
end

# Carries out a Kepler step for bodies i & j
function keplerij!(m::Array{T,1},x::Array{T,2},v::Array{T,2},i::Int64,j::Int64,h::T,jac_ij::Array{T,2},dqdt::Array{T,1}) where {T <: Real}
# The state vector has: 1 time; 2-4 position; 5-7 velocity; 8 r0; 9 dr0dt; 10 beta; 11 s; 12 ds
# Initial state:
state0 = zeros(typeof(h),12)
# Final state (after a step):
state = zeros(typeof(h),12)
delx = zeros(typeof(h),NDIM)
delv = zeros(typeof(h),NDIM)
# jac_ij should be the Jacobian for going from (x_{0,i},v_{0,i},m_i) &  (x_{0,j},v_{0,j},m_j)
# to  (x_i,v_i,m_i) &  (x_j,v_j,m_j), a 14x14 matrix for the 3-dimensional case. 
# Fill with zeros for now:
fill!(jac_ij,0.0)
for k=1:NDIM
  state0[1+k     ] = x[k,i] - x[k,j]
  state0[1+k+NDIM] = v[k,i] - v[k,j]
end
gm = GNEWT*(m[i]+m[j])
# The following jacobian is just computed for the Keplerian coordinates (i.e. doesn't include
# center-of-mass motion, or scale to motion of bodies about their common center of mass):
jac_kepler = zeros(typeof(h),7,7)
kepler_step!(gm, h, state0, state, jac_kepler)
for k=1:NDIM
  delx[k] = state[1+k] - state0[1+k]
  delv[k] = state[1+NDIM+k] - state0[1+NDIM+k]
end
# Compute COM coords:
mijinv =1.0/(m[i] + m[j])
xcm = zeros(typeof(h),NDIM)
vcm = zeros(typeof(h),NDIM)
mi = m[i]*mijinv # Normalize the masses
mj = m[j]*mijinv
#println("Masses: ",i," ",j)
for k=1:NDIM
  xcm[k] = mi*x[k,i] + mj*x[k,j]
  vcm[k] = mi*v[k,i] + mj*v[k,j]
end
# Compute the Jacobian:
jac_ij[ 7, 7] = 1.0  # the masses don't change with time!
jac_ij[14,14] = 1.0
for k=1:NDIM
   jac_ij[   k,   k] =   mi
   jac_ij[   k, 3+k] = h*mi
   jac_ij[   k, 7+k] =   mj
   jac_ij[   k,10+k] = h*mj
   jac_ij[ 3+k, 3+k] =   mi
   jac_ij[ 3+k,10+k] =   mj
   jac_ij[ 7+k,   k] =   mi
   jac_ij[ 7+k, 3+k] = h*mi
   jac_ij[ 7+k, 7+k] =   mj
   jac_ij[ 7+k,10+k] = h*mj
   jac_ij[10+k, 3+k] =   mi
   jac_ij[10+k,10+k] =   mj
end
for l=1:NDIM, k=1:NDIM
# Compute derivatives of \delta x_i with respect to initial conditions:
     jac_ij[   k,   l] += mj*jac_kepler[  k,  l]
     jac_ij[   k, 3+l] += mj*jac_kepler[  k,3+l]
     jac_ij[   k, 7+l] -= mj*jac_kepler[  k,  l]
     jac_ij[   k,10+l] -= mj*jac_kepler[  k,3+l]
# Compute derivatives of \delta v_i with respect to initial conditions:
     jac_ij[ 3+k,   l] += mj*jac_kepler[3+k,  l]
     jac_ij[ 3+k, 3+l] += mj*jac_kepler[3+k,3+l]
     jac_ij[ 3+k, 7+l] -= mj*jac_kepler[3+k,  l]
     jac_ij[ 3+k,10+l] -= mj*jac_kepler[3+k,3+l]
# Compute derivatives of \delta x_j with respect to initial conditions:
     jac_ij[ 7+k,   l] -= mi*jac_kepler[  k,  l]
     jac_ij[ 7+k, 3+l] -= mi*jac_kepler[  k,3+l]
     jac_ij[ 7+k, 7+l] += mi*jac_kepler[  k,  l]
     jac_ij[ 7+k,10+l] += mi*jac_kepler[  k,3+l]
# Compute derivatives of \delta v_j with respect to initial conditions:
     jac_ij[10+k,   l] -= mi*jac_kepler[3+k,  l]
     jac_ij[10+k, 3+l] -= mi*jac_kepler[3+k,3+l]
     jac_ij[10+k, 7+l] += mi*jac_kepler[3+k,  l]
     jac_ij[10+k,10+l] += mi*jac_kepler[3+k,3+l]
end
for k=1:NDIM
# Compute derivatives of \delta x_i with respect to the masses:
   jac_ij[   k, 7] = (x[k,i]+h*v[k,i]-xcm[k]-h*vcm[k]-mj*state[1+k])*mijinv + GNEWT*mj*jac_kepler[  k,7]
   jac_ij[   k,14] = (x[k,j]+h*v[k,j]-xcm[k]-h*vcm[k]+mi*state[1+k])*mijinv + GNEWT*mj*jac_kepler[  k,7]
# Compute derivatives of \delta v_i with respect to the masses:
   jac_ij[ 3+k, 7] = (v[k,i]-vcm[k]-mj*state[4+k])*mijinv + GNEWT*mj*jac_kepler[3+k,7]
   jac_ij[ 3+k,14] = (v[k,j]-vcm[k]+mi*state[4+k])*mijinv + GNEWT*mj*jac_kepler[3+k,7]
# Compute derivatives of \delta x_j with respect to the masses:
   jac_ij[ 7+k, 7] = (x[k,i]+h*v[k,i]-xcm[k]-h*vcm[k]-mj*state[1+k])*mijinv - GNEWT*mi*jac_kepler[  k,7]
   jac_ij[ 7+k,14] = (x[k,j]+h*v[k,j]-xcm[k]-h*vcm[k]+mi*state[1+k])*mijinv - GNEWT*mi*jac_kepler[  k,7]
# Compute derivatives of \delta v_j with respect to the masses:
   jac_ij[10+k, 7] = (v[k,i]-vcm[k]-mj*state[4+k])*mijinv - GNEWT*mi*jac_kepler[3+k,7]
   jac_ij[10+k,14] = (v[k,j]-vcm[k]+mi*state[4+k])*mijinv - GNEWT*mi*jac_kepler[3+k,7]
end
# Advance center of mass & individual Keplerian motions:
centerm!(m,mijinv,x,v,vcm,delx,delv,i,j,h)
# Compute the time derivatives:
for k=1:NDIM
  # Define relative velocity and acceleration:
  vij = state[1+NDIM+k]
  acc_ij = gm*state[1+k]/state[8]^3
  # Position derivative, body i:
  dqdt[   k] = vcm[k] + mj*vij
  # Velocity derivative, body i:
  dqdt[ 3+k] = -mj*acc_ij
  # Time derivative of mass is zero, so we skip this.
  # Position derivative, body j:
  dqdt[ 7+k] = vcm[k] - mi*vij
  # Velocity derivative, body j:
  dqdt[10+k] =  mi*acc_ij
  # Time derivative of mass is zero, so we skip this.
end
return
end

# Drifts all particles:
#function drift!(x::Array{Float64,2},v::Array{Float64,2},h::Float64,n::Int64)
function drift!(x::Array{T,2},v::Array{T,2},h::T,n::Int64) where {T <: Real}
@inbounds for i=1:n, j=1:NDIM
  x[j,i] += h*v[j,i]
end
return
end

# Drifts all particles:
function drift!(x::Array{T,2},v::Array{T,2},h::T,n::Int64,jac_step::Array{T,2}) where {T <: Real}
indi = 0
@inbounds for i=1:n
  indi = (i-1)*7
  for j=1:NDIM
    x[j,i] += h*v[j,i]
  end
  # Now for Jacobian:
  for k=1:7*n, j=1:NDIM
    jac_step[indi+j,k] += h*jac_step[indi+3+j,k]
  end
end
return
end

function kickfast!(x::Array{T,2},v::Array{T,2},h::T,m::Array{T,1},n::Int64,pair::Array{Bool,2}) where {T <: Real}
rij = zeros(typeof(h),3)
@inbounds for i=1:n-1
  for j = i+1:n
    if pair[i,j]
      r2 = 0.0
      for k=1:3
        rij[k] = x[k,i] - x[k,j]
        r2 += rij[k]^2
      end
      r3_inv = 1.0/(r2*sqrt(r2))
      for k=1:3
        fac = h*GNEWT*rij[k]*r3_inv
        v[k,i] -= m[j]*fac
        v[k,j] += m[i]*fac
      end
    end
  end
end
return
end

function kickfast!(x::Array{T,2},v::Array{T,2},h::T,m::Array{T,1},n::Int64,jac_step::Array{T,2},
    dqdt_kick::Array{T,1},pair::Array{Bool,2}) where {T <: Real}
rij = zeros(typeof(h),3)
#fill!(jac_step,zero(typeof(h)))
jac_step.=eye(typeof(h),7*n)
@inbounds for i=1:n-1
  indi = (i-1)*7
  for j=i+1:n
    indj = (j-1)*7
    if pair[i,j]
      for k=1:3
        rij[k] = x[k,i] - x[k,j]
      end
      r2inv = 1.0/(rij[1]*rij[1]+rij[2]*rij[2]+rij[3]*rij[3])
      r3inv = r2inv*sqrt(r2inv)
      for k=1:3
        fac = h*GNEWT*rij[k]*r3inv
        # Apply impulses:
        v[k,i] -= m[j]*fac
        v[k,j] += m[i]*fac
        # Compute time derivative:
        dqdt_kick[indi+3+k] -= m[j]*fac/h
        dqdt_kick[indj+3+k] += m[i]*fac/h
        # Computing the derivative
        # Mass derivative of acceleration vector (10/6/17 notes):
        # Impulse of ith particle depends on mass of jth particle:
        jac_step[indi+3+k,indj+7] -= fac
        # Impulse of jth particle depends on mass of ith particle:
        jac_step[indj+3+k,indi+7] += fac
        # x derivative of acceleration vector:
        fac *= 3.0*r2inv
        # Dot product x_ij.\delta x_ij means we need to sum over components:
        for p=1:3
          jac_step[indi+3+k,indi+p] += fac*m[j]*rij[p]
          jac_step[indi+3+k,indj+p] -= fac*m[j]*rij[p]
          jac_step[indj+3+k,indj+p] += fac*m[i]*rij[p]
          jac_step[indj+3+k,indi+p] -= fac*m[i]*rij[p]
        end
        # Final term has no dot product, so just diagonal:
        fac = h*GNEWT*r3inv
        jac_step[indi+3+k,indi+k] -= fac*m[j]
        jac_step[indi+3+k,indj+k] += fac*m[j]
        jac_step[indj+3+k,indj+k] -= fac*m[i]
        jac_step[indj+3+k,indi+k] += fac*m[i]
      end
    end
  end
end
return
end

function phisalpha!(x::Array{T,2},v::Array{T,2},h::T,m::Array{T,1},alpha::T,n::Int64,pair::Array{Bool,2}) where {T <: Real}
# Computes the 4th-order correction:
#function [v] = phisalpha(x,v,h,m,alpha,pair)
#n = size(m,2);
a = zeros(typeof(h),3,n)
rij = zeros(typeof(h),3)
aij = zeros(typeof(h),3)
coeff = alpha*h^3/96*2*GNEWT
zero = 0.0*alpha
fac = zero; fac1 = zero; fac2 = zero; r1 = zero; r2 = zero; r3 = zero
@inbounds for i=1:n-1
  for j = i+1:n
    if ~pair[i,j] # correction for Kepler pairs
      for k=1:3
        rij[k] = x[k,i] - x[k,j]
      end
      r2 = rij[1]*rij[1]+rij[2]*rij[2]+rij[3]*rij[3]
      r3 = r2*sqrt(r2)
      for k=1:3
        fac = GNEWT*rij[k]/r3
        a[k,i] -= m[j]*fac
        a[k,j] += m[i]*fac
      end
    end
  end
end
# Next, compute \tilde g_i acceleration vector (this is rewritten
# slightly to avoid reference to \tilde a_i):
@inbounds for i=1:n-1
  for j=i+1:n
    if ~pair[i,j] # correction for Kepler pairs
      for k=1:3
        aij[k] = a[k,i] - a[k,j]
  #      aij[k] = 0.0
        rij[k] = x[k,i] - x[k,j]
      end
      r2 = rij[1]*rij[1]+rij[2]*rij[2]+rij[3]*rij[3]
      r1 = sqrt(r2)
      ardot = aij[1]*rij[1]+aij[2]*rij[2]+aij[3]*rij[3]
      fac1 = coeff/r1^5
      fac2 = (2*GNEWT*(m[i]+m[j])/r1 + 3*ardot)
      for k=1:3
  #      fac = coeff/r1^5*(rij[k]*(2*GNEWT*(m[i]+m[j])/r1 + 3*ardot) - r2*aij[k])
        fac = fac1*(rij[k]*fac2- r2*aij[k])
        v[k,i] += m[j]*fac
        v[k,j] -= m[i]*fac
      end
    end
  end
end
return
end

function phisalpha!(x::Array{T,2},v::Array{T,2},h::T,m::Array{T,1},
  alpha::T,n::Int64,jac_step::Array{T,2},dqdt_phi::Array{T,1},pair::Array{Bool,2}) where {T <: Real}
# Computes the 4th-order correction:
#function [v] = phisalpha(x,v,h,m,alpha,pair)
#n = size(m,2);
a = zeros(typeof(h),3,n)
dadq = zeros(typeof(h),3,n,4,n)  # There is no velocity dependence
dotdadq = zeros(typeof(h),4,n)  # There is no velocity dependence
rij = zeros(typeof(h),3)
aij = zeros(typeof(h),3)
coeff = alpha*h^3/96*2*GNEWT
fac = 0.0; fac1 = 0.0; fac2 = 0.0; fac3 = 0.0; r1 = 0.0; r2 = 0.0; r3 = 0.0
#jac_step.=eye(typeof(h),7*n)
@inbounds for i=1:n-1
  indi = (i-1)*7
  for j=i+1:n
    if ~pair[i,j] # correction for Kepler pairs
      indj = (j-1)*7
      for k=1:3
        rij[k] = x[k,i] - x[k,j]
      end
      r2 = rij[1]*rij[1]+rij[2]*rij[2]+rij[3]*rij[3]
      r3 = r2*sqrt(r2)
      for k=1:3
        fac = GNEWT*rij[k]/r3
        a[k,i] -= m[j]*fac
        a[k,j] += m[i]*fac
        # Mass derivative of acceleration vector (10/6/17 notes):
        # Since there is no velocity dependence, this is fourth parameter.
        # Acceleration of ith particle depends on mass of jth particle:
        dadq[k,i,4,j] -= fac
        dadq[k,j,4,i] += fac
        # x derivative of acceleration vector:
        fac *= 3.0/r2
        # Dot product x_ij.\delta x_ij means we need to sum over components:
        for p=1:3
          dadq[k,i,p,i] += fac*m[j]*rij[p]
          dadq[k,i,p,j] -= fac*m[j]*rij[p]
          dadq[k,j,p,j] += fac*m[i]*rij[p]
          dadq[k,j,p,i] -= fac*m[i]*rij[p]
        end
        # Final term has no dot product, so just diagonal:
        fac = GNEWT/r3
        dadq[k,i,k,i] -= fac*m[j]
        dadq[k,i,k,j] += fac*m[j]
        dadq[k,j,k,j] -= fac*m[i]
        dadq[k,j,k,i] += fac*m[i]
      end
    end
  end
end
# Next, compute \tilde g_i acceleration vector (this is rewritten
# slightly to avoid reference to \tilde a_i):
# Note that jac_step[(i-1)*7+k,(j-1)*7+p] is the derivative of the kth coordinate
# of planet i with respect to the pth coordinate of planet j.
indi = 0; indj=0; indd = 0
@inbounds for i=1:n-1
  indi = (i-1)*7
  for j=i+1:n
    if ~pair[i,j] # correction for Kepler pairs
      indj = (j-1)*7
      for k=1:3
        aij[k] = a[k,i] - a[k,j]
  #      aij[k] = 0.0
        rij[k] = x[k,i] - x[k,j]
      end
      # Compute dot product of r_ij with \delta a_ij:
      fill!(dotdadq,0.0)
      @inbounds for d=1:n, p=1:4, k=1:3
        dotdadq[p,d] += rij[k]*(dadq[k,i,p,d]-dadq[k,j,p,d])
      end
      r2 = rij[1]*rij[1]+rij[2]*rij[2]+rij[3]*rij[3]
      r1 = sqrt(r2)
      ardot = aij[1]*rij[1]+aij[2]*rij[2]+aij[3]*rij[3]
      fac1 = coeff/r1^5
      fac2 = (2*GNEWT*(m[i]+m[j])/r1 + 3*ardot) 
      for k=1:3
  #      fac = coeff/r1^5*(rij[k]*(2*GNEWT*(m[i]+m[j])/r1 + 3*ardot) - r2*aij[k])
        fac = fac1*(rij[k]*fac2- r2*aij[k])
        v[k,i] += m[j]*fac
        v[k,j] -= m[i]*fac
        # Compute time derivative:
        dqdt_phi[indi+3+k] += 3.0/h*m[j]*fac
        dqdt_phi[indj+3+k] -= 3.0/h*m[i]*fac
        # Mass derivative (first part is easy):
        jac_step[indi+3+k,indj+7] += fac
        jac_step[indj+3+k,indi+7] -= fac
        # Position derivatives:
        fac *= 5.0/r2
        for p=1:3
          jac_step[indi+3+k,indi+p] -= fac*m[j]*rij[p]
          jac_step[indi+3+k,indj+p] += fac*m[j]*rij[p]
          jac_step[indj+3+k,indj+p] -= fac*m[i]*rij[p]
          jac_step[indj+3+k,indi+p] += fac*m[i]*rij[p]
        end
        # Second mass derivative:
        fac = 2*GNEWT*fac1*rij[k]/r1
        jac_step[indi+3+k,indi+7] += fac*m[j]
        jac_step[indi+3+k,indj+7] += fac*m[j]
        jac_step[indj+3+k,indj+7] -= fac*m[i]
        jac_step[indj+3+k,indi+7] -= fac*m[i]
        #  (There's also a mass term in dadq [x]. See below.)
        # Diagonal position terms:
        fac = fac1*fac2
        jac_step[indi+3+k,indi+k] += fac*m[j]
        jac_step[indi+3+k,indj+k] -= fac*m[j]
        jac_step[indj+3+k,indj+k] += fac*m[i]
        jac_step[indj+3+k,indi+k] -= fac*m[i]
        # Dot product \delta rij terms:
        fac = -2*fac1*(rij[k]*GNEWT*(m[i]+m[j])/(r2*r1)+aij[k])
        for p=1:3
          fac3 = fac*rij[p] + fac1*3.0*rij[k]*aij[p]
          jac_step[indi+3+k,indi+p] += m[j]*fac3
          jac_step[indi+3+k,indj+p] -= m[j]*fac3
          jac_step[indj+3+k,indj+p] += m[i]*fac3
          jac_step[indj+3+k,indi+p] -= m[i]*fac3
        end
        # Diagonal acceleration terms:
        fac = -fac1*r2
        # Duoh.  For dadq, have to loop over all other parameters!
        @inbounds for d=1:n
          indd = (d-1)*7
          for p=1:3
            jac_step[indi+3+k,indd+p] += fac*m[j]*(dadq[k,i,p,d]-dadq[k,j,p,d])
            jac_step[indj+3+k,indd+p] -= fac*m[i]*(dadq[k,i,p,d]-dadq[k,j,p,d])
          end
          # Don't forget mass-dependent term:
          jac_step[indi+3+k,indd+7] += fac*m[j]*(dadq[k,i,4,d]-dadq[k,j,4,d])
          jac_step[indj+3+k,indd+7] -= fac*m[i]*(dadq[k,i,4,d]-dadq[k,j,4,d])
        end
        # Now, for the final term:  (\delta a_ij . r_ij ) r_ij
        fac = 3.*fac1*rij[k]
        @inbounds for d=1:n
          indd = (d-1)*7
          for p=1:3
            jac_step[indi+3+k,indd+p] += fac*m[j]*dotdadq[p,d]
            jac_step[indj+3+k,indd+p] -= fac*m[i]*dotdadq[p,d]
          end
          jac_step[indi+3+k,indd+7] += fac*m[j]*dotdadq[4,d]
          jac_step[indj+3+k,indd+7] -= fac*m[i]*dotdadq[4,d]
        end
      end
    end
  end
end
return
end

function phic!(x::Array{T,2},v::Array{T,2},h::T,m::Array{T,1},n::Int64,pair::Array{Bool,2}) where {T <: Real}
a = zeros(typeof(h),3,n)
rij = zeros(typeof(h),3)
aij = zeros(typeof(h),3)
@inbounds for i=1:n-1, j = i+1:n
  if pair[i,j] # kick group
    r2 = 0.0
    for k=1:3
      rij[k] = x[k,i] - x[k,j]
      r2 += rij[k]^2
    end
    r3_inv = 1.0/(r2*sqrt(r2))
    for k=1:3
      fac = GNEWT*rij[k]*r3_inv
      v[k,i] -= m[j]*fac*2h/3
      v[k,j] += m[i]*fac*2h/3
      a[k,i] -= m[j]*fac
      a[k,j] += m[i]*fac
    end
  end
end
coeff = h^3/36*GNEWT
@inbounds for i=1:n-1 ,j=i+1:n
  if pair[i,j] # kick group
    for k=1:3
      aij[k] = a[k,i] - a[k,j]
      rij[k] = x[k,i] - x[k,j]
    end
    r2 = dot(rij,rij)
    r5inv = 1.0/(r2^2*sqrt(r2))
    ardot = dot(aij,rij)
    for k=1:3
      fac = coeff*r5inv*(rij[k]*3*ardot-r2*aij[k])
      v[k,i] += m[j]*fac
      v[k,j] -= m[i]*fac
    end
  end
end
end

function phic!(x::Array{T,2},v::Array{T,2},h::T,m::Array{T,1},n::Int64,jac_step::Array{T,2},pair::Array{Bool,2}) where {T <: Real}
a = zeros(typeof(h),3,n)
rij = zeros(typeof(h),3)
aij = zeros(typeof(h),3)
dadq = zeros(typeof(h),3,n,4,n)  # There is no velocity dependence
dotdadq = zeros(typeof(h),4,n)  # There is no velocity dependence
jac_step.=eye(typeof(h),7*n)
fac = 0.0; fac1 = 0.0; fac2 = 0.0; fac3 = 0.0; r1 = 0.0; r2 = 0.0; r3 = 0.0
coeff = h^3/36*GNEWT
@inbounds for i=1:n-1
  indi = (i-1)*7
  for j=i+1:n
    if pair[i,j]
      indj = (j-1)*7
      for k=1:3
        rij[k] = x[k,i] - x[k,j]
      end
      r2inv = inv(dot(rij,rij))
      r3inv = r2inv*sqrt(r2inv)
      for k=1:3
        # Apply impulses:
        fac = GNEWT*rij[k]*r3inv
        facv = fac*2*h/3
        v[k,i] -= m[j]*facv
        v[k,j] += m[i]*facv
        a[k,i] -= m[j]*fac
        a[k,j] += m[i]*fac
        # Impulse of ith particle depends on mass of jth particle:
        jac_step[indi+3+k,indj+7] -= facv
        # Impulse of jth particle depends on mass of ith particle:
        jac_step[indj+3+k,indi+7] += facv
        # x derivative of acceleration vector:
        facv *= 3.0*r2inv
        # Dot product x_ij.\delta x_ij means we need to sum over components:
        for p=1:3
          jac_step[indi+3+k,indi+p] += facv*m[j]*rij[p]
          jac_step[indi+3+k,indj+p] -= facv*m[j]*rij[p]
          jac_step[indj+3+k,indj+p] += facv*m[i]*rij[p]
          jac_step[indj+3+k,indi+p] -= facv*m[i]*rij[p]
        end
        # Final term has no dot product, so just diagonal:
        facv = 2h/3*GNEWT*r3inv
        jac_step[indi+3+k,indi+k] -= facv*m[j]
        jac_step[indi+3+k,indj+k] += facv*m[j]
        jac_step[indj+3+k,indj+k] -= facv*m[i]
        jac_step[indj+3+k,indi+k] += facv*m[i]
        # Mass derivative of acceleration vector (10/6/17 notes):
        # Since there is no velocity dependence, this is fourth parameter.
        # Acceleration of ith particle depends on mass of jth particle:
        dadq[k,i,4,j] -= fac
        dadq[k,j,4,i] += fac
        # x derivative of acceleration vector:
        fac *= 3.0*r2inv
        # Dot product x_ij.\delta x_ij means we need to sum over components:
        for p=1:3
          dadq[k,i,p,i] += fac*m[j]*rij[p]
          dadq[k,i,p,j] -= fac*m[j]*rij[p]
          dadq[k,j,p,j] += fac*m[i]*rij[p]
          dadq[k,j,p,i] -= fac*m[i]*rij[p]
        end
        # Final term has no dot product, so just diagonal:
        fac = GNEWT*r3inv
        dadq[k,i,k,i] -= fac*m[j]
        dadq[k,i,k,j] += fac*m[j]
        dadq[k,j,k,j] -= fac*m[i]
        dadq[k,j,k,i] += fac*m[i]
      end
    end
  end
end
# Next, compute g_i acceleration vector.
# Note that jac_step[(i-1)*7+k,(j-1)*7+p] is the derivative of the kth coordinate
# of planet i with respect to the pth coordinate of planet j.
indi = 0; indj=0; indd = 0
@inbounds for i=1:n-1
  indi = (i-1)*7
  for j=i+1:n
    if pair[i,j] # correction for Kepler pairs
      indj = (j-1)*7
      for k=1:3
        aij[k] = a[k,i] - a[k,j]
        rij[k] = x[k,i] - x[k,j]
      end
      # Compute dot product of r_ij with \delta a_ij:
      fill!(dotdadq,0.0)
      @inbounds for d=1:n, p=1:4, k=1:3
        dotdadq[p,d] += rij[k]*(dadq[k,i,p,d]-dadq[k,j,p,d])
      end
      r2 = dot(rij,rij)
      r1 = sqrt(r2)
      ardot = dot(aij,rij)
      fac1 = coeff/r1^5
      fac2 = 3*ardot
      for k=1:3
        fac = fac1*(rij[k]*fac2- r2*aij[k])
        v[k,i] += m[j]*fac
        v[k,j] -= m[i]*fac
        # Mass derivative (first part is easy):
        jac_step[indi+3+k,indj+7] += fac
        jac_step[indj+3+k,indi+7] -= fac
        # Position derivatives:
        fac *= 5.0/r2
        for p=1:3
          jac_step[indi+3+k,indi+p] -= fac*m[j]*rij[p]
          jac_step[indi+3+k,indj+p] += fac*m[j]*rij[p]
          jac_step[indj+3+k,indj+p] -= fac*m[i]*rij[p]
          jac_step[indj+3+k,indi+p] += fac*m[i]*rij[p]
        end
        # Diagonal position terms:
        fac = fac1*fac2
        jac_step[indi+3+k,indi+k] += fac*m[j]
        jac_step[indi+3+k,indj+k] -= fac*m[j]
        jac_step[indj+3+k,indj+k] += fac*m[i]
        jac_step[indj+3+k,indi+k] -= fac*m[i]
        # Dot product \delta rij terms:
        fac = -2*fac1*aij[k]
        for p=1:3
          fac3 = fac*rij[p] + fac1*3.0*rij[k]*aij[p]
          jac_step[indi+3+k,indi+p] += m[j]*fac3
          jac_step[indi+3+k,indj+p] -= m[j]*fac3
          jac_step[indj+3+k,indj+p] += m[i]*fac3
          jac_step[indj+3+k,indi+p] -= m[i]*fac3
        end
        # Diagonal acceleration terms:
        fac = -fac1*r2
        # Duoh.  For dadq, have to loop over all other parameters!
        @inbounds for d=1:n
          indd = (d-1)*7
          for p=1:3
            jac_step[indi+3+k,indd+p] += fac*m[j]*(dadq[k,i,p,d]-dadq[k,j,p,d])
            jac_step[indj+3+k,indd+p] -= fac*m[i]*(dadq[k,i,p,d]-dadq[k,j,p,d])
          end
          # Don't forget mass-dependent term:
          jac_step[indi+3+k,indd+7] += fac*m[j]*(dadq[k,i,4,d]-dadq[k,j,4,d])
          jac_step[indj+3+k,indd+7] -= fac*m[i]*(dadq[k,i,4,d]-dadq[k,j,4,d])
        end
        # Now, for the final term:  (\delta a_ij . r_ij ) r_ij
        fac = 3.*fac1*rij[k]
        @inbounds for d=1:n
          indd = (d-1)*7
          for p=1:3
            jac_step[indi+3+k,indd+p] += fac*m[j]*dotdadq[p,d]
            jac_step[indj+3+k,indd+p] -= fac*m[i]*dotdadq[p,d]
          end
          jac_step[indi+3+k,indd+7] += fac*m[j]*dotdadq[4,d]
          jac_step[indj+3+k,indd+7] -= fac*m[i]*dotdadq[4,d]
        end
      end
    end
  end
end
return
end

# Carries out the AH18 mapping:
function ah18!(x::Array{T,2},v::Array{T,2},h::T,m::Array{T,1},n::Int64,pair::Array{Bool,2}) where {T <: Real}
# New version of solver that consolidates keplerij and driftij, and sets
# alpha = 0:
h2 = 0.5*h
drift!(x,v,h2,n)
kickfast!(x,v,h2,m,n,pair)
@inbounds for i=1:n-1
  for j=i+1:n
    if ~pair[i,j]
      kepler_driftij!(m,x,v,i,j,h2,true)
    end
  end
end
phisalpha!(x,v,h,m,convert(typeof(h),2),n,pair)
for i=n-1:-1:1
  for j=n:-1:i+1
    if ~pair[i,j]
      kepler_driftij!(m,x,v,i,j,h2,false)
    end
  end
end
kickfast!(x,v,h2,m,n,pair)
drift!(x,v,h2,n)
return
end

# Carries out the AH18 mapping & computes the Jacobian:
function ah18!(x::Array{T,2},v::Array{T,2},h::T,m::Array{T,1},n::Int64,jac_step::Array{T,2},pair::Array{Bool,2}) where {T <: Real}
zero = convert(typeof(h),0.0); one = convert(typeof(h),1.0); half = convert(typeof(h),0.5); two = convert(typeof(h),2.0)
h2 = half*h
sevn = 7*n
jac_phi = zeros(typeof(h),sevn,sevn)
jac_kick = zeros(typeof(h),sevn,sevn)
jac_copy = zeros(typeof(h),sevn,sevn)
jac_ij = zeros(typeof(h),14,14)
dqdt_ij = zeros(typeof(h),14)
dqdt_phi = zeros(typeof(h),sevn)
dqdt_kick = zeros(typeof(h),sevn)
jac_tmp1 = zeros(typeof(h),14,sevn)
jac_tmp2 = zeros(typeof(h),14,sevn)
drift!(x,v,h2,n,jac_step)
kickfast!(x,v,h2,m,n,jac_kick,dqdt_kick,pair)
# Multiply Jacobian from kick step:
@inbounds for i in eachindex(jac_step)
  jac_copy[i] = jac_step[i]
end
if typeof(h) == BigFloat
  jac_step = *(jac_kick,jac_copy)
else
  BLAS.gemm!('N','N',one,jac_kick,jac_copy,zero,jac_step)
end
indi = 0; indj = 0
@inbounds for i=1:n-1
  indi = (i-1)*7
  for j=i+1:n
    indj = (j-1)*7
    if ~pair[i,j]  # Check to see if kicks have not been applied
      kepler_driftij!(m,x,v,i,j,h2,jac_ij,true)
    # Pick out indices for bodies i & j:
      @inbounds for k2=1:sevn, k1=1:7
        jac_tmp1[k1,k2] = jac_step[indi+k1,k2]
      end
      @inbounds for k2=1:sevn, k1=1:7
        jac_tmp1[7+k1,k2] = jac_step[indj+k1,k2]
      end
      # Carry out multiplication on the i/j components of matrix:
      if typeof(h) == BigFloat
        jac_tmp2 = *(jac_ij,jac_tmp1)
      else
        BLAS.gemm!('N','N',one,jac_ij,jac_tmp1,zero,jac_tmp2)
      end
      # Copy back to the Jacobian:
      @inbounds for k2=1:sevn, k1=1:7
         jac_step[indi+k1,k2]=jac_tmp2[k1,k2]
      end
      @inbounds for k2=1:sevn, k1=1:7
        jac_step[indj+k1,k2]=jac_tmp2[7+k1,k2]
      end
    end
  end
end
# Need to set jac_phi to identity before calling this. [ ]
jac_phi = eye(typeof(h),sevn)
phisalpha!(x,v,h,m,two,n,jac_phi,dqdt_phi,pair) # 10%
@inbounds for i in eachindex(jac_step)
  jac_copy[i] = jac_step[i]
end
#  jac_step .= jac_phi*jac_step # < 1%  Perhaps use gemm?! [ ]
if typeof(h) == BigFloat
  jac_step = *(jac_phi,jac_copy)
else
  BLAS.gemm!('N','N',one,jac_phi,jac_copy,zero,jac_step)
end
indi=0; indj=0
for i=n-1:-1:1
  indi=(i-1)*7
  for j=n:-1:i+1
    indj=(j-1)*7
    if ~pair[i,j]  # Check to see if kicks have not been applied
      kepler_driftij!(m,x,v,i,j,h2,jac_ij,false)
      # Pick out indices for bodies i & j:
      # Carry out multiplication on the i/j components of matrix:
      @inbounds for k2=1:sevn, k1=1:7
        jac_tmp1[k1,k2] = jac_step[indi+k1,k2]
      end
      @inbounds for k2=1:sevn, k1=1:7
        jac_tmp1[7+k1,k2] = jac_step[indj+k1,k2]
      end
      # Carry out multiplication on the i/j components of matrix:
      if typeof(h) == BigFloat
        jac_tmp2 = *(jac_ij,jac_tmp1)
      else
        BLAS.gemm!('N','N',one,jac_ij,jac_tmp1,zero,jac_tmp2)
      end
      # Copy back to the Jacobian:
      @inbounds for k2=1:sevn, k1=1:7
         jac_step[indi+k1,k2]=jac_tmp2[k1,k2]
      end
      @inbounds for k2=1:sevn, k1=1:7
        jac_step[indj+k1,k2]=jac_tmp2[7+k1,k2]
      end
    end
  end
end
kickfast!(x,v,h2,m,n,jac_kick,dqdt_kick,pair)
# Multiply Jacobian from kick step:
@inbounds for i in eachindex(jac_step)
  jac_copy[i] = jac_step[i]
end
if typeof(h) == BigFloat
  jac_step = *(jac_kick,jac_copy)
else
  BLAS.gemm!('N','N',one,jac_kick,jac_copy,zero,jac_step)
end
drift!(x,v,h2,n,jac_step)
return
end

# Carries out the DH17 mapping:
function dh17!(x::Array{T,2},v::Array{T,2},h::T,m::Array{T,1},n::Int64,pair::Array{Bool,2}) where {T <: Real}
alpha = convert(typeof(h),alpha0)
h2 = 0.5*h
# alpha = 0. is similar in precision to alpha=0.25
kickfast!(x,v,h/6,m,n,pair)
if alpha != 0.0
  phisalpha!(x,v,h,m,alpha,n,pair)
end
drift!(x,v,h2,n)
@inbounds for i=1:n-1
  for j=i+1:n
    if ~pair[i,j]
      driftij!(x,v,i,j,-h2)
      keplerij!(m,x,v,i,j,h2)
    end
  end
end
# kick and correction for pairs which are kicked:
phic!(x,v,h,m,n,pair)
if alpha != 1.0
  phisalpha!(x,v,h,m,2.*(1.-alpha),n,pair)
end
for i=n-1:-1:1
  for j=n:-1:i+1
    if ~pair[i,j]
      keplerij!(m,x,v,i,j,h2)
      driftij!(x,v,i,j,-h2)
    end
  end
end
drift!(x,v,h2,n)
if alpha != 0.0
  phisalpha!(x,v,h,m,alpha,n,pair)
end
kickfast!(x,v,h/6,m,n,pair)
return
end

# Used in computing the transit time:
function g!(i::Int64,j::Int64,x::Array{T,2},v::Array{T,2}) where {T <: Real}
# See equation 8-10 Fabrycky (2008) in Seager Exoplanets book
g = (x[1,j]-x[1,i])*(v[1,j]-v[1,i])+(x[2,j]-x[2,i])*(v[2,j]-v[2,i])
return g
end

# Carries out the DH17 mapping & computes the Jacobian:
function dh17!(x::Array{T,2},v::Array{T,2},h::T,m::Array{T,1},n::Int64,jac_step::Array{T,2},pair::Array{Bool,2}) where {T <: Real}
zero = convert(typeof(h),0.0); one = convert(typeof(h),1.0); half = convert(typeof(h),0.5); two = convert(typeof(h),2.0)
h2 = half*h
alpha = convert(typeof(h),alpha0)
sevn = 7*n
jac_phi = zeros(typeof(h),sevn,sevn)
jac_kick = zeros(typeof(h),sevn,sevn)
jac_copy = zeros(typeof(h),sevn,sevn)
jac_ij = zeros(typeof(h),14,14)
dqdt_ij = zeros(typeof(h),14)
dqdt_phi = zeros(typeof(h),sevn)
dqdt_kick = zeros(typeof(h),sevn)
jac_tmp1 = zeros(typeof(h),14,sevn)
jac_tmp2 = zeros(typeof(h),14,sevn)
kickfast!(x,v,h/6,m,n,jac_phi,dqdt_kick,pair)
# alpha = 0. is similar in precision to alpha=0.25
if alpha != zero
  phisalpha!(x,v,h,m,alpha,n,jac_phi,dqdt_phi,pair)
#  jac_step .= jac_phi*jac_step # < 1%
end
# Multiply Jacobian from kick/phisalpha steps:
@inbounds for i in eachindex(jac_step)
  jac_copy[i] = jac_step[i]
end
if typeof(h) == BigFloat
  jac_step = *(jac_phi,jac_copy)
else
  BLAS.gemm!('N','N',one,jac_phi,jac_copy,zero,jac_step)
end
drift!(x,v,h2,n,jac_step)
indi = 0; indj = 0
@inbounds for i=1:n-1
  indi = (i-1)*7
  for j=i+1:n
    indj = (j-1)*7
    if ~pair[i,j]  # Check to see if kicks have not been applied
      driftij!(x,v,i,j,-h2,jac_step,n)
      keplerij!(m,x,v,i,j,h2,jac_ij,dqdt_ij) # 21%
    # Pick out indices for bodies i & j:
      @inbounds for k2=1:sevn, k1=1:7
        jac_tmp1[k1,k2] = jac_step[indi+k1,k2]
      end
      @inbounds for k2=1:sevn, k1=1:7
        jac_tmp1[7+k1,k2] = jac_step[indj+k1,k2]
      end
      # Carry out multiplication on the i/j components of matrix:
#    jac_tmp2 = BLAS.gemm('N','N',jac_ij,jac_tmp1)
      if typeof(h) == BigFloat
        jac_tmp2 = *(jac_ij,jac_tmp1)
      else
        BLAS.gemm!('N','N',one,jac_ij,jac_tmp1,zero,jac_tmp2)
      end
      # Copy back to the Jacobian:
      @inbounds for k2=1:sevn, k1=1:7
         jac_step[indi+k1,k2]=jac_tmp2[k1,k2]
      end
      @inbounds for k2=1:sevn, k1=1:7
        jac_step[indj+k1,k2]=jac_tmp2[7+k1,k2]
      end
    end
  end
end
# kick and correction for pairs which are kicked:
phic!(x,v,h,m,n,jac_phi,pair)
if alpha != one
#  phisalpha!(x,v,h,m,2.*(1.-alpha),n,jac_phi,pair) # 10%
  phisalpha!(x,v,h,m,two*(one-alpha),n,jac_phi,dqdt_phi,pair) # 10%
#  @inbounds for i in eachindex(jac_step)
#    jac_copy[i] = jac_step[i]
#  end
#  jac_step .= jac_phi*jac_step # < 1%  Perhaps use gemm?! [ ]
#  if typeof(h) == BigFloat
#    jac_step = *(jac_phi,jac_copy)
#  else
#    BLAS.gemm!('N','N',one,jac_phi,jac_copy,zero,jac_step)
#  end
end
@inbounds for i in eachindex(jac_step)
  jac_copy[i] = jac_step[i]
end
if typeof(h) == BigFloat
  jac_step = *(jac_phi,jac_copy)
else
  BLAS.gemm!('N','N',one,jac_phi,jac_copy,zero,jac_step)
end
indi=0; indj=0
for i=n-1:-1:1
  indi=(i-1)*7
  for j=n:-1:i+1
    indj=(j-1)*7
    if ~pair[i,j]  # Check to see if kicks have not been applied
      keplerij!(m,x,v,i,j,h2,jac_ij,dqdt_ij) # 23%
      # Pick out indices for bodies i & j:
      # Carry out multiplication on the i/j components of matrix:
      @inbounds for k2=1:sevn, k1=1:7
        jac_tmp1[k1,k2] = jac_step[indi+k1,k2]
      end
      @inbounds for k2=1:sevn, k1=1:7
        jac_tmp1[7+k1,k2] = jac_step[indj+k1,k2]
      end
      # Carry out multiplication on the i/j components of matrix:
  #    jac_tmp2 = BLAS.gemm('N','N',jac_ij,jac_tmp1)
      if typeof(h) == BigFloat
        jac_tmp2 = *(jac_ij,jac_tmp1)
      else
        BLAS.gemm!('N','N',one,jac_ij,jac_tmp1,zero,jac_tmp2)
      end
      # Copy back to the Jacobian:
      @inbounds for k2=1:sevn, k1=1:7
         jac_step[indi+k1,k2]=jac_tmp2[k1,k2]
      end
      @inbounds for k2=1:sevn, k1=1:7
        jac_step[indj+k1,k2]=jac_tmp2[7+k1,k2]
      end
      driftij!(x,v,i,j,-h2,jac_step,n) 
    end
  end
end
drift!(x,v,h2,n,jac_step)
if alpha != zero
#  phisalpha!(x,v,h,m,alpha,n,jac_phi)
  phisalpha!(x,v,h,m,alpha,n,jac_phi,dqdt_phi,pair) # 10%
#  jac_step .= jac_phi*jac_step # < 1%
end
kickfast!(x,v,h/6,m,n,jac_phi,dqdt_kick,pair)
# Multiply Jacobian from kick step:
@inbounds for i in eachindex(jac_step)
  jac_copy[i] = jac_step[i]
end
if typeof(h) == BigFloat
  jac_step = *(jac_phi,jac_copy)
else
  BLAS.gemm!('N','N',one,jac_phi,jac_copy,zero,jac_step)
end
#println("jac_step: ",typeof(h)," ",convert(Array{Float64,2},jac_step))
return #jac_step
end

# Carries out the DH17 mapping & computes the derivative with respect to time step, h:
function dh17!(x::Array{T,2},v::Array{T,2},h::T,m::Array{T,1},n::Int64,dqdt::Array{T,1},pair::Array{Bool,2}) where {T <: Real}
zero = convert(typeof(h),0.0); one = convert(typeof(h),1.0); half = convert(typeof(h),0.5); two = convert(typeof(h),2.0)
h2 = half*h
# This routine assumes that alpha = 0.0
sevn = 7*n
jac_phi = zeros(typeof(h),sevn,sevn)
jac_kick = zeros(typeof(h),sevn,sevn)
jac_ij = zeros(typeof(h),14,14)
dqdt_ij = zeros(typeof(h),14)
dqdt_phi = zeros(typeof(h),sevn)
dqdt_kick = zeros(typeof(h),sevn)
dqdt_tmp1 = zeros(typeof(h),14)
fill!(dqdt,zero)
# dqdt_save is for debugging:
#dqdt_save = copy(dqdt)
drift!(x,v,h2,n)
# Compute time derivative of drift step:
for i=1:n, k=1:3
  dqdt[(i-1)*7+k] = half*v[k,i] + h2*dqdt[(i-1)*7+3+k]
end
#println("dqdt 1: ",dqdt-dqdt_save)
#dqdt_save .= dqdt
kickfast!(x,v,h/6,m,n,jac_kick,dqdt_kick,pair)
dqdt_kick /= 6 # Since step is h/6
dqdt_kick .+= *(jac_kick,dqdt)
# Copy result to dqdt:
dqdt .= dqdt_kick
#println("dqdt 2: ",dqdt-dqdt_save)
# dqdt_save .= dqdt
@inbounds for i=1:n-1
  indi = (i-1)*7
  for j=i+1:n
    indj = (j-1)*7
    if ~pair[i,j]  # Check to see if kicks have not been applied
      driftij!(x,v,i,j,-h2,dqdt,-half)
#      println("dqdt 3: i: ",i," j: ",j," diff: ",dqdt-dqdt_save)
#      dqdt_save .= dqdt
      keplerij!(m,x,v,i,j,h2,jac_ij,dqdt_ij) # 21%
      # Copy current time derivatives for multiplication purposes:
      @inbounds for k1=1:7
        dqdt_tmp1[  k1] = dqdt[indi+k1]
        dqdt_tmp1[7+k1] = dqdt[indj+k1]
      end
      # Add in partial derivatives with respect to time:
      # Need to multiply by 1/2 since we're taking 1/2 time step:
  #    BLAS.gemm!('N','N',one,jac_ij,dqdt_tmp1,half,dqdt_ij)
      dqdt_ij .*= half
      dqdt_ij .+= *(jac_ij,dqdt_tmp1)
      # Copy back time derivatives:
      @inbounds for k1=1:7
        dqdt[indi+k1] = dqdt_ij[  k1]
        dqdt[indj+k1] = dqdt_ij[7+k1]
      end
#      println("dqdt 4: i: ",i," j: ",j," diff: ",dqdt-dqdt_save)
#      dqdt_save .= dqdt
    end
  end
end
# Looks like we are missing phic here: [ ]  
# Since I haven't added dqdt to phic yet, for now, set jac_phi equal to identity matrix
# (since this is commented out within phisalpha):
jac_phi .= eye(typeof(h),sevn)
phisalpha!(x,v,h,m,two,n,jac_phi,dqdt_phi,pair) # 10%
# Add in time derivative with respect to prior parameters:
#BLAS.gemm!('N','N',one,jac_phi,dqdt,one,dqdt_phi)
dqdt_phi .+= *(jac_phi,dqdt)
# Copy result to dqdt:
dqdt .= dqdt_phi
#println("dqdt 5: ",dqdt-dqdt_save)
#dqdt_save .= dqdt
indi=0; indj=0
for i=n-1:-1:1
  indi=(i-1)*7
  for j=n:-1:i+1
    if ~pair[i,j]  # Check to see if kicks have not been applied
      indj=(j-1)*7
      keplerij!(m,x,v,i,j,h2,jac_ij,dqdt_ij) # 23%
      # Copy current time derivatives for multiplication purposes:
      @inbounds for k1=1:7
        dqdt_tmp1[  k1] = dqdt[indi+k1]
        dqdt_tmp1[7+k1] = dqdt[indj+k1]
      end
      # Add in partial derivatives with respect to time:
      # Need to multiply by 1/2 since we're taking 1/2 time step:
      #BLAS.gemm!('N','N',one,jac_ij,dqdt_tmp1,half,dqdt_ij)
      dqdt_ij .*= half
      dqdt_ij .+= *(jac_ij,dqdt_tmp1)
      # Copy back time derivatives:
      @inbounds for k1=1:7
        dqdt[indi+k1] = dqdt_ij[  k1]
        dqdt[indj+k1] = dqdt_ij[7+k1]
      end
#      println("dqdt 6: i: ",i," j: ",j," diff: ",dqdt-dqdt_save)
#      dqdt_save .= dqdt
      driftij!(x,v,i,j,-h2,dqdt,-half)
#      println("dqdt 7: ",dqdt-dqdt_save)
#      dqdt_save .= dqdt
    end
  end
end
fill!(dqdt_kick,zero)
kickfast!(x,v,h/6,m,n,jac_kick,dqdt_kick,pair)
dqdt_kick /= 6 # Since step is h/6
dqdt_kick .+= *(jac_kick,dqdt)
#println("dqdt 8: ",dqdt-dqdt_save)
#dqdt_save .= dqdt
# Copy result to dqdt:
dqdt .= dqdt_kick
drift!(x,v,h2,n)
# Compute time derivative of drift step:
for i=1:n, k=1:3
  dqdt[(i-1)*7+k] += half*v[k,i] + h2*dqdt[(i-1)*7+3+k]
end
#println("dqdt 9: ",dqdt-dqdt_save)
return
end

# Finds the transit by taking a partial dh17 step from prior times step:
#function findtransit2!(i::Int64,j::Int64,n::Int64,h::Float64,tt::Float64,m::Array{Float64,1},x1::Array{Float64,2},v1::Array{Float64,2})
function findtransit2!(i::Int64,j::Int64,n::Int64,h::T,tt::T,m::Array{T,1},x1::Array{T,2},v1::Array{T,2},pair::Array{Bool,2}) where {T <: Real}
# Computes the transit time, approximating the motion as a fraction of a DH17 step forward in time.
# Initial guess using linear interpolation:
dt = one(h)
iter = 0
r3 = zero(h)
gdot = zero(h)
gsky = zero(h)
x = copy(x1)
v = copy(v1)
#TRANSIT_TOL = 10*sqrt(eps(one(typeof(h))))
TRANSIT_TOL = 10*eps(one(typeof(h)))
#println("h: ",h,", TRANSIT_TOL: ",TRANSIT_TOL)
while abs(dt) > TRANSIT_TOL && iter < 20
  x .= x1
  v .= v1
  # Advance planet state at start of step to estimated transit time:
  dh17!(x,v,tt,m,n,pair)
  # Compute time offset:
  gsky = g!(i,j,x,v)
  # Compute gravitational acceleration in sky plane dotted with sky position:
  gdot = zero(h)
  for k=1:n
    if k != i
      r3 = sqrt((x[1,k]-x[1,i])^2+(x[2,k]-x[2,i])^2 +(x[3,k]-x[3,i])^2)^3
      gdot -= GNEWT*m[k]*((x[1,k]-x[1,i])*(x[1,j]-x[1,i])+(x[2,k]-x[2,i])*(x[2,j]-x[2,i]))/r3
    end
    if k != j
      r3 = sqrt((x[1,k]-x[1,j])^2+(x[2,k]-x[2,j])^2 +(x[3,k]-x[3,j])^2)^3
      gdot += GNEWT*m[k]*((x[1,k]-x[1,j])*(x[1,j]-x[1,i])+(x[2,k]-x[2,j])*(x[2,j]-x[2,i]))/r3
    end
  end
  # Compute derivative of g with respect to time:
  gdot += (v[1,j]-v[1,i])^2+(v[2,j]-v[2,i])^2
  # Refine estimate of transit time with Newton's method:
  dt = -gsky/gdot
  # Add refinement to estimated time:
  tt += dt
  iter +=1
end
#x = copy(x1)
#v = copy(v1)
#dh17!(x,v,tt,m,n,pair)
# Compute time offset:
#gsky = g!(i,j,x,v)
#println("gsky: ",convert(Float64,gsky))
if iter >= 20
  println("Exceeded iterations: planet ",j," iter ",iter," dt ",dt," gsky ",gsky," gdot ",gdot)
end
# Note: this is the time elapsed *after* the beginning of the timestep:
return tt::typeof(h)
end

# Finds the transit by taking a partial dh17 step from prior times step, computes timing Jacobian, dtdq, wrt initial cartesian coordinates, masses:
#function findtransit2!(i::Int64,j::Int64,n::Int64,h::Float64,tt::Float64,m::Array{Float64,1},x1::Array{Float64,2},v1::Array{Float64,2},jac_step::Array{Float64,2},dtdq::Array{Float64,2},pair::Array{Bool,2})
function findtransit2!(i::Int64,j::Int64,n::Int64,h::T,tt::T,m::Array{T,1},x1::Array{T,2},v1::Array{T,2},jac_step::Array{T,2},dtdq::Array{T,2},pair::Array{Bool,2})  where {T <: Real}

# Computes the transit time, approximating the motion as a fraction of a DH17 step forward in time.
# Also computes the Jacobian of the transit time with respect to the initial parameters, dtdq[7,n].
# Initial guess using linear interpolation:
#dt = 1.0
dt = one(h)
iter = 0
#r3 = 0.0
r3 = zero(h)
#gdot = 0.0
gdot = zero(h)
#gsky = 0.0
gsky = zero(h)
x = copy(x1)
v = copy(v1)
#TRANSIT_TOL = 10*sqrt(eps(one(typeof(h))))
TRANSIT_TOL = 10*eps(one(typeof(h)))
#println("h: ",h,", TRANSIT_TOL: ",TRANSIT_TOL)
while abs(dt) > TRANSIT_TOL && iter < 20
  x .= x1
  v .= v1
  # Advance planet state at start of step to estimated transit time:
  dh17!(x,v,tt,m,n,pair)
  # Compute time offset:
  gsky = g!(i,j,x,v)
  # Compute gravitational acceleration in sky plane dotted with sky position:
  #gdot = 0.0
  gdot = zero(h)
  for k=1:n
    if k != i
      r3 = sqrt((x[1,k]-x[1,i])^2+(x[2,k]-x[2,i])^2 +(x[3,k]-x[3,i])^2)^3
      gdot -= GNEWT*m[k]*((x[1,k]-x[1,i])*(x[1,j]-x[1,i])+(x[2,k]-x[2,i])*(x[2,j]-x[2,i]))/r3
# g = (x[1,j]-x[1,i])*(v[1,j]-v[1,i])+(x[2,j]-x[2,i])*(v[2,j]-v[2,i])
    end
    if k != j
      r3 = sqrt((x[1,k]-x[1,j])^2+(x[2,k]-x[2,j])^2 +(x[3,k]-x[3,j])^2)^3
      gdot += GNEWT*m[k]*((x[1,k]-x[1,j])*(x[1,j]-x[1,i])+(x[2,k]-x[2,j])*(x[2,j]-x[2,i]))/r3
    end
  end
  # Compute derivative of g with respect to time:
  gdot += (v[1,j]-v[1,i])^2+(v[2,j]-v[2,i])^2
  # Refine estimate of transit time with Newton's method:
  dt = -gsky/gdot
  # Add refinement to estimated time:
  tt += dt
  iter +=1
end
if iter >= 20
  println("Exceeded iterations: planet ",j," iter ",iter," dt ",dt," gsky ",gsky," gdot ",gdot)
end
# Compute time derivatives:
x = copy(x1)
v = copy(v1)
# Compute dgdt with the updated time step.
dh17!(x,v,tt,m,n,jac_step,pair)
#println("\Delta t: ",tt)
#println("jac_step: ",jac_step)
# Compute time offset:
gsky = g!(i,j,x,v)
#println("gsky: ",convert(Float64,gsky))
# Compute gravitational acceleration in sky plane dotted with sky position:
#gdot = 0.0
gdot = (v[1,j]-v[1,i])^2+(v[2,j]-v[2,i])^2
gdot_acc = zero(h)
for k=1:n
  if k != i
    r3 = sqrt((x[1,k]-x[1,i])^2+(x[2,k]-x[2,i])^2 +(x[3,k]-x[3,i])^2)^3
    # Does this have the wrong sign?!
    gdot_acc -= GNEWT*m[k]*((x[1,k]-x[1,i])*(x[1,j]-x[1,i])+(x[2,k]-x[2,i])*(x[2,j]-x[2,i]))/r3
# g = (x[1,j]-x[1,i])*(v[1,j]-v[1,i])+(x[2,j]-x[2,i])*(v[2,j]-v[2,i])
  end
  if k != j
    r3 = sqrt((x[1,k]-x[1,j])^2+(x[2,k]-x[2,j])^2 +(x[3,k]-x[3,j])^2)^3
    # Does this have the wrong sign?!
    gdot_acc += GNEWT*m[k]*((x[1,k]-x[1,j])*(x[1,j]-x[1,i])+(x[2,k]-x[2,j])*(x[2,j]-x[2,i]))/r3
  end
end
gdot += gdot_acc
#println("gdot_acc/gdot: ",gdot_acc/gdot)
# Compute derivative of g with respect to time:
# Set dtdq to zero:
fill!(dtdq,zero(h))
indj = (j-1)*7+1
indi = (i-1)*7+1
for p=1:n
  indp = (p-1)*7
  for k=1:7
    # Compute derivatives:
    #g = (x[1,j]-x[1,i])*(v[1,j]-v[1,i])+(x[2,j]-x[2,i])*(v[2,j]-v[2,i])
    dtdq[k,p] = -((jac_step[indj  ,indp+k]-jac_step[indi  ,indp+k])*(v[1,j]-v[1,i])+(jac_step[indj+1,indp+k]-jac_step[indi+1,indp+k])*(v[2,j]-v[2,i])+
                  (jac_step[indj+3,indp+k]-jac_step[indi+3,indp+k])*(x[1,j]-x[1,i])+(jac_step[indj+4,indp+k]-jac_step[indi+4,indp+k])*(x[2,j]-x[2,i]))/gdot
  end
end
# Note: this is the time elapsed *after* the beginning of the timestep:
return tt::typeof(h),gdot::typeof(h)
end

# Finds the transit by taking a partial dh17 step from prior times step, computes timing Jacobian, dtdq, wrt initial cartesian coordinates, masses:
function findtransit3!(i::Int64,j::Int64,n::Int64,h::T,tt::T,m::Array{T,1},x1::Array{T,2},v1::Array{T,2},pair::Array{Bool,2}) where {T <: Real}
# Computes the transit time, approximating the motion as a fraction of a DH17 step forward in time.
# Also computes the Jacobian of the transit time with respect to the initial parameters, dtdq[7,n].
# This version is same as findtransit2, but uses the derivative of dh17 step with respect to time
# rather than the instantaneous acceleration for finding transit time derivative (gdot).
# Initial guess using linear interpolation:
dt = 1.0
iter = 0
r3 = 0.0
gdot = 0.0
gsky = 0.0
x = copy(x1)
v = copy(v1)
dqdt = zeros(typeof(h),7*n)
#TRANSIT_TOL = 10*sqrt(eps(one(typeof(h))))
TRANSIT_TOL = 10*eps(one(typeof(h)))
while abs(dt) > TRANSIT_TOL && iter < 20
  x .= x1
  v .= v1
  # Advance planet state at start of step to estimated transit time:
#  dh17!(x,v,tt,m,n,pair)
  dh17!(x,v,tt,m,n,dqdt,pair)
  # Compute time offset:
  gsky = g!(i,j,x,v)
#  # Compute derivative of g with respect to time:
  gdot = ((x[1,j]-x[1,i])*(dqdt[(j-1)*7+4]-dqdt[(i-1)*7+4])+(x[2,j]-x[2,i])*(dqdt[(j-1)*7+5]-dqdt[(i-1)*7+5])
       +  (v[1,j]-v[1,i])*(dqdt[(j-1)*7+1]-dqdt[(i-1)*7+1])+(v[2,j]-v[2,i])*(dqdt[(j-1)*7+2]-dqdt[(i-1)*7+2]))
  # Refine estimate of transit time with Newton's method:
  dt = -gsky/gdot
  # Add refinement to estimated time:
  tt += dt
  iter +=1
end
if iter >= 20
  println("Exceeded iterations: planet ",j," iter ",iter," dt ",dt," gsky ",gsky," gdot ",gdot)
end
# Note: this is the time elapsed *after* the beginning of the timestep:
return tt::typeof(h)
end

# Finds the transit by taking a partial dh17 step from prior times step, computes timing Jacobian, dtdq, wrt initial cartesian coordinates, masses:
function findtransit3!(i::Int64,j::Int64,n::Int64,h::Float64,tt::Float64,m::Array{Float64,1},x1::Array{Float64,2},v1::Array{Float64,2},jac_step::Array{Float64,2},dtdq::Array{Float64,2},pair::Array{Bool,2})
# Computes the transit time, approximating the motion as a fraction of a DH17 step forward in time.
# Also computes the Jacobian of the transit time with respect to the initial parameters, dtdq[7,n].
# This version is same as findtransit2, but uses the derivative of dh17 step with respect to time
# rather than the instantaneous acceleration for finding transit time derivative (gdot).
# Initial guess using linear interpolation:
dt = 1.0
iter = 0
r3 = 0.0
gdot = 0.0
gsky = 0.0
x = copy(x1)
v = copy(v1)
dqdt = zeros(Float64,7*n)
#TRANSIT_TOL = 10*sqrt(eps(one(typeof(h))))
TRANSIT_TOL = 10*eps(one(typeof(h)))
while abs(dt) > TRANSIT_TOL && iter < 20
  x .= x1
  v .= v1
  # Advance planet state at start of step to estimated transit time:
#  dh17!(x,v,tt,m,n,pair)
  dh17!(x,v,tt,m,n,dqdt,pair)
  # Compute time offset:
  gsky = g!(i,j,x,v)
#  # Compute derivative of g with respect to time:
  gdot = ((x[1,j]-x[1,i])*(dqdt[(j-1)*7+4]-dqdt[(i-1)*7+4])+(x[2,j]-x[2,i])*(dqdt[(j-1)*7+5]-dqdt[(i-1)*7+5])
       +  (v[1,j]-v[1,i])*(dqdt[(j-1)*7+1]-dqdt[(i-1)*7+1])+(v[2,j]-v[2,i])*(dqdt[(j-1)*7+2]-dqdt[(i-1)*7+2]))
  # Refine estimate of transit time with Newton's method:
  dt = -gsky/gdot
  # Add refinement to estimated time:
  tt += dt
  iter +=1
end
if iter >= 20
  println("Exceeded iterations: planet ",j," iter ",iter," dt ",dt," gsky ",gsky," gdot ",gdot)
end
# Compute time derivatives:
x = copy(x1); v = copy(v1)
# Compute dgdt with the updated time step.
dh17!(x,v,tt,m,n,jac_step,pair)
# Need to reset to compute dqdt:
x = copy(x1); v = copy(v1)
dh17!(x,v,tt,m,n,dqdt,pair)
# Compute time offset:
gsky = g!(i,j,x,v)
# Compute derivative of g with respect to time:
gdot  = ((x[1,j]-x[1,i])*(dqdt[(j-1)*7+4]-dqdt[(i-1)*7+4])+(x[2,j]-x[2,i])*(dqdt[(j-1)*7+5]-dqdt[(i-1)*7+5])
      +  (v[1,j]-v[1,i])*(dqdt[(j-1)*7+1]-dqdt[(i-1)*7+1])+(v[2,j]-v[2,i])*(dqdt[(j-1)*7+2]-dqdt[(i-1)*7+2]))
# Set dtdq to zero:
fill!(dtdq,0.0)
indj = (j-1)*7+1
indi = (i-1)*7+1
for p=1:n
  indp = (p-1)*7
  for k=1:7
    # Compute derivatives:
    #g = (x[1,j]-x[1,i])*(v[1,j]-v[1,i])+(x[2,j]-x[2,i])*(v[2,j]-v[2,i])
    dtdq[k,p] = -((jac_step[indj  ,indp+k]-jac_step[indi  ,indp+k])*(v[1,j]-v[1,i])+(jac_step[indj+1,indp+k]-jac_step[indi+1,indp+k])*(v[2,j]-v[2,i])+
                  (jac_step[indj+3,indp+k]-jac_step[indi+3,indp+k])*(x[1,j]-x[1,i])+(jac_step[indj+4,indp+k]-jac_step[indi+4,indp+k])*(x[2,j]-x[2,i]))/gdot
  end
end
# Note: this is the time elapsed *after* the beginning of the timestep:
return tt::Float64
end
