include("kepler_init.jl")

# Initializes N-body integration for a plane-parallel hierarchical system 
# (see Hamers & Portugies-Zwart 2016 (HPZ16); Beust 2003).
# We want to define a "simplex" hierarchy which can be decomposed into N-1 Keplerian binaries.
# This can be diagramed with a "mobile" diagram, pioneered by Evans (1968).  Here's an example:
#   Level             |
#    4         _______|_______
#    3   _____|____           |     
#    2  |          |      ____|_____     
#    1  |          |     |       ___|____
#       |          |     |      |        |
#       5          4     3      2        1

# Number of levels:  n_level
# Number of bodies:  n_body
#  - Problem is divided up into N-1 Kepler problems:  there is a single Kepler problem at each level.
#  - For example, in the above "mobile" diagram, 1-2 could be a binary star,
#    while 3 is an interior planets orbiting the stars, and 4/5 are a planet/moon orbiting exterior.
#  - We compute the N-body positions with each Keplerian connection (one at each level), starting
#    at the bottom and moving upwards.
#  

function init_nbody(elements,t0,n_body)
# the "_plane" is to remind us that this is currently plane-parallel, so inclination & Omega are zero
n_level = n_body-1
# Input -
# elements: masses & orbital elements for each Keplerian (in this case, each planet plus star)
# Output -
# x: NDIM x n_body array of positions  of each planet.
# v: NDIM x n_body array of velocities "   "      "
#
# Read in the orbital elements:
# elements = readdlm("elements.txt",',')
# Initialize N-body for each Keplerian:
# Get the indices:
indices = get_indices_planetary(n_body)
# Set up "A" matrix (Hamers & Portegies-Zwart 2016) which transforms from
# cartesian coordinates to Keplerian orbits (we are using opposite sign
# convention of HPZ16, for example, r_1 = R_2-R_1).
amat = zeros(Float64,n_body,n_body)
# Mass vector:
mass = vcat(elements[:,1])
# Set up array for orbital positions of each Keplerian:
rkepler = zeros(Float64,n_body,NDIM)
rdotkepler = zeros(Float64,n_body,NDIM)
# Fill in the A matrix & compute the Keplerian elements:
for i=1:n_body-1
  # Sums of masses for two components of Keplerian:
  m1 = 0.0
  m2 = 0.0
  for j=1:n_body
    if indices[i,j] == 1
      m1 += mass[j]
    end
    if indices[i,j] == -1
      m2 += mass[j]
    end
  end
  # Compute Kepler problem: r is a vector of positions of "body" 2 with respect to "body" 1; rdot is velocity vector
  # For now set inclination to Inclination = pi/2 and longitude of nodes to Omega = pi:
#  r,rdot = kepler_init(t0,m1+m2,[elements[i+1,2:5];pi/2;pi])
  r,rdot = kepler_init(t0,m1+m2,elements[i+1,2:7])
  for j=1:NDIM
    rkepler[i,j] = r[j]
    rdotkepler[i,j] = rdot[j]
  end
  # Now, fill in the A matrix
  for j=1:n_body
    if indices[i,j] == 1
      amat[i,j] = -mass[j]/m1
    end
    if indices[i,j] == -1
      amat[i,j] =  mass[j]/m2
    end
  end
end
mtot = sum(mass)
# Final row is for center-of-mass of system:
for j=1:n_body
  amat[n_body,j] = mass[j]/mtot
end
# Compute inverse of A matrix to convert from Keplerian coordinates
# to Cartesian coordinates:
ainv = inv(amat)
# Now, compute the Cartesian coordinates (eqn A6 from HPZ16):
x = zeros(Float64,NDIM,n_body)
v = zeros(Float64,NDIM,n_body)
#for i=1:n_body
#  for j=1:NDIM
#    for k=1:n_body
#      x[j,i] += ainv[i,k]*rkepler[k,j]
#      v[j,i] += ainv[i,k]*rdotkepler[k,j]
#    end
#  end
#end
x = transpose(*(ainv,rkepler))
v = transpose(*(ainv,rdotkepler))
# Return the cartesian position & velocity matrices:
#return x,v,amat,ainv
return x,v
end

# Version including derivatives:
function init_nbody(elements,t0,n_body,jac_init)

# the "_plane" is to remind us that this is currently plane-parallel, so inclination & Omega are zero
n_level = n_body-1
# Input -
# elements: masses & orbital elements for each Keplerian (in this case, each planet plus star)
# Output -
# x: NDIM x n_body array of positions  of each planet.
# v: NDIM x n_body array of velocities "   "      "
# jac_init: derivative of cartesian coordinates, (x,v,m) for each body, with respect to initial conditions
#  n_body x (period, t0, e*cos(omega), e*sin(omega), inclination, Omega, mass) for each Keplerian, with
#  the first body having orbital elements set to zero, so the first six derivatives are zero.
#jac_init = zeros(Float64,7*n_body,7*n_body)
fill!(jac_init,0.0)
# 
# Read in the orbital elements:
# elements = readdlm("elements.txt",',')
# Initialize N-body for each Keplerian:
# Get the indices:
indices = get_indices_planetary(n_body)
#println("Indices: ",indices)
# Set up "A" matrix (Hamers & Portegies-Zwart 2016) which transforms from
# cartesian coordinates to Keplerian orbits (we are using opposite sign
# convention of HPZ16, for example, r_1 = R_2-R_1).
amat = zeros(Float64,n_body,n_body)
# Derivative of i,jth element of A matrix with respect to mass of body k:
damatdm = zeros(Float64,n_body,n_body,n_body)
# Mass vector:
mass = vcat(elements[:,1])
# Set up array for orbital positions of each Keplerian:
rkepler = zeros(Float64,n_body,NDIM)
rdotkepler = zeros(Float64,n_body,NDIM)
# Set up Jacobian for transformation from n_body-1 Keplerian elements & masses
# to (x,v,m) - the last is center-of-mass, which is taken to be zero.
jac_21 = zeros(Float64,7,7)
# jac_kepler saves jac_21 for each set of bodies:
jac_kepler = zeros(Float64,n_body*6,n_body*7)
# Fill in the A matrix & compute the Keplerian elements:
for i=1:n_body-1  # i labels the row of matrix, which weights masses in current Keplerian
  # Sums of masses for two components of Keplerian:
  m1 = 0.0 ; m2 = 0.0
  for j=1:n_body
    if indices[i,j] == 1
      m1 += mass[j]
    end
    if indices[i,j] == -1
      m2 += mass[j]
    end
  end
  # Compute Kepler problem: r is a vector of positions of "body" 2 with respect to "body" 1; rdot is velocity vector:
  r,rdot = kepler_init(t0,m1+m2,elements[i+1,2:7],jac_21)
  for j=1:NDIM
    rkepler[i,j] = r[j]
    rdotkepler[i,j] = rdot[j]
  end
  # Save Keplerian Jacobian to a matrix.  First, positions/velocities vs. elements:
  for j=1:6, k=1:6
    jac_kepler[(i-1)*6+j,i*7+k] = jac_21[j,k]
  end
  # Now add in mass derivatives:
  for j=1:n_body
    # Check which bodies participate in current Keplerian:
    if indices[i,j] != 0
      for k=1:6
        jac_kepler[(i-1)*6+k,j*7] = jac_21[k,7]
      end
    end
  end
  # Now, fill in the A matrix:
  for j=1:n_body
    if indices[i,j] == 1
      amat[i,j] = -mass[j]/m1
      damatdm[i,j,j] += -1.0/m1
      for k=1:n_body
        if indices[i,k] == 1
          damatdm[i,j,k] += mass[j]/m1^2
        end
      end
    end
    if indices[i,j] == -1
      amat[i,j] =  mass[j]/m2
      damatdm[i,j,j] +=  1.0/m2
      for k=1:n_body
        if indices[i,k] == -1
          damatdm[i,j,k] -= mass[j]/m2^2
        end
      end
    end
  end
end
mtot = sum(mass)
for j=1:n_body
  amat[n_body,j] = mass[j]/mtot
  damatdm[n_body,j,j] = 1.0/mtot
  for k=1:n_body
    damatdm[n_body,j,k] -= mass[j]/mtot^2
  end
end
ainv = inv(amat)
# Propagate mass uncertainties (11/17/17 notes):
dainvdm = zeros(Float64,n_body,n_body,n_body)
#dainvdm_num = zeros(Float64,n_body,n_body,n_body)
#damatdm_num = zeros(Float64,n_body,n_body,n_body)
#dlnq = 2e-3
#elements_tmp = zeros(Float64,n_body,7)
for k=1:n_body
  dainvdm[:,:,k]=-ainv*damatdm[:,:,k]*ainv
# Compute derivatives numerically:
#  elements_tmp = copy(elements)
#  elements_tmp .= elements
#  elements_tmp[k,1] *= (1.0-dlnq)
#  x_minus,v_minus,amat_minus,ainv_minus = init_nbody(elements_tmp,t0,n_body)
#  elements_tmp[k,1] *= (1.0+dlnq)/(1.0-dlnq)
#  x_plus,v_plus,amat_plus,ainv_plus = init_nbody(elements_tmp,t0,n_body)
#  dainvdm_num[:,:,k] = (ainv_plus.-ainv_minus)/(2dlnq*elements[k,1])
#  damatdm_num[:,:,k] = (amat_plus.-amat_minus)/(2dlnq*elements[k,1])
end
#println("damatdm: ",maximum(abs.(damatdm-damatdm_num)))
#println("dainvdm: ",maximum(abs.(dainvdm-dainvdm_num)))
#for k=1:n_body
#  println("damatdm: ",k," ",abs.(damatdm[k,:,:]-damatdm_num[k,:,:]))
##  println("damatdm_num: ",k," ",abs.(damatdm_num[k,:,:]))
#end
#println("dainvdm: ",abs.(dainvdm))
#println("dainvdm_num: ",abs.(dainvdm_num))
# Now, compute the Cartesian coordinates (eqn A6 from HPZ16):
x = zeros(Float64,NDIM,n_body)
v = zeros(Float64,NDIM,n_body)
#for i=1:n_body
#  for j=1:NDIM
#    for k=1:n_body
#      x[j,i] += ainv[i,k]*rkepler[k,j]
#      v[j,i] += ainv[i,k]*rdotkepler[k,j]
#    end
#  end
#end
x = transpose(*(ainv,rkepler))
v = transpose(*(ainv,rdotkepler))
# Finally, compute the overall Jacobian.
# First, compute it for the orbital elements:
dxdm = zeros(Float64,3,n_body); dvdm = zeros(Float64,3,n_body)
for i=1:n_body
  # Propagate derivatives of Kepler coordinates with respect to
  # orbital elements to the Cartesian coordinates:
  for k=1:n_body
    for j=1:3, l=1:7*n_body
      jac_init[(i-1)*7+j,l] += ainv[i,k]*jac_kepler[(k-1)*6+j,l]
      jac_init[(i-1)*7+3+j,l] += ainv[i,k]*jac_kepler[(k-1)*6+3+j,l]
    end
  end
  # Include the mass derivatives of the A matrix:
#  dainvdm .= -ainv*damatdm[:,:,i]*ainv  # derivative with respect to the ith body
  # Multiply the derivative time Kepler positions/velocities to convert to
  # Cartesian coordinates:
  # Cartesian coordinates of all of the bodies can depend on the masses 
  # of all the others so we need to loop over indices of each body, k:
  for k=1:n_body
    dxdm = transpose(dainvdm[:,:,k]*rkepler)
    dvdm = transpose(dainvdm[:,:,k]*rdotkepler)
    jac_init[(i-1)*7+1:(i-1)*7+3,k*7] += dxdm[1:3,i]
    jac_init[(i-1)*7+4:(i-1)*7+6,k*7] += dvdm[1:3,i]
  end
  # Masses are conserved:
  jac_init[i*7,i*7] = 1.0
end

return x,v
end

function get_indices_planetary(n_body)
# This sets up a planetary-hierarchy index matrix
indices = zeros(Int64,n_body,n_body)
for i=1:n_body-1
 for j=1:i
  indices[i,j ]=-1
 end
 indices[i,i+1]= 1
 indices[n_body,i]=1
end
indices[n_body,n_body]=1
# This is an example for TRAPPIST-1
# indices = [[-1, 1, 0, 0, 0, 0, 0, 0],  # first two bodies orbit in a binary
#            [-1,-1, 1, 0, 0, 0, 0, 0],  # next planet orbits about these
#            [-1,-1,-1, 1, 0, 0, 0, 0],  # etc...
#            [-1,-1,-1,-1, 1, 0, 0, 0],
#            [-1,-1,-1,-1,-1, 1, 0, 0],
#            [-1,-1,-1,-1,-1,-1, 1, 0],
#            [-1,-1,-1,-1,-1,-1,-1, 1],
#            [ 1, 1, 1, 1, 1, 1, 1, 1]  # center of mass of the system
return indices
end
