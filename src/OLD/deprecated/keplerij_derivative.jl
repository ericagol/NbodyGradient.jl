# Carries out a Kepler step for bodies i & j
function keplerij!(m::Array{Float64,1},x::Array{Float64,2},v::Array{Float64,2},i::Int64,j::Int64,h::Float64,jac_ij::Array{Float64,2})
# The state vector has: 1 time; 2-4 position; 5-7 velocity; 8 r0; 9 dr0dt; 10 beta; 11 s; 12 ds
# Initial state:
state0 = zeros(Float64,12)
# Final state (after a step):
state = zeros(Float64,12)
delx = zeros(Float64,NDIM)
delv = zeros(Float64,NDIM)
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
jac_kepler = zeros(Float64,7,7)
kepler_step!(gm, h, state0, state, jac_kepler)
for k=1:NDIM
  delx[k] = state[1+k] - state0[1+k]
  delv[k] = state[1+NDIM+k] - state0[1+NDIM+k]
end
# Compute COM coords:
mijinv =1.0/(m[i] + m[j])
xcm = zeros(Float64,NDIM)
vcm = zeros(Float64,NDIM)
mi = m[i]*mijinv # Normalize the masses
mj = m[j]*mijinv
for k=1:NDIM
  xcm[k] = mi*x[k,i] + mj*x[k,j]
  vcm[k] = mi*v[k,i] + mj*v[k,j]
end
# Compute the Jacobian:
jac_ij[ 7, 7] = 1.0  # the masses don't change with time!
jac_ij[14,14] = 1.0
for k=1:NDIM
   jac_ij[   k,   k] +=   mi
   jac_ij[   k, 3+k] += h*mi
   jac_ij[   k, 7+k] +=   mj
   jac_ij[   k,10+k] += h*mj
   jac_ij[ 3+k, 3+k] +=   mi
   jac_ij[ 3+k,10+k] +=   mj
   jac_ij[ 7+k,   k] +=   mi
   jac_ij[ 7+k, 3+k] += h*mi
   jac_ij[ 7+k, 7+k] +=   mj
   jac_ij[ 7+k,10+k] += h*mj
   jac_ij[10+k, 3+k] +=   mi
   jac_ij[10+k,10+k] +=   mj
   for l=1:NDIM
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
# Compute derivatives of \delta x_i with respect to the masses:
   jac_ij[   k, 7] += (x[k,i]+h*v[k,i]-xcm[k]-mj*state[1+k])*mijinv + GNEWT*mj*jac_kepler[  k,7]
   jac_ij[   k,14] += (x[k,j]+h*v[k,j]-xcm[k]+mi*state[1+k])*mijinv + GNEWT*mj*jac_kepler[  k,7]
# Compute derivatives of \delta v_i with respect to the masses:
   jac_ij[ 3+k, 7] += (v[k,i]-vcm[k]-mj*state[4+k])*mijinv + GNEWT*mj*jac_kepler[3+k,7]
   jac_ij[ 3+k,14] += (v[k,j]-vcm[k]+mi*state[4+k])*mijinv + GNEWT*mj*jac_kepler[3+k,7]
# Compute derivatives of \delta x_j with respect to the masses:
   jac_ij[ 7+k, 7] += (x[k,i]+h*v[k,i]-xcm[k]-mj*state[1+k])*mijinv - GNEWT*mi*jac_kepler[  k,7]
   jac_ij[ 7+k,14] += (x[k,j]+h*v[k,j]-xcm[k]+mi*state[1+k])*mijinv - GNEWT*mi*jac_kepler[  k,7]
# Compute derivatives of \delta v_j with respect to the masses:
   jac_ij[10+k, 7] += (v[k,i]-vcm[k]-mj*state[4+k])*mijinv - GNEWT*mi*jac_kepler[3+k,7]
   jac_ij[10+k,14] += (v[k,j]-vcm[k]+mi*state[4+k])*mijinv - GNEWT*mi*jac_kepler[3+k,7]
end
# Advance center of mass & individual Keplerian motions:
centerm!(m,mijinv,x,v,vcm,delx,delv,i,j,h)
return
end
