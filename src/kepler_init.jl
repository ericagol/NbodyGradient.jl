include("kepler.jl")

function kepler_init(time::Float64,mass::Float64,elements::Array{Float64,1})
# Takes orbital elements of a single Keplerian; returns positions & velocities.
# This is 3D), so 6 orbital elements specified, the code returns 3D.  For
# Inclination = pi/2, motion is in X-Z plane; sky plane is X-Y.
# Elements are given by: period, t0, e*cos(omega), e*sin(omega), Inclination, Omega
period = elements[1]
# Compute the semi-major axis in AU (or other units specified by GNEWT):
semi = (GNEWT*mass*period^2/4/pi^2)^third
# Convert to eccentricity & longitude of periastron:
ecc=sqrt(elements[3]^2+elements[4]^2)
omega = atan2(elements[4],elements[3])
# The true anomaly at the time of transit:
f1 = 1.5*pi-omega
# Compute the time of periastron passage:
sqrt1mecc2 = sqrt(1.0-ecc^2)
tp=(elements[2]+period*sqrt1mecc2/2.0/pi*(ecc*sin(f1)/(1.0+ecc*cos(f1))
    -2.0/sqrt1mecc2*atan2(sqrt1mecc2*tan(0.5*f1),1.0+ecc)))
# Compute the mean anomaly
n = 2pi/period
m=n*(time-tp)
# Kepler solver:
f=kepler(m,ecc)
ecosfp1 = 1.0+ecc*cos(f)
fdot = n*ecosfp1^2/sqrt1mecc2^3
# Compute the radial distance:
r=semi*(1.0-ecc^2)/ecosfp1
rdot = semi*n/sqrt1mecc2*ecc*sin(f)
# For now assume plane-parallel:
#inc = pi/2
#capomega = pi
inc = elements[5]
capomega = elements[6]
# Now, compute the positions
x = zeros(Float64,3)
v = zeros(Float64,3)
if abs(capomega-pi) > 1e-15
  coscapomega = cos(capomega) ; sincapomega = sin(capomega)
else
  coscapomega = -1.0 ; sincapomega = 0.0
end
cosomegapf = cos(omega+f) ; sinomegapf = sin(omega+f) 
if abs(inc-pi/2) > 1e-15
  cosinc = cos(inc) ; sininc = sin(inc)
else
  cosinc = 0.0 ; sininc = 1.0
end
x[1]=r*(coscapomega*cosomegapf-sincapomega*sinomegapf*cosinc)
x[2]=r*(sincapomega*cosomegapf+coscapomega*sinomegapf*cosinc)
x[3]=r*sinomegapf*sininc
rdotonr = rdot/r
# Compute the velocities:
rfdot = r*fdot
v[1]=x[1]*rdotonr+rfdot*(-coscapomega*sinomegapf-sincapomega*cosomegapf*cosinc)
v[2]=x[2]*rdotonr+rfdot*(-sincapomega*sinomegapf+coscapomega*cosomegapf*cosinc)
v[3]=x[3]*rdotonr+rfdot*cosomegapf*sininc
return x,v
end

function kepler_init(time::Float64,mass::Float64,elements::Array{Float64,1},jac_init::Array{Float64,2})
# Takes orbital elements of a single Keplerian; returns positions & velocities.
# This is 3D), so 6 orbital elements specified, the code returns 3D.  For
# Inclination = pi/2, motion is in X-Z plane; sky plane is X-Y.
# Elements are given by: period, t0, e*cos(omega), e*sin(omega), Inclination, Omega
# Returns the position & velocity of Keplerian.
# jac_init is the derivative of (x,v,m) with respect to (elements,m).
period = elements[1]
n = 2pi/period
t0 = elements[2]
# Compute the semi-major axis in AU (or other units specified by GNEWT):
semi = (GNEWT*mass*period^2/4/pi^2)^third
dsemidp = 2third*semi/period
dsemidm = third*semi/mass
# Convert to eccentricity & longitude of periastron:
ecosomega = elements[3]
esinomega = elements[4]
ecc=sqrt(esinomega^2+ecosomega^2)
deccdecos = ecosomega/ecc
deccdesin = esinomega/ecc
#omega = atan2(esinomega,ecosomega)
# The true anomaly at the time of transit:
#f1 = 1.5*pi-omega
# Compute the time of periastron passage:
sqrt1mecc2 = sqrt(1.0-ecc^2)
#tp=(t0+period*sqrt1mecc2/2.0/pi*(ecc*sin(f1)/(1.0+ecc*cos(f1))
#    -2.0/sqrt1mecc2*atan2(sqrt1mecc2*tan(0.5*f1),1.0+ecc)))
den1 = esinomega-ecosomega-ecc
tp = (t0 - sqrt1mecc2/n*ecosomega/(1.0-esinomega)-2/n*
     atan2(sqrt(1.0-ecc)*(esinomega+ecosomega+ecc),sqrt(1.0+ecc)*den1))
dtpdp = (tp-t0)/period
fac = sqrt((1.0-ecc)/(1.0+ecc))
den2 = 1.0/den1^2
theta = fac*(esinomega+ecosomega+ecc)/den1
#println("theta: ",theta," tan(atan2()): ",tan(atan2(sqrt(1.0-ecc)*(esinomega+ecosomega+ecc),sqrt(1.0+ecc)*den1)))
dthetadecc = ((ecc+ecosomega)^2+2*(1.0-ecc^2)*esinomega-esinomega^2)/(sqrt1mecc2*(1.0+ecc))*den2
dthetadecos = 2fac*esinomega*den2
dthetadesin = -2fac*(ecosomega+ecc)*den2
dtpdecc = ecc/sqrt1mecc2/n*ecosomega/(1.0-esinomega)-2/n/(1.0+theta^2)*dthetadecc
dtpdecos = dtpdecc*deccdecos -sqrt1mecc2/n/(1.0-esinomega)-2/n/(1.0+theta^2)*dthetadecos
dtpdesin = dtpdecc*deccdesin -sqrt1mecc2/n*ecosomega/(1.0-esinomega)^2-2/n/(1.0+theta^2)*dthetadesin
dtpdt0 = 1.0
# Compute the mean anomaly
m=n*(time-tp)
dmdp = -m/period
dmdtp = -n
# Kepler solver: instead of true anomly, return eccentric anomaly:
ekep=ekepler(m,ecc)
cosekep = cos(ekep); sinekep = sin(ekep)
# Compute the radial distance:
r=semi*(1.0-ecc*cosekep)
drdekep = semi*ecc*sinekep
drdecc = -semi*cosekep
denom = semi/r
dekepdecos = sinekep*denom*deccdecos
dekepdesin = sinekep*denom*deccdesin
dekepdm   = denom
inc = elements[5]
capomega = elements[6]
# Now, compute the positions
coscapomega = cos(capomega) ; sincapomega = sin(capomega)
cosomega = ecosomega/ecc; sinomega = esinomega/ecc
cosinc = cos(inc) ; sininc = sin(inc)
# Define rotation matrices (M&D 2.119-2.120):
P1 = [cosomega -sinomega 0.0; sinomega cosomega 0.0; 0.0 0.0 1.0]
P2 = [1.0 0.0 0.0; 0.0 cosinc -sininc; 0.0 sininc cosinc]
P3 = [coscapomega -sincapomega 0.0; sincapomega coscapomega 0.0; 0.0 0.0 1.0]
P321 = P3*P2*P1
# Express x & v in terms of eccentric anomaly (2.121, 2.41, 2.68):
xplane = semi*[cosekep-ecc;  sqrt1mecc2*sinekep;  0.0] # position vector in orbital plane
vplane = [-sinekep; sqrt1mecc2*cosekep; 0.0]
x = P321*xplane
# Now derivatives:
dxda = x/semi
dxdekep = P321*semi*vplane
P32 = P3*P2
dxdecc  = -x/ecc + P321*semi*[-1.0; -ecc/sqrt1mecc2*sinekep; 0.0]
# These may need to be rewritten to flag ecc = 0.0 case:
dxdecos = dxdecc*deccdecos + P32/ecc*xplane
dxdesin = dxdecc*deccdesin + P32/ecc*[-xplane[2]; xplane[1]; 0.0]
dxdinc  = P3*[0.0 0.0 0.0; 0.0 -sininc -cosinc; 0.0 cosinc -sininc]*P1*xplane
dxdcom  = [-sincapomega -coscapomega 0.0; coscapomega -sincapomega 0.0; 0.0 0.0 0.0]*P2*P1*xplane
# Compute the velocities:
v = P321*n*semi*denom*vplane
#decc = 1e-5
#dvp = P321*ecc/(ecc+decc)*n*semi/(1.0-(ecc+decc)*cosekep)*[-sinekep; sqrt(1.0-(ecc+decc)^2)*cosekep; 0.0]
#dvm = P321*ecc/(ecc-decc)*n*semi/(1.0-(ecc-decc)*cosekep)*[-sinekep; sqrt(1.0-(ecc-decc)^2)*cosekep; 0.0]
dvda    = v/semi
dvdp    = -v/period
dvdekep = -v*ecc*sinekep*denom+P321*n*semi*denom*[-cosekep; -sqrt1mecc2*sinekep; 0.0]
dvdecc  = -v/ecc + v*cosekep*denom + P321*n*semi*denom*[0.0; -ecc/sqrt1mecc2*cosekep; 0.0]
#dvdecc_num = .5*(dvp-dvm)/decc
#println("dvdecc: ",dvdecc," dvdecc_num: ",dvdecc_num)
#dvdecc = dvdecc_num
dvdecos = dvdecc*deccdecos + P32*n*semi*denom/ecc*vplane
dvdesin = dvdecc*deccdesin + P32*n*semi*denom/ecc*[-vplane[2]; vplane[1]; 0.0]
dvdinc  = P3*[0.0 0.0 0.0; 0.0 -sininc -cosinc; 0.0 cosinc -sininc]*P1*n*semi*denom*vplane
dvdcom  = [-sincapomega -coscapomega 0.0; coscapomega -sincapomega 0.0; 0.0 0.0 0.0]*P2*P1*n*semi*denom*vplane
# Now, take derivatives (11/15/2017 notes):
# Elements are given by: period, t0, e*cos(omega), e*sin(omega), Inclination, Omega
fill!(jac_init,0.0)
jac_init[1:3,1] = dxda*dsemidp + dxdekep*dekepdm*(dmdp+dmdtp*dtpdp)
jac_init[1:3,2] = dxdekep*dekepdm*dmdtp*dtpdt0
jac_init[1:3,3] = dxdecos + dxdekep*(dekepdm*dmdtp*dtpdecos + dekepdecos)
jac_init[1:3,4] = dxdesin + dxdekep*(dekepdm*dmdtp*dtpdesin + dekepdesin)
jac_init[1:3,5] = dxdinc
jac_init[1:3,6] = dxdcom
jac_init[1:3,7] = dxda*dsemidm
jac_init[4:6,1] = dvdp + dvda*dsemidp +  dvdekep*dekepdm*(dmdp+dmdtp*dtpdp)
jac_init[4:6,2] = dvdekep*dekepdm*dmdtp*dtpdt0
jac_init[4:6,3] = dvdecos + dvdekep*(dekepdm*dmdtp*dtpdecos + dekepdecos) # <-- These two lines still have a bug!
jac_init[4:6,4] = dvdesin + dvdekep*(dekepdm*dmdtp*dtpdesin + dekepdesin) # <-----'
jac_init[4:6,5] = dvdinc
jac_init[4:6,6] = dvdcom
jac_init[4:6,7] = dvda*dsemidm
jac_init[7,7] = 1.0
#return x,v,tp,dtpdecos,dtpdesin
#return x,v,ekep,dekepdm*dmdtp*dtpdecos + dekepdecos,dekepdm*dmdtp*dtpdesin + dekepdesin
return x,v
end
