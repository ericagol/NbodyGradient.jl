
using NbodyGradient, LinearAlgebra

export State

#include("../src/ttv.jl")
#include("/Users/ericagol/Software/TRAPPIST1_Spitzer/src/NbodyGradient/src/ttv.jl")
# Specify the initial conditions for the outer solar
# system 
#n=6
n=5
xout = zeros(3,n)
# Positions at time September 5, 1994 at 0h00 in days (from Hairer, Lubich & Wanner 
# 2006, Geometric Numerical Integration, 2nd Edition, Springer, pp. 13-14):

xout .= transpose([-2.079997415328555E-04  7.127853194812450E-03 -1.352450694676177E-05;
        -3.502576700516146E+00 -4.111754741095586E+00  9.546978009906396E-02;
         9.075323061767737E+00 -3.443060862268533E+00 -3.008002403885198E-01;
         8.309900066449559E+00 -1.782348877489204E+01 -1.738826162402036E-01;
         1.147049510166812E+01 -2.790203169301273E+01  3.102324955757055E-01]) #;
        #-1.553841709421204E+01 -2.440295115792555E+01  7.105854443660053E+00])
vout = zeros(3,n)
vout .= transpose([-6.227982601533108E-06  2.641634501527718E-06  1.564697381040213E-07;
         5.647185656190083E-03 -4.540768041260330E-03 -1.077099720398784E-04;
         1.677252499111402E-03  5.205044577942047E-03 -1.577215030049337E-04;
         3.535508197097127E-03  1.479452678720917E-03 -4.019422185567764E-05;
         2.882592399188369E-03  1.211095412047072E-03 -9.118527716949448E-05]) #;
         #2.754640676017983E-03 -2.105690992946069E-03 -5.607958889969929E-04]);
# Units of velocity are AU/day

# Specify masses, including terrestrial planets in the Sun:
m = [1.00000597682,0.000954786104043,0.000285583733151,
         0.0000437273164546,0.0000517759138449] #,6.58086572e-9];

# Compute the center-of-mass:
vcm = zeros(3);
xcm = zeros(3);
for j=1:n
    vcm .+= m[j]*vout[:,j];
    xcm .+= m[j]*xout[:,j];
end
vcm ./= sum(m);
xcm ./= sum(m);
# Adjust so CoM is stationary
for j=1:n
    vout[:,j] .-= vcm[:];
    xout[:,j] .-= xcm[:];
end

struct CartesianElements{T} <: NbodyGradient.InitialConditions{T}
    x::Matrix{T}
    v::Matrix{T}
    m::Vector{T}
    nbody::Int64
end

ic = CartesianElements(xout,vout,m,5);

function NbodyGradient.State(ic::NbodyGradient.InitialConditions{T}) where T<:AbstractFloat
    n = ic.nbody
    x = copy(ic.x)
    v = copy(ic.v)
    jac_init = zeros(7*n,7*n)
    xerror = zeros(T,size(x))
    verror = zeros(T,size(v))
    jac_step = Matrix{T}(I,7*n,7*n)
    dqdt = zeros(T,7*n)
    dqdt_error = zeros(T,size(dqdt))
    jac_error = zeros(T,size(jac_step))

    rij = zeros(T,3)
    a = zeros(T,3,n)
    aij = zeros(T,3)
    x0 = zeros(T,3)
    v0 = zeros(T,3)
    input = zeros(T,8)
    delxv = zeros(T,6)
    rtmp = zeros(T,3)
    return State(x,v,[0.0],copy(ic.m),jac_step,dqdt,jac_init,xerror,verror,dqdt_error,jac_error,n,
    rij,a,aij,x0,v0,input,delxv,rtmp)
end

function compute_energy(m::Array{T,1},x::Array{T,2},v::Array{T,2},n::Int64) where {T <: Real}
  KE = 0.0
  for j=1:n
    KE += 0.5*m[j]*(v[1,j]^2+v[2,j]^2+v[3,j]^2)
  end
  PE = 0.0
  for j=1:n-1
    for k=j+1:n
       PE += -NbodyGradient.GNEWT*m[j]*m[k]/norm(x[:,j] .- x[:,k])
    end
  end
  ang_mom = zeros(3)
  for j=1:n
    ang_mom .+= m[j]*cross(x[:,j],v[:,j])
  end
  return KE,PE,ang_mom
end

# Now, integrate this forward in time:
xerror = zeros(3,n); verror = zeros(3,n); 
#h = 200.0 # 200-day time-step chosen to be <1/20 of the orbital period of Jupiter
#h = 100.0 # 100-day time-step chosen to check conservation of energy/angular momentum with time step
h = 50.0 # 50-day time-step chosen to check conservation of energy/angular momentum with time step
#h = 25.0 # 25-day time-step chosen to check conservation of energy/angular momentum with time step
#h = 12.5 # 12.5-day time-step chosen to check conservation of energy/angular momentum with time step
#h = 6.25 # 6.25-day time-step chosen to check conservation of energy/angular momentum with time step
#h = 3.125 # 3.125-day time-step chosen to check conservation of energy/angular momentum with time step
#h = 1.5625 # 1.5625-day time-step chosen to check conservation of energy/angular momentum with time step

# 50 days x 1e6 time steps ~ 137,000 yr (takes about 15 seconds to run)
nstep = 1000000; pair = zeros(Bool,n,n)

grad = false
s = State(ic)
pair = zeros(Bool,s.n,s.n)
if grad; d = Derivatives(T,s.n); end

# Set up array to save the state as a function of time:
xsave = zeros(3,n,nstep)
vsave = zeros(3,n,nstep)
# Save the potential & kinetic energy, as well as angular momentum:
PE = zeros(nstep); KE=zeros(nstep); ang_mom = zeros(3,nstep)
# Time the integration:
tstart = time()
# Carry out the integration:
for i=1:nstep
  if grad
    ah18!(s,d,h,pair)
  else
    ah18!(s,h,pair)
  end
  #ah18!(xout,vout,xerror,verror,h,m,n,pair)
  xsave[:,:,i] .= s.x
  vsave[:,:,i] .= s.v
  KE_step,PE_step,ang_mom_step=compute_energy(s.m,s.x,s.v,n)
  KE[i] = KE_step
  PE[i] = PE_step
  ang_mom[:,i] = ang_mom_step
end
s.t[1] = h*nstep
telapse = time()- tstart

