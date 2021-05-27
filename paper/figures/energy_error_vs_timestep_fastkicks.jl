


using PyPlot, Statistics, NbodyGradient, LinearAlgebra

export State

GNEWT = NbodyGradient.GNEWT

function compute_energy(m::Array{T,1},x::Array{T,2},v::Array{T,2},n::Int64) where {T <: Real}
  KE = 0.0
  for j=1:n
    KE += 0.5*m[j]*(v[1,j]^2+v[2,j]^2+v[3,j]^2)
  end
  PE = 0.0
  for j=1:n-1
    for k=j+1:n
       PE += -GNEWT*m[j]*m[k]/norm(x[:,j] .- x[:,k])
    end
  end
  ang_mom = zeros(3)
  for j=1:n
    ang_mom .+= m[j]*cross(x[:,j],v[:,j])
  end
  return KE,PE,ang_mom
end

n=5
xout = zeros(3,n)
# Positions at time September 5, 1994 at 0h00 in days (from Hairer, Lubich & Wanner
# 2006, Geometric Numerical Integration, 2nd Edition, Springer, pp. 13-14):

xout .= transpose([0.0 0.0 0.0;
  -3.5023653  -3.8169847  -1.5507963;
  9.0755314  -3.0458353  -1.6483708;
  8.3101420  -16.2901086  -7.2521278;
  11.4707666  -25.7294829  -10.8169456])

vout = zeros(3,n)
vout .= transpose([0.0 0.0 0.0;
  0.00565429  -0.00412490  -0.00190589;
  .00168318  .00483525  .00192462;
  .00354178  .00137102  .00055029;
  .00288930  .00114527  .00039677])

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

#= 
struct CartesianElements{T} <: NbodyGradient.InitialConditions{T}
    x::Matrix{T}
    v::Matrix{T}
    m::Vector{T}
    nbody::Int64
end


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
    pair = zeros(Bool,n,n)

    rij = zeros(T,3)
    a = zeros(T,3,n)
    aij = zeros(T,3)
    x0 = zeros(T,3)
    v0 = zeros(T,3)
    input = zeros(T,8)
    delxv = zeros(T,6)
    rtmp = zeros(T,3)
    return State(x,v,[0.0],copy(ic.m),jac_step,dqdt,jac_init,xerror,verror,dqdt_error,jac_error,n,
    pair,rij,a,aij,x0,v0,input,delxv,rtmp)
end
=#

ic = CartesianIC(xout,vout,m,n,0.0);

# Now, integrate this forward in time:
xerror = zeros(3,n); verror = zeros(3,n); 
hgrid = [1.5625,3.125,6.25,12.5,25.0,50.0,100.0,200.0]

power = 20
nstep = 2^power
grad = false
ngrid = length(hgrid)

# 4e6 time steps ~ 2 Myr (takes about 9 minutes to run)
# I've reducd that to ~400,000 to take ~1 minute.

# Set up array to save the state as a function of time:
# Save the potential & kinetic energy, as well as angular momentum:
egrid = zeros(nstep); ang_mom = zeros(3,nstep)
sig_energy = zeros(ngrid); sig_ang_mom = zeros(3,ngrid)
mean_energy = zeros(ngrid); mean_ang_mom = zeros(3,ngrid)
# Time the integration:
for j=1:length(hgrid)
  h = hgrid[j]
  s = State(ic)
  # Specify fast kicks between all bodies:
  s.pair .= ones(Bool,s.n,s.n)
  if grad; d = Derivatives(T,s.n); end

  tstart = time()
  # Carry out the integration:
  for i=1:nstep
       if grad
      ahl21!(s,d,h)
    else
      ahl21!(s,h)
    end
    KE_step,PE_step,ang_mom_step=compute_energy(s.m,s.x,s.v,n)
    egrid[i] = KE_step+PE_step
    ang_mom[:,i] = ang_mom_step
  end
  mean_energy[j] = mean(egrid)
  sig_energy[j] = std(egrid)
  mean_ang_mom[1,j] = mean(ang_mom[1,:])
  mean_ang_mom[2,j] = mean(ang_mom[2,:])
  mean_ang_mom[3,j] = mean(ang_mom[3,:])
  sig_ang_mom[1,j] = std(ang_mom[1,:])
  sig_ang_mom[2,j] = std(ang_mom[2,:])
  sig_ang_mom[3,j] = std(ang_mom[3,:])
  telapse = time()- tstart
  println(j," ",telapse," ",sig_energy[j]," ",sig_ang_mom[:,j])
end


# These integrations have already been carried out:  so,
# just plotting the results:

clf()
tstep = hgrid
loglog(tstep,sig_energy,"o",label="Measured scatter in energy")
loglog(tstep,1.403145134013028e-16 .* (tstep ./ tstep[8]).^4,label="Quartic scaling with time step")
xlabel("Time step [days]")
ylabel(L"Energy error [$M_\odot$ AU$^2$ day$^{-2}$]")
loglog([1.5625,200],abs(mean_energy[end])*2.2e-16*[1,1],linestyle=":",label="Double precision floor")
legend()

tight_layout()
savefig("Energy_error_vs_timestep_fastkicks.pdf",bbox_inches="tight")
read(stdin,Char)

clf()

loglog(tstep,sig_ang_mom[1,:],label=L"$L_x$")
loglog(tstep,sig_ang_mom[2,:],label=L"$L_y$")
loglog(tstep,sig_ang_mom[3,:],label=L"$L_z$")
loglog(tstep,mean_ang_mom[1,end]*2.2e-16*ones(8),linestyle=":",color="blue")
loglog(tstep,mean_ang_mom[2,end]*2.2e-16*ones(8),linestyle=":",color="orange")
loglog(tstep,mean_ang_mom[3,end]*2.2e-16*ones(8),linestyle=":",color="green")
loglog(tstep,1e-21 .* tstep, color="k",linestyle="--")
xlabel("Time step [days]")
ylabel(L"Angular momentum error [$M_\odot$ AU$^2$ day$^{-1}$]")
legend()

tight_layout()
savefig("Angular_momentum_error_vs_timestep_fastkicks.pdf",bbox_inches="tight")
