


using PyPlot, Statistics

include("/Users/ericagol/Software/TRAPPIST1_Spitzer/src/NbodyGradient/src/ttv.jl")

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

# 4e6 time steps ~ 2 Myr (takes about 9 minutes to run)
# I've reducd that to ~400,000 to take ~1 minute.
nstep = 1000000; pair = zeros(Bool,n,n)

# Set up array to save the state as a function of time:
xsave = zeros(3,n,nstep)
vsave = zeros(3,n,nstep)
# Save the potential & kinetic energy, as well as angular momentum:
PE = zeros(nstep); KE=zeros(nstep); ang_mom = zeros(3,nstep)
# Time the integration:
tstart = time()
# Carry out the integration:
for i=1:nstep
  ah18!(xout,vout,xerror,verror,h,m,n,pair)
  xsave[:,:,i] .= xout
  vsave[:,:,i] .= vout
  KE_step,PE_step,ang_mom_step=compute_energy(m,xout,vout,n)
  KE[i] = KE_step
  PE[i] = PE_step
  ang_mom[:,i] = ang_mom_step
end
telapse = time()- tstart


# These integrations have already been carried out:  so,
# just plotting the results:

tstep = [1.5625,3.125,6.25,12.5,25.0,50.0,100.0,200.0]
loglog(tstep,[1.4006621658556116e-23,1.7001179303989925e-23,1.3001711407790665e-22,1.998439479375678e-21,3.2843598552483375e-20,
        5.2089934752740415e-19,8.409447080119717e-18,1.403145134013028e-16],"o",label="Measured scatter in energy")
loglog(tstep,1.403145134013028e-16 .* (tstep ./ tstep[8]).^4,label="Quartic scaling with time step")
xlabel("Time step [days]")
ylabel(L"Energy error [$M_\odot$ AU$^2$ day$^{-2}$]")
loglog([1.5625,200],abs(mean(PE .+ KE))*2.2e-16*[1,1],linestyle=":",label="Double precision floor")
legend()

tight_layout()
savefig("Energy_error_vs_timestep.pdf",bbox_inches="tight")

clf()

ang_mom_error = [
    1.2142752536899133e-21 8.967806905848364e-23 3.518983383652709e-20;
    1.2636522904660814e-21 3.937722858230106e-22 2.8271651344049584e-20;
    1.0283851649218377e-21 4.431264216227097e-22 3.439586397896587e-20;
    5.826084063314492e-22 5.605861897002911e-22 2.7015478453960177e-20;
    2.063306397483572e-21 7.644568975834805e-22 3.383173853389063e-20; 
2.2656944739009946e-21 7.159672839723679e-22 6.878756018442181e-20;
2.058909762658079e-21 1.5933370189240744e-21 8.673068698213327e-20;
5.4980605689569064e-21 8.590090554403878e-21 1.677635913327141e-19]
loglog(tstep,ang_mom_error[:,1],label=L"$L_x$")
loglog(tstep,ang_mom_error[:,2],label=L"$L_y$")
loglog(tstep,ang_mom_error[:,3],label=L"$L_z$")
loglog(tstep,mean(ang_mom[1,:])*2.2e-16*ones(8),linestyle=":",color="blue")
loglog(tstep,mean(ang_mom[2,:])*2.2e-16*ones(8),linestyle=":",color="orange")
loglog(tstep,mean(ang_mom[3,:])*2.2e-16*ones(8),linestyle=":",color="green")
loglog(tstep,1e-21 .* sqrt.(tstep), color="k",linestyle="--")
xlabel("Time step [days]")
ylabel(L"Angular momentum error [$M_\odot$ AU$^2$ day$^{-1}$]")
legend()

tight_layout()
savefig("Angular_momentum_error_vs_timestep.pdf",bbox_inches="tight")
