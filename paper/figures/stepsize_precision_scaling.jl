# This computes the scaling of the TTVs with the mass-ratios
# of the planets to the star, "epsilon", as well as the stepsize, h.
# 

include("../../src/ttv.jl")
include("/Users/ericagol/Computer/Julia/regress.jl")

using PyPlot

# This routine takes derivative of transit times with respect
# to the initial orbital elements.
n = 3
n_body = n
t0 = 7257.93115525
h  = 0.07
tmax = 100.0

# Read in initial conditions:
elements = readdlm("elements.txt",',')

ntt = zeros(Int64,n)

# Make an array, tt,  to hold transit times:
# First, though, make sure it is large enough:
for i=2:n
  ntt[i] = ceil(Int64,tmax/elements[i,2])+3
end
dtdq0 = zeros(n,maximum(ntt),7,n)
tt  = zeros(n,maximum(ntt))
tt1 = zeros(n,maximum(ntt))

# Make an array to save the variables as a function of step
# size and mass:
nh = 5
nmass = 5
tt_save = zeros(BigFloat,nh,nmass,n,maximum(ntt))
# Save a counter for the actual number of transit times of each planet:
count = zeros(Int64,n)
count1 = zeros(Int64,n)
# Call the ttv function:
rstar = 1e12

mass_fac = big.([1.0,0.5,0.25,0.125,0.0625])
h_list = [h,h/2.,h/4.,h/8.,h/32.]
t0big = big(t0)
tmaxbig = big(tmax)
tt1big = big.(tt1)
rstarbig = big(rstar)
for i=1:nh, j=1:nmass
  tic()
# Create BigFloat versions of the input variables:
  elements_big = convert(Array{BigFloat,2},elements)
  elements_big[2:n,1] *= mass_fac[j]
  hbig = big(h_list[i])
  # Call the routine 
  dqbig = ttv_elements!(n,t0big,hbig,tmaxbig,elements_big,tt1big,count,big(0.0),0,0,rstarbig)
  # Save this:
#  tt_save[i,j,:,:]=convert(Array{Float64,2},tt1big)
  tt_save[i,j,:,:]=tt1big
  println("finished h: ",convert(Float64,h_list[i])," mass_fac: ",convert(Float64,mass_fac[j])," elapsed: ",toq())
end

# Make a plot of transit time errors versus stepsize:
ntrans = sum(count)
clf()
tab = 0
hlabel = ["h-h/32","h/2-h/32","h/4-h/32","h/8-h/32"]
ch = ["black","red","green","blue","orange"]

# Make a plot of timing errors versus stepsize and epsilon:
ntrans = sum(count)
clf()
sigt = zeros(n-1,nh-1,nmass)
tab = 0
hlabel = ["h-h/32","h/2-h/32","h/4-h/32","h/8-h/32","h/16-big(h/32)"]
ch = ["black","red","green","blue","orange"]
for i=2:n
  for k=1:nmass
#    fn = zeros(Float64,2,count[i])
    fn = zeros(BigFloat,2,count[i])
#    sig = ones(count[i])
    sig = ones(BigFloat,count[i])
    tti16 = tt_save[nh,k,i,1:count[i]]
    for j=1:count[i]
#      fn[1,j] = 1.0
      fn[1,j] = big(1.0)
      fn[2,j] = round(Int64,(tti16[j]-elements[i,3])/elements[i,2])
    end
    for j=1:nh-1
      tti1 = tt_save[j,k,i,1:count[i]]
      coeff,cov = regress(fn,tti1-tti16,sig)
      diff = tti1-tti16-coeff[1]-coeff[2]*fn[2,:]
      sigt[i-1,j,k]=std(diff)
      if i == n
        plot(tti1,diff/h_list[j]^4/mass_fac[k],linestyle="dashed",c=ch[j])
      else
        plot(tti1,diff/h_list[j]^4/mass_fac[k],label=hlabel[j],c=ch[j])
      end
    end
  end
end
legend(loc="lower left")

read(STDIN,Char)
clf()


# Make a plot of some TTVs vs mass scaling:
# Try p=1:
p = 1
for j=1:nh-1
  loglog(mass_fac[1:nmass],sigt[1,j,:]*24.*3600./(h_list[j]/h[1]).^4,".",markersize=15,label="Inner planet")
  loglog(mass_fac[1:nmass],sigt[1,j,1]*24.*3600./(h_list[j]/h[1]).^4*mass_fac[1:nmass].^p,label=L"$\propto m^p$")
  loglog(mass_fac[1:nmass],sigt[2,j,:]*24.*3600./(h_list[j]/h[1]).^4,".",markersize=15,label="Outer planet")
  loglog(mass_fac[1:nmass],sigt[2,j,1]*24.*3600./(h_list[j]/h[1]).^4*mass_fac[1:nmass].^p,label=L"$\propto m^p$")
end
legend(loc = "upper left")
ylabel("RMS timing error [sec]")
xlabel("Mass ratio [day]")

PyPlot.savefig("timing_error_vs_mass_big.pdf",bbox_inches="tight")
