    


# We are trying to identify the source of the kink
# after ~900 days in the time integration.

include("../../src/ttv.jl")
include("/Users/ericagol/Computer/Julia/regress.jl")

using PyPlot

# This routine takes derivative of transit times with respect
# to the initial orbital elements.
n = 3
t0 = 7257.93115525
h  = 0.07
tmax = 2000.0

# Read in initial conditions:
elements = readdlm("elements.txt",',')

ntt = zeros(Int64,n)

# Make an array, tt,  to hold transit times:
# First, though, make sure it is large enough:
for i=2:n
  ntt[i] = ceil(Int64,tmax/elements[i,2])+3
end
tt  = zeros(n,maximum(ntt))
tt1 = zeros(n,maximum(ntt))
tt_save = zeros(5,n,maximum(ntt))
tt_save_big = zeros(5,n,maximum(ntt))
# Save a counter for the actual number of transit times of each planet:
count = zeros(Int64,n)
count1 = zeros(Int64,n)
# Call the ttv function:
rstar = 1e12
dq = ttv_elements!(n,t0,h,tmax,elements,tt1,count1,0.0,0,0,rstar)
tt_save[1,:,:]=tt1


# Create BigFloat versions of the variables:
elements_big = convert(Array{BigFloat,2},elements)
hbig = big(h)
t0big = big(t0)
tmaxbig = big(tmax)
tt1big = big.(tt1)
rstarbig = big(rstar)
dqbig = ttv_elements!(n,t0big,hbig,tmaxbig,elements_big,tt1big,count,big(0.0),0,0,rstarbig)
tt_save_big[1,:,:]=convert(Array{Float64,2},tt1big)

# Now, compute derivatives (with respect to initial cartesian positions/masses):
dq = ttv_elements!(n,t0,h/2.,tmax,elements,tt1,count,0.0,0,0,rstar)
tt_save[2,:,:]=tt1
dqbig = ttv_elements!(n,t0big,hbig,tmaxbig/2,elements_big,tt1big,count,big(0.0),0,0,rstarbig)
tt_save_big[2,:,:]=convert(Array{Float64,2},tt1big)
dq = ttv_elements!(n,t0,h/4.,tmax,elements,tt1,count,0.0,0,0,rstar)
tt_save[3,:,:]=tt1
dqbig = ttv_elements!(n,t0big,hbig,tmaxbig/4,elements_big,tt1big,count,big(0.0),0,0,rstarbig)
tt_save_big[3,:,:]=convert(Array{Float64,2},tt1big)
dq = ttv_elements!(n,t0,h/8.,tmax,elements,tt1,count,0.0,0,0,rstar)
tt_save[4,:,:]=tt1
dqbig = ttv_elements!(n,t0big,hbig/8,tmaxbig,elements_big,tt1big,count,big(0.0),0,0,rstarbig)
tt_save_big[4,:,:]=convert(Array{Float64,2},tt1big)

dq = ttv_elements!(n,t0,h/16.,tmax,elements,tt1,count,0.0,0,0,rstar)
tt_save[5,:,:]=tt1
# Compute the h/16 case in BigFloat precision:
dqbig = ttv_elements!(n,t0big,hbig/16,tmaxbig,elements_big,tt1big,count,big(0.0),0,0,rstarbig)
tt_save_big[5,:,:]=convert(Array{Float64,2},tt1big)


# Make a plot of transit time errors versus stepsize:
ntrans = sum(count)
clf()
sigt = zeros(n-1,5)
tab = 0
#h_list = [h,h/2.,h/4.,h/8.]
h_list = [h,h/2.,h/4.,h/8,h/16]
hlabel = ["h-big(h)","h/2-big(h/2)","h/4-big(h/4)","h/8-big(h/8)","h/16-big(h/16)"]
ch = ["black","red","green","blue"]
for i=2:n
  for j=1:5
    tti1 = tt_save[j,i,1:count[i]]
    tti16 = tt_save_big[j,i,1:count[i]]
    diff = tti1-tti16
    sigt[i-1,j]=std(diff)
    if i == n
      plot(tti1,diff,linestyle="dashed",c=ch[j])
    else
      plot(tti1,diff,label=hlabel[j],c=ch[j])
    end
  end
end
xlabel("Transit time")
ylabel("Difference in transit time")
legend(loc="lower left")

#read(STDIN,Char)
#
## Make a plot of timing errors versus stepsize:
#ntrans = sum(count)
#clf()
#sigt = zeros(n-1,4)
#tab = 0
##h_list = [h,h/2.,h/4.,h/8.]
#h_list = [h,h/2.,h/4.,h/8]
#hlabel = ["h-h/16","h/2-h/16","h/4-h/16","h/8-h/16"]
#ch = ["black","red","green","blue"]
#for i=2:n
#  fn = zeros(Float64,2,count[i])
#  sig = ones(count[i])
#  tti16 = tt_save[5,i,1:count[i]]
#  for j=1:count[i]
#    fn[1,j] = 1.0
#    fn[2,j] = round(Int64,(tti16[j]-elements[i,3])/elements[i,2])
#  end
#  for j=1:4
#    tti1 = tt_save[j,i,1:count[i]]
#    coeff,cov = regress(fn,tti1-tti16,sig)
#    diff = tti1-tti16-coeff[1]-coeff[2]*fn[2,:]
#    sigt[i-1,j]=std(diff)
#    if i == n
#      plot(tti1,diff/h_list[j]^4,linestyle="dashed",c=ch[j])
#    else
#      plot(tti1,diff/h_list[j]^4,label=hlabel[j],c=ch[j])
#    end
#  end
#end
#legend(loc="lower left")
#
#read(STDIN,Char)
#clf()
#
#
## Make a plot of some TTVs:
#loglog(h_list,sigt[1,:]*24.*3600.,".",markersize=15,label="Inner planet")
#loglog(h_list,sigt[1,1]*24.*3600.*(h_list/h[1]).^4,label=L"$\propto h^4$")
#loglog(h_list,sigt[2,:]*24.*3600.,".",markersize=15,label="Outer planet")
#loglog(h_list,sigt[2,1]*24.*3600.*(h_list/h[1]).^4,label=L"$\propto h^4$")
#legend(loc = "upper left")
#ylabel("RMS timing error [sec]")
#xlabel("Step size [day]")
#
#PyPlot.savefig("timing_error_vs_h.pdf",bbox_inches="tight")
