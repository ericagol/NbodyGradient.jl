
#include("../../src/ttv.jl")
#include("/Users/ericagol/Computer/Julia/regress.jl")

using NbodyGradient, PyPlot, DelimitedFiles
include("linreg.jl")

# This routine takes derivative of transit times with respect
# to the initial orbital elements.
#n = 8
n = 3
n_body = n
#t0 = 7257.93115525
#t0 = 7260.0
# Try a larger value of t0:
#t0 = 15000.0
# For some reason changing t0 makes the scalings worse:
t0 =  -200.0
# Wow, t0 = 3000.0 makes things *really* bad! Between h0/2 & h0/4, we lost the first transit of the inner planet!
# I'm looking at how kink in TTVs scales with t0 - it seems fairly stable.
# t0 =  4000.0  
# t0 =  3000.0  
#t0 =  randn()
#h  = 0.12
h  = 0.07
#tmax = 600.0
#tmax = 800.0
#tmax = 600.0
#tmax = 800.0
#tmax = 2000.0
tmax = 400.0

# Read in initial conditions:
#elements = readdlm("elements_88.txt",',')[1:n,:]
elements = readdlm("elements.txt",',')[1:n,:]
# Make masses of planets bigger
#elements[2,1] *= 10.0
#elements[3,1] *= 10.0

ntt = zeros(Int64,n)

# Make an array, tt,  to hold transit times:
# First, though, make sure it is large enough:
for i=2:n
  ntt[i] = ceil(Int64,tmax/elements[i,2])+3
end
tt  = zeros(n,maximum(ntt))
tt1 = zeros(n,maximum(ntt))
nsub = 7 # How many times to divide the timestep (/2^(nsub-1)).
tt_save = zeros(nsub,n,maximum(ntt))
# Save a counter for the actual number of transit times of each planet:
count = zeros(Int64,n)
count1 = zeros(Int64,n)
# Call the ttv function:
rstar = 1e12
dq = ttv_elements!(n,t0,h,tmax,elements,tt1,count1,0.0,0,0,rstar;fout="test_output_h.txt",iout=10)
tt_save[1,:,:]=tt1


# Create BigFloat versions of the variables:
elements_big = convert(Array{BigFloat,2},elements)
hbig = big(h)
t0big = big(t0)
tmaxbig = big(tmax)
tt1big = big.(tt1)
rstarbig = big(rstar)

# Now, compute derivatives (with respect to initial cartesian positions/masses):
dq = ttv_elements!(n,t0,h/2.,tmax,elements,tt1,count,0.0,0,0,rstar;fout="test_output_h2.txt",iout=20)
tt_save[2,:,:]=tt1
dq = ttv_elements!(n,t0,h/4.,tmax,elements,tt1,count,0.0,0,0,rstar;fout="test_output_h4.txt",iout=40)
tt_save[3,:,:]=tt1
dq = ttv_elements!(n,t0,h/8.,tmax,elements,tt1,count,0.0,0,0,rstar;fout="test_output_h8.txt",iout=80)
tt_save[4,:,:]=tt1
#dqbig = ttv_elements!(n,t0big,hbig/8,tmaxbig,elements_big,tt1big,count,big(0.0),0,0,rstarbig)
#tt_save[4,:,:]=convert(Array{Float64,2},tt1big)

#dq = ttv_elements!(n,t0,h/16.,tmax,elements,tt1,count,0.0,0,0,rstar)
dq = ttv_elements!(n,t0,h/16.,tmax,elements,tt1,count,0.0,0,0,rstar;fout="test_output_h16.txt",iout=160)
tt_save[5,:,:]=tt1
dq = ttv_elements!(n,t0,h/32.,tmax,elements,tt1,count,0.0,0,0,rstar;fout="test_output_h32.txt",iout=320)
tt_save[6,:,:]=tt1
dq = ttv_elements!(n,t0,h/64.,tmax,elements,tt1,count,0.0,0,0,rstar;fout="test_output_h64.txt",iout=640)
tt_save[7,:,:]=tt1
# Compute the h/16 case in BigFloat precision:
#dqbig = ttv_elements!(n,t0big,hbig/16,tmaxbig,elements_big,tt1big,count,big(0.0),0,0,rstarbig)
#tt_save[5,:,:]=convert(Array{Float64,2},tt1big)


# Make a plot of transit time errors versus stepsize:
ntrans = sum(count)
clf()
sigt = zeros(n-1,nsub-1)
tab = 0
#h_list = [h,h/2.,h/4.,h/8.]
h_list = [h,h/2.,h/4.,h/8,h/16,h/32]
hlabel = ["h-h/64","h/2-h/64","h/4-h/64","h/8-h/64","h/16-h/64","h/32-h/64"]
ch = ["black","red","green","blue","orange","magenta"]
for i=2:n
  for j=1:nsub-1
    tti1 = tt_save[j,i,1:count[i]]
    tti_max = tt_save[nsub,i,1:count[i]]
    diff = tti1 .- tti_max
    sigt[i-1,j]=std(diff)
    if i == n
      plot(tti1,diff ./ h_list[j]^4,linestyle="dashed",c=ch[j])
    else
      plot(tti1,diff ./ h_list[j]^4,label=hlabel[j],c=ch[j])
    end
  end
end
xlabel("Transit time")
ylabel("Difference in transit time / h^4")
legend(loc="lower left")

#read(stdin,Char)

# Make a plot of timing errors versus stepsize:
ntrans = sum(count)
clf()
sigt = zeros(n-1,nsub-1)
tab = 0
for i=2:n
  fn = zeros(Float64,2,count[i])
  sig = ones(count[i])
  tti_max = tt_save[nsub,i,1:count[i]]
  for j=1:count[i]
    fn[1,j] = 1.0
    fn[2,j] = round(Int64,(tti_max[j]-elements[i,3])/elements[i,2])
  end
  for j=1:nsub-1
    tti1 = tt_save[j,i,1:count[i]]
    #coeff,cov = regress(fn,tti1-tti_max,sig)
    p = Regression(fn[2,:],tti1 .- tti_max)
    coeff = [p.α, p.β]
    diff = tti1 .- tti_max .- coeff[1] .- (coeff[2] * fn[2,:])
    sigt[i-1,j]=std(diff)
#    print("planet: ",i-1," stepsize: ",j," sig: ",sigt[i-1,j])
    if i == n
      plot(tti1,diff ./ h_list[j]^4,linestyle="dashed",c=ch[j])
    else
      plot(tti1,diff ./ h_list[j]^4,label=hlabel[j],c=ch[j])
    end
  end
end
ylabel("TTV difference / h^4")
xlabel("Transit time")
legend(loc="lower left")

 #read(stdin,Char)
 clf()
 
 
 # Make a plot of some TTVs:
 loglog(h_list,sigt[1,:] .* 24.0 .* 3600.0,".",markersize=15,label="planet b")
 loglog(h_list,sigt[1,1] .* 24.0 .* 3600.0 .* (h_list ./ h[1]).^4,label=L"$\propto h^4$")
 loglog(h_list,sigt[2,:] .* 24.0 .* 3600.0,".",markersize=15,label="planet c")
 loglog(h_list,sigt[2,1] .* 24.0 .* 3600.0 .* (h_list ./ h[1]).^4,label=L"$\propto h^4$")
 legend(loc = "upper left")
 ylabel("RMS TTV error [sec]")
 xlabel("Step size [day]")
 
 PyPlot.savefig("timing_error_vs_h_kep88_no_outer.pdf",bbox_inches="tight")
# read(STDIN,Char)
# 
# clf()
# # Plot differences across phase space positions:
# data1 = readdlm("test_output_h.txt");
# data2 = readdlm("test_output_h2.txt");
# data4 = readdlm("test_output_h4.txt");
# data8 = readdlm("test_output_h8.txt");
# data16 = readdlm("test_output_h16.txt");
# 
# for i=8:16
#   plot(data1[:,1],data2[:, i]-data8[:,i],".")
# end
