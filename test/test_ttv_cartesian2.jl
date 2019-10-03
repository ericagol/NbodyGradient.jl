


#include("../src/ttv.jl")
#include("/Users/ericagol/Computer/Julia/regress.jl")

#@testset "ttv_cartesian" begin

# This routine takes derivative of transit times with respect
# to the initial cartesian coordinates of bodies. [x]
#n = 8
n = 3
t0 = 7257.93115525-7300.0
#h  = 0.12
h  = 0.04
#h  = 0.02
#tmax = 600.0
tmax = 40000.0
#tmax = 100.0
#tmax = 10.0

# Read in initial conditions:
elements = readdlm("elements.txt",',')

# Make an array, tt,  to hold transit times:
# First, though, make sure it is large enough:
ntt = zeros(Int64,n)
for i=2:n
  ntt[i] = ceil(Int64,tmax/elements[i,2])+3
end
println("ntt: ",ntt)
tt  = zeros(n,maximum(ntt))
tt2 = zeros(n,maximum(ntt))
# Save a counter for the actual number of transit times of each planet:
count = zeros(Int64,n)
# Call the ttv function:
rstar = 1e12

# Now, compute derivatives (with respect to initial cartesian positions/masses):
dtdq0 = zeros(n,maximum(ntt),7,n)
@time dtdelements = ttv_elements!(n,t0,h,tmax,elements,tt,count,dtdq0,rstar)
# Now with half the timestep:
#dtdq2 = zeros(n,maximum(ntt),7,n)
#@time dtdelements = ttv_elements!(n,t0,h/2,tmax,elements,tt2,count,dtdq2,rstar)

tt_big = big.(tt); elementsbig = big.(elements); rstarbig = big(rstar)
tmaxbig = big(tmax); t0big = big(t0); hbig = big(h)

# Compute the derivatives in BigFloat precision to see whether finite difference
# derivatives or Float64 derivatives are imprecise at the start:
dtdq0_big = zeros(BigFloat,n,maximum(ntt),7,n)
hbig = big(h); tt_big = big.(tt); elementsbig = big.(elements); rstarbig = big(rstar)
@time dtdelements_big = ttv_elements!(n,t0big,hbig,tmaxbig,elementsbig,tt_big,count,dtdq0_big,rstarbig)
# Now with half the timestep:
dtdq2_big = zeros(BigFloat,n,maximum(ntt),7,n)
tt2_big = big.(tt2)
#@time dtdelements_big = ttv_elements!(n,t0big,hbig/2,tmaxbig,elementsbig,tt2_big,count,dtdq2_big,rstarbig)

using PyPlot

clf()
# Plot the difference in the TTVs:
for i=2:3
  diff1 = abs.(tt[i,2:count[i]].-tt_big[i,2:count[i]])/elements[i,2];
  loglog(tt[i,2:count[i]]-tt[i,1],diff1);
#  diff2 = abs.(tt2[i,2:count[i]].-tt2_big[i,2:count[i]])/elements[i,2];
#  loglog(tt[i,2:count[i]]-tt[i,1],diff2);
end
#loglog([1.0,1024.0],2e-15*[1,2^15],":")
loglog([1.0,40000.0],2.2e-16*([1.0,40000.0]/h).^1.5,":")
for i=2:3, k=1:7, l=1:3
  if maximum(abs.(dtdq0[i,2:count[i],k,l])) > 0
    diff1 = abs.(asinh.(dtdq0_big[i,2:count[i],k,l])-asinh.(dtdq0[i,2:count[i],k,l]));
#    diff3 = abs.(asinh.(dtdq2_big[i,2:count[i],k,l])-asinh.(dtdq2[i,2:count[i],k,l]));
    loglog(tt[i,2:count[i]]-tt[i,1],diff1,linestyle=":");
#    loglog(tt[i,2:count[i]]-tt[i,1],diff3);
    println(i," ",k," ",l," asinh error h  : ",convert(Float64,maximum(diff1))); #read(STDIN,Char);
#    println(i," ",k," ",l," asinh error h/2: ",convert(Float64,maximum(diff3))); #read(STDIN,Char);
  end
end

#axis([1,1024,1e-19,1e-9])
axis([1,31600,1e-19,2.5e-7])
# Plot a line that scales as time^{3/2}:

# loglog([1.0,1024.0],1e-12*[1,2^15],":",linewidth=3)
# loglog([1.0,1024.0],1e-12*[1,2^15],":",linewidth=3)


println("Max diff asinh(dtdq0): ",maximum(abs.(asinh.(dtdq0_big)-asinh.(dtdq0))))
@test isapprox(asinh.(dtdq0),asinh.(convert(Array{Float64,4},dtdq0_big));norm=maxabs)
#end
