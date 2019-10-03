
#include("../src/ttv.jl")
#include("/Users/ericagol/Computer/Julia/regress.jl")

run(`rm -f coord_jac_double.txt`)
run(`rm -f coord_jac_bigfloat.txt`)

#@testset "test_coordinates" begin

# This routine takes derivative of coordinates with respect
# to the initial cartesian coordinates of bodies, output to a *large* file.
#n = 8
n = 3
t0 = 7257.93115525-7300.0
#h  = 0.12
h  = 0.04
#h  = 0.02
#tmax = 600.0
#tmax = 1000.0
tmax = 100.0
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
# Save a counter for the actual number of transit times of each planet:
count = zeros(Int64,n)
# Call the ttv function & compute derivatives (with respect to initial cartesian positions/masses):
rstar = 1e12
dtdq0 = zeros(n,maximum(ntt),7,n)
@time dtdelements = ttv_elements!(n,t0,h,tmax,elements,tt,count,dtdq0,rstar;fout="coord_jac_double.txt",iout=10)

# Compute the derivatives in BigFloat precision to see whether finite difference
# derivatives or Float64 derivatives are imprecise at the start:
hbig = big(h); t0big = big(t0); tmaxbig=big(tmax); tt_big = big.(tt); elementsbig = big.(elements); rstarbig = big(rstar)
dtdq0_big = zeros(BigFloat,n,maximum(ntt),7,n)
@time dtdelements_big = ttv_elements!(n,t0big,hbig,tmaxbig,elementsbig,tt_big,count,dtdq0_big,rstarbig;fout="coord_jac_bigfloat.txt",iout=10)

using PyPlot

clf()
# Plot the difference in the TTVs:
for i=2:3
  diff1 = abs.(tt[i,2:count[i]].-tt_big[i,2:count[i]])/elements[i,2];
  loglog(tt[i,2:count[i]]-tt[i,1],diff1);
end
loglog([1.0,1024.0],2e-15*[1,2^15],":")
for i=2:3, k=1:7, l=1:3
  if maximum(abs.(dtdq0[i,2:count[i],k,l])) > 0
    diff3 = abs.(asinh.(dtdq0_big[i,2:count[i],k,l])-asinh.(dtdq0[i,2:count[i],k,l]));
    loglog(tt[i,2:count[i]]-tt[i,1],diff3,linestyle=":");
    println(i," ",k," ",l," asinh error: ",convert(Float64,maximum(diff3))); read(STDIN,Char);
  end
end
mederror = zeros(size(tt))

# Plot a line that scales as time^{3/2}:

loglog([1.0,1024.0],1e-12*[1,2^15],":",linewidth=3)

println("Max diff asinh(dtdq0): ",maximum(abs.(asinh.(dtdq0_big)-asinh.(dtdq0))))
@test isapprox(asinh.(dtdq0),asinh.(convert(Array{Float64,4},dtdq0_big));norm=maxabs)
#end


read(STDIN,Char)

# Now read in coordinate Jacobians:
data_dbl = readdlm("coord_jac_double.txt")
data_big = readdlm("coord_jac_bigfloat.txt")


# And plot the results
ncol = 49*n^2+6*n+1
nt = size(data_dbl)[1]
for i=1:ncol
  loglog(data_dbl[2:nt,1]-data_dbl[1,1],abs.(asinh.(data_big[2:nt,i])-asinh.(data_dbl[2:nt,i])))
end
