2
using JLD2

#include("../src/ttv.jl")
#include("/Users/ericagol/Computer/Julia/regress.jl")

@testset "ttvbv_cartesian" begin

# This routine takes derivative of transit times, sky velocity & 
# impact parameter squared with respect to the initial cartesian coordinates of bodies. [x]
#n = 8
n = 3
H =[3,1,1]
t0 = 7257.93115525-7300.0
#h  = 0.12
h  = 0.04
#h  = 0.02
#tmax = 600.0
#tmax = 1000.0
#tmax = 100.0
tmax = 10.0

# Read in initial conditions:
elements = readdlm("elements.txt",',')
elements[:,3] .-= 7300.0

# Make an array, tt,  to hold transit times:
# First, though, make sure it is large enough:
ntt = zeros(Int64,n)
for i=2:n
  ntt[i] = ceil(Int64,tmax/elements[i,2])+3
end
println("ntt: ",ntt)
ntbv = 3
ttbv  = zeros(ntbv,n,maximum(ntt))
ttbv1 = zeros(ntbv,n,maximum(ntt))
ttbv2 = zeros(ntbv,n,maximum(ntt))
ttbv3 = zeros(ntbv,n,maximum(ntt))
# Save a counter for the actual number of transit times of each planet:
count = zeros(Int64,n)
count1 = zeros(Int64,n)
# Call the ttv function:
rstar = 1e12
dq = ttvbv_elements!(H,t0,h,tmax,elements,ttbv1,count1,0.0,0,0,rstar)
# Now call with half the timestep:
count2 = zeros(Int64,n)
count3 = zeros(Int64,n)
dq = ttvbv_elements!(H,t0,h/2,tmax,elements,ttbv2,count2,0.0,0,0,rstar)

# Now, compute derivatives (with respect to initial cartesian positions/masses):
dtbvdq0 = zeros(ntbv,n,maximum(ntt),7,n)
dtbvdelements = ttvbv_elements!(H,t0,h,tmax,elements,ttbv,count,dtbvdq0,rstar)

# Compute derivatives numerically:
# Compute the numerical derivative:
dtbvdq0_num = zeros(BigFloat,ntbv,n,maximum(ntt),7,n)
dlnq = big(1e-15)
hbig = big(h); t0big = big(t0); tmaxbig=big(tmax); ttbv2big = big.(ttbv2); ttbv3big = big.(ttbv3)
for jq=1:n
  for iq=1:7
    elements2  = big.(elements)
    dq_plus = ttvbv_elements!(H,t0big,hbig,tmaxbig,elements2,ttbv2big,count2,dlnq,iq,jq,big(rstar))
    elements3  = big.(elements)
    dq_minus = ttvbv_elements!(H,t0big,hbig,tmaxbig,elements3,ttbv3big,count3,-dlnq,iq,jq,big(rstar))
    for i=1:n
      for k=1:count2[i]
        # Compute double-sided derivative for more accuracy:
        for itbv=1:ntbv
          dtbvdq0_num[itbv,i,k,iq,jq] = (ttbv2big[itbv,i,k]-ttbv3big[itbv,i,k])/(dq_plus-dq_minus)
        end
#        println(i," ",k," ",iq," ",jq," ",tt2big[i,k]," ",tt3big[i,k]," ")
      end
    end
  end
end

nbad = 0
ntot = 0
diff_dtbvdq0 = zeros(ntbv,n,maximum(ntt),7,n)
mask = zeros(Bool, size(dtbvdq0))
for i=2:n, j=1:count[i], k=1:7, l=1:n
  for itbv=1:ntbv
    if abs(dtbvdq0[itbv,i,j,k,l]-dtbvdq0_num[itbv,i,j,k,l]) > 0.1*abs(dtbvdq0[itbv,i,j,k,l]) && ~(abs(dtbvdq0[itbv,i,j,k,l]) == 0.0  && abs(dtbvdq0_num[itbv,i,j,k,l]) < 1e-3)
      nbad +=1
    end
    diff_dtbvdq0[itbv,i,j,k,l] = abs(dtbvdq0[itbv,i,j,k,l]-dtbvdq0_num[itbv,i,j,k,l])
    if k != 2 && k != 5
      mask[itbv,i,j,k,l] = true
    end
    ntot +=1
  end
end

ttbv_big = big.(ttbv); elementsbig = big.(elements); rstarbig = big(rstar)
dqbig = ttvbv_elements!(H,t0big,hbig,tmaxbig,elementsbig,ttbv_big,count,big(0.0),0,0,rstarbig)
# Now halve the time steps:
ttbv_big_half = copy(ttbv_big)
dqbig = ttvbv_elements!(H,t0big,hbig/2,tmaxbig,elementsbig,ttbv_big_half,count1,big(0.0),0,0,rstarbig)

# Compute the derivatives in BigFloat precision to see whether finite difference
# derivatives or Float64 derivatives are imprecise at the start:
dtbvdq0_big = zeros(BigFloat,ntbv,n,maximum(ntt),7,n)
hbig = big(h); ttbv_big = big.(ttbv); elementsbig = big.(elements); rstarbig = big(rstar)
dtbvdelements_big = ttvbv_elements!(H,t0big,hbig,tmaxbig,elementsbig,ttbv_big,count,dtbvdq0_big,rstarbig)

using PyPlot

clf()
# Plot the difference in the TTVs:
for i=2:3
#  diff1 = abs.(tt1[i,2:count1[i]]./tt_big[i,2:count1[i]]-1.0);
  for itbv=1:ntbv
    diff1 = convert(Array{Float64,1},abs.(ttbv1[itbv,i,2:count1[i]].-ttbv_big[itbv,i,2:count1[i]])/elements[i,2]);
    dtt=ttbv[itbv,i,2:count1[i]].-ttbv[itbv,i,1]
    loglog(dtt,diff1);
  end
#  diff2 = abs.(tt2[i,2:count1[i]]./tt_big_half[i,2:count1[i]]-1.0);
#  diff2 = abs.(tt2[i,2:count1[i]].-tt_big_half[i,2:count1[i]])/elements[i,2];
#  loglog(tt[i,2:count[i]]-tt[i,1],diff2);
end
loglog([1.0,1024.0],2e-15*[1,2^15],":")
for i=2:3, k=1:7, l=1:3
  for itbv=1:ntbv
    if maximum(abs.(dtbvdelements_big[itbv,i,2:count[i],k,l])) > 0
      diff1 = convert(Array{Float64,1},abs.(dtbvdelements_big[itbv,i,2:count[i],k,l]./dtbvdelements[itbv,i,2:count[i],k,l].-1));
      diff3 = convert(Array{Float64,1},abs.(asinh.(dtbvdelements_big[itbv,i,2:count[i],k,l])-asinh.(dtbvdelements[itbv,i,2:count[i],k,l])));
      dtt = ttbv[itbv,i,2:count[i]].-ttbv[itbv,i,1]
      loglog(dtt,diff3);
      println(itbv," ",i," ",k," ",l," frac error: ",convert(Float64,maximum(diff1))," asinh error: ",convert(Float64,maximum(diff3))); #read(STDIN,Char);
    end
    if maximum(abs.(dtbvdq0_big[itbv,i,2:count[i],k,l])) > 0
      diff1 = convert(Array{Float64,1},abs.(dtbvdq0[itbv,i,2:count[i],k,l]./dtbvdq0_big[itbv,i,2:count[i],k,l].-1.0));
      diff3 = convert(Array{Float64,1},abs.(asinh.(dtbvdq0_big[itbv,i,2:count[i],k,l]).-asinh.(dtbvdq0[itbv,i,2:count[i],k,l])));
      dtt = ttbv[itbv,i,2:count[i]].-ttbv[itbv,i,1]
      loglog(dtt,diff3,linestyle=":");
      println(itbv," ",i," ",k," ",l," frac error: ",convert(Float64,maximum(diff1))," asinh error: ",convert(Float64,maximum(diff3))); #read(STDIN,Char);
    end
  end
end
#mederror = zeros(size(ttbv))
#for i=2:3
#  for j=1:count1[i]
#    data_list = Float64[]
#    for itbv=1:ntbv, k=1:7, l=1:3
#      if abs(dtbvdq0_num[itbv,i,j,k,l]) > 0
#        push!(data_list,abs(dtbvdq0[itbv,i,j,k,l]/dtbvdq0_num[itbv,i,j,k,l]-1.0))
#      end
#      mederror[itbv,i,j] = median(data_list)
#    end
#  end
#end

# Plot a line that scales as time^{3/2}:

loglog([1.0,1024.0],1e-12*[1,2^15],":",linewidth=3)


println("Max diff asinh(dtbvdq0): ",maximum(abs.(asinh.(dtbvdq0_num[mask]).-asinh.(dtbvdq0[mask]))))
@test isapprox(asinh.(dtbvdq0[mask]),asinh.(convert(Array{Float64,5},dtbvdq0_num)[mask]);norm=maxabs)
#@save "test_ttbv.jld2" dtbvdq0 dtbvdq0_num
end
