  #include("../src/ttv.jl"2
#include("/Users/ericagol/Computer/Julia/regress.jl")

#@testset "ttv_elements" begin

# This routine takes derivative of transit times with respect
# to the initial orbital elements.
#n = 8
n = 3
n_body = n
t0 = 7257.93115525-7300.0
#h  = 0.12
h  = 0.04
#tmax = 600.0
#tmax = 800.0
#tmax = 1000.0
tmax = 10.0

# Read in initial conditions:
elements = readdlm("elements.txt",',')
elements[:,3] -= 7300.0
# Make masses of planets bigger
#elements[2,1] *= 10.0
#elements[3,1] *= 10.0

ntt = zeros(Int64,n)

# Make an array, tt,  to hold transit times:
# First, though, make sure it is large enough:
for i=2:n
  ntt[i] = ceil(Int64,tmax/elements[i,2])+3
end
dtdq0 = zeros(n,maximum(ntt),7,n)
tt  = zeros(n,maximum(ntt))
tt1 = zeros(n,maximum(ntt))
tt2 = zeros(n,maximum(ntt))
tt3 = zeros(n,maximum(ntt))
tt4 = zeros(n,maximum(ntt))
tt8 = zeros(n,maximum(ntt))
# Save a counter for the actual number of transit times of each planet:
count = zeros(Int64,n)
count1 = zeros(Int64,n)
# Call the ttv function:
rstar = 1e12
dq = ttv_elements!(n,t0,h,tmax,elements,tt1,count1,0.0,0,0,rstar)
# Now call with half the timestep:
count2 = zeros(Int64,n)
count3 = zeros(Int64,n)
dq = ttv_elements!(n,t0,h/10.,tmax,elements,tt2,count2,0.0,0,0,rstar)

mask = zeros(Bool, size(dtdq0))
for jq=1:n_body
  for iq=1:7
    if iq == 7; ivary = 1; else; ivary = iq+1; end  # Shift mass variation to end
    for i=2:n
      for k=1:count2[i]
        # Ignore inclination & longitude of nodes variations:
        if iq != 5 && iq != 6 && ~(jq == 1 && iq < 7) && ~(jq == i && iq == 7)
          mask[i,k,iq,jq] = true
        end
      end
    end
  end
end

# Now, compute derivatives (with respect to initial cartesian positions/masses):
dtdelements0 = zeros(n,maximum(ntt),7,n)
dtdelements0 = ttv_elements!(n,t0,h,tmax,elements,tt,count,dtdq0,rstar)
dtdq2 = zeros(n,maximum(ntt),7,n)
dtdelements2 = zeros(n,maximum(ntt),7,n)
dtdelements2 = ttv_elements!(n,t0,h/2.,tmax,elements,tt2,count,dtdq2,rstar)
dtdq4 = zeros(n,maximum(ntt),7,n)
dtdelements4 = zeros(n,maximum(ntt),7,n)
dtdelements4 = ttv_elements!(n,t0,h/4.,tmax,elements,tt4,count,dtdq4,rstar)
dtdq8 = zeros(n,maximum(ntt),7,n)
dtdelements8 = zeros(n,maximum(ntt),7,n)
dtdelements8 = ttv_elements!(n,t0,h/8.,tmax,elements,tt8,count,dtdq8,rstar)
#println("Maximum error on derivative: ",maximum(abs.(dtdelements0-dtdelements2)))
#println("Maximum error on derivative: ",maximum(abs.(dtdelements2-dtdelements4)))
#println("Maximum error on derivative: ",maximum(abs.(dtdelements4-dtdelements8)))
println("Maximum error on derivative: ",maximum(abs.(asinh.(dtdelements0[mask])-asinh.(dtdelements2[mask]))))
println("Maximum error on derivative: ",maximum(abs.(asinh.(dtdq0)-asinh.(dtdq2))))
println("Maximum error on derivative: ",maximum(abs.(asinh.(dtdelements2[mask])-asinh.(dtdelements4[mask]))))
println("Maximum error on derivative: ",maximum(abs.(asinh.(dtdq2)-asinh.(dtdq4))))
println("Maximum error on derivative: ",maximum(abs.(asinh.(dtdelements4[mask])-asinh.(dtdelements8[mask]))))
println("Maximum error on derivative: ",maximum(abs.(asinh.(dtdq4)-asinh.(dtdq8))))
#read(STDIN,Char)

# Check that this is working properly:
#for i=1:n
#  for j=1:count2[i]
#    println(i," ",j," ",tt[i,j]," ",tt2[i,j]," ",tt[i,j]-tt2[i,j]," ",tt1[i,j]-tt2[i,j])
#  end
#end
#read(STDIN,Char)

# Compute derivatives numerically:
#nq = 15
# This "summarizes" best numerical derivative:
dtdelements0_num = zeros(BigFloat,n,maximum(ntt),7,n)

# Compute derivatives with BigFloat for additional precision:
elements0 = copy(elements)
#delement = big.([1e-15,1e-15,1e-15,1e-15,1e-15,1e-15,1e-15])
#dq0 = big(1e-20)
dq0 = big(1e-10)
tt2 = big.(tt2)
tt3 = big.(tt3)
t0big = big(t0); tmaxbig = big(tmax); hbig = big(h)
zilch = big(0.0)
# Compute the transit times in BigFloat precision:
tt_big = big.(tt); elementsbig = big.(elements0); rstarbig = big(rstar)
dqbig = ttv_elements!(n,t0big,hbig,tmaxbig,elementsbig,tt_big,count,big(0.0),0,0,rstarbig)

mask = zeros(Bool, size(dtdq0))
for jq=1:n_body
  for iq=1:7
    if iq == 7; ivary = 1; else; ivary = iq+1; end  # Shift mass variation to end
    for i=2:n
      for k=1:count2[i]
        # Ignore inclination & longitude of nodes variations:
        if iq != 5 && iq != 6 && ~(jq == 1 && iq < 7) && ~(jq == i && iq == 7)
          mask[i,k,iq,jq] = true
        end
      end
    end
  end
end

# Now, compute derivatives (with respect to initial cartesian positions/masses):
dtdelements0 = zeros(n,maximum(ntt),7,n)
dtdelements0 = ttv_elements!(n,t0,h,tmax,elements,tt,count,dtdq0,rstar)
dtdq2 = zeros(n,maximum(ntt),7,n)
dtdelements2 = zeros(n,maximum(ntt),7,n)
dtdelements2 = ttv_elements!(n,t0,h/2.,tmax,elements,tt2,count,dtdq2,rstar)
dtdq4 = zeros(n,maximum(ntt),7,n)
dtdelements4 = zeros(n,maximum(ntt),7,n)
dtdelements4 = ttv_elements!(n,t0,h/4.,tmax,elements,tt4,count,dtdq4,rstar)
dtdq8 = zeros(n,maximum(ntt),7,n)
dtdelements8 = zeros(n,maximum(ntt),7,n)
dtdelements8 = ttv_elements!(n,t0,h/8.,tmax,elements,tt8,count,dtdq8,rstar)
#println("Maximum error on derivative: ",maximum(abs.(dtdelements0-dtdelements2)))
#println("Maximum error on derivative: ",maximum(abs.(dtdelements2-dtdelements4)))
#println("Maximum error on derivative: ",maximum(abs.(dtdelements4-dtdelements8)))
println("Maximum error on derivative: ",maximum(abs.(asinh.(dtdelements0[mask])-asinh.(dtdelements2[mask]))))
println("Maximum error on derivative: ",maximum(abs.(asinh.(dtdq0)-asinh.(dtdq2))))
println("Maximum error on derivative: ",maximum(abs.(asinh.(dtdelements2[mask])-asinh.(dtdelements4[mask]))))
println("Maximum error on derivative: ",maximum(abs.(asinh.(dtdq2)-asinh.(dtdq4))))
println("Maximum error on derivative: ",maximum(abs.(asinh.(dtdelements4[mask])-asinh.(dtdelements8[mask]))))
println("Maximum error on derivative: ",maximum(abs.(asinh.(dtdq4)-asinh.(dtdq8))))
#read(STDIN,Char)

# Check that this is working properly:
#for i=1:n
#  for j=1:count2[i]
#    println(i," ",j," ",tt[i,j]," ",tt2[i,j]," ",tt[i,j]-tt2[i,j]," ",tt1[i,j]-tt2[i,j])
#  end
#end
#read(STDIN,Char)

# Compute derivatives numerically:
#nq = 15
# This "summarizes" best numerical derivative:
dtdelements0_num = zeros(BigFloat,n,maximum(ntt),7,n)

# Compute derivatives with BigFloat for additional precision:
elements0 = copy(elements)
#delement = big.([1e-15,1e-15,1e-15,1e-15,1e-15,1e-15,1e-15])
#dq0 = big(1e-20)
dq0 = big(1e-10)
tt2 = big.(tt2)
tt3 = big.(tt3)
t0big = big(t0); tmaxbig = big(tmax); hbig = big(h)
zilch = big(0.0)
# Compute the transit times in BigFloat precision:
tt_big = big(tt); elementsbig = big.(elements0)
# Now, compute derivatives numerically:
for jq=1:n_body
  for iq=1:7
    elementsbig = big.(elements0)
#    dq0 = delement[iq]; if jq==1 && iq==7 ; dq0 = big(1e-10); end  # Vary mass of star by a larger factor
    if iq == 7; ivary = 1; else; ivary = iq+1; end  # Shift mass variation to end
    elementsbig[jq,ivary] += dq0
    dq_plus = ttv_elements!(n,t0big,hbig,tmaxbig,elementsbig,tt2,count2,zilch,0,0,big(rstar))
    elementsbig[jq,ivary] -= 2dq0
    dq_minus = ttv_elements!(n,t0big,hbig,tmaxbig,elementsbig,tt3,count2,zilch,0,0,big(rstar))
    #xm,vm = init_nbody(elements,t0,n_body)
    for i=2:n
      for k=1:count2[i]
        # Compute double-sided derivative for more accuracy:
        dtdelements0_num[i,k,iq,jq] = (tt2[i,k]-tt3[i,k])/(2.*dq0)
        # Ignore inclination & longitude of nodes variations:
        if iq != 5 && iq != 6 && ~(jq == 1 && iq < 7) && ~(jq == i && iq == 7)
          mask[i,k,iq,jq] = true
        end
      end
    end
  end
end

#println("Max diff dtdelements: ",maximum(abs.(dtdelements0[mask]./dtdelements0_num[mask]-1.0)))
println("Max diff asinh(dtdelements): ",maximum(abs.(asinh.(dtdelements0[mask])-asinh.(dtdelements0_num[mask]))))

#ntot = 0
#diff_dtdelements0 = zeros(n,maximum(ntt),7,n)
#for i=1:n, j=1:count[i], k=1:7, l=2:n
#  diff_dtdelements0[i,j,k,l] = abs(dtdelements0[i,j,k,l]-convert(Float64,dtdelements0_num[i,j,k,l]))
#  ntot +=1
#end

#using PyPlot
#
#nderiv = n^2*7*maximum(ntt)
##mask[:,:,2,:] = false
#loglog(abs.(reshape(dtdelements0,nderiv)),abs.(reshape(convert(Array{Float64,4},dtdelements0_num),nderiv)),".")
#axis([1e-6,1e2,1e-12,1e2])
#loglog(abs.(reshape(dtdelements0,nderiv)),abs.(reshape(diff_dtdelements0,nderiv)),".")
#println("Maximum error: ",maximum(diff_dtdelements0))

using PyPlot

# Make a plot of the fractional errors:
for i=2:3, k=1:7, l=1:3
  if maximum(abs.(dtdelements0_num[i,2:count1[i],k,l])) > 0
    loglog(tt[i,2:count1[i]]-tt[i,1],abs.(dtdelements0[i,2:count1[i],k,l]./dtdelements0_num[i,2:count1[i],k,l]-1.))
  end
end

clf()
# Plot the difference in the TTVs:
for i=2:3
  diff1 = abs.(tt1[i,2:count1[i]]./tt_big[i,2:count1[i]]-1.0);
  loglog(tt[i,2:count1[i]]-tt[i,1],diff1);
end
for i=2:3, k=1:7, l=1:3
  if maximum(abs.(dtdelements0_num[i,2:count1[i],k,l])) > 0 && k != 5 && k != 6
    diff1 = abs.(dtdelements0[i,2:count1[i],k,l]./dtdelements0_num[i,2:count1[i],k,l]-1.);
    diff2 = abs.(asinh.(dtdelements0[i,2:count1[i],k,l])-asinh.(dtdelements0_num[i,2:count1[i],k,l]));
    loglog(tt[i,2:count1[i]]-tt[i,1],diff1); 
#    loglog(tt[i,2:count1[i]]-tt[i,1],diff2,"."); 
    println(i," ",k," ",l," frac error: ",convert(Float64,maximum(diff1))," asinh error: ",convert(Float64,maximum(diff2))); #read(STDIN,Char);
  end
end

# Plot a line that scales as time^{3/2}:

loglog([1.0,1024.0],1e-09*[1,2^15],":")
loglog([1.0,1024.0],1e-12*[1,2^15],":")
loglog([1.0,1024.0],1e-15*[1,2^15],":")

#@test isapprox(dtdelements0[mask],dtdelements0_num[mask];norm=maxabs)
@test isapprox(asinh.(dtdelements0[mask]),asinh.(dtdelements0_num[mask]);norm=maxabs)
#unit = ones(dtdelements0[mask])
#@test isapprox(dtdelements0[mask]./dtdelements0_num[mask],unit;norm=maxabs)
#end

## Make a plot of some TTVs:
#
#fig,axes = subplots(4,2)
#
#for i=2:8
#  ax = axes[i-1]
#  fn = zeros(Float64,2,count1[i])
#  sig = ones(count1[i])
#  tti1 = tt1[i,1:count1[i]]
#  tti2 = tt2[i,1:count2[i]]
#  for j=1:count1[i]
#    fn[1,j] = 1.0
#    fn[2,j] = round(Int64,(tti1[j]-elements[i,3])/elements[i,2])
#  end
#  coeff,cov = regress(fn,tti1,sig)
#  tt_ref1 = coeff[1]+coeff[2]*fn[2,:]
#  ttv1 = (tti1-tt_ref1)*24.*60.
#  coeff,cov = regress(fn,tti2,sig)
#  tt_ref2 = coeff[1]+coeff[2]*fn[2,:]
#  ttv2 = (tti2-tt_ref2)*24.*60.
#  ax[:plot](tti1,ttv1)
##  ax[:plot](tti2,ttv2)
#  ax[:plot](tti2,((ttv1-ttv2)-mean(ttv1-ttv2)))
#  println(i," ",coeff," ",elements[i,2:3]," ",coeff[1]-elements[i,3]," ",coeff[2]-elements[i,2])
#  println(i," ",maximum(ttv1-ttv2-mean(ttv1-ttv2))*60.," sec ", minimum(ttv1-ttv2-mean(ttv1-ttv2))*60.," sec" )
#end
