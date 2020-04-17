
#include("../src/ttv.jl")
#include("/Users/ericagol/Computer/Julia/regress.jl")

@testset "ttvbv_elements" begin

# This routine takes derivative of transit times, vsky & bsky^2 with respect
# to the initial orbital elements.
#n = 8
n = 3
H = [3,1,1]
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
elements[:,3] .-= 7300.0
# Make masses of planets bigger
#elements[2,1] *= 10.0
#elements[3,1] *= 10.0

ntt = zeros(Int64,n)

# Make an array, ttbv,  to hold transit times, vsky & b^2:
# First, though, make sure it is large enough:
for i=2:n
  ntt[i] = ceil(Int64,tmax/elements[i,2])+3
end
dtbvdq0 = zeros(3,n,maximum(ntt),7,n)
ttbv  = zeros(3,n,maximum(ntt))
ttbv1 = zeros(3,n,maximum(ntt))
ttbv2 = zeros(3,n,maximum(ntt))
ttbv3 = zeros(3,n,maximum(ntt))
ttbv4 = zeros(3,n,maximum(ntt))
ttbv8 = zeros(3,n,maximum(ntt))
# Save a counter for the actual number of transit times of each planet:
count = zeros(Int64,n)
count1 = zeros(Int64,n)
# Call the ttv function:
rstar = 1e12
dq = ttvbv_elements!(H,t0,h,tmax,elements,ttbv1,count1,0.0,0,0,rstar)
# Now call with half the timestep:
count2 = zeros(Int64,n)
count3 = zeros(Int64,n)
dq = ttvbv_elements!(H,t0,h/10.,tmax,elements,ttbv2,count2,0.0,0,0,rstar)

mask = zeros(Bool, size(dtbvdq0))
for itbv=1:3, jq=1:n_body
  for iq=1:7
    if iq == 7; ivary = 1; else; ivary = iq+1; end  # Shift mass variation to end
    for i=2:n
      for k=1:count2[i]
        # Ignore inclination & longitude of nodes variations:
        if iq != 5 && iq != 6 && ~(jq == 1 && iq < 7) && ~(jq == i && iq == 7)
          mask[itbv,i,k,iq,jq] = true
        end
      end
    end
  end
end

# Now, compute derivatives (with respect to initial cartesian positions/masses):
dtbvdelements0 = zeros(3,n,maximum(ntt),7,n)
dtbvdelements0 = ttvbv_elements!(H,t0,h,tmax,elements,ttbv,count,dtbvdq0,rstar)
dtbvdq2 = zeros(3,n,maximum(ntt),7,n)
dtbvdelements2 = zeros(3,n,maximum(ntt),7,n)
dtbvdelements2 = ttvbv_elements!(H,t0,h/2.,tmax,elements,ttbv2,count,dtbvdq2,rstar)
dtbvdq4 = zeros(3,n,maximum(ntt),7,n)
dtbvdelements4 = zeros(3,n,maximum(ntt),7,n)
dtbvdelements4 = ttvbv_elements!(H,t0,h/4.,tmax,elements,ttbv4,count,dtbvdq4,rstar)
dtbvdq8 = zeros(3,n,maximum(ntt),7,n)
dtbvdelements8 = zeros(3,n,maximum(ntt),7,n)
dtbvdelements8 = ttvbv_elements!(H,t0,h/8.,tmax,elements,ttbv8,count,dtbvdq8,rstar)
println("Maximum error on derivative: ",maximum(abs.(asinh.(dtbvdelements0[mask])-asinh.(dtbvdelements2[mask]))))
println("Maximum error on derivative: ",maximum(abs.(asinh.(dtbvdq0)-asinh.(dtbvdq2))))
println("Maximum error on derivative: ",maximum(abs.(asinh.(dtbvdelements2[mask])-asinh.(dtbvdelements4[mask]))))
println("Maximum error on derivative: ",maximum(abs.(asinh.(dtbvdq2)-asinh.(dtbvdq4))))
println("Maximum error on derivative: ",maximum(abs.(asinh.(dtbvdelements4[mask])-asinh.(dtbvdelements8[mask]))))
println("Maximum error on derivative: ",maximum(abs.(asinh.(dtbvdq4)-asinh.(dtbvdq8))))


# Compute derivatives numerically:
#nq = 15
# This "summarizes" best numerical derivative:
dtbvdelements0_num = zeros(BigFloat,3,n,maximum(ntt),7,n)

# Compute derivatives with BigFloat for additional precision:
elements0 = copy(elements)
dq0 = big(1e-10)
t0big = big(t0); tmaxbig = big(tmax); hbig = big(h)
zilch = big(0.0)
# Compute the transit times in BigFloat precision:
ttbv_big = big.(ttbv); elementsbig = big.(elements0); rstarbig = big(rstar)
dqbig = ttvbv_elements!(H,t0big,hbig,tmaxbig,elementsbig,ttbv_big,count,big(0.0),0,0,rstarbig)

mask = zeros(Bool, size(dtbvdq0))
for itbv=1:3, jq=1:n_body
  for iq=1:7
    if iq == 7; ivary = 1; else; ivary = iq+1; end  # Shift mass variation to end
    for i=2:n
      for k=1:count2[i]
        # Ignore inclination & longitude of nodes variations:
        if iq != 5 && iq != 6 && ~(jq == 1 && iq < 7) && ~(jq == i && iq == 7)
          mask[itbv,i,k,iq,jq] = true
        end
      end
    end
  end
end

# Now, compute derivatives (with respect to initial cartesian positions/masses):
dtbvdelements0 = zeros(3,n,maximum(ntt),7,n)
dtbvdelements0 = ttvbv_elements!(H,t0,h,tmax,elements,ttbv,count,dtbvdq0,rstar)
dtbvdq2 = zeros(3,n,maximum(ntt),7,n)
dtbvdelements2 = zeros(3,n,maximum(ntt),7,n)
dtbvdelements2 = ttvbv_elements!(H,t0,h/2.,tmax,elements,ttbv2,count,dtbvdq2,rstar)
dtbvdq4 = zeros(3,n,maximum(ntt),7,n)
dtbvdelements4 = zeros(3,n,maximum(ntt),7,n)
dtbvdelements4 = ttvbv_elements!(H,t0,h/4.,tmax,elements,ttbv4,count,dtbvdq4,rstar)
dtbvdq8 = zeros(3,n,maximum(ntt),7,n)
dtbvdelements8 = zeros(3,n,maximum(ntt),7,n)
dtbvdelements8 = ttvbv_elements!(H,t0,h/8.,tmax,elements,ttbv8,count,dtbvdq8,rstar)
println("Maximum error on derivative: ",maximum(abs.(asinh.(dtbvdelements0[mask])-asinh.(dtbvdelements2[mask]))))
println("Maximum error on derivative: ",maximum(abs.(asinh.(dtbvdq0)-asinh.(dtbvdq2))))
println("Maximum error on derivative: ",maximum(abs.(asinh.(dtbvdelements2[mask])-asinh.(dtbvdelements4[mask]))))
println("Maximum error on derivative: ",maximum(abs.(asinh.(dtbvdq2)-asinh.(dtbvdq4))))
println("Maximum error on derivative: ",maximum(abs.(asinh.(dtbvdelements4[mask])-asinh.(dtbvdelements8[mask]))))
println("Maximum error on derivative: ",maximum(abs.(asinh.(dtbvdq4)-asinh.(dtbvdq8))))


# Compute derivatives numerically:
#nq = 15
# This "summarizes" best numerical derivative:
dtbvdelements0_num = zeros(BigFloat,3,n,maximum(ntt),7,n)

# Compute derivatives with BigFloat for additional precision:
elements0 = copy(elements)
dq0 = big(1e-10)
ttbv2 = big.(ttbv2)
ttbv3 = big.(ttbv3)
t0big = big(t0); tmaxbig = big(tmax); hbig = big(h)
zilch = big(0.0)
# Compute the transit times in BigFloat precision:
ttbv_big = big.(ttbv); elementsbig = big.(elements0)
# Now, compute derivatives numerically:
for jq=1:n_body
  for iq=1:7
    elementsbig = big.(elements0)
    if iq == 7; ivary = 1; else; ivary = iq+1; end  # Shift mass variation to end
    elementsbig[jq,ivary] += dq0
    dq_plus = ttvbv_elements!(H,t0big,hbig,tmaxbig,elementsbig,ttbv2,count2,zilch,0,0,big(rstar))
    elementsbig[jq,ivary] -= 2dq0
    dq_minus = ttvbv_elements!(H,t0big,hbig,tmaxbig,elementsbig,ttbv3,count2,zilch,0,0,big(rstar))
    #xm,vm = init_nbody(elements,t0,n_body)
    for i=2:n
      for k=1:count2[i]
        # Compute double-sided derivative for more accuracy:
        for itbv=1:3
          dtbvdelements0_num[itbv,i,k,iq,jq] = (ttbv2[itbv,i,k]-ttbv3[itbv,i,k])/(2dq0)
        # Ignore inclination & longitude of nodes variations:
          if iq != 5 && iq != 6 && ~(jq == 1 && iq < 7) && ~(jq == i && iq == 7)
            mask[itbv,i,k,iq,jq] = true
          end
        end
      end
    end
  end
end

println("Max diff asinh(dtbvdelements): ",maximum(abs.(asinh.(dtbvdelements0[mask])-asinh.(dtbvdelements0_num[mask]))))

using PyPlot

# Make a plot of the fractional errors:
for itbv=1:3, i=2:3, k=1:7, l=1:3
  if maximum(abs.(dtbvdelements0_num[itbv,i,2:count1[i],k,l])) > 0
    loglog(ttbv[itbv,i,2:count1[i]].-ttbv[itbv,i,1],convert(Array{Float64,1},abs.(dtbvdelements0[itbv,i,2:count1[i],k,l]./dtbvdelements0_num[itbv,i,2:count1[i],k,l].-1.)))
  end
end

clf()
# Plot the difference in the TTVs:
for itbv=1:3, i=2:3
  diff1 = convert(Array{Float64,1},abs.(ttbv1[itbv,i,2:count1[i]]./ttbv_big[itbv,i,2:count1[i]].-1.0));
  loglog(ttbv[itbv,i,2:count1[i]].-ttbv[itbv,i,1],diff1);
end
for itbv=1:3, i=2:3, k=1:7, l=1:3
  if maximum(abs.(dtbvdelements0_num[itbv,i,2:count1[i],k,l])) > 0 && k != 5 && k != 6
    diff1 = convert(Array{Float64,1},abs.(dtbvdelements0[itbv,i,2:count1[i],k,l]./dtbvdelements0_num[itbv,i,2:count1[i],k,l].-1));
    diff2 = convert(Array{Float64,1},abs.(asinh.(dtbvdelements0[itbv,i,2:count1[i],k,l])-asinh.(dtbvdelements0_num[itbv,i,2:count1[i],k,l])));
    loglog(ttbv[itbv,i,2:count1[i]].-ttbv[itbv,i,1],diff1); 
    println(i," ",k," ",l," frac error: ",convert(Float64,maximum(diff1))," asinh error: ",convert(Float64,maximum(diff2))); #read(STDIN,Char);
  end
end

# Plot a line that scales as time^{3/2}:

loglog([1.0,1024.0],1e-09*[1,2^15],":")
loglog([1.0,1024.0],1e-12*[1,2^15],":")
loglog([1.0,1024.0],1e-15*[1,2^15],":")

@test isapprox(asinh.(dtbvdelements0[mask]),asinh.(dtbvdelements0_num[mask]);norm=maxabs)
dtbvdelements0_num = convert(Array{Float64,5},dtbvdelements0_num)
#@save "ttvbv_elements.jld2" dtbvdelements0 dtbvdelements0_num ttbv
end
