include("/Users/ericagol/Computer/Julia/regress.jl")

n = 8
include("ttv.jl")
t0 = 7257.93115525
#h  = 0.12
h  = 0.05
#h  = 0.025
tmax = 600.0
#tmax = 80.0

# Read in initial conditions:
elements = readdlm("../test/elements.txt",',')

# Make an array, tt,  to hold transit times:
# First, though, make sure it is large enough:
ntt = zeros(Int64,n)
for i=2:n
  ntt[i] = ceil(Int64,tmax/elements[i,2])+3
end
println("ntt: ",ntt)
tt  = zeros(n,maximum(ntt))
tt1 = zeros(n,maximum(ntt))
tt2 = zeros(n,maximum(ntt))
tt3 = zeros(n,maximum(ntt))
tt4 = zeros(n,maximum(ntt))
# Save a counter for the actual number of transit times of each planet:
count = zeros(Int64,n)
count1 = zeros(Int64,n)
# Call the ttv function:
rstar = 1e12
dq = ttv_elements!(n,t0,h,tmax,elements,tt1,count1,0.0,0,0,rstar)
@time dq = ttv_elements!(n,t0,h,tmax,elements,tt1,count1,0.0,0,0,rstar)
# Now, try calling with kicks between planets rather than -drift+Kepler:
pair_input = ones(Bool,n,n)
# We want Keplerian between star & planets, and impulses between
# planets.  Impulse is indicated with 'true', -drift+Kepler with 'false':
for i=2:n
  pair_input[1,i] = false
  # We don't need to define this, but let's anyways:
  pair_input[i,1] = false
end
# Now, only include Kepler solver for adjacent planets:
for i=2:n-1
  pair_input[i,i+1] = false
  pair_input[i+1,i] = false
end

# Now call with smaller timestep:
count2 = zeros(Int64,n)
count3 = zeros(Int64,n)
dq = ttv_elements!(n,t0,h/4.,tmax,elements,tt2,count2,0.0,0,0,rstar)
for i=2:n
  println("Timing error -drift+Kepler: ",i-1," ",maximum(abs.(tt2[i,:]-tt1[i,:]))*24.*3600.)
end
@time dq = ttv_elements!(n,t0,h,tmax,elements,tt3,count1,0.0,0,0,rstar;pair=pair_input)
dq = ttv_elements!(n,t0,h/4.,tmax,elements,tt4,count2,0.0,0,0,rstar;pair=pair_input)
for i=2:n
  println("Timing error kickfast:           ",i-1," ",maximum(abs.(tt3[i,:]-tt4[i,:]))*24.*3600.)
end
for i=2:n
  println("Timing kickfast-(-drift+Kepler): ",i-1," ",maximum(abs.(tt1[i,:]-tt3[i,:]))*24.*3600.)
end

# Now, compute derivatives (with respect to initial cartesian positions/masses):
dtdq1 = zeros(n,maximum(ntt),7,n)
dtdq2 = zeros(n,maximum(ntt),7,n)
dtdq_kick1 = zeros(n,maximum(ntt),7,n)
dtdq_kick2 = zeros(n,maximum(ntt),7,n)
dtdelements1 = zeros(n,maximum(ntt),7,n)
dtdelements2 = zeros(n,maximum(ntt),7,n)
dtdelements_kick1 = zeros(n,maximum(ntt),7,n)
dtdelements_kick2 = zeros(n,maximum(ntt),7,n)

dtdelements1 = ttv_elements!(n,t0,h,tmax,elements,tt,count,dtdq1,rstar)
@time dtdelements1 = ttv_elements!(n,t0,h,tmax,elements,tt,count,dtdq1,rstar)
dtdelements2 = ttv_elements!(n,t0,h/2,tmax,elements,tt,count,dtdq2,rstar)
@time dtdelements_kick1 = ttv_elements!(n,t0,h,tmax,elements,tt,count,dtdq_kick1,rstar;pair=pair_input)
@time dtdelements_kick2 = ttv_elements!(n,t0,h/2,tmax,elements,tt,count,dtdq_kick2,rstar;pair=pair_input)
println(maximum(abs.(asinh.(dtdelements1)-asinh.(dtdelements2))))
println(maximum(abs.(asinh.(dtdelements_kick1)-asinh.(dtdelements_kick2))))
println(maximum(abs.(asinh.(dtdelements2)-asinh.(dtdelements_kick2))))

Profile.clear()
Profile.init(10^7,0.01)
#@profile dtdelements0 = ttv_elements!(n,t0,h,tmax,elements,tt,count,dtdq0,rstar);
#Profile.print()
