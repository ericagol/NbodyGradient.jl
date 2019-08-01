# Compares the time of simulation and 
include("../../src/ttv.jl")
include("/Users/ericagol/Computer/Julia/regress.jl")

# Turn of fastkicks:
nopair = false

elements = readdlm("elements_rebound.txt",',')

#n = 4
rstar = -1e12
for n=2:11
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

# Loop over the number of planets in the simulation from 2 to 8:
t0 = 0.0
h  = 0.05
# Rebound integration is for 5000 code units;  inner planet has
# a period of 2pi, while in our case 1-day, so differs by 5000/(2pi):
tmax = 2500/pi


ntt = zeros(Int64,n)
for i=2:n
  ntt[i] = ceil(Int64,tmax/elements[i,2])+2
end
#println("ntt: ",ntt)
tt1 = zeros(n,maximum(ntt))
tt = zeros(n,maximum(ntt))
# Save a counter for the actual number of transit times of each planet:
count1 = zeros(Int64,n)
count = zeros(Int64,n)
# Call the ttv function, without gradient:
rstar = -1e12
if nopair
  ttv_elements!(n,t0,h,tmax,elements,tt1,count1,0.0,0,0,rstar)
else
  ttv_elements!(n,t0,h,tmax,elements,tt1,count1,0.0,0,0,rstar;pair=pair_input)
end

tic()
if nopair
  ttv_elements!(n,t0,h,tmax,elements,tt1,count1,0.0,0,0,rstar)
else
  ttv_elements!(n,t0,h,tmax,elements,tt1,count1,0.0,0,0,rstar;pair=pair_input)
end
var_0 = toq()

# Now call with gradient:

dtdq0 = zeros(n,maximum(ntt),7,n)
dtdq0 = zeros(n,maximum(ntt),7,n)
#dtdelements0 = zeros(n,maximum(ntt),7,n)
#@time dtdelements0 = ttv_elements!(n,t0,h,tmax,elements,tt2,count1,dtdq0,rstar)
#dtdelements0 = ttv_elements!(n,t0,h,tmax,elements,tt2,count1,dtdq0,rstar)
#println("n ",n," t0 ",t0," h ",h," tmax ",tmax," elements ",elements)
if nopair
  dtdelements = ttv_elements!(n,t0,h,tmax,elements,tt,count,dtdq0,rstar)
else 
  dtdelements = ttv_elements!(n,t0,h,tmax,elements,tt,count,dtdq0,rstar;pair=pair_input)
end
tic()
if nopair
  dtdelements = ttv_elements!(n,t0,h,tmax,elements,tt,count,dtdq0,rstar)
else
  dtdelements = ttv_elements!(n,t0,h,tmax,elements,tt,count,dtdq0,rstar;pair=pair_input)
end
var_1 = toq()

println("Nplanet: ",n-1," No gradient: ",var_0, " gradient: ", var_1, " ratio: ", var_1/var_0)
end
