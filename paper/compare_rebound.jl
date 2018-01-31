# Compares the time of simulation and 
include("../src/ttv.jl")
include("/Users/ericagol/Computer/Julia/regress.jl")

elements = readdlm("elements_rebound.txt",',')

#n = 4
rstar = -1e12
for n=2:8
t0 = 0.0
h  = 0.05
tmax = 800.0


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
rstar = 1e12
ttv_elements!(n,t0,h,tmax,elements,tt1,count1,0.0,0,0,rstar)
tic()
ttv_elements!(n,t0,h,tmax,elements,tt1,count1,0.0,0,0,rstar)
var_0 = toq()

# Now call with gradient:

dtdq0 = zeros(n,maximum(ntt),7,n)
dtdq0 = zeros(n,maximum(ntt),7,n)
#dtdelements0 = zeros(n,maximum(ntt),7,n)
#@time dtdelements0 = ttv_elements!(n,t0,h,tmax,elements,tt2,count1,dtdq0,rstar)
#dtdelements0 = ttv_elements!(n,t0,h,tmax,elements,tt2,count1,dtdq0,rstar)
#println("n ",n," t0 ",t0," h ",h," tmax ",tmax," elements ",elements)
dtdelements = ttv_elements!(n,t0,h,tmax,elements,tt,count,dtdq0,rstar)
tic()
dtdelements = ttv_elements!(n,t0,h,tmax,elements,tt,count,dtdq0,rstar)
var_1 = toq()

println("Nplanet: ",n-1," No gradient: ",var_0, " gradient: ", var_1, " ratio: ", var_1/var_0)
end
