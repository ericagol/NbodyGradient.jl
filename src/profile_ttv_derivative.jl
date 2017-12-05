include("/Users/ericagol/Computer/Julia/regress.jl")

n = 8
include("ttv.jl")
t0 = 7257.93115525
#h  = 0.12
h  = 0.075
tmax = 600.0
#tmax = 80.0

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
tt1 = zeros(n,maximum(ntt))
tt2 = zeros(n,maximum(ntt))
tt3 = zeros(n,maximum(ntt))
# Save a counter for the actual number of transit times of each planet:
count = zeros(Int64,n)
count1 = zeros(Int64,n)
# Call the ttv function:
dq = ttv_elements!(n,t0,h,tmax,elements,tt1,count1,0.0,0,0)
@time dq = ttv_elements!(n,t0,h,tmax,elements,tt1,count1,0.0,0,0)
# Now call with half the timestep:
count2 = zeros(Int64,n)
count3 = zeros(Int64,n)
dq = ttv_elements!(n,t0,h/10.,tmax,elements,tt2,count2,0.0,0,0)
println("Timing error: ",maximum(abs.(tt2-tt1))*24.*3600.)

# Now, compute derivatives (with respect to initial cartesian positions/masses):
dtdq0 = zeros(n,maximum(ntt),7,n)
ttv_elements!(n,t0,h,tmax,elements,tt,count,dtdq0)
@time ttv_elements!(n,t0,h,tmax,elements,tt,count,dtdq0)

Profile.clear()
Profile.init(10^7,0.01)
@profile ttv_elements!(n,t0,h,tmax,elements,tt,count,dtdq0)
#Profile.print()
