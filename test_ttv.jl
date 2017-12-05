include("ttv.jl")
include("/Users/ericagol/Computer/Julia/regress.jl")

n = 8
t0 = 7257.93115525
#h  = 0.12
#h  = 0.075
h  = 0.05
tmax = 600.0

# Read in initial conditions:
elements = readdlm("elements.txt",',')

# Make an array, tt,  to hold transit times:
# First, though, make sure it is large enough:
ntt = zeros(Int64,n)
for i=2:n
  ntt[i] = ceil(Int64,tmax/elements[i,2])+2
end
println("ntt: ",ntt)
tt1 = zeros(n,maximum(ntt))
tt2 = zeros(n,maximum(ntt))
# Save a counter for the actual number of transit times of each planet:
count1 = zeros(Int64,n)
# Call the ttv function:
@time ttv_elements!(n,t0,h,tmax,elements,tt1,count1,0.0,0,0)
@time ttv_elements!(n,t0,h,tmax,elements,tt1,count1,0.0,0,0)
# Now call with half the timestep:
count2 = zeros(Int64,n)
ttv_elements!(n,t0,h/10.,tmax,elements,tt2,count2,0.0,0,0)
println("Timing error: ",maximum(abs.(tt1-tt2))*24.*3600.," sec")

using PyPlot

# Make a plot of some TTVs:

fig,axes = subplots(4,2)

for i=2:8
  ax = axes[i-1]
  fn = zeros(Float64,2,count1[i])
  sig = ones(count1[i])
  tti1 = tt1[i,1:count1[i]]
  tti2 = tt2[i,1:count2[i]]
  for j=1:count1[i]
    fn[1,j] = 1.0
    fn[2,j] = round(Int64,(tti1[j]-elements[i,3])/elements[i,2])
  end
  coeff,cov = regress(fn,tti1,sig)
  tt_ref1 = coeff[1]+coeff[2]*fn[2,:]
  ttv1 = (tti1-tt_ref1)*24.*60.
#  coeff,cov = regress(fn,tti2,sig)
  tt_ref2 = coeff[1]+coeff[2]*fn[2,:]
  ttv2 = (tti2-tt_ref2)*24.*60.
  ax[:plot](tti1,ttv1)
  ax[:plot](tti2,ttv2)
#  ax[:plot](tti2,((ttv1-ttv2)-mean(ttv1-ttv2)))
  ax[:plot](tti2,ttv1-ttv2)
  println(i," ",coeff," ",elements[i,2:3]," ",coeff[1]-elements[i,3]," ",coeff[2]-elements[i,2])
#  println(i," ",maximum(ttv1-ttv2-mean(ttv1-ttv2))*60.," sec ", minimum(ttv1-ttv2-mean(ttv1-ttv2))*60.," sec" )
  println(i," ",maximum(ttv1-ttv2)*60.," sec ", minimum(ttv1-ttv2)*60.," sec")
end
