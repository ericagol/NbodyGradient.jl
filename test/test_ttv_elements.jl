include("../src/ttv.jl")
include("/Users/ericagol/Computer/Julia/regress.jl")

# This routine takes derivative of transit times with respect
# to the initial orbital elements.
#n = 8
n = 3
t0 = 7257.93115525
#h  = 0.12
h  = 0.05
#tmax = 600.0
#tmax = 800.0
tmax = 10.0

# Read in initial conditions:
elements = readdlm("elements.txt",',')

# Make an array, tt,  to hold transit times:
# First, though, make sure it is large enough:
ntt = zeros(Int64,n)
for i=2:n
  ntt[i] = ceil(Int64,tmax/elements[i,2])+3
end
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
dq = ttv_elements!(n,t0,h,tmax,elements,tt1,count1,0.0,0,0)
@time dq = ttv_elements!(n,t0,h,tmax,elements,tt1,count1,0.0,0,0)
@time dq = ttv_elements!(n,t0,h,tmax,elements,tt1,count1,0.0,0,0)
# Now call with half the timestep:
count2 = zeros(Int64,n)
count3 = zeros(Int64,n)
dq = ttv_elements!(n,t0,h/10.,tmax,elements,tt2,count2,0.0,0,0)

# Now, compute derivatives (with respect to initial cartesian positions/masses):
dtdq0 = zeros(n,maximum(ntt),7,n)
dtdelements0 = zeros(n,maximum(ntt),7,n)
dtdelements0 = ttv_elements!(n,t0,h,tmax,elements,tt,count,dtdq0)
@time dtdelements0 = ttv_elements!(n,t0,h,tmax,elements,tt,count,dtdq0)
@time dtdelements0 = ttv_elements!(n,t0,h,tmax,elements,tt,count,dtdq0)
dtdq2 = zeros(n,maximum(ntt),7,n)
dtdelements2 = zeros(n,maximum(ntt),7,n)
@time dtdelements2 = ttv_elements!(n,t0,h/2.,tmax,elements,tt2,count,dtdq2)
dtdq4 = zeros(n,maximum(ntt),7,n)
dtdelements4 = zeros(n,maximum(ntt),7,n)
@time dtdelements4 = ttv_elements!(n,t0,h/4.,tmax,elements,tt4,count,dtdq4)
dtdq8 = zeros(n,maximum(ntt),7,n)
dtdelements8 = zeros(n,maximum(ntt),7,n)
@time dtdelements8 = ttv_elements!(n,t0,h/8.,tmax,elements,tt8,count,dtdq8)
println("Maximum error on derivative: ",maximum(abs.(dtdelements0-dtdelements2)))
println("Maximum error on derivative: ",maximum(abs.(dtdelements2-dtdelements4)))
println("Maximum error on derivative: ",maximum(abs.(dtdelements4-dtdelements8)))
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
dtdelements0_sum = zeros(BigFloat,n,maximum(ntt),7,n)

n_body = n
# Compute derivatives with BigFloat for additional precision:
elements0 = copy(elements)
delement = big.([1e-15,1e-15,1e-15,1e-15,1e-15,1e-15,1e-15])
tt2 = big.(tt2)
tt3 = big.(tt3)
t0big = big(t0); tmaxbig = big(tmax); hbig = big(h)
zero = big(0.0)
# Now, compute derivatives numerically:
for jq=2:n_body
  for iq=1:7
    elementsbig = big.(elements0)
    dq0 = delement[iq]; if jq==1 && iq==7 ; dq0 = big(1e-10); end  # Vary mass of star by a larger factor
    if iq == 7; ivary = 1; else; ivary = iq+1; end  # Shift mass variation to end
    elementsbig[jq,ivary] += dq0
    @time dq_plus = ttv_elements!(n,t0big,hbig,tmaxbig,elementsbig,tt2,count2,zero,0,0)
    elementsbig[jq,ivary] -= 2dq0
    @time dq_minus = ttv_elements!(n,t0big,hbig,tmaxbig,elementsbig,tt3,count2,zero,0,0)
    #xm,vm = init_nbody(elements,t0,n_body)
    for i=1:n
      for k=1:count2[i]
        # Compute double-sided derivative for more accuracy:
        dtdelements0_sum[i,k,iq,jq] = (tt2[i,k]-tt3[i,k])/(2.*dq0)
      end
    end
  end
end

println(maximum(abs.(dtdelements0-dtdelements0_sum)))

nbad = 0
ntot = 0
diff_dtdelements0 = zeros(n,maximum(ntt),7,n)
for i=1:n, j=1:count[i], k=1:7, l=2:n
  if abs(dtdelements0[i,j,k,l]-dtdelements0_sum[i,j,k,l]) > 0.1*abs(dtdelements0[i,j,k,l]) && ~(abs(dtdelements0[i,j,k,l]) == 0.0  && abs(dtdelements0_sum[i,j,k,l]) < 1e-3)
    nbad +=1
  end
  if dtdelements0[i,j,k,l] != 0.0
    diff_dtdelements0[i,j,k,l] = minimum([abs(dtdelements0[i,j,k,l]-dtdelements0_sum[i,j,k,l]);abs(dtdelements0_sum[i,j,k,l]/dtdelements0[i,j,k,l]-1.0)])
  else
    diff_dtdelements0[i,j,k,l] = abs(dtdelements0[i,j,k,l]-dtdelements0_sum[i,j,k,l])
  end
  ntot +=1
end

using PyPlot

nderiv = n^2*7*maximum(ntt)
#mask[:,:,2,:] = false
loglog(abs.(reshape(dtdelements0,nderiv)),abs.(reshape(convert(Array{Float64,4},dtdelements0_sum),nderiv)),".")
axis([1e-6,1e2,1e-12,1e2])
loglog(abs.(reshape(dtdelements0,nderiv)),abs.(reshape(diff_dtdelements0,nderiv)),".")
println("Maximum error: ",maximum(diff_dtdelements0))

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
