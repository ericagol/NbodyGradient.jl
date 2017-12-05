include("ttv.jl")
include("/Users/ericagol/Computer/Julia/regress.jl")

# This routine takes derivative of transit times with respect
# to the initial cartesian coordinates of bodies. [ ]
# (I still need to write this routine since now dtdq0 is turning
# into derivative wrt elements).
#n = 8
n = 3
t0 = 7257.93115525
#h  = 0.12
h  = 0.05
#tmax = 600.0
tmax = 80.0
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
tt1 = zeros(n,maximum(ntt))
tt2 = zeros(n,maximum(ntt))
tt3 = zeros(n,maximum(ntt))
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
ttv_elements!(n,t0,h,tmax,elements,tt,count,dtdq0)
@time ttv_elements!(n,t0,h,tmax,elements,tt,count,dtdq0)
@time ttv_elements!(n,t0,h,tmax,elements,tt,count,dtdq0)
#read(STDIN,Char)

# Check that this is working properly:
for i=1:n
  for j=1:count2[i]
    println(i," ",j," ",tt[i,j]," ",tt2[i,j]," ",tt[i,j]-tt2[i,j]," ",tt1[i,j]-tt2[i,j])
  end
end
#read(STDIN,Char)

# Compute derivatives numerically:
nq = 15
dtdq0_num = zeros(n,maximum(ntt),7,n,nq)
# This "summarizes" best numerical derivative:
dtdq0_sum = zeros(n,maximum(ntt),7,n)
itdq0 = zeros(Int64,n,maximum(ntt),7,n)
dlnq = [1e-2,3.16e-3,1e-3,3.16e-4,1e-4,3.16e-5,1e-5,3.16e-6,1e-6,3.16e-7,1e-7,3.16e-8,1e-8,3.16e-9,1e-9]
for jq=1:n
  for iq=1:7
    for inq = 1:nq
      elements2  = copy(elements)
      dq_plus = ttv_elements!(n,t0,h,tmax,elements2,tt2,count2,dlnq[inq],iq,jq)
      elements3  = copy(elements)
      dq_minus = ttv_elements!(n,t0,h,tmax,elements3,tt3,count3,-dlnq[inq],iq,jq)
#      if iq == 2 || iq == 5
#       println("timing difference: ",iq," ",maximum(abs.(tt2-tt3)))
#      end
      for i=1:n
        for k=1:count2[i]
#          dtdq0_num[i,k,iq,jq,inq] = (tt2[i,k]-tt1[i,k])/dq
          # Compute double-sided derivative for more accuracy:
          dtdq0_num[i,k,iq,jq,inq] = (tt2[i,k]-tt3[i,k])/(2.*dq_plus)
        end
      end
    end
    for i=1:n
      for k=1:count2[i]
        # Compare with analytic derivative (minimize over the finite difference):
        dmin = Inf
        imin = 0
        for inq=1:nq
          if abs(dtdq0_num[i,k,iq,jq,inq]-dtdq0[i,k,iq,jq]) < dmin
            imin = inq
            dmin = abs(dtdq0_num[i,k,iq,jq,inq]-dtdq0[i,k,iq,jq])
          end
        end
        dtdq0_sum[i,k,iq,jq] = dtdq0_num[i,k,iq,jq,imin]
        itdq0[i,k,iq,jq] = imin
#        println(iq," ",jq," ",i," ",k," ",tt1[i,k]," ",dtdq0_sum[i,k,iq,jq]," ",dtdq0[i,k,iq,jq]," ",dtdq0_sum[i,k,iq,jq]/dtdq0[i,k,iq,jq]-1.)
      end
    end
  end
end

nbad = 0
ntot = 0
diff_dtdq0 = zeros(n,maximum(ntt),7,n)
for i=2:n, j=1:count[i], k=1:7, l=1:n
  if abs(dtdq0[i,j,k,l]-dtdq0_sum[i,j,k,l]) > 0.1*abs(dtdq0[i,j,k,l]) && ~(abs(dtdq0[i,j,k,l]) == 0.0  && abs(dtdq0_sum[i,j,k,l]) < 1e-3)
    println(i," ",j," ",k," ",l," ",dtdq0[i,j,k,l]," ",dtdq0_sum[i,j,k,l]," ",itdq0[i,j,k,l])
    nbad +=1
  end
  if dtdq0[i,j,k,l] != 0.0
    diff_dtdq0[i,j,k,l] = minimum([abs(dtdq0[i,j,k,l]-dtdq0_sum[i,j,k,l]);abs(dtdq0_sum[i,j,k,l]/dtdq0[i,j,k,l]-1.0)])
  else
    diff_dtdq0[i,j,k,l] = abs(dtdq0[i,j,k,l]-dtdq0_sum[i,j,k,l])
  end
  ntot +=1
end

using PyPlot

nderiv = n^2*7*maximum(ntt)
#nderiv = n^2*5*maximum(ntt)
#mask = ones(Bool, dtdq0)
#mask[:,:,2,:] = false
#mask[:,:,5,:] = false
#loglog(abs.(reshape(dtdq0[mask],nderiv)),abs.(reshape(dtdq0_sum[mask],nderiv)),".")
loglog(abs.(reshape(dtdq0,nderiv)),abs.(reshape(dtdq0_sum,nderiv)),".")
#loglog(abs.(reshape(dtdq0[mask],nderiv)),abs.(reshape(dtdq0[mask]-dtdq0_sum[mask],nderiv)),".")
loglog(abs.(reshape(dtdq0,nderiv)),abs.(reshape(diff_dtdq0,nderiv)),".")
#loglog(abs.(reshape(dtdq0[mask],nderiv)),abs.(reshape(dtdq0_sum[mask]./dtdq0[mask]-1.,nderiv)),".")


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
