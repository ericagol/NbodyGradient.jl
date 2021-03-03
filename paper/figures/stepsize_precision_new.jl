
#include("../../src/ttv.jl")
#include("/Users/ericagol/Computer/Julia/regress.jl")

using NbodyGradient, PyPlot, DelimitedFiles
include("linreg.jl")

# This routine takes derivative of transit times with respect
# to the initial orbital elements.
n = 3
# Try a larger value of t0:
#t0 =  -200.0
t0 =  0.0
h  = 0.06
tmax = 400.0

# Read in initial conditions:
elements = readdlm("elements.txt",',')[1:n,:]

ntt = zeros(Int64,n)

# Make an array, tt,  to hold transit times:
# First, though, make sure it is large enough:
for i=2:n
  ntt[i] = ceil(Int64,tmax/elements[i,2])+3
end

nsub = 7 # How many times to divide the timestep (/2^(nsub-1)).
tt_save = zeros(nsub,n,maximum(ntt))
# Call the ttv function:

ic = ElementsIC(t0, n, elements[1:n,:])
intr = Integrator(h, t0, tmax)
s = State(ic)
tt = TransitTiming(intr.tmax, ic)
@time intr(s,tt)
tt_save[1,:,:] .= tt.tt


# Now, compute derivatives (with respect to initial cartesian positions/masses):
intr = Integrator(h/2, t0, tmax)
s = State(ic)
tt = TransitTiming(intr.tmax, ic)
@time intr(s,tt)
tt_save[2,:,:] .= tt.tt

intr = Integrator(h/4, t0, tmax)
s = State(ic)
tt = TransitTiming(intr.tmax, ic)
@time intr(s,tt)
tt_save[3,:,:] .= tt.tt

intr = Integrator(h/8, t0, tmax)
s = State(ic)
tt = TransitTiming(intr.tmax, ic)
@time intr(s,tt)
tt_save[4,:,:] .= tt.tt

intr = Integrator(h/16, t0, tmax)
s = State(ic)
tt = TransitTiming(intr.tmax, ic)
@time intr(s,tt)
tt_save[5,:,:] .= tt.tt

intr = Integrator(h/32, t0, tmax)
s = State(ic)
tt = TransitTiming(intr.tmax, ic)
@time intr(s,tt)
tt_save[6,:,:] .= tt.tt

intr = Integrator(h/128, t0, tmax)
s = State(ic)
tt = TransitTiming(intr.tmax, ic)
@time intr(s,tt)
tt_save[7,:,:] .= tt.tt


# Make a plot of transit time errors versus stepsize:
clf()
sigt = zeros(n-1,nsub-1)
tab = 0
#h_list = [h,h/2.,h/4.,h/8.]
h_list = [h,h/2.,h/4.,h/8,h/16,h/32]
#hlabel = ["h-h/64","h/2-h/64","h/4-h/64","h/8-h/64","h/16-h/64","h/32-h/64"]
hlabel = ["h-h/128","h/2-h/128","h/4-h/128","h/8-h/128","h/16-h/128","h/32-h/128"]
ch = ["black","red","green","blue","orange","magenta"]
for i=2:n
  for j=1:nsub-1
    tti1 = tt_save[j,i,1:tt.count[i]]
    tti_max = tt_save[nsub,i,1:tt.count[i]]
    diff = (tti1 .- tti_max)*24*3600 # seconds
    sigt[i-1,j]=std(diff)
    if i == n
#      plot(tti1,diff ./ h_list[j]^4,linestyle="dashed",c=ch[j])
#      plot(tti1,diff ,linestyle="dashed",c=ch[j])
      semilogy(tti1,abs.(diff) ,linestyle="dashed",c=ch[j])
    else
#      plot(tti1,diff ./ h_list[j]^4,label=hlabel[j],c=ch[j])
#      plot(tti1,diff,label=hlabel[j],c=ch[j])
      plot(tti1,abs.(diff),label=hlabel[j],c=ch[j])
    end
  end
end
xlabel("Elapsed time")
ylabel("Difference in transit time [sec]")
plot(tti1,diff .* h^4,".")
plot([50,50,45,50,55],[1e-2,1e-2/2^4,1e-2/2^4*1.5,1e-2/2^4,1e-2/2^4*1.5])
text(55,3e-3,L"$2^4$")
plot(tti1,tti1 .* 1e-3, linestyle="dotted")
text(100,0.2,L"$10^{-3}t$")
axis([-10,410,0.5e-10,1.0])
legend(ncol=2)

#read(stdin,Char)
#
## Make a plot of timing errors versus stepsize:
#clf()
#sigt = zeros(n-1,nsub-1)
#tab = 0
#for i=2:n
#  fn = zeros(Float64,2,count[i])
#  sig = ones(count[i])
#  tti_max = tt_save[nsub,i,1:count[i]]
#  for j=1:count[i]
#    fn[1,j] = 1.0
#    fn[2,j] = round(Int64,(tti_max[j]-elements[i,3])/elements[i,2])
#  end
#  for j=1:nsub-1
#    tti1 = tt_save[j,i,1:count[i]]
#    #coeff,cov = regress(fn,tti1-tti_max,sig)
#    p = Regression(fn[2,:],tti1 .- tti_max)
#    coeff = [p.α, p.β]
#    diff = tti1 .- tti_max .- coeff[1] .- (coeff[2] * fn[2,:])
#    sigt[i-1,j]=std(diff)
##    print("planet: ",i-1," stepsize: ",j," sig: ",sigt[i-1,j])
#    if i == n
#      plot(tti1,diff ./ h_list[j]^4,linestyle="dashed",c=ch[j])
#    else
#      plot(tti1,diff ./ h_list[j]^4,label=hlabel[j],c=ch[j])
#    end
#  end
#end
#ylabel("TTV difference / h^4")
#xlabel("Transit time")
#legend(loc="lower left")
#
# #read(stdin,Char)
# clf()
# 
# 
# # Make a plot of some TTVs:
# loglog(h_list,sigt[1,:] .* 24.0 .* 3600.0,".",markersize=15,label="planet b")
# loglog(h_list,sigt[1,1] .* 24.0 .* 3600.0 .* (h_list ./ h[1]).^4,label=L"$\propto h^4$")
# loglog(h_list,sigt[2,:] .* 24.0 .* 3600.0,".",markersize=15,label="planet c")
# loglog(h_list,sigt[2,1] .* 24.0 .* 3600.0 .* (h_list ./ h[1]).^4,label=L"$\propto h^4$")
# legend(loc = "upper left")
# ylabel("RMS TTV error [sec]")
# xlabel("Step size [day]")
# 
## PyPlot.savefig("timing_error_vs_h_kep88_no_outer.pdf",bbox_inches="tight")
