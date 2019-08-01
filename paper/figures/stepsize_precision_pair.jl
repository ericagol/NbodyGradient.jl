
include("../../src/ttv.jl")
include("/Users/ericagol/Computer/Julia/regress.jl")

using PyPlot

# This routine takes derivative of transit times with respect
# to the initial orbital elements.
#n = 8
n = 3
n_body = n
t0 = 7257.93115525
#t0 =  0.0
#t0 =  randn()
#h  = 0.12
h  = 0.07
#tmax = 600.0
#tmax = 800.0
#tmax = 600.0
tmax = 700.0
#tmax = 100.0

# Read in initial conditions:
elements = readdlm("elements.txt",',')
# Make masses of planets bigger
#elements[2,1] *= 10.0
#elements[3,1] *= 10.0

ntt = zeros(Int64,n)

pair_input = ones(Bool,n,n)
pair_input[1,2] = false
pair_input[1,3] = false
pair_input[2,1] = false
pair_input[3,1] = false

# Make an array, tt,  to hold transit times:
# First, though, make sure it is large enough:
for i=2:n
  ntt[i] = ceil(Int64,tmax/elements[i,2])+3
end
dtdq0 = zeros(n,maximum(ntt),7,n)
tt  = zeros(n,maximum(ntt))
tt1 = zeros(n,maximum(ntt))
tt_save = zeros(5,n,maximum(ntt))
# Save a counter for the actual number of transit times of each planet:
count = zeros(Int64,n)
count1 = zeros(Int64,n)
# Call the ttv function:
rstar = 1e12
dq = ttv_elements!(n,t0,h,tmax,elements,tt1,count1,0.0,0,0,rstar;pair=pair_input)
tt_save[1,:,:]=tt1
# Now call with 1/10 the timestep:
#dq = ttv_elements!(n,t0,h/10.,tmax,elements,tt1,count,0.0,0,0,rstar;pair=pair_input)

mask = zeros(Bool, size(dtdq0))
for jq=1:n_body
  for iq=1:7
    if iq == 7; ivary = 1; else; ivary = iq+1; end  # Shift mass variation to end
    for i=2:n
      for k=1:count[i]
        # Ignore inclination & longitude of nodes variations:
        if iq != 5 && iq != 6 && ~(jq == 1 && iq < 7) && ~(jq == i && iq == 7)
          mask[i,k,iq,jq] = true
        end
      end
    end
  end
end

# Now, compute derivatives (with respect to initial cartesian positions/masses):
dq = ttv_elements!(n,t0,h/2.,tmax,elements,tt1,count,0.0,0,0,rstar;pair=pair_input)
tt_save[2,:,:]=tt1
dq = ttv_elements!(n,t0,h/4.,tmax,elements,tt1,count,0.0,0,0,rstar;pair=pair_input)
tt_save[3,:,:]=tt1
dq = ttv_elements!(n,t0,h/8.,tmax,elements,tt1,count,0.0,0,0,rstar;pair=pair_input)
tt_save[4,:,:]=tt1
dq = ttv_elements!(n,t0,h/16.,tmax,elements,tt1,count,0.0,0,0,rstar;pair=pair_input)
tt_save[5,:,:]=tt1

# Make a plot of transit time errors versus stepsize:
ntrans = sum(count)
clf()
sigt = zeros(n-1,5)
tab = 0
h_list = [h,h/2.,h/4.,h/8.,h/8]
hlabel = ["h-h/16","h/2-h/16","h/4-h/16","h/8-h/16","h/16-big(h/16)"]
ch = ["black","red","green","blue","orange"]
for i=2:n
  for j=1:4
    tti1 = tt_save[j,i,1:count[i]]
    tti16 = tt_save[5,i,1:count[i]]
    diff = tti1-tti16
    sigt[i-1,j]=std(diff)
    if i == n
      plot(tti1,diff/h_list[j]^4,linestyle="dashed",c=ch[j])
    else
      plot(tti1,diff/h_list[j]^4,label=hlabel[j],c=ch[j])
    end
  end
end
xlabel("Transit time")
ylabel("Difference in transit time / h^4")
legend(loc="lower left")

read(STDIN,Char)

# Make a plot of timing errors versus stepsize:
ntrans = sum(count)
clf()
sigt = zeros(n-1,5)
tab = 0
h_list = [h,h/2.,h/4.,h/8.,h/8]
hlabel = ["h-h/16","h/2-h/16","h/4-h/16","h/8-h/16","h/16-big(h/16)"]
ch = ["black","red","green","blue","orange"]
for i=2:n
  fn = zeros(Float64,2,count[i])
  sig = ones(count[i])
  tti16 = tt_save[5,i,1:count[i]]
  for j=1:count[i]
    fn[1,j] = 1.0
    fn[2,j] = round(Int64,(tti16[j]-elements[i,3])/elements[i,2])
  end
  for j=1:4
    tti1 = tt_save[j,i,1:count[i]]
    coeff,cov = regress(fn,tti1-tti16,sig)
    diff = tti1-tti16-coeff[1]-coeff[2]*fn[2,:]
    sigt[i-1,j]=std(diff)
    if i == n
      plot(tti1,diff/h_list[j]^4,linestyle="dashed",c=ch[j])
    else
      plot(tti1,diff/h_list[j]^4,label=hlabel[j],c=ch[j])
    end
  end
end
legend(loc="lower left")

read(STDIN,Char)
clf()


# Make a plot of some TTVs:
loglog(h_list,sigt[1,:]*24.*3600.,".",markersize=15,label="Inner planet")
loglog(h_list,sigt[1,1]*24.*3600.*(h_list/h[1]).^4,label=L"$\propto h^4$")
loglog(h_list,sigt[2,:]*24.*3600.,".",markersize=15,label="Outer planet")
loglog(h_list,sigt[2,1]*24.*3600.*(h_list/h[1]).^4,label=L"$\propto h^4$")
legend(loc = "upper left")
ylabel("RMS timing error [sec]")
xlabel("Step size [day]")

PyPlot.savefig("timing_error_vs_h_pair.pdf",bbox_inches="tight")
