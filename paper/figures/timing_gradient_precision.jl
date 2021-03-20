

# Makes plot for paper showing that transit timing
# precision obeys Brouwer's law:

using PyPlot
using JLD

@load "../../test/test_ttv_cartesian2_4e4days.jld"

using PyPlot

clf()
planet = ["Planet b","Planet c"]
# Plot the difference in the TTVs:
for i=2:3
  diff1 = convert(Array{Float64,1},abs.(tt[i,2:count[i]].-tt_big[i,2:count[i]])./h);
  loglog(tt[i,2:count[i]] .-tt[i,1],diff1,label=planet[i-1]);
#  diff2 = abs.(tt2[i,2:count[i]].-tt2_big[i,2:count[i]])/elements[i,2];
#  loglog(tt[i,2:count[i]]-tt[i,1],diff2);
end
xlabel("Time since start [days]",fontsize=12)
ylabel(L"$abs(t\vert_\mathrm{dbl}-t\vert_\mathrm{big})/h$",fontsize=15)
#loglog([1.0,1024.0],2e-15*[1,2^15],":")
loglog([1.0,40000.0],0.5e-16*([1.0,40000.0]/h).^1.5,":",label=L"$2^{-52}N^{3/2}$")
legend()
tight_layout()
savefig("transit_time_errors_4e4.pdf",bbox_inches="tight")
read(stdin,Char)

clf()
nmed = 10
for i=2:3, k=1:7, l=1:3
  if maximum(abs.(dtdq0[i,2:count[i],k,l])) > 0
#    diff1 = abs.(asinh.(dtdq0_big[i,2:count[i],k,l])-asinh.(dtdq0[i,2:count[i],k,l]));
#    diff1 = abs.(dtdq0_big[i,2:count[i],k,l]-dtdq0[i,2:count[i],k,l])/maximum(dtdq0[i,2:count[i],k,l]);
    diff1 = abs.(convert(Array{Float64,1},dtdq0_big[i,2:count[i],k,l]) .-dtdq0[i,2:count[i],k,l])
    diff2 = copy(diff1);
    diff3 = abs.(convert(Array{Float64,1},dtdelements_big[i,2:count[i],k,l]) .-dtdelements[i,2:count[i],k,l])
#    diff3 = abs.(asinh.(dtdelements_big[i,2:count[i],k,l])-asinh.(dtdelements[i,2:count[i],k,l]))
    diff4 = copy(diff3);
    ntt1 = size(diff1)[1];
    for it=nmed:ntt1-nmed
      # Divide by a median smoothed dtdq0:
      diff2[it] = diff1[it]/maximum(abs.(dtdq0[i,it+1-nmed+1:it+1+nmed,k,l]))
      maxit = maximum(abs.(dtdelements[i,it+1-nmed+1:it+1+nmed,k,l]))
      if maxit > 0
        diff4[it] = diff3[it]/maximum(abs.(dtdelements[i,it+1-nmed+1:it+1+nmed,k,l]))
      end
    end;
    diff2[1:nmed-1] .= diff2[nmed]; diff2[ntt1-nmed+1:ntt1] .= diff2[ntt1-nmed];
    diff4[1:nmed-1] .= diff4[nmed]; diff4[ntt1-nmed+1:ntt1] .= diff4[ntt1-nmed];
#    diff3 = abs.(asinh.(dtdq2_big[i,2:count[i],k,l])-asinh.(dtdq2[i,2:count[i],k,l]));
    telapsed = tt[i,2:count[i]] .-tt[i,1]
#    loglog(telapsed,diff2,".",alpha=0.2);
    # Only plot the points which are local maxima:
    indx = collect(1:nmed)
    for i=nmed+1:count[i]-nmed-1
      if diff2[i] == maximum(diff2[i-nmed:i+nmed])
        push!(indx,i)
      end
    end
    push!(indx,count[i]-1)
    loglog(telapsed[indx],diff2[indx]);
#    loglog(telapsed,diff4,linestyle="--");
#    loglog(tt[i,2:count[i]]-tt[i,1],diff3);
    println("xvs: ",i," ",k," ",l," asinh error h  : ",convert(Float64,maximum(diff2))); #read(STDIN,Char);
    println("els: ",i," ",k," ",l," asinh error h  : ",convert(Float64,maximum(diff4))); #read(STDIN,Char);
#    println(i," ",k," ",l," asinh error h/2: ",convert(Float64,maximum(diff3))); #read(STDIN,Char);
  end
end
loglog([1.0,40000.0],0.5e-16*([1.0,40000.0]/h).^1.5,linestyle=":")

#axis([1,1024,1e-19,1e-9])
axis([1,31600,1e-16,1e-7])
xlabel("Time since start [days]",fontsize=12)
#ylabel("median(asinh(dt/dq)_dbl - asinh(dt/dq)_big,20)")
ylabel(L"$\frac{maxabs(\frac{dt}{dq}\vert_\mathrm{dbl}-\frac{dt}{dq}\vert_\mathrm{big})[i-9:i+10]}{maxabs(\frac{dt}{dq}\vert_\mathrm{dbl}[i-9:i+10]}$",fontsize=15)
tight_layout()
savefig("Timing_derivative_errors_4e4days_maxabs20.pdf",bbox_inches="tight")
