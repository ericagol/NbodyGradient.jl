
clf()
nmed = 10
for i=2:3, k=1:7, l=1:3
  if maximum(abs.(dtdq0[i,2:count[i],k,l])) > 0
#    diff1 = abs.(asinh.(dtdq0_big[i,2:count[i],k,l])-asinh.(dtdq0[i,2:count[i],k,l]));
#    diff1 = abs.(dtdq0_big[i,2:count[i],k,l]-dtdq0[i,2:count[i],k,l])/maximum(dtdq0[i,2:count[i],k,l]);
    diff1 = abs.(dtdq0_big[i,2:count[i],k,l]-dtdq0[i,2:count[i],k,l])
    diff2 = copy(diff1);
    diff3 = abs.(dtdelements_big[i,2:count[i],k,l]-dtdelements_mixed[i,2:count[i],k,l])
#    diff3 = abs.(asinh.(dtdelements_big[i,2:count[i],k,l])-asinh.(dtdelements_mixed[i,2:count[i],k,l]))
    diff4 = copy(diff3);
    ntt1 = size(diff1)[1];
    for it=nmed:ntt1-nmed
      # Divide by a median smoothed dtdq0:
      diff2[it] = diff1[it]/maximum(abs.(dtdq0[i,it+1-nmed+1:it+1+nmed,k,l]))
      maxit = maximum(abs.(dtdelements_mixed[i,it+1-nmed+1:it+1+nmed,k,l]))
      if maxit > 0
        diff4[it] = diff3[it]/maximum(abs.(dtdelements_mixed[i,it+1-nmed+1:it+1+nmed,k,l]))
      end
    end;
    diff2[1:nmed-1] .= diff2[nmed]; diff2[ntt1-nmed+1:ntt1] .= diff2[ntt1-nmed];
    diff4[1:nmed-1] .= diff4[nmed]; diff4[ntt1-nmed+1:ntt1] .= diff4[ntt1-nmed];
#    diff3 = abs.(asinh.(dtdq2_big[i,2:count[i],k,l])-asinh.(dtdq2[i,2:count[i],k,l]));
    loglog(tt[i,2:count[i]]-tt[i,1],diff2,linestyle=":");
    loglog(tt[i,2:count[i]]-tt[i,1],diff4,linestyle="--");
#    loglog(tt[i,2:count[i]]-tt[i,1],diff3);
    println("xvs: ",i," ",k," ",l," asinh error h  : ",convert(Float64,maximum(diff2))); #read(STDIN,Char);
    println("els: ",i," ",k," ",l," asinh error h  : ",convert(Float64,maximum(diff4))); #read(STDIN,Char);
#    println(i," ",k," ",l," asinh error h/2: ",convert(Float64,maximum(diff3))); #read(STDIN,Char);
  end
end
loglog([1.0,40000.0],0.5e-16*([1.0,40000.0]/h).^1.5)

clf()
nmed = 10
for i=2:3, k=1:7, l=1:3
  if maximum(abs.(dtdq0[i,2:count[i],k,l])) > 0
    diff1 = abs.(asinh.(dtdq0_big[i,2:count[i],k,l])-asinh.(dtdq0[i,2:count[i],k,l]));
    diff3 = abs.(asinh.(dtdelements_big[i,2:count[i],k,l])-asinh.(dtdelements_mixed[i,2:count[i],k,l]))
    loglog(tt[i,2:count[i]]-tt[i,1],diff1,linestyle=":");
    loglog(tt[i,2:count[i]]-tt[i,1],diff3,linestyle="--");
    println("xvs: ",i," ",k," ",l," asinh error h  : ",convert(Float64,maximum(diff1))); #read(STDIN,Char);
    println("els: ",i," ",k," ",l," asinh error h  : ",convert(Float64,maximum(diff3))); #read(STDIN,Char);
  end
end
loglog([1.0,40000.0],0.5e-16*([1.0,40000.0]/h).^1.5)

@inbounds for i=1:n, j=1:count[i]
  @inbounds for k=1:n, l=1:7; dtdelements_mixed[i,j,l,k] = 0.0
    @inbounds for p=1:n, q=1:7
      dtdelements_mixed[i,j,l,k] += dtdq0[i,j,q,p]*convert(Float64,jac_init_big[(p-1)*7+q,(k-1)*7+l])
    end
  end
end

dtdelements_mixed_big = copy(dtdelements_big)
@inbounds for i=1:n, j=1:count[i]
  @inbounds for k=1:n, l=1:7; dtdelements_mixed_big[i,j,l,k] = 0.0
    @inbounds for p=1:n, q=1:7
      dtdelements_mixed_big[i,j,l,k] += dtdq0_big[i,j,q,p]*jac_init_big[(p-1)*7+q,(k-1)*7+l]
    end
  end
end


