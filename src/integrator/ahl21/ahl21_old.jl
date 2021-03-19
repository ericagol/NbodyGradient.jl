# Old versions of AHL21 methods

function ahl21!(x::Array{T,2},v::Array{T,2},xerror::Array{T,2},verror::Array{T,2},h::T,m::Array{T,1},n::Int64,jac_step::Array{T,2},jac_error::Array{T,2},pair::Array{Bool,2}) where {T <: Real}
    zilch = zero(T); uno = one(T); half = convert(T,0.5); two = convert(T,2.0)
    h2 = half*h; sevn = 7*n

    ## Replace with PreAllocArrays
    jac_phi = zeros(T,sevn,sevn)
    jac_kick = zeros(T,sevn,sevn)
    jac_copy = zeros(T,sevn,sevn)
    jac_ij = zeros(T,14,14)
    dqdt_ij = zeros(T,14)
    dqdt_phi = zeros(T,sevn)
    dqdt_kick = zeros(T,sevn)
    jac_tmp1 = zeros(T,14,sevn)
    jac_tmp2 = zeros(T,14,sevn)
    jac_err1 = zeros(T,14,sevn)
    ##

    drift!(x,v,xerror,verror,h2,n,jac_step,jac_error)
    kickfast!(x,v,xerror,verror,h/6,m,n,jac_kick,dqdt_kick,pair)
    # Multiply Jacobian from kick step:
    if T == BigFloat
        jac_copy .= *(jac_kick,jac_step)
    else
        BLAS.gemm!('N','N',uno,jac_kick,jac_step,zilch,jac_copy)
    end
    # Add back in the identity portion of the Jacobian with compensated summation:
    comp_sum_matrix!(jac_step,jac_error,jac_copy)
    indi = 0; indj = 0
    @inbounds for i=1:n-1
        indi = (i-1)*7
        @inbounds for j=i+1:n
            indj = (j-1)*7
            if ~pair[i,j]  # Check to see if kicks have not been applied
                kepler_driftij_gamma!(m,x,v,xerror,verror,i,j,h2,jac_ij,dqdt_ij,true)
                # Pick out indices for bodies i & j:
                @inbounds for k2=1:sevn, k1=1:7
                    jac_tmp1[k1,k2] = jac_step[ indi+k1,k2]
                    jac_err1[k1,k2] = jac_error[indi+k1,k2]
                end
                @inbounds for k2=1:sevn, k1=1:7
                    jac_tmp1[7+k1,k2] = jac_step[ indj+k1,k2]
                    jac_err1[7+k1,k2] = jac_error[indj+k1,k2]
                end
                # Carry out multiplication on the i/j components of matrix:
                if T == BigFloat
                    jac_tmp2 .= *(jac_ij,jac_tmp1)
                else
                    BLAS.gemm!('N','N',uno,jac_ij,jac_tmp1,zilch,jac_tmp2)
                end
                # Add back in the Jacobian with compensated summation:
                comp_sum_matrix!(jac_tmp1,jac_err1,jac_tmp2)
                # Copy back to the Jacobian:
                @inbounds for k2=1:sevn, k1=1:7
                    jac_step[ indi+k1,k2]=jac_tmp1[k1,k2]
                    jac_error[indi+k1,k2]=jac_err1[k1,k2]
                end
                @inbounds for k2=1:sevn, k1=1:7
                    jac_step[ indj+k1,k2]=jac_tmp1[7+k1,k2]
                    jac_error[indj+k1,k2]=jac_err1[7+k1,k2]
                end
            end
        end
    end
    phic!(x,v,xerror,verror,h,m,n,jac_phi,dqdt_phi,pair)
    phisalpha!(x,v,xerror,verror,h,m,two,n,jac_phi,dqdt_phi,pair) # 10%
    if T == BigFloat
        jac_copy .= *(jac_phi,jac_step)
    else
        BLAS.gemm!('N','N',uno,jac_phi,jac_step,zilch,jac_copy)
    end
    # Add back in the identity portion of the Jacobian with compensated summation:
    comp_sum_matrix!(jac_step,jac_error,jac_copy)
    indi=0; indj=0
    @inbounds for i=n-1:-1:1
        indi=(i-1)*7
        @inbounds for j=n:-1:i+1
            indj=(j-1)*7
            if ~pair[i,j]  # Check to see if kicks have not been applied
                kepler_driftij_gamma!(m,x,v,xerror,verror,i,j,h2,jac_ij,dqdt_ij,false)
                # Pick out indices for bodies i & j:
                # Carry out multiplication on the i/j components of matrix:
                @inbounds for k2=1:sevn, k1=1:7
                    jac_tmp1[k1,k2] = jac_step[ indi+k1,k2]
                    jac_err1[k1,k2] = jac_error[indi+k1,k2]
                end
                @inbounds for k2=1:sevn, k1=1:7
                    jac_tmp1[7+k1,k2] = jac_step[ indj+k1,k2]
                    jac_err1[7+k1,k2] = jac_error[indj+k1,k2]
                end
                # Carry out multiplication on the i/j components of matrix:
                if T == BigFloat
                    jac_tmp2 .= *(jac_ij,jac_tmp1)
                else
                    BLAS.gemm!('N','N',uno,jac_ij,jac_tmp1,zilch,jac_tmp2)
                end
                # Add back in the Jacobian with compensated summation:
                comp_sum_matrix!(jac_tmp1,jac_err1,jac_tmp2)
                # Copy back to the Jacobian:
                @inbounds for k2=1:sevn, k1=1:7
                    jac_step[ indi+k1,k2]=jac_tmp1[k1,k2]
                    jac_error[indi+k1,k2]=jac_err1[k1,k2]
                end
                @inbounds for k2=1:sevn, k1=1:7
                    jac_step[ indj+k1,k2]=jac_tmp1[7+k1,k2]
                    jac_error[indj+k1,k2]=jac_err1[7+k1,k2]
                end
            end
        end
    end
    #kickfast!(x,v,h2,m,n,jac_kick,dqdt_kick,pair)
    kickfast!(x,v,xerror,verror,h/6,m,n,jac_kick,dqdt_kick,pair)
    # Multiply Jacobian from kick step:
    if T == BigFloat
        jac_copy .= *(jac_kick,jac_step)
    else
        BLAS.gemm!('N','N',uno,jac_kick,jac_step,zilch,jac_copy)
    end
    # Add back in the identity portion of the Jacobian with compensated summation:
    comp_sum_matrix!(jac_step,jac_error,jac_copy)
    # Edit this routine to do compensated summation for Jacobian [x]
    drift!(x,v,xerror,verror,h2,n,jac_step,jac_error)
    return
end

function ahl21!(x::Array{T,2},v::Array{T,2},xerror::Array{T,2},verror::Array{T,2},h::T,m::Array{T,1},n::Int64,dqdt::Array{T,1},pair::Array{Bool,2}) where {T <: Real}
    # [Currently this routine is not giving the correct dqdt values. -EA 8/12/2019]
    zilch = zero(T); uno = one(T); half = convert(T,0.5); two = convert(T,2.0)
    h2 = half*h
    # This routine assumes that alpha = 0.0
    sevn = 7*n
    jac_phi = zeros(T,sevn,sevn)
    jac_kick = zeros(T,sevn,sevn)
    jac_ij = zeros(T,14,14)
    dqdt_ij = zeros(T,14)
    dqdt_phi = zeros(T,sevn)
    dqdt_kick = zeros(T,sevn)
    dqdt_tmp1 = zeros(T,14)
    dqdt_tmp2 = zeros(T,14)
    fill!(dqdt,zilch)
    #dqdt_save =copy(dqdt)
    drift!(x,v,xerror,verror,h2,n)
    # Compute time derivative of drift step:
    @inbounds for i=1:n, k=1:3
        dqdt[(i-1)*7+k] = half*v[k,i] + h2*dqdt[(i-1)*7+3+k]
    end
    #println("dqdt 1: ",dqdt-dqdt_save)
    #dqdt_save .= dqdt
    kickfast!(x,v,xerror,verror,h/6,m,n,jac_kick,dqdt_kick,pair)
    dqdt_kick /= 6 # Since step is h/6
    # Since I removed identity from kickfast, need to add in dqdt:
    dqdt .+= dqdt_kick + *(jac_kick,dqdt)
    #println("dqdt 2: ",dqdt-dqdt_save)
    #dqdt_save .= dqdt
    @inbounds for i=1:n-1
        indi = (i-1)*7
        @inbounds for j=i+1:n
            indj = (j-1)*7
            if ~pair[i,j]  # Check to see if kicks have not been applied
                #      kepler_driftij!(m,x,v,xerror,verror,i,j,h2,jac_ij,dqdt_ij,true)
                kepler_driftij_gamma!(m,x,v,xerror,verror,i,j,h2,jac_ij,dqdt_ij,true)
                # Copy current time derivatives for multiplication purposes:
                @inbounds for k1=1:7
                    dqdt_tmp1[  k1] = dqdt[indi+k1]
                    dqdt_tmp1[7+k1] = dqdt[indj+k1]
                end
                # Add in partial derivatives with respect to time:
                # Need to multiply by 1/2 since we're taking 1/2 time step:
                #    BLAS.gemm!('N','N',uno,jac_ij,dqdt_tmp1,half,dqdt_ij)
                dqdt_ij .*= half
                dqdt_ij .+= *(jac_ij,dqdt_tmp1) + dqdt_tmp1
                # Copy back time derivatives:
                @inbounds for k1=1:7
                    dqdt[indi+k1] = dqdt_ij[  k1]
                    dqdt[indj+k1] = dqdt_ij[7+k1]
                end
                #      println("dqdt 4: i: ",i," j: ",j," diff: ",dqdt-dqdt_save)
                #      dqdt_save .= dqdt
            end
        end
    end
    # Looks like we are missing phic! here: [ ]
    # Since I haven't added dqdt to phic yet, for now, set jac_phi equal to identity matrix
    # (since this is commented out within phisalpha):
    #jac_phi .= eye(T,sevn)
    phic!(x,v,xerror,verror,h,m,n,jac_phi,dqdt_phi,pair)
    phisalpha!(x,v,xerror,verror,h,m,two,n,jac_phi,dqdt_phi,pair) # 10%
    # Add in time derivative with respect to prior parameters:
    #BLAS.gemm!('N','N',uno,jac_phi,dqdt,uno,dqdt_phi)
    # Copy result to dqdt:
    dqdt .+= dqdt_phi + *(jac_phi,dqdt)
    #println("dqdt 5: ",dqdt-dqdt_save)
    #dqdt_save .= dqdt
    indi=0; indj=0
    @inbounds for i=n-1:-1:1
        indi=(i-1)*7
        @inbounds for j=n:-1:i+1
            if ~pair[i,j]  # Check to see if kicks have not been applied
                indj=(j-1)*7
                #      kepler_driftij!(m,x,v,xerror,verror,i,j,h2,jac_ij,dqdt_ij,false) # 23%
                kepler_driftij_gamma!(m,x,v,xerror,verror,i,j,h2,jac_ij,dqdt_ij,false) # 23%
                # Copy current time derivatives for multiplication purposes:
                @inbounds for k1=1:7
                    dqdt_tmp1[  k1] = dqdt[indi+k1]
                    dqdt_tmp1[7+k1] = dqdt[indj+k1]
                end
                # Add in partial derivatives with respect to time:
                # Need to multiply by 1/2 since we're taking 1/2 time step:
                #BLAS.gemm!('N','N',uno,jac_ij,dqdt_tmp1,half,dqdt_ij)
                dqdt_ij .*= half
                dqdt_ij .+= *(jac_ij,dqdt_tmp1) + dqdt_tmp1
                # Copy back time derivatives:
                @inbounds for k1=1:7
                    dqdt[indi+k1] = dqdt_ij[  k1]
                    dqdt[indj+k1] = dqdt_ij[7+k1]
                end
                #      dqdt_save .= dqdt
                #      println("dqdt 7: ",dqdt-dqdt_save)
                #      println("dqdt 6: i: ",i," j: ",j," diff: ",dqdt-dqdt_save)
                #      dqdt_save .= dqdt
            end
        end
    end
    fill!(dqdt_kick,zilch)
    kickfast!(x,v,xerror,verror,h/6,m,n,jac_kick,dqdt_kick,pair)
    dqdt_kick /= 6 # Since step is h/6
    # Copy result to dqdt:
    dqdt .+= dqdt_kick + *(jac_kick,dqdt)
    #println("dqdt 8: ",dqdt-dqdt_save)
    #dqdt_save .= dqdt
    drift!(x,v,xerror,verror,h2,n)
    # Compute time derivative of drift step:
    @inbounds for i=1:n, k=1:3
        dqdt[(i-1)*7+k] += half*v[k,i] + h2*dqdt[(i-1)*7+3+k]
    end
    #println("dqdt 9: ",dqdt-dqdt_save)
    return
end

function ahl21!(x::Array{T,2},v::Array{T,2},xerror::Array{T,2},verror::Array{T,2},h::T,m::Array{T,1},n::Int64,pair::Array{Bool,2}) where {T <: Real}
    h2 = 0.5*h
    drift!(x,v,xerror,verror,h2,n)
    kickfast!(x,v,xerror,verror,h/6,m,n,pair)
    @inbounds for i=1:n-1
        for j=i+1:n
            if ~pair[i,j]
                kepler_driftij_gamma!(m,x,v,xerror,verror,i,j,h2,true)
            end
        end
    end
    phic!(x,v,xerror,verror,h,m,n,pair)
    phisalpha!(x,v,xerror,verror,h,m,convert(T,2),n,pair)
    for i=n-1:-1:1
        for j=n:-1:i+1
            if ~pair[i,j]
                kepler_driftij_gamma!(m,x,v,xerror,verror,i,j,h2,false)
            end
        end
    end
    kickfast!(x,v,xerror,verror,h/6,m,n,pair)
    drift!(x,v,xerror,verror,h2,n)
    return
end

function drift!(x::Array{T,2},v::Array{T,2},xerror::Array{T,2},verror::Array{T,2},h::T,n::Int64,jac_step::Array{T,2},jac_error::Array{T,2}) where {T <: Real}
    indi = 0
    @inbounds for i=1:n
        indi = (i-1)*7
        for j=1:NDIM
            x[j,i],xerror[j,i] = comp_sum(x[j,i],xerror[j,i],h*v[j,i])
        end
        # Now for Jacobian:
        for k=1:7*n, j=1:NDIM
            jac_step[indi+j,k],jac_error[indi+j,k] = comp_sum(jac_step[indi+j,k],jac_error[indi+j,k],h*jac_step[indi+3+j,k])
        end
    end
    return
end

function drift!(x::Array{T,2},v::Array{T,2},xerror::Array{T,2},verror::Array{T,2},h::T,n::Int64) where {T <: Real}
    @inbounds for i=1:n, j=1:NDIM
        x[j,i],xerror[j,i] = comp_sum(x[j,i],xerror[j,i],h*v[j,i])
    end
    return
end

function kickfast!(x::Array{T,2},v::Array{T,2},xerror::Array{T,2},verror::Array{T,2},h::T,m::Array{T,1},n::Int64,jac_step::Array{T,2},dqdt_kick::Array{T,1},pair::Array{Bool,2}) where {T <: Real}
    rij = zeros(T,3)
    # Getting rid of identity since we will add that back in in calling routines:
    fill!(jac_step,zero(T))
    #jac_step.=eye(T,7*n)
    @inbounds for i=1:n-1
        indi = (i-1)*7
        for j=i+1:n
            indj = (j-1)*7
            if pair[i,j]
                for k=1:3
                    rij[k] = x[k,i] - x[k,j]
                end
                r2inv = 1.0/(rij[1]*rij[1]+rij[2]*rij[2]+rij[3]*rij[3])
                r3inv = r2inv*sqrt(r2inv)
                for k=1:3
                    fac = h*GNEWT*rij[k]*r3inv
                    # Apply impulses:
                    #v[k,i] -= m[j]*fac
                    v[k,i],verror[k,i] = comp_sum(v[k,i],verror[k,i],-m[j]*fac)
                    #v[k,j] += m[i]*fac
                    v[k,j],verror[k,j] = comp_sum(v[k,j],verror[k,j], m[i]*fac)
                    # Compute time derivative:
                    dqdt_kick[indi+3+k] -= m[j]*fac/h
                    dqdt_kick[indj+3+k] += m[i]*fac/h
                    # Computing the derivative
                    # Mass derivative of acceleration vector (10/6/17 notes):
                    # Impulse of ith particle depends on mass of jth particle:
                    jac_step[indi+3+k,indj+7] -= fac
                    # Impulse of jth particle depends on mass of ith particle:
                    jac_step[indj+3+k,indi+7] += fac
                    # x derivative of acceleration vector:
                    fac *= 3.0*r2inv
                    # Dot product x_ij.\delta x_ij means we need to sum over components:
                    for p=1:3
                        jac_step[indi+3+k,indi+p] += fac*m[j]*rij[p]
                        jac_step[indi+3+k,indj+p] -= fac*m[j]*rij[p]
                        jac_step[indj+3+k,indj+p] += fac*m[i]*rij[p]
                        jac_step[indj+3+k,indi+p] -= fac*m[i]*rij[p]
                    end
                    # Final term has no dot product, so just diagonal:
                    fac = h*GNEWT*r3inv
                    jac_step[indi+3+k,indi+k] -= fac*m[j]
                    jac_step[indi+3+k,indj+k] += fac*m[j]
                    jac_step[indj+3+k,indj+k] -= fac*m[i]
                    jac_step[indj+3+k,indi+k] += fac*m[i]
                end
            end
        end
    end
    return
end

function kickfast!(x::Array{T,2},v::Array{T,2},xerror::Array{T,2},verror::Array{T,2},h::T,m::Array{T,1},n::Int64,pair::Array{Bool,2}) where {T <: Real}
    rij = zeros(T,3)
    @inbounds for i=1:n-1
        for j = i+1:n
            if pair[i,j]
                r2 = zero(T)
                for k=1:3
                    rij[k] = x[k,i] - x[k,j]
                    r2 += rij[k]*rij[k]
                end
                r3_inv = 1.0/(r2*sqrt(r2))
                for k=1:3
                    fac = h*GNEWT*rij[k]*r3_inv
                    v[k,i],verror[k,i] = comp_sum(v[k,i],verror[k,i],-m[j]*fac)
                    v[k,j],verror[k,j] = comp_sum(v[k,j],verror[k,j], m[i]*fac)
                end
            end
        end
    end
    return
end

function phic!(x::Array{T,2},v::Array{T,2},xerror::Array{T,2},verror::Array{T,2},h::T,m::Array{T,1},n::Int64,jac_step::Array{T,2},dqdt_phi::Array{T,1},pair::Array{Bool,2}) where {T <: Real}
    a = zeros(T,3,n)
    rij = zeros(T,3)
    aij = zeros(T,3)
    dadq = zeros(T,3,n,4,n)  # There is no velocity dependence
    dotdadq = zeros(T,4,n)  # There is no velocity dependence
    # Set jac_step to zeros:
    #jac_step.=eye(T,7*n)
    fill!(jac_step,zero(T))
    fac = 0.0; fac1 = 0.0; fac2 = 0.0; fac3 = 0.0; r1 = 0.0; r2 = 0.0; r3 = 0.0
    coeff = h^3/36*GNEWT
    @inbounds for i=1:n-1
        indi = (i-1)*7
        for j=i+1:n
            if pair[i,j]
                indj = (j-1)*7
                for k=1:3
                    rij[k] = x[k,i] - x[k,j]
                end
                r2inv = inv(dot(rij,rij))
                r3inv = r2inv*sqrt(r2inv)
                for k=1:3
                    # Apply impulses:
                    fac = GNEWT*rij[k]*r3inv
                    facv = fac*2*h/3
                    #v[k,i] -= m[j]*facv
                    v[k,i],verror[k,i] = comp_sum(v[k,i],verror[k,i],-m[j]*facv)
                    #v[k,j] += m[i]*facv
                    v[k,j],verror[k,j] = comp_sum(v[k,j],verror[k,j],m[i]*facv)
                    # Compute time derivative:
                    dqdt_phi[indi+3+k] -= 3/h*m[j]*facv
                    dqdt_phi[indj+3+k] += 3/h*m[i]*facv
                    a[k,i] -= m[j]*fac
                    a[k,j] += m[i]*fac
                    # Impulse of ith particle depends on mass of jth particle:
                    jac_step[indi+3+k,indj+7] -= facv
                    # Impulse of jth particle depends on mass of ith particle:
                    jac_step[indj+3+k,indi+7] += facv
                    # x derivative of acceleration vector:
                    facv *= 3.0*r2inv
                    # Dot product x_ij.\delta x_ij means we need to sum over components:
                    for p=1:3
                        jac_step[indi+3+k,indi+p] += facv*m[j]*rij[p]
                        jac_step[indi+3+k,indj+p] -= facv*m[j]*rij[p]
                        jac_step[indj+3+k,indj+p] += facv*m[i]*rij[p]
                        jac_step[indj+3+k,indi+p] -= facv*m[i]*rij[p]
                    end
                    # Final term has no dot product, so just diagonal:
                    facv = 2h/3*GNEWT*r3inv
                    jac_step[indi+3+k,indi+k] -= facv*m[j]
                    jac_step[indi+3+k,indj+k] += facv*m[j]
                    jac_step[indj+3+k,indj+k] -= facv*m[i]
                    jac_step[indj+3+k,indi+k] += facv*m[i]
                    # Mass derivative of acceleration vector (10/6/17 notes):
                    # Since there is no velocity dependence, this is fourth parameter.
                    # Acceleration of ith particle depends on mass of jth particle:
                    dadq[k,i,4,j] -= fac
                    dadq[k,j,4,i] += fac
                    # x derivative of acceleration vector:
                    fac *= 3.0*r2inv
                    # Dot product x_ij.\delta x_ij means we need to sum over components:
                    for p=1:3
                        dadq[k,i,p,i] += fac*m[j]*rij[p]
                        dadq[k,i,p,j] -= fac*m[j]*rij[p]
                        dadq[k,j,p,j] += fac*m[i]*rij[p]
                        dadq[k,j,p,i] -= fac*m[i]*rij[p]
                    end
                    # Final term has no dot product, so just diagonal:
                    fac = GNEWT*r3inv
                    dadq[k,i,k,i] -= fac*m[j]
                    dadq[k,i,k,j] += fac*m[j]
                    dadq[k,j,k,j] -= fac*m[i]
                    dadq[k,j,k,i] += fac*m[i]
                end
            end
        end
    end
    # Next, compute g_i acceleration vector.
    # Note that jac_step[(i-1)*7+k,(j-1)*7+p] is the derivative of the kth coordinate
    # of planet i with respect to the pth coordinate of planet j.
    indi = 0; indj=0; indd = 0
    @inbounds for i=1:n-1
        indi = (i-1)*7
        for j=i+1:n
            if pair[i,j] # correction for Kepler pairs
                indj = (j-1)*7
                for k=1:3
                    aij[k] = a[k,i] - a[k,j]
                    rij[k] = x[k,i] - x[k,j]
                end
                # Compute dot product of r_ij with \delta a_ij:
                fill!(dotdadq,0.0)
                @inbounds for d=1:n, p=1:4, k=1:3
                    dotdadq[p,d] += rij[k]*(dadq[k,i,p,d]-dadq[k,j,p,d])
                end
                r2 = dot(rij,rij)
                r1 = sqrt(r2)
                ardot = dot(aij,rij)
                fac1 = coeff/r1^5
                fac2 = 3*ardot
                for k=1:3
                    fac = fac1*(rij[k]*fac2- r2*aij[k])
                    #v[k,i] += m[j]*fac
                    v[k,i],verror[k,i] = comp_sum(v[k,i],verror[k,i],m[j]*fac)
                    #v[k,j] -= m[i]*fac
                    v[k,j],verror[k,j] = comp_sum(v[k,j],verror[k,j],-m[i]*fac)
                    # Mass derivative (first part is easy):
                    jac_step[indi+3+k,indj+7] += fac
                    jac_step[indj+3+k,indi+7] -= fac
                    # Position derivatives:
                    fac *= 5.0/r2
                    for p=1:3
                        jac_step[indi+3+k,indi+p] -= fac*m[j]*rij[p]
                        jac_step[indi+3+k,indj+p] += fac*m[j]*rij[p]
                        jac_step[indj+3+k,indj+p] -= fac*m[i]*rij[p]
                        jac_step[indj+3+k,indi+p] += fac*m[i]*rij[p]
                    end
                    # Diagonal position terms:
                    fac = fac1*fac2
                    jac_step[indi+3+k,indi+k] += fac*m[j]
                    jac_step[indi+3+k,indj+k] -= fac*m[j]
                    jac_step[indj+3+k,indj+k] += fac*m[i]
                    jac_step[indj+3+k,indi+k] -= fac*m[i]
                    # Dot product \delta rij terms:
                    fac = -2*fac1*aij[k]
                    for p=1:3
                        fac3 = fac*rij[p] + fac1*3.0*rij[k]*aij[p]
                        jac_step[indi+3+k,indi+p] += m[j]*fac3
                        jac_step[indi+3+k,indj+p] -= m[j]*fac3
                        jac_step[indj+3+k,indj+p] += m[i]*fac3
                        jac_step[indj+3+k,indi+p] -= m[i]*fac3
                    end
                    # Diagonal acceleration terms:
                    fac = -fac1*r2
                    # Duoh.  For dadq, have to loop over all other parameters!
                    @inbounds for d=1:n
                        indd = (d-1)*7
                        for p=1:3
                            jac_step[indi+3+k,indd+p] += fac*m[j]*(dadq[k,i,p,d]-dadq[k,j,p,d])
                            jac_step[indj+3+k,indd+p] -= fac*m[i]*(dadq[k,i,p,d]-dadq[k,j,p,d])
                        end
                        # Don't forget mass-dependent term:
                        jac_step[indi+3+k,indd+7] += fac*m[j]*(dadq[k,i,4,d]-dadq[k,j,4,d])
                        jac_step[indj+3+k,indd+7] -= fac*m[i]*(dadq[k,i,4,d]-dadq[k,j,4,d])
                    end
                    # Now, for the final term:  (\delta a_ij . r_ij ) r_ij
                    fac = 3.0*fac1*rij[k]
                    @inbounds for d=1:n
                        indd = (d-1)*7
                        for p=1:3
                            jac_step[indi+3+k,indd+p] += fac*m[j]*dotdadq[p,d]
                            jac_step[indj+3+k,indd+p] -= fac*m[i]*dotdadq[p,d]
                        end
                        jac_step[indi+3+k,indd+7] += fac*m[j]*dotdadq[4,d]
                        jac_step[indj+3+k,indd+7] -= fac*m[i]*dotdadq[4,d]
                    end
                end
            end
        end
    end
    return
end

function phic!(x::Array{T,2},v::Array{T,2},xerror::Array{T,2},verror::Array{T,2},h::T,m::Array{T,1},n::Int64,pair::Array{Bool,2}) where {T <: Real}
    a = zeros(T,3,n)
    rij = zeros(T,3)
    aij = zeros(T,3)
    @inbounds for i=1:n-1, j = i+1:n
        if pair[i,j] # kick group
            r2 = 0.0
            for k=1:3
                rij[k] = x[k,i] - x[k,j]
                r2 += rij[k]^2
            end
            r3_inv = 1.0/(r2*sqrt(r2))
            for k=1:3
                fac = GNEWT*rij[k]*r3_inv
                facv = fac*2*h/3
                v[k,i],verror[k,i] = comp_sum(v[k,i],verror[k,i],-m[j]*facv)
                v[k,j],verror[k,j] = comp_sum(v[k,j],verror[k,j],m[i]*facv)
                a[k,i] -= m[j]*fac
                a[k,j] += m[i]*fac
            end
        end
    end
    coeff = h^3/36*GNEWT
    @inbounds for i=1:n-1 ,j=i+1:n
        if pair[i,j] # kick group
            for k=1:3
                aij[k] = a[k,i] - a[k,j]
                rij[k] = x[k,i] - x[k,j]
            end
            r2 = dot(rij,rij)
            r5inv = 1.0/(r2^2*sqrt(r2))
            ardot = dot(aij,rij)
            for k=1:3
                fac = coeff*r5inv*(rij[k]*3*ardot-r2*aij[k])
                v[k,i],verror[k,i] = comp_sum(v[k,i],verror[k,i],m[j]*fac)
                v[k,j],verror[k,j] = comp_sum(v[k,j],verror[k,j],-m[i]*fac)
            end
        end
    end
    return
end

function phisalpha!(x::Array{T,2},v::Array{T,2},xerror::Array{T,2},verror::Array{T,2},h::T,m::Array{T,1},alpha::T,n::Int64,jac_step::Array{T,2},dqdt_phi::Array{T,1},pair::Array{Bool,2}) where {T <: Real}
    a = zeros(T,3,n)
    dadq = zeros(T,3,n,4,n)  # There is no velocity dependence
    dotdadq = zeros(T,4,n)  # There is no velocity dependence
    rij = zeros(T,3)
    aij = zeros(T,3)
    coeff = alpha*h^3/96*2*GNEWT
    fac = 0.0; fac1 = 0.0; fac2 = 0.0; fac3 = 0.0; r1 = 0.0; r2 = 0.0; r3 = 0.0
    @inbounds for i=1:n-1
        indi = (i-1)*7
        for j=i+1:n
            if ~pair[i,j] # correction for Kepler pairs
                indj = (j-1)*7
                for k=1:3
                    rij[k] = x[k,i] - x[k,j]
                end
                r2 = rij[1]*rij[1]+rij[2]*rij[2]+rij[3]*rij[3]
                r3 = r2*sqrt(r2)
                for k=1:3
                    fac = GNEWT*rij[k]/r3
                    a[k,i] -= m[j]*fac
                    a[k,j] += m[i]*fac
                    # Mass derivative of acceleration vector (10/6/17 notes):
                    # Since there is no velocity dependence, this is fourth parameter.
                    # Acceleration of ith particle depends on mass of jth particle:
                    dadq[k,i,4,j] -= fac
                    dadq[k,j,4,i] += fac
                    # x derivative of acceleration vector:
                    fac *= 3.0/r2
                    # Dot product x_ij.\delta x_ij means we need to sum over components:
                    for p=1:3
                        dadq[k,i,p,i] += fac*m[j]*rij[p]
                        dadq[k,i,p,j] -= fac*m[j]*rij[p]
                        dadq[k,j,p,j] += fac*m[i]*rij[p]
                        dadq[k,j,p,i] -= fac*m[i]*rij[p]
                    end
                    # Final term has no dot product, so just diagonal:
                    fac = GNEWT/r3
                    dadq[k,i,k,i] -= fac*m[j]
                    dadq[k,i,k,j] += fac*m[j]
                    dadq[k,j,k,j] -= fac*m[i]
                    dadq[k,j,k,i] += fac*m[i]
                end
            end
        end
    end
    # Next, compute \tilde g_i acceleration vector (this is rewritten
    # slightly to avoid reference to \tilde a_i):
    # Note that jac_step[(i-1)*7+k,(j-1)*7+p] is the derivative of the kth coordinate
    # of planet i with respect to the pth coordinate of planet j.
    indi = 0; indj=0; indd = 0
    @inbounds for i=1:n-1
        indi = (i-1)*7
        for j=i+1:n
            if ~pair[i,j] # correction for Kepler pairs
                indj = (j-1)*7
                for k=1:3
                    aij[k] = a[k,i] - a[k,j]
                    rij[k] = x[k,i] - x[k,j]
                end
                # Compute dot product of r_ij with \delta a_ij:
                fill!(dotdadq,0.0)
                @inbounds for d=1:n, p=1:4, k=1:3
                    dotdadq[p,d] += rij[k]*(dadq[k,i,p,d]-dadq[k,j,p,d])
                end
                r2 = rij[1]*rij[1]+rij[2]*rij[2]+rij[3]*rij[3]
                r1 = sqrt(r2)
                ardot = aij[1]*rij[1]+aij[2]*rij[2]+aij[3]*rij[3]
                fac1 = coeff/r1^5
                fac2 = (2*GNEWT*(m[i]+m[j])/r1 + 3*ardot)
                for k=1:3
                    fac = fac1*(rij[k]*fac2- r2*aij[k])
                    v[k,i],verror[k,i] = comp_sum(v[k,i],verror[k,i], m[j]*fac)
                    v[k,j],verror[k,j] = comp_sum(v[k,j],verror[k,j],-m[i]*fac)
                    # Compute time derivative:
                    dqdt_phi[indi+3+k] += 3/h*m[j]*fac
                    dqdt_phi[indj+3+k] -= 3/h*m[i]*fac
                    # Mass derivative (first part is easy):
                    jac_step[indi+3+k,indj+7] += fac
                    jac_step[indj+3+k,indi+7] -= fac
                    # Position derivatives:
                    fac *= 5.0/r2
                    for p=1:3
                        jac_step[indi+3+k,indi+p] -= fac*m[j]*rij[p]
                        jac_step[indi+3+k,indj+p] += fac*m[j]*rij[p]
                        jac_step[indj+3+k,indj+p] -= fac*m[i]*rij[p]
                        jac_step[indj+3+k,indi+p] += fac*m[i]*rij[p]
                    end
                    # Second mass derivative:
                    fac = 2*GNEWT*fac1*rij[k]/r1
                    jac_step[indi+3+k,indi+7] += fac*m[j]
                    jac_step[indi+3+k,indj+7] += fac*m[j]
                    jac_step[indj+3+k,indj+7] -= fac*m[i]
                    jac_step[indj+3+k,indi+7] -= fac*m[i]
                    #  (There's also a mass term in dadq [x]. See below.)
                    # Diagonal position terms:
                    fac = fac1*fac2
                    jac_step[indi+3+k,indi+k] += fac*m[j]
                    jac_step[indi+3+k,indj+k] -= fac*m[j]
                    jac_step[indj+3+k,indj+k] += fac*m[i]
                    jac_step[indj+3+k,indi+k] -= fac*m[i]
                    # Dot product \delta rij terms:
                    fac = -2*fac1*(rij[k]*GNEWT*(m[i]+m[j])/(r2*r1)+aij[k])
                    for p=1:3
                        fac3 = fac*rij[p] + fac1*3.0*rij[k]*aij[p]
                        jac_step[indi+3+k,indi+p] += m[j]*fac3
                        jac_step[indi+3+k,indj+p] -= m[j]*fac3
                        jac_step[indj+3+k,indj+p] += m[i]*fac3
                        jac_step[indj+3+k,indi+p] -= m[i]*fac3
                    end
                    # Diagonal acceleration terms:
                    fac = -fac1*r2
                    # Duoh.  For dadq, have to loop over all other parameters!
                    @inbounds for d=1:n
                        indd = (d-1)*7
                        for p=1:3
                            jac_step[indi+3+k,indd+p] += fac*m[j]*(dadq[k,i,p,d]-dadq[k,j,p,d])
                            jac_step[indj+3+k,indd+p] -= fac*m[i]*(dadq[k,i,p,d]-dadq[k,j,p,d])
                        end
                        # Don't forget mass-dependent term:
                        jac_step[indi+3+k,indd+7] += fac*m[j]*(dadq[k,i,4,d]-dadq[k,j,4,d])
                        jac_step[indj+3+k,indd+7] -= fac*m[i]*(dadq[k,i,4,d]-dadq[k,j,4,d])
                    end
                    # Now, for the final term:  (\delta a_ij . r_ij ) r_ij
                    fac = 3.0*fac1*rij[k]
                    @inbounds for d=1:n
                        indd = (d-1)*7
                        for p=1:3
                            jac_step[indi+3+k,indd+p] += fac*m[j]*dotdadq[p,d]
                            jac_step[indj+3+k,indd+p] -= fac*m[i]*dotdadq[p,d]
                        end
                        jac_step[indi+3+k,indd+7] += fac*m[j]*dotdadq[4,d]
                        jac_step[indj+3+k,indd+7] -= fac*m[i]*dotdadq[4,d]
                    end
                end
            end
        end
    end
    return
end

function phisalpha!(x::Array{T,2},v::Array{T,2},xerror::Array{T,2},verror::Array{T,2},h::T,m::Array{T,1},alpha::T,n::Int64,pair::Array{Bool,2}) where {T <: Real}
    a = zeros(T,3,n)
    rij = zeros(T,3)
    aij = zeros(T,3)
    coeff = alpha*h^3/96*2*GNEWT

    fac = zero(T); fac1 = zero(T); fac2 = zero(T); r1 = zero(T); r2 = zero(T); r3 = zero(T)
    @inbounds for i=1:n-1
        for j = i+1:n
            if ~pair[i,j] # correction for Kepler pairs
                for k=1:3
                    rij[k] = x[k,i] - x[k,j]
                end
                r2 = rij[1]*rij[1]+rij[2]*rij[2]+rij[3]*rij[3]
                r3 = r2*sqrt(r2)
                for k=1:3
                    fac = GNEWT*rij[k]/r3
                    a[k,i] -= m[j]*fac
                    a[k,j] += m[i]*fac
                end
            end
        end
    end
    # Next, compute \tilde g_i acceleration vector (this is rewritten
    # slightly to avoid reference to \tilde a_i):
    @inbounds for i=1:n-1
        for j=i+1:n
            if ~pair[i,j] # correction for Kepler pairs
                for k=1:3
                    aij[k] = a[k,i] - a[k,j]
                    rij[k] = x[k,i] - x[k,j]
                end
                r2 = rij[1]*rij[1]+rij[2]*rij[2]+rij[3]*rij[3]
                r1 = sqrt(r2)
                ardot = aij[1]*rij[1]+aij[2]*rij[2]+aij[3]*rij[3]
                fac1 = coeff/r1^5
                fac2 = (2*GNEWT*(m[i]+m[j])/r1 + 3*ardot)
                for k=1:3
                    fac = fac1*(rij[k]*fac2- r2*aij[k])
                    v[k,i],verror[k,i] = comp_sum(v[k,i],verror[k,i], m[j]*fac)
                    v[k,j],verror[k,j] = comp_sum(v[k,j],verror[k,j],-m[i]*fac)
                end
            end
        end
    end
    return
end

function kepler_driftij_gamma!(m::Array{T,1},x::Array{T,2},v::Array{T,2},xerror::Array{T,2},verror::Array{T,2},i::Int64,j::Int64,h::T,jac_ij::Array{T,2},dqdt::Array{T,1},drift_first::Bool) where {T <: Real}
    # Initial state:
    x0 = zeros(T,NDIM) # x0 = positions of body i relative to j
    v0 = zeros(T,NDIM) # v0 = velocities of body i relative to j
    @inbounds for k=1:NDIM
        x0[k] = x[k,i] - x[k,j]
        v0[k] = v[k,i] - v[k,j]
    end
    gm = GNEWT*(m[i]+m[j])
    # jac_ij should be the Jacobian for going from (x_{0,i},v_{0,i},m_i) &  (x_{0,j},v_{0,j},m_j)
    # to  (x_i,v_i,m_i) &  (x_j,v_j,m_j), a 14x14 matrix for the 3-dimensional case.
    # Fill with zeros for now:
    #jac_ij .= eye(T,14)
    fill!(jac_ij,zero(T))
    if gm == 0
        #  Do nothing
        #  for k=1:3
        #    x[k,i] += h*v[k,i]
        #    x[k,j] += h*v[k,j]
        #  end
    else
        delxv = zeros(T,6)
        jac_kepler = zeros(T,6,8)
        jac_mass = zeros(T,6)
        jac_delxv_gamma!(x0,v0,gm,h,drift_first,delxv,jac_kepler,jac_mass,false)

        #  kepler_drift_step!(gm, h, state0, state,jac_kepler,drift_first)
        mijinv::T =one(T)/(m[i] + m[j])
        mi::T = m[i]*mijinv # Normalize the masses
        mj::T = m[j]*mijinv
        @inbounds for k=1:3
            # Add kepler-drift differences, weighted by masses, to start of step:
            x[k,i],xerror[k,i] = comp_sum(x[k,i],xerror[k,i], mj*delxv[k])
            x[k,j],xerror[k,j] = comp_sum(x[k,j],xerror[k,j],-mi*delxv[k])
        end
        @inbounds for k=1:3
            v[k,i],verror[k,i] = comp_sum(v[k,i],verror[k,i], mj*delxv[3+k])
            v[k,j],verror[k,j] = comp_sum(v[k,j],verror[k,j],-mi*delxv[3+k])
        end
        # Compute Jacobian:
        @inbounds for l=1:6, k=1:6
            # Compute derivatives of x_i,v_i with respect to initial conditions:
            jac_ij[  k,  l] += mj*jac_kepler[k,l]
            jac_ij[  k,7+l] -= mj*jac_kepler[k,l]
            # Compute derivatives of x_j,v_j with respect to initial conditions:
            jac_ij[7+k,  l] -= mi*jac_kepler[k,l]
            jac_ij[7+k,7+l] += mi*jac_kepler[k,l]
        end
        @inbounds for k=1:6
            # Compute derivatives of x_i,v_i with respect to the masses:
            #    println("Old dxv/dm_i: ",-mj*delxv[k]*mijinv + GNEWT*mj*jac_kepler[  k,7])
            #    jac_ij[   k, 7] = -mj*delxv[k]*mijinv + GNEWT*mj*jac_kepler[  k,7]
            #    println("New dx/dm_i: ",jac_mass[k]*m[j])
            jac_ij[   k, 7] = jac_mass[k]*m[j]
            jac_ij[   k,14] =  mi*delxv[k]*mijinv + GNEWT*mj*jac_kepler[  k,7]
            # Compute derivatives of x_j,v_j with respect to the masses:
            jac_ij[ 7+k, 7] = -mj*delxv[k]*mijinv - GNEWT*mi*jac_kepler[  k,7]
            #    println("Old dxv/dm_j: ",mi*delxv[k]*mijinv - GNEWT*mi*jac_kepler[  k,7])
            #    jac_ij[ 7+k,14] =  mi*delxv[k]*mijinv - GNEWT*mi*jac_kepler[  k,7]
            #     println("New dxv/dm_j: ",-jac_mass[k]*m[i])
            jac_ij[ 7+k,14] = -jac_mass[k]*m[i]
        end
    end
    # The following lines are meant to compute dq/dt for kepler_driftij,
    # but they currently contain an error (test doesn't pass in test_ahl21.jl). [ ]
    @inbounds for k=1:6
        # Position/velocity derivative, body i:
        dqdt[  k] =  mj*jac_kepler[k,8]
        # Position/velocity derivative, body j:
        dqdt[7+k] = -mi*jac_kepler[k,8]
    end
    return
end

function kepler_driftij_gamma!(m::Array{T,1},x::Array{T,2},v::Array{T,2},xerror::Array{T,2},verror::Array{T,2},i::Int64,j::Int64,h::T,drift_first::Bool) where {T <: Real}
    x0 = zeros(T,NDIM) # x0 = positions of body i relative to j
    v0 = zeros(T,NDIM) # v0 = velocities of body i relative to j
    @inbounds for k=1:NDIM
        x0[k] = x[k,i] - x[k,j]
        v0[k] = v[k,i] - v[k,j]
    end
    gm = GNEWT*(m[i]+m[j])
    if gm == 0
        #  Do nothing
        #  for k=1:3
        #    x[k,i] += h*v[k,i]
        #    x[k,j] += h*v[k,j]
        #  end
    else
        # Compute differences in x & v over time step:
        delxv = jac_delxv_gamma!(x0,v0,gm,h,drift_first)
        mijinv =1.0/(m[i] + m[j])
        mi = m[i]*mijinv # Normalize the masses
        mj = m[j]*mijinv
        @inbounds for k=1:3
            # Add kepler-drift differences, weighted by masses, to start of step:
            x[k,i],xerror[k,i] = comp_sum(x[k,i],xerror[k,i], mj*delxv[k])
            x[k,j],xerror[k,j] = comp_sum(x[k,j],xerror[k,j],-mi*delxv[k])
            v[k,i],verror[k,i] = comp_sum(v[k,i],verror[k,i], mj*delxv[3+k])
            v[k,j],verror[k,j] = comp_sum(v[k,j],verror[k,j],-mi*delxv[3+k])
        end
    end
    return
end

function jac_delxv_gamma!(x0::Array{T,1},v0::Array{T,1},k::T,h::T,drift_first::Bool;grad::Bool=false,auto::Bool=false,dlnq::T=convert(T,0.0),debug=false) where {T <: Real}
    # Using autodiff, computes Jacobian of delx & delv with respect to x0, v0, k & h.

    # Autodiff requires a single-vector input, so create an array to hold the independent variables:
    input = zeros(T,8)
    input[1:3]=x0; input[4:6]=v0; input[7]=k; input[8]=h
    if grad
        if debug
            # Also output gamma, r, fm1, dfdt, gmh, dgdtm1, and for debugging:
            delxv_jac = zeros(T,12,8)
        else
            # Output \delta x & \delta v only:
            delxv_jac = zeros(T,6,8)
        end
    end

    # Create a closure so that the function knows value of drift_first:

    function delx_delv(input::Array{T2,1}) where {T2 <: Real} # input = x0,v0,k,h,drift_first
        # Compute delx and delv from h, s, k, beta0, x0 and v0:
        x0 = input[1:3]; v0 = input[4:6]; k = input[7]; h = input[8]
        # Compute r0:
        drift_first ?  r0 = norm(x0-h*v0) : r0 = norm(x0)
        # And its inverse:
        r0inv = inv(r0)
        # Compute beta_0:
        beta0 = 2k*r0inv-dot(v0,v0)
        beta0inv = inv(beta0)
        signb = sign(beta0)
        sqb = sqrt(signb*beta0)
        zeta = k-r0*beta0
        gamma_guess = zero(T2)
        # Compute \eta_0 = x_0 . v_0:
        drift_first ?  eta = dot(x0-h*v0,v0) : eta = dot(x0,v0)
        if zeta != zero(T2)
            # Make sure we have a cubic in gamma (and don't divide by zero):
            gamma_guess = cubic1(3eta*sqb/zeta,6r0*signb*beta0/zeta,-6h*signb*beta0*sqb/zeta)
        else
            # Check that we have a quadratic in gamma (and don't divide by zero):
            if eta != zero(T2)
                reta = r0/eta
                disc = reta^2+2h/eta
                disc > zero(T2) ?  gamma_guess = sqb*(-reta+sqrt(disc)) : gamma_guess = h*r0inv*sqb
            else
                gamma_guess = h*r0inv*sqb
            end
        end
        gamma  = copy(gamma_guess)
        # Make sure prior two steps differ:
        gamma1 = 2*copy(gamma)
        gamma2 = 3*copy(gamma)
        iter = 0
        ITMAX = 20
        # Compute coefficients: (8/28/19 notes)
        #  c1 = k; c2 = -zeta; c3 = -eta*sqb; c4 = sqb*(eta-h*beta0); c5 = eta*signb*sqb
        c1 = k; c2 = -2zeta; c3 = 2*eta*signb*sqb; c4 = -sqb*h*beta0; c5 = 2eta*signb*sqb
        # Solve for gamma:
        while true
            gamma2 = gamma1
            gamma1 = gamma
            xx = 0.5*gamma
            #    xx = gamma
            if beta0 > 0
                sx = sin(xx); cx = cos(xx)
            else
                sx = sinh(xx); cx = exp(-xx)+sx
            end
            #    gamma -= (c1*gamma+c2*sx+c3*cx+c4)/(c2*cx+c5*sx+c1)
            gamma -= (k*gamma+c2*sx*cx+c3*sx^2+c4)/(2signb*zeta*sx^2+c5*sx*cx+r0*beta0)
            iter +=1
            if iter >= ITMAX || gamma == gamma2 || gamma == gamma1
                break
            end
        end
        #  if typeof(gamma) ==  Float64
        #    println("s: ",gamma/sqb)
        #  end
        # Set up a single output array for delx and delv:
        if debug
            delxv = zeros(T2,12)
        else
            delxv = zeros(T2,6)
        end
        # Since we updated gamma, need to recompute:
        xx = 0.5*gamma
        if beta0 > 0
            sx = sin(xx); cx = cos(xx)
        else
            sx = sinh(xx); cx = exp(-xx)+sx
        end
        # Now, compute final values.  Compute Wisdom/Hernandez G_i^\beta(s) functions:
        g1bs = 2sx*cx/sqb
        g2bs = 2signb*sx^2*beta0inv
        g0bs = one(T2)-beta0*g2bs
        g3bs = G3(gamma,beta0)
        h1 = zero(T2); h2 = zero(T2)
        #  if typeof(g1bs) == Float64
        #    println("g1: ",g1bs," g2: ",g2bs," g3: ",g3bs)
        #  end
        # Compute r from equation (35):
        r = r0*g0bs+eta*g1bs+k*g2bs
        #  if typeof(r) == Float64
        #    println("r: ",r)
        #  end
        rinv = inv(r)
        dfdt = -k*g1bs*rinv*r0inv # [x]
        if drift_first
            # Drift backwards before Kepler step: (1/22/2018)
            fm1 = -k*r0inv*g2bs # [x]
            # This is given in 2/7/2018 notes: g-h*f
            #    gmh = k*r0inv*(r0*(g1bs*g2bs-g3bs)+eta*g2bs^2+k*g3bs*g2bs)  # [x]
            #    println("Old gmh: ",gmh," new gmh: ",k*r0inv*(h*g2bs-r0*g3bs))  # [x]
            gmh = k*r0inv*(h*g2bs-r0*g3bs)  # [x]
        else
            # Drift backwards after Kepler step: (1/24/2018)
            # The following line is f-1-h fdot:
            h1= H1(gamma,beta0); h2= H2(gamma,beta0)
            #    fm1 =  k*rinv*(g2bs-k*r0inv*H1(gamma,beta0))  # [x]
            fm1 =  k*rinv*(g2bs-k*r0inv*h1)  # [x]
            # This is g-h*dgdt
            #    gmh = k*rinv*(r0*H2(gamma,beta0)+eta*H1(gamma,beta0)) # [x]
            gmh = k*rinv*(r0*h2+eta*h1) # [x]
        end
        # Compute velocity component functions:
        if drift_first
            # This is gdot - h fdot - 1:
            #    dgdtm1 = k*r0inv*rinv*(r0*g0bs*g2bs+eta*g1bs*g2bs+k*g1bs*g3bs) # [x]
            #    println("gdot-h fdot-1: ",dgdtm1," alternative expression: ",k*r0inv*rinv*(h*g1bs-r0*g2bs))
            dgdtm1 = k*r0inv*rinv*(h*g1bs-r0*g2bs)
        else
            # This is gdot - 1:
            dgdtm1 = -k*rinv*g2bs # [x]
        end
        #  if typeof(fm1) == Float64
        #    println("fm1: ",fm1," dfdt: ",dfdt," gmh: ",gmh," dgdt-1: ",dgdtm1)
        #  end
        @inbounds for j=1:3
            # Compute difference vectors (finish - start) of step:
            delxv[  j] = fm1*x0[j]+gmh*v0[j]        # position x_ij(t+h)-x_ij(t) - h*v_ij(t) or -h*v_ij(t+h)
        end
        @inbounds for j=1:3
            delxv[3+j] = dfdt*x0[j]+dgdtm1*v0[j]    # velocity v_ij(t+h)-v_ij(t)
        end
        if debug
            delxv[7] = gamma
            delxv[8] = r
            delxv[9] = fm1
            delxv[10] = dfdt
            delxv[11] = gmh
            delxv[12] = dgdtm1
        end
        if grad == true && auto == false && dlnq == 0.0
            # Compute gradient analytically:
            jac_mass = zeros(T,6)
            compute_jacobian_gamma!(gamma,g0bs,g1bs,g2bs,g3bs,h1,h2,dfdt,fm1,gmh,dgdtm1,r0,r,r0inv,rinv,k,h,beta0,beta0inv,eta,sqb,zeta,x0,v0,delxv_jac,jac_mass,drift_first,debug)
        end
        return delxv
    end

    # Use autodiff to compute Jacobian:
    if grad
        if auto
            #    delxv_jac = ForwardDiff.jacobian(delx_delv,input)
            if debug
                delxv = zeros(T,12)
            else
                delxv = zeros(T,6)
            end
            out = DiffResults.JacobianResult(delxv,input)
            ForwardDiff.jacobian!(out,delx_delv,input)
            delxv_jac = DiffResults.jacobian(out)
            delxv = DiffResults.value(out)
        elseif dlnq != 0.0
            # Use finite differences to compute Jacobian:
            if debug
                delxv_jac = zeros(T,12,8)
            else
                delxv_jac = zeros(T,6,8)
            end
            delxv = delx_delv(input)
            @inbounds for j=1:8
                # Difference the jth component:
                inputp = copy(input); dp = dlnq*inputp[j]; inputp[j] += dp
                delxvp = delx_delv(inputp)
                inputm = copy(input); inputm[j] -= dp
                delxvm = delx_delv(inputm)
                delxv_jac[:,j] = (delxvp-delxvm)/(2*dp)
            end
        else
            # If grad = true and dlnq = 0.0, then the above routine will compute Jacobian analytically.
            delxv = delx_delv(input)
        end
        # Return Jacobian:
        return  delxv::Array{T,1},delxv_jac::Array{T,2}
    else
        return delx_delv(input)::Array{T,1}
    end
end

function jac_delxv_gamma!(x0::Array{T,1},v0::Array{T,1},k::T,h::T,drift_first::Bool,delxv::Array{T,1},delxv_jac::Array{T,2},jac_mass::Array{T,1},debug::Bool) where {T <: Real}
    # Compute r0:
    drift_first ?  r0 = norm(x0-h*v0) : r0 = norm(x0)
    # And its inverse:
    r0inv = inv(r0)
    # Compute beta_0:
    beta0 = 2k*r0inv-dot(v0,v0)
    beta0inv = inv(beta0)
    signb = sign(beta0)
    sqb = sqrt(signb*beta0)
    zeta = k-r0*beta0
    gamma_guess = zero(T)
    # Compute \eta_0 = x_0 . v_0:
    drift_first ?  eta = dot(x0-h*v0,v0) : eta = dot(x0,v0)
    if zeta != zero(T)
        # Make sure we have a cubic in gamma (and don't divide by zero):
        gamma_guess = cubic1(3eta*sqb/zeta,6r0*signb*beta0/zeta,-6h*signb*beta0*sqb/zeta)
    else
        # Check that we have a quadratic in gamma (and don't divide by zero):
        if eta != zero(T)
            reta = r0/eta
            disc = reta^2+2h/eta
            disc > zero(T) ?  gamma_guess = sqb*(-reta+sqrt(disc)) : gamma_guess = h*r0inv*sqb
        else
            gamma_guess = h*r0inv*sqb
        end
    end
    gamma  = copy(gamma_guess)
    # Make sure prior two steps differ:
    gamma1 = 2*copy(gamma)
    gamma2 = 3*copy(gamma)
    iter = 0
    ITMAX = 20
    # Compute coefficients: (8/28/19 notes)
    c1 = k; c2 = -2zeta; c3 = 2*eta*signb*sqb; c4 = -sqb*h*beta0; c5 = 2eta*signb*sqb
    # Solve for gamma:
    while true
        gamma2 = gamma1
        gamma1 = gamma
        xx = 0.5*gamma
        if beta0 > 0
            sx = sin(xx); cx = cos(xx)
        else
            sx = sinh(xx); cx = exp(-xx)+sx
        end
        gamma -= (k*gamma+c2*sx*cx+c3*sx^2+c4)/(2signb*zeta*sx^2+c5*sx*cx+r0*beta0)
        iter +=1
        if iter >= ITMAX || gamma == gamma2 || gamma == gamma1
            break
        end
    end
    # Set up a single output array for delx and delv:
    fill!(delxv,zero(T))
    # Since we updated gamma, need to recompute:
    xx = 0.5*gamma
    if beta0 > 0
        sx = sin(xx); cx = cos(xx)
    else
        sx = sinh(xx); cx = exp(-xx)+sx
    end
    # Now, compute final values.  Compute Wisdom/Hernandez G_i^\beta(s) functions:
    g1bs = 2sx*cx/sqb
    g2bs = 2signb*sx^2*beta0inv
    g0bs = one(T)-beta0*g2bs
    g3bs = G3(gamma,beta0)
    h1 = zero(T); h2 = zero(T)
    # Compute r from equation (35):
    r = r0*g0bs+eta*g1bs+k*g2bs
    rinv = inv(r)
    dfdt = -k*g1bs*rinv*r0inv # [x]
    if drift_first
        # Drift backwards before Kepler step: (1/22/2018)
        fm1 = -k*r0inv*g2bs # [x]
        # This is given in 2/7/2018 notes: g-h*f
        gmh = k*r0inv*(h*g2bs-r0*g3bs)  # [x]
    else
        # Drift backwards after Kepler step: (1/24/2018)
        # The following line is f-1-h fdot:
        h1= H1(gamma,beta0); h2= H2(gamma,beta0)
        fm1 =  k*rinv*(g2bs-k*r0inv*h1)  # [x]
        # This is g-h*dgdt
        gmh = k*rinv*(r0*h2+eta*h1) # [x]
    end
    # Compute velocity component functions:
    if drift_first
        # This is gdot - h fdot - 1:
        dgdtm1 = k*r0inv*rinv*(h*g1bs-r0*g2bs)
    else
        # This is gdot - 1:
        dgdtm1 = -k*rinv*g2bs # [x]
    end
    @inbounds for j=1:3
        # Compute difference vectors (finish - start) of step:
        delxv[  j] = fm1*x0[j]+gmh*v0[j]        # position x_ij(t+h)-x_ij(t) - h*v_ij(t) or -h*v_ij(t+h)
    end
    @inbounds for j=1:3
        delxv[3+j] = dfdt*x0[j]+dgdtm1*v0[j]    # velocity v_ij(t+h)-v_ij(t)
    end
    if debug
        delxv[7] = gamma
        delxv[8] = r
        delxv[9] = fm1
        delxv[10] = dfdt
        delxv[11] = gmh
        delxv[12] = dgdtm1
    end
    # Compute gradient analytically:
    compute_jacobian_gamma!(gamma,g0bs,g1bs,g2bs,g3bs,h1,h2,dfdt,fm1,gmh,dgdtm1,r0,r,r0inv,rinv,k,h,beta0,beta0inv,eta,sqb,zeta,x0,v0,delxv_jac,jac_mass,drift_first,debug)
end