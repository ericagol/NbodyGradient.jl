# The AH18 integrator with derivatives and associated functions.

"""

The AH18 integrator top level function. Carries out the AH18 mapping and computes the Jacobian.
"""
function ah18!(x::Array{T,2},v::Array{T,2},xerror::Array{T,2},verror::Array{T,2},h::T,m::Array{T,1},n::Int64,jac_step::Array{T,2},jac_error::Array{T,2},pair::Array{Bool,2}) where {T <: Real}
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


"""

Carries out the AH18 mapping & computes the derivative with respect to time step, h.
"""
function ah18!(x::Array{T,2},v::Array{T,2},xerror::Array{T,2},verror::Array{T,2},h::T,m::Array{T,1},n::Int64,dqdt::Array{T,1},pair::Array{Bool,2}) where {T <: Real}
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

"""

AH18 drift step. Drifts all particles with compensated summation. (with Jacobian)
"""
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

"""

AH18 kick step. Computes "fast" kicks for pairs of bodies (in lieu of -drift+Kepler). Include Jacobian and compensated summation.
"""
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

"""

Computes correction for pairs which are kicked, with Jacobian, dq/dt, and compensated summation.
"""
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

"""

Computes the 4th-order correction, with Jacobian, dq/dt, and compensated summation.
"""
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

"""

Carries out a Kepler step and reverse drift for bodies i & j, and computes Jacobian. Uses new version of the code with gamma in favor of s, and full auto-diff of Kepler step.
"""
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
    # but they currently contain an error (test doesn't pass in test_ah18.jl). [ ]
    @inbounds for k=1:6
        # Position/velocity derivative, body i:
        dqdt[  k] =  mj*jac_kepler[k,8]
        # Position/velocity derivative, body j:
        dqdt[7+k] = -mi*jac_kepler[k,8]
    end
    return
end

"""

Computes analytic Jacobian of delx and delv with respect to x0, v0, k and h.
"""
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

"""

Computes the gradient analytically.  
"""
function compute_jacobian_gamma!(gamma::T,g0::T,g1::T,g2::T,g3::T,h1::T,h2::T,dfdt::T,fm1::T,gmh::T,dgdtm1::T,r0::T,r::T,r0inv::T,rinv::T,k::T,h::T,beta::T,betainv::T,eta::T,sqb::T,zeta::T,x0::Array{T,1},v0::Array{T,1},delxv_jac::Array{T,2},jac_mass::Array{T,1},drift_first::Bool,debug::Bool) where {T <: Real}
    # Computes Jacobian:
    if drift_first
        # First, x0 derivatives:
        #  delxv[  j] = fm1*x0[j]+gmh*v0[j]        # position x_ij(t+h)-x_ij(t) - h*v_ij(t) or -h*v_ij(t+h)
        #  delxv[3+j] = dfdt*x0[j]+dgdtm1*v0[j]    # velocity v_ij(t+h)-v_ij(t)
        # First, the diagonal terms:
        # Now the off-diagonal terms:
        d   = (h + eta*g2 + 2*k*g3)*betainv
        c1 = d-r0*g3
        c2 = eta*g0+g1*zeta
        c3  = d*k+g1*r0^2
        c4 = eta*g1+2*g0*r0
        c13 = g1*h-g2*r0
        #  c9  = g2*r-h1*k
        #  c10 = c3*c9*rinv+g2*r0*h-k*(2*g2*h+3*g3*r0)*betainv
        c9 = 2*g2*h-3*g3*r0
        c10 = k*r0inv^4*(-g2*r0*h+k*c9*betainv-c3*c13*rinv)
        c24 = r0inv^3*(r0*(2*k*r0inv-beta)*betainv-g1*c3*rinv/g2)
        h6 = H6(gamma,beta)
        # Derivatives of \delta x with respect to x0, v0, k & h:
        dfm1dxx = fm1*c24
        dfm1dxv = -fm1*(g1*rinv+h*c24)
        dfm1dvx = dfm1dxv
        #  dfm1dvv = fm1*(2*betainv-g1*rinv*(d/g2-2*h)+h^2*c24)
        dfm1dvv = fm1*rinv*(-r0*g2 + k*h6*betainv/g2 + h*(2*g1+h*r*c24))
        dfm1dh  = fm1*(g1*rinv*(1/g2+2*k*r0inv-beta)-eta*c24)
        dfm1dk  = fm1*(1/k+g1*c1*rinv*r0inv/g2-2*betainv*r0inv)
        #  dfm1dk2  = 2g2*betainv-c1*g1*rinv
        h4 = -H1(gamma,beta)*beta
        #  h5 = g1*g2-g3*(2+g0)
        h5 = H5(gamma,beta)
        #  println("H5 : ",g1*g2-g3*(2+g0)," ",H2(gamma,beta)-2*G3(gamma,beta)," ",h5)
        #  h6 = 2*g2^2-3*g1*g3
        #  println("H6: ",2*g2^2-3*g1*g3," ",h6)
        #  dfm1dk2  = (r0*h4+k*h6)*betainv*rinv
        dfm1dk2  = (r0*h4+k*h6)
        #  println("dfm1dk2: ",dfm1dk2," ", (r0*h4+k*h6)*betainv*rinv)
        dgmhdxx = c10
        dgmhdxv =  -g2*k*c13*rinv*r0inv-h*c10
        dgmhdvx =  dgmhdxv
        # dgmhdvv =  -d*k*c13*rinv*r0inv+2*g2*h*k*c13*rinv*r0inv+k*c9*betainv*r0inv+h^2*c10
        h8 = H8(gamma,beta)
        dgmhdvv =  2*g2*h*k*c13*rinv*r0inv+h^2*c10+
        k*betainv*rinv*r0inv*(r0^2*h8-beta*h*r0*g2^2 + (h*k+eta*r0)*h6)
        dgmhdh  =  g2*k*r0inv+k*c13*rinv*r0inv+g2*k*(2*k*r0inv-beta)*c13*rinv*r0inv-eta*c10
        dgmhdk  =  r0inv*(k*c1*c13*rinv*r0inv+g2*h-g3*r0-k*c9*betainv*r0inv)
        #  dgmhdk2  =  c1*c13*rinv-c9*betainv
        #  dgmhdk2 = -betainv*rinv*(h6*g3*k^2+eta*r0*(h6+g2*h4)+r0^2*g0*h5+k*eta*g2*h6+(g1*h6+g3*h4)*k*r0)
        dgmhdk2 = -(h6*g3*k^2+eta*r0*(h6+g2*h4)+r0^2*g0*h5+k*eta*g2*h6+(g1*h6+g3*h4)*k*r0)
        #  println("dgmhdk2: ",dgmhdk2," ",-betainv*rinv*(h6*g3*k^2+eta*r0*(h6+g2*h4)+r0^2*g0*h5+k*eta*g2*h6+(g1*h6+g3*h4)*k*r0))
        @inbounds for j=1:3
            # First, compute the \delta x-derivatives:
            delxv_jac[  j,  j] = fm1
            delxv_jac[  j,3+j] = gmh
            @inbounds for i=1:3
                delxv_jac[j,  i] += (dfm1dxx*x0[i]+dfm1dxv*v0[i])*x0[j] + (dgmhdxx*x0[i]+dgmhdxv*v0[i])*v0[j]
                delxv_jac[j,3+i] += (dfm1dvx*x0[i]+dfm1dvv*v0[i])*x0[j] + (dgmhdvx*x0[i]+dgmhdvv*v0[i])*v0[j]
            end
            delxv_jac[  j,  7] = dfm1dk*x0[j] + dgmhdk*v0[j]
            delxv_jac[  j,  8] = dfm1dh*x0[j] + dgmhdh*v0[j]
            # Compute the mass jacobian separately since otherwise cancellations happen in kepler_driftij_gamma:
            jac_mass[  j] = (GNEWT*r0inv)^2*betainv*rinv*(dfm1dk2*x0[j]+dgmhdk2*v0[j])
        end
        # Derivatives of \delta v with respect to x0, v0, k & h:
        c5 = (r0-k*g2)*rinv/g1
        c6 = (r0*g0-k*g2)*betainv
        c7 = g2*(1/g1+c2*rinv)
        c8 = (k*c6+r*r0+c3*c5)*r0inv^3
        c12 = g0*h-g1*r0
        c17 = r0-r-g2*k
        c18 = eta*g1+2*g2*k
        c20 = k*(g2*k+r)-g0*r0*zeta
        c21 = (g2*k-r0)*(beta*c3-k*g1*r)*betainv*rinv^2*r0inv^3/g1+eta*g1*rinv*r0inv^2-2r0inv^2
        c22 = rinv*(-g1-g0*g2/g1+g2*c2*rinv)
        c25 = k*rinv*r0inv^2*(-g2+k*(c13-g2*r0)*betainv*r0inv^2-c13*r0inv-c12*c3*rinv*r0inv^2+
                              c13*c2*c3*rinv^2*r0inv^2-c13*(k*(g2*k+r)-g0*r0*zeta)*betainv*rinv*r0inv^2)
        c26 = k*rinv^2*r0inv*(-g2*c12-g1*c13+g2*c13*c2*rinv)
        ddfdtdxx = dfdt*c21
        ddfdtdxv = dfdt*(c22-h*c21)
        ddfdtdvx = ddfdtdxv
        #  ddfdtdvv = dfdt*(betainv*(1-c18*rinv)+d*(g1*c2-g0*r)*rinv^2/g1-h*(2c22-h*c21))
        c34 = (-beta*eta^2*g2^2-eta*k*h8-h6*k^2-2beta*eta*r0*g1*g2+(g2^2-3*g1*g3)*beta*k*r0
               - beta*g1^2*r0^2)*betainv*rinv^2+(eta*g2^2)*rinv/g1 + (k*h8)*betainv*rinv/g1 
        ddfdtdvv = dfdt*(c34 - 2*h*c22 +h^2*c21)
        ddfdtdk  = dfdt*(1/k-betainv*r0inv-c17*betainv*rinv*r0inv-c1*(g1*c2-g0*r)*rinv^2*r0inv/g1)
        #  ddfdtdk2  = -g1*(-betainv*r0inv-c17*betainv*rinv*r0inv-c1*(g1*c2-g0*r)*rinv^2*r0inv/g1)
        #ddfdtdk2  = -(g2*k-r0)*(g1*r-beta*c1)*betainv*rinv^2*r0inv
        ddfdtdk2 = -(g2*k-r0)*(beta*r0*(g3-g1*g2)-beta*eta*g2^2+k*H3(gamma,beta))*betainv*rinv^2*r0inv
        ddfdtdh  = dfdt*(g0*rinv/g1-c2*rinv^2-(2*k*r0inv-beta)*c22-eta*c21)
        dgdtmhdfdtm1dxx = c25
        dgdtmhdfdtm1dxv = c26-h*c25
        dgdtmhdfdtm1dvx = c26-h*c25
        h2 = H2(gamma,beta)
        #  dgdtmhdfdtm1dvv = d*k*rinv^3*r0inv*(c13*c2-r*c12)+k*(c13*(r0*g0-k*g2)-g2*r*r0)*betainv*rinv^2*r0inv-2*h*c26+h^2*c25
        #  dgdtmhdfdtm1dvv = d*k*rinv^3*r0inv*k*(r0*(g1*g2-g3)+eta*g2^2+k*g2*g3)+k*(c13*(r0*g0-k*g2)-g2*r*r0)*betainv*rinv^2*r0inv-2*h*c26+h^2*c25
        #  println("\dot g - h \dot f -1, dv0 terms: ",d*k*rinv^3*r0inv*c13*c2," ",-d*k*rinv^3*r0inv*r*c12," ",k*c13*r0*g0*betainv*rinv^2*r0inv," ",-k*c13*k*g2*betainv*rinv^2*r0inv," ",-k*g2*r*r0*betainv*rinv^2*r0inv," ",-2*h*c26," ",h^2*c25)
        #  println("second version of dv0 terms:     ",d*k*rinv^3*r0inv*k*r0*g1*g2," ",-d*k*rinv^3*r0inv*k*r0*g3," ",d*k*rinv^3*r0inv*k*eta*g2^2," ",d*k*rinv^3*r0inv*k*k*g2*g3," ",-k*eta*k*g1*g2^2*betainv*rinv^2*r0inv," ",-k*g1*g2*g3*k^2*betainv*rinv^2*r0inv," ",-k*r0*eta*beta*g1*g2^2*betainv*rinv^2*r0inv," ",-r0*k*k*g1*h2*betainv*rinv^2*r0inv," ",-beta*k*g2^2*g0*r0^2*betainv*rinv^2*r0inv," ",-2*h*c26," ",h^2*c25)
        c33 = d*k*rinv^3*r0inv*k*(h*g2- r0*g3)+k*(-eta*k*g1*g2^2-g1*g2*g3*k^2-r0*eta*beta*g1*g2^2-r0*k*g1*h2 - beta*g2^2*g0*r0^2)*betainv*rinv^2*r0inv
        dgdtmhdfdtm1dvv = c33-2*h*c26+h^2*c25
        dgdtmhdfdtm1dk = rinv*r0inv*(-k*(c13-g2*r0)*betainv*r0inv+c13-k*c13*c17*betainv*rinv*r0inv+k*c1*c12*rinv*r0inv-k*c1*c2*c13*rinv^2*r0inv)
        #  dgdtmhdfdtm1dk2 = -(c13-g2*r0)*betainv*r0inv-c13*c17*betainv*rinv*r0inv+c1*c12*rinv*r0inv-c1*c2*c13*rinv^2*r0inv
        #dgdtmhdfdtm1dk2 = g2*betainv+rinv*r0inv*(c1*c12+c13*((k*g2-r0)*betainv-c1*c2*rinv))
        h3 = H3(gamma,beta)
        dgdtmhdfdtm1dk2 = k*betainv*rinv^2*r0inv*(-beta*eta^2*g2^4+eta*g2*(g1*g2^2+g1^2*g3-5*g2*g3)*k+g2*g3*h3*k^2+ 
                                                  2eta*r0*beta*g2^2*(g3-g1*g2)+(4g3-g0*g3-g1*g2)*(g3-g1*g2)*r0*k+beta*(2g1*g3*g2-g1^2*g2^2-g3^2)*r0^2)
        dgdtmhdfdtm1dh = g1*k*rinv*r0inv+k*c12*rinv^2*r0inv-k*c2*c13*rinv^3*r0inv-(2*k*r0inv-beta)*c26-eta*c25
        @inbounds for j=1:3
            # Next, compute the \delta v-derivatives:
            delxv_jac[3+j,  j] = dfdt
            delxv_jac[3+j,3+j] = dgdtm1
            @inbounds for i=1:3
                delxv_jac[3+j,  i] += (ddfdtdxx*x0[i]+ddfdtdxv*v0[i])*x0[j] + (dgdtmhdfdtm1dxx*x0[i]+dgdtmhdfdtm1dxv*v0[i])*v0[j]
                delxv_jac[3+j,3+i] += (ddfdtdvx*x0[i]+ddfdtdvv*v0[i])*x0[j] + (dgdtmhdfdtm1dvx*x0[i]+dgdtmhdfdtm1dvv*v0[i])*v0[j]
            end
            delxv_jac[3+j,  7] = ddfdtdk*x0[j] + dgdtmhdfdtm1dk*v0[j]
            delxv_jac[3+j,  8] = ddfdtdh*x0[j] + dgdtmhdfdtm1dh*v0[j]
            # Compute the mass jacobian separately since otherwise cancellations happen in kepler_driftij_gamma:
            jac_mass[3+j] = GNEWT^2*r0inv*rinv*(ddfdtdk2*x0[j]+dgdtmhdfdtm1dk2*v0[j])
        end
        if debug
            # Now include derivatives of gamma, r, fm1, dfdt, gmh, and dgdtmhdfdtm1:
            @inbounds for i=1:3
                delxv_jac[ 7,i] = -sqb*rinv*((g2-h*c3*r0inv^3)*v0[i]+c3*x0[i]*r0inv^3); delxv_jac[7,3+i] = sqb*rinv*((-d+2*g2*h-h^2*c3*r0inv^3)*v0[i]+(-g2+h*c3*r0inv^3)*x0[i])
                delxv_jac[ 8,i] = (c20*betainv-c2*c3*rinv)*r0inv^3*x0[i]+((eta*g2+g1*r0)*rinv+h*r0inv^3*(c2*c3*rinv-c20*betainv))*v0[i]
                #      delxv_jac[8,3+i] = (g1-g2*c2*rinv+h*r0inv^3*(c2*c3*rinv-c20*betainv))*x0[i]+(-2g1*h+c18*betainv+(2g2*h-d)*c2*rinv+h^2*r0inv^3*(c20*betainv-c2*c3*rinv))*v0[i]
                #      delxv_jac[8,3+i] = ((g1*r0+eta*g2)*rinv+h*r0inv^3*(c2*c3*rinv-c20*betainv))*x0[i]+(-2g1*h+c18*betainv+(2g2*h-d)*c2*rinv+h^2*r0inv^3*(c20*betainv-c2*c3*rinv))*v0[i]
                drdv0x0 = (beta*g1*g2+((eta*g2+k*g3)*eta*g0*c3)*rinv*r0inv^3 + (g1*g0*(2k*eta^2*g2+3eta*k^2*g3))*betainv*rinv*r0inv^2- 
                           k*betainv*r0inv^3*(eta*g1*(eta*g2+k*g3)+g3*g0*r0^2*beta+2h*g2*k)+(g1*zeta)*rinv*((h*c3)*r0inv^3 - g2) - 
                           (eta*(beta*g2*g0*r0+k*g1^2)*(eta*g1+k*g2))*betainv*rinv*r0inv^2)
                #      delxv_jac[8,3+i] = drdv0x0*x0[i]+ (k*betainv*rinv*(eta*(2g0*g3-g1*g2+g3) - h6*k^2 + (g2^2 - 2*g1*g3)*beta*k*r0) + h*drdv0x0)*v0[i]
                delxv_jac[8,3+i] = drdv0x0*x0[i] - (k*betainv*rinv*(eta*(beta*g2*g3-h8) - h6*k + (g2^2 - 2*g1*g3)*beta*r0) + h*drdv0x0)*v0[i]
                #                         +(-2g1*h+c18*betainv+(2g2*h-d)*c2*rinv+h^2*r0inv^3*(c20*betainv-c2*c3*rinv))*v0[i]
                #      delxv_jac[8,3+i] = ((eta*g2+g1*r0)*rinv+h*betainv*rinv*r0inv^3*((3g1*g3 - 2g2^2)*k^3 - k^2*eta*h8 + 
                #            beta*k^2*(g2^2 - 3*g1*g3)*r0 - beta*r0^3 - k*g2*beta*(eta^2*g2 + 2eta*g1*r0 + g0*r0^2)))*x0[i]+(-2g1*h+c18*betainv+((g1*r0-g3*k-2*g0*h)*c2)*betainv*rinv + 
                #         h^2*((2*g2^2 - 3*g1*g3)*k^3 + beta*r0^3 + k^2*(eta*(g1*g2 - 3*g0*g3) + beta*(3*g1*g3 - g2^2)*r0) + 
                #         k*beta*g2*(eta*(eta*g2 + g1*r0) + r0*(eta*g1 + g0*r0)))*betainv*rinv*r0inv^3)*v0[i]
                delxv_jac[ 9,i] = dfm1dxx*x0[i]+dfm1dxv*v0[i]; delxv_jac[ 9,3+i]=dfm1dvx*x0[i]+dfm1dvv*v0[i]
                delxv_jac[10,i] = ddfdtdxx*x0[i]+ddfdtdxv*v0[i]; delxv_jac[10,3+i]=ddfdtdvx*x0[i]+ddfdtdvv*v0[i]
                delxv_jac[11,i] = dgmhdxx*x0[i]+dgmhdxv*v0[i]; delxv_jac[11,3+i]=dgmhdvx*x0[i]+dgmhdvv*v0[i]
                delxv_jac[12,i] = dgdtmhdfdtm1dxx*x0[i]+dgdtmhdfdtm1dxv*v0[i]; delxv_jac[12,3+i]=dgdtmhdfdtm1dvx*x0[i]+dgdtmhdfdtm1dvv*v0[i]
            end
            delxv_jac[ 7,7] = sqb*c1*r0inv*rinv; delxv_jac[7,8] = sqb*rinv*(1+eta*c3*r0inv^3+g2*(2k*r0inv-beta))
            #    delxv_jac[ 8,7] = (c17*betainv+c1*c2*rinv)*r0inv
            delxv_jac[ 8,7] = betainv*r0inv*rinv*(-g2*r0^2*beta-eta*g1*g2*(k+beta*r0)+eta*g0*g3*(2*k+zeta)-
                                                  g2^2*(beta*eta^2+2*k*zeta)+g1*g3*zeta*(3*k-beta*r0)); delxv_jac[8,8] = c2*rinv
            delxv_jac[ 8,8] = ((r0*g1+eta*g2)*rinv)*(beta-2*k*r0inv)+c2*rinv+eta*r0inv^3*(c2*c3*rinv-c20*betainv)
            delxv_jac[ 9,7] = dfm1dk; delxv_jac[ 9,8] = dfm1dh
            delxv_jac[10,7] = ddfdtdk; delxv_jac[10,8] = ddfdtdh
            delxv_jac[11,7] = dgmhdk; delxv_jac[11,8] = dgmhdh
            delxv_jac[12,7] = dgdtmhdfdtm1dk; delxv_jac[12,8] = dgdtmhdfdtm1dh
        end
    else
        # Now compute the Kepler-Drift Jacobian terms:
        # First, x0 derivatives:
        #  delxv[  j] = fm1*x0[j]+gmh*v0[j]        # position x_ij(t+h)-x_ij(t) - h*v_ij(t) or -h*v_ij(t+h)
        #  delxv[3+j] = dfdt*x0[j]+dgdtm1*v0[j]    # velocity v_ij(t+h)-v_ij(t)
        # First, the diagonal terms:
        # Now the off-diagonal terms:
        d   = (h + eta*g2 + 2*k*g3)*betainv
        c1 = d-r0*g3
        c2 = eta*g0+g1*zeta
        c3  = d*k+g1*r0^2
        c4 = eta*g1+2*g0*r0
        c6 = (r0*g0-k*g2)*betainv
        c9  = g2*r-h1*k
        c14 = r0*g2-k*h1
        c15 = eta*h1+h2*r0
        c16 = eta*h2+g1*gamma*r0/sqb
        c17 = r0-r-g2*k
        c18 = eta*g1+2*g2*k
        c19 = 4*eta*h1+3*h2*r0
        c23 = h2*k-r0*g1
        h6 = H6(gamma,beta)
        h8 = H8(gamma,beta)
        # Derivatives of \delta x with respect to x0, v0, k & h:
        dfm1dxx = k*rinv^3*betainv*r0inv^4*(k*h1*r^2*r0*(beta-2*k*r0inv)+beta*c3*(r*c23+c14*c2)+c14*r*(k*(r-g2*k)+g0*r0*zeta))
        dfm1dxv = k*rinv^2*r0inv*(k*(g2*h2+g1*h1)-2g1*g2*r0+g2*c14*c2*rinv)
        dfm1dvx = dfm1dxv
        #  dfm1dvv = k*r0inv*rinv^2*betainv*(r*(2*g2*r0-4*h1*k)+d*beta*c23-c18*c14+d*beta*c14*c2*rinv)
        dfm1dvv = k*r0inv*rinv^2*betainv*(2eta*k*(g2*g3-g1*h1)+(3g3*h2-4h1*g2)*k^2 + 
                                          beta*g2*r0*(3h1*k-g2*r0)+c14*rinv*(-beta*g2^2*eta^2+eta*k*(2g0*g3-h2)-
                                                                             h6*k^2+(-2eta*g1*g2+k*(h1-2g1*g3))*beta*r0-beta*g1^2*r0^2))
        dfm1dh  = (g1*k-h2*k^2*r0inv-k*c14*c2*rinv*r0inv)*rinv^2
        dfm1dk  = rinv*r0inv*(4*h1*k^2*betainv*r0inv-k*h1-2*g2*k*betainv+c14-k*c14*c17*betainv*rinv*r0inv+
                              k*(g1*r0-k*h2)*c1*rinv*r0inv-k*c14*c1*c2*rinv^2*r0inv)
        #  dfm1dk2_old  = 4*h1*k*betainv*r0inv-h1-2*g2*betainv-c14*c17*betainv*rinv*r0inv+ (g1*r0-k*h2)*c1*rinv*r0inv-c14*c1*c2*rinv^2*r0inv
        # New expression for d(f-1-h \dot f)/dk with cancellations of higher order terms in gamma is:
        dfm1dk2  = betainv*r0inv*rinv^2*(r*(2eta*k*(g1*h1-g3*g2)+(4g2*h1-3g3*h2)*k^2-eta*r0*beta*g1*h1 + (g3*h2-4g2*h1)*beta*k*r0 + g2*h1*beta^2*r0^2) - 
                                         # In the following line I need to replace 3g0*g3-g1*g2 by -H8:
                                         c14*(-eta^2*beta*g2^2 - k*eta*h8 - k^2*h6 - eta*r0*beta*(g1*g2 + g0*g3) + 2*(h1 - g1*g3)*beta*k*r0 - (g2 - beta*g1*g3)*beta*r0^2))
        #  println("dfm1dk2: old ",dfm1dk2_old," new: ",dfm1dk2)
        dgmhdxx = k*rinv*r0inv*(h2+k*c19*betainv*r0inv^2-c16*c3*rinv*r0inv^2+c2*c3*c15*(rinv*r0inv)^2-c15*(k*(g2*k+r)-g0*r0*zeta)*betainv*rinv*r0inv^2)
        dgmhdxv = k*rinv^2*(h1*r-g2*c16-g1*c15+g2*c2*c15*rinv)
        dgmhdvx = dgmhdxv
        #  dgmhdvv = k*rinv^2*(-d*c16-c15*c18*betainv+r*c19*betainv+d*c2*c15*rinv)
        dgmhdvv = k*betainv*rinv^2*(2*eta^2*(g1*h1-g2*g3)+eta*k*(4g2*h1-3h2*g3)+r0*eta*(4g0*h1-2g1*g3)+ 
                                    # In the following lines I need to replace g1*g2-3g0*g3-g1*g2 by H8:
                                    3r0*k*((g1+beta*g3)*h1-g3*g2)+(g0*h8-beta*g1*(g2^2+g1*g3))*r0^2 -
                                    c15*rinv*(beta*g2^2*eta^2+eta*k*h8+h6*k^2+(2eta*g1*g2-k*(g2^2-3g1*g3))*beta*r0+beta*g1^2*r0^2))
        dgmhdk  = rinv*(k*c1*c16*rinv*r0inv+c15-k*c15*c17*betainv*rinv*r0inv-k*c19*betainv*r0inv-k*c1*c2*c15*rinv^2*r0inv)
        #  dgmhdk2_old  = c1*c16*rinv-c15*c17*betainv*rinv-c19*betainv-c1*c2*c15*rinv^2
        h7 = H7(gamma,beta)
        dgmhdk2 =  betainv*rinv^2*(r*(2eta^2*(g3*g2-g1*h1) + eta*k*(3g3*h2 - 4g2*h1) +
                                      r0*eta*(beta*g3*(g1*g2 + g0*g3) - 2g0*h6) + (-h6*(g1 + beta*g3) + g2*(2g3 - h2))*r0*k + 
                                      (h7 - beta^2*g1*g3^2)*r0^2)- c15*(-beta*eta^2*g2^2 + eta*k*(-h2 + 2g0*g3) - h6*k^2 - 
                                                                        r0*eta*beta*(h2 + 2g0*g3) + 2beta*(2*h1 - g2^2)*r0*k + beta*(beta*g1*g3 - g2)*r0^2))
        #  println("gmhgdot: old ",dgmhdk2_old," new: ",dgmhdk2)
        dgmhdh  = k*rinv^3*(r*c16-c2*c15)
        @inbounds for j=1:3
            # First, compute the \delta x-derivatives:
            delxv_jac[  j,  j] = fm1
            delxv_jac[  j,3+j] = gmh
            @inbounds for i=1:3
                delxv_jac[j,  i] += (dfm1dxx*x0[i]+dfm1dxv*v0[i])*x0[j] + (dgmhdxx*x0[i]+dgmhdxv*v0[i])*v0[j]
                delxv_jac[j,3+i] += (dfm1dvx*x0[i]+dfm1dvv*v0[i])*x0[j] + (dgmhdvx*x0[i]+dgmhdvv*v0[i])*v0[j]
            end
            delxv_jac[  j,  7] = dfm1dk*x0[j] + dgmhdk*v0[j]
            delxv_jac[  j,  8] = dfm1dh*x0[j] + dgmhdh*v0[j] 
            jac_mass[  j] = GNEWT^2*rinv*r0inv*(dfm1dk2*x0[j]+dgmhdk2*v0[j])
        end
        # Derivatives of \delta v with respect to x0, v0, k & h:
        c5 = (r0-k*g2)*rinv/g1
        c7 = g2*(1/g1+c2*rinv)
        c8 = (k*c6+r*r0+c3*c5)*r0inv^3
        c12 = g0*h-g1*r0
        c20 = k*(g2*k+r)-g0*r0*zeta
        ddfdtdxx = dfdt*(eta*g1*rinv-2-g0*c3*rinv*r0inv/g1+c2*c3*r0inv*rinv^2-k*(k*g2-r0)*betainv*rinv*r0inv)*r0inv^2
        ddfdtdxv = -dfdt*(g0*g2/g1+(r0*g1+eta*g2)*rinv)*rinv
        ddfdtdvx = ddfdtdxv
        #  ddfdtdvv = dfdt*(betainv-d*g0*rinv/g1-c18*betainv*rinv+d*c2*rinv^2)
        ddfdtdvv = -k*rinv^3*r0inv*betainv*((beta*eta*g2^2+k*h8)*(r0*g0+k*g2)+ 
                                            g1*(- h6*k^2 + (-2eta*g1*g2+(h1-2g1*g3)*k)*beta*r0 - beta*g1^2*r0^2))
        ddfdtdk  = dfdt*(1/k+c1*(r0-g2*k)*r0inv*rinv^2/g1-betainv*r0inv*(1+c17*rinv))
        #  ddfdtdk2  = -g1*(c1*(r0-g2*k)*r0inv*rinv^2/g1-betainv*r0inv*(1+c17*rinv))
        #  ddfdtdk2  = r0inv*(g1*c17*betainv*rinv+g1*betainv-g1*c1*c2*rinv^2-c1*g0*rinv)
        h3 = H3(gamma,beta)
        ddfdtdk2  = (r0-g2*k)*betainv*r0inv*rinv^2*(-eta*beta*g2^2+h3*k+(g3-g1*g2)*beta*r0)
        ddfdtdh  = dfdt*(r0-g2*k)*rinv^2/g1
        dgdotm1dxx = rinv^2*r0inv^3*((eta*g2+g1*r0)*k*c3*rinv+g2*k*(k*(g2*k-r)-g0*r0*zeta)*betainv)
        dgdotm1dxv = k*g2*rinv^3*(r*g1+r0*g1+eta*g2)
        dgdotm1dvx = dgdotm1dxv
        #  dgdotm1dvv = k*rinv^2*(d*g1+g2*c18*betainv-2*r*g2*betainv-d*g2*c2*rinv)
        dgdotm1dvv = k*betainv*rinv^3*(eta^2*beta*g2^3-eta*k*g2*h3+3r0*eta*beta*g1*g2^2 +
                                       r0*k*(-g0*h6+3beta*g1*g2*g3)+beta*g2*(g0*g2+g1^2)*r0^2)
        dgdotm1dk = rinv*r0inv*(-r0*g2+g2*k*(r+r0-g2*k)*betainv*rinv-k*g1*c1*rinv+k*g2*c1*c2*rinv^2)
        #  dgdotm1dk2 = rinv*(g2*(r+r0-g2*k)*betainv-g1*c1+g2*c1*c2*rinv)
        dgdotm1dk2 = betainv*rinv^2*(-beta*eta^2*g2^3+eta*k*g2*h3+eta*r0*beta*g2*(g3-2g1*g2)+ 
                                     #       (g2^2*(1+g0)-3*g1*g3)*r0*k + beta*g1*(g3 - g1*g2)*r0^2)
                                     (h6-beta*g2^3)*r0*k + beta*g1*(g3 - g1*g2)*r0^2)
        dgdotm1dh = k*rinv^3*(g2*c2-r*g1)
        @inbounds for j=1:3
            # Next, compute the \delta v-derivatives:
            delxv_jac[3+j,  j] = dfdt
            delxv_jac[3+j,3+j] = dgdtm1
            @inbounds for i=1:3
                delxv_jac[3+j,  i] += (ddfdtdxx*x0[i]+ddfdtdxv*v0[i])*x0[j] + (dgdotm1dxx*x0[i]+dgdotm1dxv*v0[i])*v0[j]
                delxv_jac[3+j,3+i] += (ddfdtdvx*x0[i]+ddfdtdvv*v0[i])*x0[j] + (dgdotm1dvx*x0[i]+dgdotm1dvv*v0[i])*v0[j]
            end
            delxv_jac[3+j,  7] = ddfdtdk*x0[j] + dgdotm1dk*v0[j]
            delxv_jac[3+j,  8] = ddfdtdh*x0[j] + dgdotm1dh*v0[j]
            jac_mass[3+j] = GNEWT^2*rinv*r0inv*(ddfdtdk2*x0[j]+dgdotm1dk2*v0[j])
        end
        if debug
            # Now include derivatives of gamma, r, fm1, gmh, dfdt, and dgdtmhdfdtm1:
            @inbounds for i=1:3
                delxv_jac[ 7,i] = -sqb*rinv*(g2*v0[i]+c3*x0[i]*r0inv^3); delxv_jac[7,3+i] = -sqb*rinv*(d*v0[i]+g2*x0[i])
                delxv_jac[ 8,i] = (c20*betainv-c2*c3*rinv)*r0inv^3*x0[i]+((r0*g1+eta*g2)*rinv)*v0[i]
                delxv_jac[8,3+i] = (c18*betainv-d*c2*rinv)*v0[i]+((r0*g1+eta*g2)*rinv)*x0[i]
                #      delxv_jac[8,3+i] = (g2*(2k*(r-r0)+beta*(r0^2+eta^2*g2)-zeta*eta*g1) + c2*g3*(2k+zeta))*betainv*rinv*r0inv*v0[i]+((r0*g1+eta*g2)*rinv)*x0[i]
                delxv_jac[ 9,i] = dfm1dxx*x0[i]+dfm1dxv*v0[i]; delxv_jac[ 9,3+i]=dfm1dvx*x0[i]+dfm1dvv*v0[i]
                delxv_jac[10,i] = ddfdtdxx*x0[i]+ddfdtdxv*v0[i]; delxv_jac[10,3+i]=ddfdtdvx*x0[i]+ddfdtdvv*v0[i]
                delxv_jac[11,i] = dgmhdxx*x0[i]+dgmhdxv*v0[i]; delxv_jac[11,3+i]=dgmhdvx*x0[i]+dgmhdvv*v0[i]
                delxv_jac[12,i] = dgdotm1dxx*x0[i]+dgdotm1dxv*v0[i]; delxv_jac[12,3+i]=dgdotm1dvx*x0[i]+dgdotm1dvv*v0[i]
            end
            delxv_jac[ 7,7] = sqb*c1*r0inv*rinv; delxv_jac[7,8] = sqb*rinv
            #    delxv_jac[ 8,7] = (c17*betainv+c1*c2*rinv)*r0inv; delxv_jac[8,8] = c2*rinv
            delxv_jac[ 8,7] = betainv*r0inv*rinv*(-g2*r0^2*beta-eta*g1*g2*(k+beta*r0)+eta*g0*g3*(2*k+zeta)-
                                                  g2^2*(beta*eta^2+2*k*zeta)+g1*g3*zeta*(3*k-beta*r0)); delxv_jac[8,8] = c2*rinv
            delxv_jac[ 9,7] = dfm1dk; delxv_jac[ 9,8] = dfm1dh
            delxv_jac[10,7] = ddfdtdk; delxv_jac[10,8] = ddfdtdh
            delxv_jac[11,7] = dgmhdk; delxv_jac[11,8] = dgmhdh
            delxv_jac[12,7] = dgdotm1dk; delxv_jac[12,8] = dgdotm1dh
        end
    end
    #return delxv_jac::Array{T,2}
    return
end
