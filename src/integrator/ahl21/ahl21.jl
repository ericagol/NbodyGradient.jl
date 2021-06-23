"""

Carry out AHL21 mapping and calculates both the Jacobian and derivative with respect to the time step.
"""
function ahl21!(s::State{T},d::Derivatives{T},h::T) where T<:AbstractFloat
    zilch = zero(T); half::T = 0.5; two::T = 2.0;
    h2::T = half*h; n = s.n; sevn = 7*n; h6::T = h/6.0
    zero_out!(d)
    fill!(s.dqdt,zilch)
    kickfast!(s,d,h6)
    d.dqdt_kick ./= 6.0 # Since step is h/6
    # Since I removed identity from kickfast, need to add in dqdt:
    mul!(d.tmp7n, d.jac_kick, s.dqdt)
    s.dqdt .+= d.dqdt_kick .+ d.tmp7n #*(d.jac_kick,s.dqdt)
    # Multiply Jacobian from kick step:
    mul!(d.jac_copy, d.jac_kick, s.jac_step)
    drift_grad!(s,h2)
    # Compute time derivative of drift step:
    @inbounds for i=1:n, k=1:3
        s.dqdt[(i-1)*7+k] = half*s.v[k,i] + h2*s.dqdt[(i-1)*7+3+k]
    end
    # Add back in the identity portion of the Jacobian with compensated summation:
    comp_sum_matrix!(s.jac_step,s.jac_error,d.jac_copy)
    indi = 0; indj = 0
    @inbounds for i=1:s.n-1
        indi = (i-1)*7
        @inbounds for j=i+1:s.n
            indj = (j-1)*7
            if ~s.pair[i,j]  # Check to see if kicks have not been applied
                kepler_driftij_gamma!(s,d,i,j,h2,true)
                copy_submatrix!(s,d,indi,indj,sevn)
                # Carry out multiplication on the i/j components of matrix:
                mul!(d.jac_tmp2, d.jac_ij, d.jac_tmp1)
                # Add back in the Jacobian with compensated summation:
                comp_sum_matrix!(d.jac_tmp1,d.jac_err1,d.jac_tmp2)
                # Add in partial derivatives with respect to time:
                # Need to multiply by 1/2 since we're taking 1/2 time step:
                d.dqdt_ij .*= half
                mul!(d.tmp14, d.jac_ij, d.dqdt_tmp1)
                d.dqdt_ij .+= d.dqdt_tmp1 .+ d.tmp14#*(d.jac_ij,d.dqdt_tmp1)
                # Copy back time derivatives:
                ypoc_submatrix!(s,d,indi,indj,sevn)
            end
        end
    end
    phic!(s,d,h)
    phisalpha!(s,d,h,two)
    mul!(d.jac_copy, d.jac_phi, s.jac_step)
    # Add in time derivative with respect to prior parameters:
    # Copy result to dqdt:
    mul!(d.tmp7n, d.jac_phi, s.dqdt)
    s.dqdt .+= d.dqdt_phi .+ d.tmp7n #*(d.jac_phi,s.dqdt)
    # Add back in the identity portion of the Jacobian with compensated summation:
    comp_sum_matrix!(s.jac_step,s.jac_error,d.jac_copy)
    indi=0; indj=0
    @inbounds for i=s.n-1:-1:1
        indi=(i-1)*7
        @inbounds for j=s.n:-1:i+1
            indj=(j-1)*7
            if ~s.pair[i,j]  # Check to see if kicks have not been applied
                kepler_driftij_gamma!(s,d,i,j,h2,false)
                # Pick out indices for bodies i & j:
                # Carry out multiplication on the i/j components of matrix:
                copy_submatrix!(s,d,indi,indj,sevn)
                # Carry out multiplication on the i/j components of matrix:
                mul!(d.jac_tmp2,d.jac_ij,d.jac_tmp1)
                # Add back in the Jacobian with compensated summation:
                comp_sum_matrix!(d.jac_tmp1,d.jac_err1,d.jac_tmp2)
                # Add in partial derivatives with respect to time:
                # Need to multiply by 1/2 since we're taking 1/2 time step:
                d.dqdt_ij .*= half
                mul!(d.tmp14, d.jac_ij, d.dqdt_tmp1)
                d.dqdt_ij .+= d.dqdt_tmp1 .+ d.tmp14#*(d.jac_ij,d.dqdt_tmp1)
                # Copy back derivatives:
                ypoc_submatrix!(s,d,indi,indj,sevn)
            end
        end
    end
    drift_grad!(s,h2)
    # Compute time derivative of drift step:
    @inbounds for i=1:n, k=1:3
        s.dqdt[(i-1)*7+k] += half*s.v[k,i] + h2*s.dqdt[(i-1)*7+3+k]
    end
    fill!(d.dqdt_kick,zilch)
    kickfast!(s,d,h6)
    d.dqdt_kick ./= 6.0 # Since step is h/6
    # Copy result to dqdt:
    mul!(d.tmp7n, d.jac_kick, s.dqdt)
    s.dqdt .+= d.dqdt_kick .+ d.tmp7n#*(d.jac_kick,s.dqdt)
    # Multiply Jacobian from kick step:
    mul!(d.jac_copy,d.jac_kick,s.jac_step)
    # Add back in the identity portion of the Jacobian with compensated summation:
    comp_sum_matrix!(s.jac_step,s.jac_error,d.jac_copy)
    # Edit this routine to do compensated summation for Jacobian [x]
end

function ahl21!(s::State{T},d::Jacobian{T},h::T) where T<:AbstractFloat
    zilch = zero(T); uno = one(T); half = convert(T,0.5); two = convert(T,2.0); h2 = half*h; sevn = 7*s.n
    zero_out!(d)

    #drift!(s.x,s.v,s.xerror,s.verror,h2,s.n,s.jac_step,s.jac_error)
    #kickfast!(s.x,s.v,s.xerror,s.verror,h/6,s.m,s.n,d.jac_kick,d.dqdt_kick,pair)
    kickfast!(s,d,h/6)
    # Multiply Jacobian from kick step:
    if T == BigFloat
        d.jac_copy .= *(d.jac_kick,s.jac_step)
    else
        start = time()
        #BLAS.gemm!('N','N',uno,d.jac_kick,s.jac_step,zilch,d.jac_copy)
        mul!(d.jac_copy,d.jac_kick,s.jac_step)
        d.ctime[1] += time()-start
    end
    # Add back in the identity portion of the Jacobian with compensated summation:
    comp_sum_matrix!(s.jac_step,s.jac_error,d.jac_copy)
    drift_grad!(s,h2)
    indi = 0; indj = 0
    @inbounds for i=1:s.n-1
        indi = (i-1)*7
        @inbounds for j=i+1:s.n
            indj = (j-1)*7
            if ~s.pair[i,j]  # Check to see if kicks have not been applied
                kepler_driftij_gamma!(s,d,i,j,h2,true)
                #kepler_driftij_gamma!(s.m,s.x,s.v,s.xerror,s.verror,i,j,h2,d.jac_ij,d.dqdt_ij,true)
                # Pick out indices for bodies i & j:
                @inbounds for k2=1:sevn, k1=1:7
                    d.jac_tmp1[k1,k2] = s.jac_step[ indi+k1,k2]
                    d.jac_err1[k1,k2] = s.jac_error[indi+k1,k2]
                end
                @inbounds for k2=1:sevn, k1=1:7
                    d.jac_tmp1[7+k1,k2] = s.jac_step[ indj+k1,k2]
                    d.jac_err1[7+k1,k2] = s.jac_error[indj+k1,k2]
                end
                # Carry out multiplication on the i/j components of matrix:
                if T == BigFloat
                    d.jac_tmp2 .= *(d.jac_ij,d.jac_tmp1)
                else
                    start = time()
                    #BLAS.gemm!('N','N',uno,d.jac_ij,d.jac_tmp1,zilch,d.jac_tmp2)
                    mul!(d.jac_tmp2,d.jac_ij,d.jac_tmp1)
                    d.ctime[1] += time()-start
                end
                # Add back in the Jacobian with compensated summation:
                comp_sum_matrix!(d.jac_tmp1,d.jac_err1,d.jac_tmp2)
                # Copy back to the Jacobian:
                @inbounds for k2=1:sevn, k1=1:7
                    s.jac_step[ indi+k1,k2]=d.jac_tmp1[k1,k2]
                    s.jac_error[indi+k1,k2]=d.jac_err1[k1,k2]
                end
                @inbounds for k2=1:sevn, k1=1:7
                    s.jac_step[ indj+k1,k2]=d.jac_tmp1[7+k1,k2]
                    s.jac_error[indj+k1,k2]=d.jac_err1[7+k1,k2]
                end
            end
        end
    end
    #phic!(s.x,s.v,s.xerror,s.verror,h,s.m,s.n,d.jac_phi,d.dqdt_phi,pair)
    #phisalpha!(s.x,s.v,s.xerror,s.verror,h,s.m,two,s.n,d.jac_phi,d.dqdt_phi,pair) # 10%
    phic!(s,d,h)
    phisalpha!(s,d,h,two)
    if T == BigFloat
        d.jac_copy .= *(d.jac_phi,s.jac_step)
    else
        start = time()
        #BLAS.gemm!('N','N',uno,d.jac_phi,s.jac_step,zilch,d.jac_copy)
        mul!(d.jac_copy,d.jac_phi,s.jac_step)
        d.ctime[1] += time()-start
    end
    # Add back in the identity portion of the Jacobian with compensated summation:
    comp_sum_matrix!(s.jac_step,s.jac_error,d.jac_copy)
    indi=0; indj=0
    @inbounds for i=s.n-1:-1:1
        indi=(i-1)*7
        @inbounds for j=s.n:-1:i+1
            indj=(j-1)*7
            if ~s.pair[i,j]  # Check to see if kicks have not been applied
                #kepler_driftij_gamma!(s.m,s.x,s.v,s.xerror,s.verror,i,j,h2,d.jac_ij,d.dqdt_ij,false)
                kepler_driftij_gamma!(s,d,i,j,h2,false)
                # Pick out indices for bodies i & j:
                # Carry out multiplication on the i/j components of matrix:
                @inbounds for k2=1:sevn, k1=1:7
                    d.jac_tmp1[k1,k2] = s.jac_step[ indi+k1,k2]
                    d.jac_err1[k1,k2] = s.jac_error[indi+k1,k2]
                end
                @inbounds for k2=1:sevn, k1=1:7
                    d.jac_tmp1[7+k1,k2] = s.jac_step[ indj+k1,k2]
                    d.jac_err1[7+k1,k2] = s.jac_error[indj+k1,k2]
                end
                # Carry out multiplication on the i/j components of matrix:
                if T == BigFloat
                    d.jac_tmp2 .= *(d.jac_ij,d.jac_tmp1)
                else
                    start = time()
                    #BLAS.gemm!('N','N',uno,d.jac_ij,d.jac_tmp1,zilch,d.jac_tmp2)
                    mul!(d.jac_tmp2,d.jac_ij,d.jac_tmp1)
                    d.ctime[1] += time()-start
                end
                # Add back in the Jacobian with compensated summation:
                comp_sum_matrix!(d.jac_tmp1,d.jac_err1,d.jac_tmp2)
                # Copy back to the Jacobian:
                @inbounds for k2=1:sevn, k1=1:7
                    s.jac_step[ indi+k1,k2]=d.jac_tmp1[k1,k2]
                    s.jac_error[indi+k1,k2]=d.jac_err1[k1,k2]
                end
                @inbounds for k2=1:sevn, k1=1:7
                    s.jac_step[ indj+k1,k2]=d.jac_tmp1[7+k1,k2]
                    s.jac_error[indj+k1,k2]=d.jac_err1[7+k1,k2]
                end
            end
        end
    end
    drift_grad!(s,h2)
    #kickfast!(x,v,h2,m,n,jac_kick,dqdt_kick,pair)
    #kickfast!(s.x,s.v,s.xerror,s.verror,h/6,s.m,s.n,d.jac_kick,d.dqdt_kick,pair)
    kickfast!(s,d,h/6)
    # Multiply Jacobian from kick step:
    if T == BigFloat
        d.jac_copy .= *(d.jac_kick,s.jac_step)
    else
        start = time()
        #BLAS.gemm!('N','N',uno,d.jac_kick,s.jac_step,zilch,d.jac_copy)
        mul!(d.jac_copy,d.jac_kick,s.jac_step)
        d.ctime[1] += time()-start
    end
    # Add back in the identity portion of the Jacobian with compensated summation:
    comp_sum_matrix!(s.jac_step,s.jac_error,d.jac_copy)
    # Edit this routine to do compensated summation for Jacobian [x]
    #drift!(s.x,s.v,s.xerror,s.verror,h2,s.n,s.jac_step,s.jac_error)
    return
end

function ahl21!(s::State{T},d::dTime{T},h::T) where T<:AbstractFloat
    # [Currently this routine is not giving the correct dqdt values. -EA 8/12/2019]
    n = s.n
    zilch = zero(T); uno = one(T); half = convert(T,0.5); two = convert(T,2.0); h2 = half*h; sevn = 7*n
    zero_out!(d)
    fill!(s.dqdt,zilch)
    kickfast!(s,d,h/6)
    d.dqdt_kick ./= 6 # Since step is h/6
    # Since I removed identity from kickfast, need to add in dqdt:
    s.dqdt .+= d.dqdt_kick + *(d.jac_kick,s.dqdt)
    drift!(s,h2)
    # Compute time derivative of drift step:
    @inbounds for i=1:n, k=1:3
        s.dqdt[(i-1)*7+k] = half*s.v[k,i] + h2*s.dqdt[(i-1)*7+3+k]
    end
    @inbounds for i=1:n-1
        indi = (i-1)*7
        @inbounds for j=i+1:n
            indj = (j-1)*7
            if ~s.pair[i,j]  # Check to see if kicks have not been applied
                kepler_driftij_gamma!(s,d,i,j,h2,true)
                # Copy current time derivatives for multiplication purposes:
                @inbounds for k1=1:7
                    d.dqdt_tmp1[  k1] = s.dqdt[indi+k1]
                    d.dqdt_tmp1[7+k1] = s.dqdt[indj+k1]
                end
                # Add in partial derivatives with respect to time:
                # Need to multiply by 1/2 since we're taking 1/2 time step:
                d.dqdt_ij .*= half
                d.dqdt_ij .+= *(d.jac_ij,d.dqdt_tmp1) + d.dqdt_tmp1
                # Copy back time derivatives:
                @inbounds for k1=1:7
                    s.dqdt[indi+k1] = d.dqdt_ij[  k1]
                    s.dqdt[indj+k1] = d.dqdt_ij[7+k1]
                end
            end
        end
    end
    # Looks like we are missing phic! here: [ ]
    # Since I haven't added dqdt to phic yet, for now, set jac_phi equal to identity matrix
    # (since this is commented out within phisalpha):
    phic!(s,d,h)
    phisalpha!(s,d,h,two)
    # Add in time derivative with respect to prior parameters:
    # Copy result to dqdt:
    s.dqdt .+= d.dqdt_phi + *(d.jac_phi,s.dqdt)
    indi=0; indj=0
    @inbounds for i=n-1:-1:1
        indi=(i-1)*7
        @inbounds for j=n:-1:i+1
            if ~s.pair[i,j]  # Check to see if kicks have not been applied
                indj=(j-1)*7
                kepler_driftij_gamma!(s,d,i,j,h2,false)
                # Copy current time derivatives for multiplication purposes:
                @inbounds for k1=1:7
                    d.dqdt_tmp1[  k1] = s.dqdt[indi+k1]
                    d.dqdt_tmp1[7+k1] = s.dqdt[indj+k1]
                end
                # Add in partial derivatives with respect to time:
                # Need to multiply by 1/2 since we're taking 1/2 time step:
                d.dqdt_ij .*= half
                d.dqdt_ij .+= *(d.jac_ij,d.dqdt_tmp1) + d.dqdt_tmp1
                # Copy back time derivatives:
                @inbounds for k1=1:7
                    s.dqdt[indi+k1] = d.dqdt_ij[  k1]
                    s.dqdt[indj+k1] = d.dqdt_ij[7+k1]
                end
            end
        end
    end
    drift_grad!(s,h2)
    # Compute time derivative of drift step:
    @inbounds for i=1:n, k=1:3
        s.dqdt[(i-1)*7+k] += half*s.v[k,i] + h2*s.dqdt[(i-1)*7+3+k]
    end
    fill!(d.dqdt_kick,zilch)
    kickfast!(s,d,h/6)
    d.dqdt_kick ./= 6 # Since step is h/6
    # Copy result to dqdt:
    s.dqdt .+= d.dqdt_kick + *(d.jac_kick,s.dqdt)
    return
end

"""

AHL21 drift step. Drifts all particles with compensated summation. (with Jacobian)
"""
function drift_grad!(s::State{T},h::T) where {T <: Real}
    indi::Int64 = 0
    @inbounds for i=1:s.n
        indi = (i-1)*7
        @inbounds for j=1:NDIM
            s.x[j,i],s.xerror[j,i] = comp_sum(s.x[j,i],s.xerror[j,i],h*s.v[j,i])
        end
        # Now for Jacobian:
        @inbounds for k=1:7*s.n, j=1:NDIM
            s.jac_step[indi+j,k],s.jac_error[indi+j,k] = comp_sum(s.jac_step[indi+j,k],s.jac_error[indi+j,k],h*s.jac_step[indi+3+j,k])
        end
    end
    return
end

"""

AHL21 kick step. Computes "fast" kicks for pairs of bodies (in lieu of -drift+Kepler). Include Jacobian and compensated summation.
"""
function kickfast!(s::State{T},d::AbstractDerivatives{T},h::T) where {T <: Real}
    n::Int64 = s.n
    s.rij .= 0.0
    # Getting rid of identity since we will add that back in in calling routines:
    d.jac_kick .= 0.0
    @inbounds for i=1:n-1
        indi = (i-1)*7
        for j=i+1:n
            indj = (j-1)*7
            if s.pair[i,j]
                for k=1:3
                    s.rij[k] = s.x[k,i] - s.x[k,j]
                end
                r2inv::T = 1.0/dot_fast(s.rij)
                r3inv::T = r2inv*sqrt(r2inv)
                fac2::T  = h*GNEWT*r3inv
                for k=1:3
                    fac::T = fac2*s.rij[k]
                    # Apply impulses:
                    s.v[k,i],s.verror[k,i] = comp_sum(s.v[k,i],s.verror[k,i],-s.m[j]*fac)
                    s.v[k,j],s.verror[k,j] = comp_sum(s.v[k,j],s.verror[k,j], s.m[i]*fac)
                    # Compute time derivative:
                    d.dqdt_kick[indi+3+k] -= s.m[j]*fac/h
                    d.dqdt_kick[indj+3+k] += s.m[i]*fac/h
                    # Computing the derivative
                    # Mass derivative of acceleration vector (10/6/17 notes):
                    # Impulse of ith particle depends on mass of jth particle:
                    d.jac_kick[indi+3+k,indj+7] -= fac
                    # Impulse of jth particle depends on mass of ith particle:
                    d.jac_kick[indj+3+k,indi+7] += fac
                    # x derivative of acceleration vector:
                    fac *= 3.0*r2inv
                    # Dot product x_ij.\delta x_ij means we need to sum over components:
                    for p=1:3
                        d.jac_kick[indi+3+k,indi+p] += fac*s.m[j]*s.rij[p]
                        d.jac_kick[indi+3+k,indj+p] -= fac*s.m[j]*s.rij[p]
                        d.jac_kick[indj+3+k,indj+p] += fac*s.m[i]*s.rij[p]
                        d.jac_kick[indj+3+k,indi+p] -= fac*s.m[i]*s.rij[p]
                    end
                    # Final term has no dot product, so just diagonal:
                    d.jac_kick[indi+3+k,indi+k] -= fac2*s.m[j]
                    d.jac_kick[indi+3+k,indj+k] += fac2*s.m[j]
                    d.jac_kick[indj+3+k,indj+k] -= fac2*s.m[i]
                    d.jac_kick[indj+3+k,indi+k] += fac2*s.m[i]
                end
            end
        end
    end
    return
end

"""

Computes correction for pairs which are kicked, with Jacobian, dq/dt, and compensated summation.
"""
function phic!(s::State{T},d::AbstractDerivatives{T},h::T) where {T <: Real}
    s.a .= 0.0
    s.rij .= 0.0
    s.aij .= 0.0
    d.dadq .= 0.0  # There is no velocity dependence
    d.dotdadq .= 0.0 # There is no velocity dependence
    # Set jac_step to zeros:
    fill!(d.jac_phi,zero(T))
    fac::T = 0.0; fac1::T = 0.0; fac2::T = 0.0; fac3::T = 0.0; r1::T = 0.0; r2::T = 0.0; r3::T = 0.0
    coeff::T = h^3/36*GNEWT
    n::Int64 = s.n
    @inbounds for i=1:n-1
        indi = (i-1)*7
        for j=i+1:n
            if s.pair[i,j]
                indj = (j-1)*7
                for k=1:3
                    s.rij[k] = s.x[k,i] - s.x[k,j]
                end
                r2inv::T = inv(dot_fast(s.rij))
                r3inv::T = r2inv*sqrt(r2inv)
                for k=1:3
                    # Apply impulses:
                    fac = GNEWT*s.rij[k]*r3inv
                    facv = fac*2*h/3
                    s.v[k,i],s.verror[k,i] = comp_sum(s.v[k,i],s.verror[k,i],-s.m[j]*facv)
                    s.v[k,j],s.verror[k,j] = comp_sum(s.v[k,j],s.verror[k,j],s.m[i]*facv)
                    # Compute time derivative:
                    d.dqdt_phi[indi+3+k] -= 1/h*s.m[j]*facv
                    d.dqdt_phi[indj+3+k] += 1/h*s.m[i]*facv
                    s.a[k,i] -= s.m[j]*fac
                    s.a[k,j] += s.m[i]*fac
                    # Impulse of ith particle depends on mass of jth particle:
                    d.jac_phi[indi+3+k,indj+7] -= facv
                    # Impulse of jth particle depends on mass of ith particle:
                    d.jac_phi[indj+3+k,indi+7] += facv
                    # x derivative of acceleration vector:
                    facv *= 3.0*r2inv
                    # Dot product x_ij.\delta x_ij means we need to sum over components:
                    for p=1:3
                        d.jac_phi[indi+3+k,indi+p] += facv*s.m[j]*s.rij[p]
                        d.jac_phi[indi+3+k,indj+p] -= facv*s.m[j]*s.rij[p]
                        d.jac_phi[indj+3+k,indj+p] += facv*s.m[i]*s.rij[p]
                        d.jac_phi[indj+3+k,indi+p] -= facv*s.m[i]*s.rij[p]
                    end
                    # Final term has no dot product, so just diagonal:
                    facv = 2h/3*GNEWT*r3inv
                    d.jac_phi[indi+3+k,indi+k] -= facv*s.m[j]
                    d.jac_phi[indi+3+k,indj+k] += facv*s.m[j]
                    d.jac_phi[indj+3+k,indj+k] -= facv*s.m[i]
                    d.jac_phi[indj+3+k,indi+k] += facv*s.m[i]
                    # Mass derivative of acceleration vector (10/6/17 notes):
                    # Since there is no velocity dependence, this is fourth parameter.
                    # Acceleration of ith particle depends on mass of jth particle:
                    d.dadq[k,i,4,j] -= fac
                    d.dadq[k,j,4,i] += fac
                    # x derivative of acceleration vector:
                    fac *= 3.0*r2inv
                    # Dot product x_ij.\delta x_ij means we need to sum over components:
                    for p=1:3
                        d.dadq[k,i,p,i] += fac*s.m[j]*s.rij[p]
                        d.dadq[k,i,p,j] -= fac*s.m[j]*s.rij[p]
                        d.dadq[k,j,p,j] += fac*s.m[i]*s.rij[p]
                        d.dadq[k,j,p,i] -= fac*s.m[i]*s.rij[p]
                    end
                    # Final term has no dot product, so just diagonal:
                    fac = GNEWT*r3inv
                    d.dadq[k,i,k,i] -= fac*s.m[j]
                    d.dadq[k,i,k,j] += fac*s.m[j]
                    d.dadq[k,j,k,j] -= fac*s.m[i]
                    d.dadq[k,j,k,i] += fac*s.m[i]
                end
            end
        end
    end
    # Next, compute g_i acceleration vector.
    # Note that jac_step[(i-1)*7+k,(j-1)*7+p] is the derivative of the kth coordinate
    # of planet i with respect to the pth coordinate of planet j.
    indi::Int64 = 0; indj::Int64 = 0; indd::Int64 = 0
    @inbounds for i=1:n-1
        indi = (i-1)*7
        for j=i+1:n
            if s.pair[i,j] # correction for Kepler pairs
                indj = (j-1)*7
                for k=1:3
                    s.aij[k] = s.a[k,i] - s.a[k,j]
                    s.rij[k] = s.x[k,i] - s.x[k,j]
                end
                # Compute dot product of r_ij with \delta a_ij:
                fill!(d.dotdadq,0.0)
                @inbounds for di=1:n, p=1:4, k=1:3
                    d.dotdadq[p,di] += s.rij[k]*(d.dadq[k,i,p,di]-d.dadq[k,j,p,di])
                end
                r2 = dot_fast(s.rij)
                r1 = sqrt(r2)
                ardot = dot_fast(s.aij,s.rij)
                fac1 = coeff/(r2*r2*r1)
                fac2 = 3*ardot
                for k=1:3
                    fac = fac1*(s.rij[k]*fac2- r2*s.aij[k])
                    s.v[k,i],s.verror[k,i] = comp_sum(s.v[k,i],s.verror[k,i],s.m[j]*fac)
                    s.v[k,j],s.verror[k,j] = comp_sum(s.v[k,j],s.verror[k,j],-s.m[i]*fac)
                    # Compute time derivative of 4th order correction
                    d.dqdt_phi[indi+3+k] += 3/h*s.m[j]*fac
                    d.dqdt_phi[indj+3+k] -= 3/h*s.m[i]*fac
                    # Mass derivative (first part is easy):
                    d.jac_phi[indi+3+k,indj+7] += fac
                    d.jac_phi[indj+3+k,indi+7] -= fac
                    # Position derivatives:
                    fac *= 5.0/r2
                    for p=1:3
                        d.jac_phi[indi+3+k,indi+p] -= fac*s.m[j]*s.rij[p]
                        d.jac_phi[indi+3+k,indj+p] += fac*s.m[j]*s.rij[p]
                        d.jac_phi[indj+3+k,indj+p] -= fac*s.m[i]*s.rij[p]
                        d.jac_phi[indj+3+k,indi+p] += fac*s.m[i]*s.rij[p]
                    end
                    # Diagonal position terms:
                    fac = fac1*fac2
                    d.jac_phi[indi+3+k,indi+k] += fac*s.m[j]
                    d.jac_phi[indi+3+k,indj+k] -= fac*s.m[j]
                    d.jac_phi[indj+3+k,indj+k] += fac*s.m[i]
                    d.jac_phi[indj+3+k,indi+k] -= fac*s.m[i]
                    # Dot product \delta rij terms:
                    fac = -2*fac1*s.aij[k]
                    for p=1:3
                        fac3 = fac*s.rij[p] + fac1*3.0*s.rij[k]*s.aij[p]
                        d.jac_phi[indi+3+k,indi+p] += s.m[j]*fac3
                        d.jac_phi[indi+3+k,indj+p] -= s.m[j]*fac3
                        d.jac_phi[indj+3+k,indj+p] += s.m[i]*fac3
                        d.jac_phi[indj+3+k,indi+p] -= s.m[i]*fac3
                    end
                    # Diagonal acceleration terms:
                    fac = -fac1*r2
                    # Duoh.  For dadq, have to loop over all other parameters!
                    @inbounds for di=1:n
                        indd = (di-1)*7
                        for p=1:3
                            d.jac_phi[indi+3+k,indd+p] += fac*s.m[j]*(d.dadq[k,i,p,di]-d.dadq[k,j,p,di])
                            d.jac_phi[indj+3+k,indd+p] -= fac*s.m[i]*(d.dadq[k,i,p,di]-d.dadq[k,j,p,di])
                        end
                        # Don't forget mass-dependent term:
                        d.jac_phi[indi+3+k,indd+7] += fac*s.m[j]*(d.dadq[k,i,4,di]-d.dadq[k,j,4,di])
                        d.jac_phi[indj+3+k,indd+7] -= fac*s.m[i]*(d.dadq[k,i,4,di]-d.dadq[k,j,4,di])
                    end
                    # Now, for the final term:  (\delta a_ij . r_ij ) r_ij
                    fac = 3.0*fac1*s.rij[k]
                    @inbounds for di=1:n
                        indd = (di-1)*7
                        for p=1:3
                            d.jac_phi[indi+3+k,indd+p] += fac*s.m[j]*d.dotdadq[p,di]
                            d.jac_phi[indj+3+k,indd+p] -= fac*s.m[i]*d.dotdadq[p,di]
                        end
                        d.jac_phi[indi+3+k,indd+7] += fac*s.m[j]*d.dotdadq[4,di]
                        d.jac_phi[indj+3+k,indd+7] -= fac*s.m[i]*d.dotdadq[4,di]
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
function phisalpha!(s::State{T},d::AbstractDerivatives{T},h::T,alpha::T) where {T <: Real}
    s.a .= 0.0
    d.dadq .= 0.0  # There is no velocity dependence
    d.dotdadq .= 0.0  # There is no velocity dependence
    s.rij .= 0.0
    s.aij .= 0.0
    coeff::T = alpha*h^3/96*2*GNEWT
    fac::T = 0.0; fac1::T = 0.0; fac2::T = 0.0; fac3::T = 0.0; r1::T = 0.0; r2::T = 0.0; r3::T = 0.0
    n::Int64 = s.n
    @inbounds for i=1:n-1
        for j=i+1:n
            if ~s.pair[i,j] # correction for Kepler pairs
                for k=1:3
                    s.rij[k] = s.x[k,i] - s.x[k,j]
                end
                r2 = dot_fast(s.rij)
                r3 = r2*sqrt(r2)
                fac2 = GNEWT/r3
                for k=1:3
                    fac = fac2*s.rij[k]
                    s.a[k,i] -= s.m[j]*fac
                    s.a[k,j] += s.m[i]*fac
                    # Mass derivative of acceleration vector (10/6/17 notes):
                    # Since there is no velocity dependence, this is fourth parameter.
                    # Acceleration of ith particle depends on mass of jth particle:
                    d.dadq[k,i,4,j] -= fac
                    d.dadq[k,j,4,i] += fac
                    # x derivative of acceleration vector:
                    fac *= 3.0/r2
                    # Dot product x_ij.\delta x_ij means we need to sum over components:
                    for p=1:3
                        d.dadq[k,i,p,i] += fac*s.m[j]*s.rij[p]
                        d.dadq[k,i,p,j] -= fac*s.m[j]*s.rij[p]
                        d.dadq[k,j,p,j] += fac*s.m[i]*s.rij[p]
                        d.dadq[k,j,p,i] -= fac*s.m[i]*s.rij[p]
                    end
                    # Final term has no dot product, so just diagonal:
                    d.dadq[k,i,k,i] -= fac2*s.m[j]
                    d.dadq[k,i,k,j] += fac2*s.m[j]
                    d.dadq[k,j,k,j] -= fac2*s.m[i]
                    d.dadq[k,j,k,i] += fac2*s.m[i]
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
            if ~s.pair[i,j] # correction for Kepler pairs
                indj = (j-1)*7
                for k=1:3
                    s.aij[k] = s.a[k,i] - s.a[k,j]
                    s.rij[k] = s.x[k,i] - s.x[k,j]
                end
                # Compute dot product of r_ij with \delta a_ij:
                fill!(d.dotdadq,0.0)
                @inbounds for di=1:n, p=1:4, k=1:3
                    d.dotdadq[p,di] += s.rij[k]*(d.dadq[k,i,p,di]-d.dadq[k,j,p,di])
                end
                r2 = s.rij[1]*s.rij[1]+s.rij[2]*s.rij[2]+s.rij[3]*s.rij[3]
                r1 = sqrt(r2)
                ardot = s.aij[1]*s.rij[1]+s.aij[2]*s.rij[2]+s.aij[3]*s.rij[3]
                fac1 = coeff/(r2*r2*r1)
                fac2 = (2*GNEWT*(s.m[i]+s.m[j])/r1 + 3*ardot)
                for k=1:3
                    fac = fac1*(s.rij[k]*fac2- r2*s.aij[k])
                    s.v[k,i],s.verror[k,i] = comp_sum(s.v[k,i],s.verror[k,i], s.m[j]*fac)
                    s.v[k,j],s.verror[k,j] = comp_sum(s.v[k,j],s.verror[k,j],-s.m[i]*fac)
                    # Compute time derivative:
                    d.dqdt_phi[indi+3+k] += 3/h*s.m[j]*fac
                    d.dqdt_phi[indj+3+k] -= 3/h*s.m[i]*fac
                    # Mass derivative (first part is easy):
                    d.jac_phi[indi+3+k,indj+7] += fac
                    d.jac_phi[indj+3+k,indi+7] -= fac
                    # Position derivatives:
                    fac *= 5.0/r2
                    for p=1:3
                        d.jac_phi[indi+3+k,indi+p] -= fac*s.m[j]*s.rij[p]
                        d.jac_phi[indi+3+k,indj+p] += fac*s.m[j]*s.rij[p]
                        d.jac_phi[indj+3+k,indj+p] -= fac*s.m[i]*s.rij[p]
                        d.jac_phi[indj+3+k,indi+p] += fac*s.m[i]*s.rij[p]
                    end
                    # Second mass derivative:
                    fac = 2*GNEWT*fac1*s.rij[k]/r1
                    d.jac_phi[indi+3+k,indi+7] += fac*s.m[j]
                    d.jac_phi[indi+3+k,indj+7] += fac*s.m[j]
                    d.jac_phi[indj+3+k,indj+7] -= fac*s.m[i]
                    d.jac_phi[indj+3+k,indi+7] -= fac*s.m[i]
                    #  (There's also a mass term in dadq [x]. See below.)
                    # Diagonal position terms:
                    fac = fac1*fac2
                    d.jac_phi[indi+3+k,indi+k] += fac*s.m[j]
                    d.jac_phi[indi+3+k,indj+k] -= fac*s.m[j]
                    d.jac_phi[indj+3+k,indj+k] += fac*s.m[i]
                    d.jac_phi[indj+3+k,indi+k] -= fac*s.m[i]
                    # Dot product \delta rij terms:
                    fac = -2*fac1*(s.rij[k]*GNEWT*(s.m[i]+s.m[j])/(r2*r1)+s.aij[k])
                    for p=1:3
                        fac3 = fac*s.rij[p] + fac1*3.0*s.rij[k]*s.aij[p]
                        d.jac_phi[indi+3+k,indi+p] += s.m[j]*fac3
                        d.jac_phi[indi+3+k,indj+p] -= s.m[j]*fac3
                        d.jac_phi[indj+3+k,indj+p] += s.m[i]*fac3
                        d.jac_phi[indj+3+k,indi+p] -= s.m[i]*fac3
                    end
                    # Diagonal acceleration terms:
                    fac = -fac1*r2
                    # Duoh.  For dadq, have to loop over all other parameters!
                    @inbounds for di=1:n
                        indd = (di-1)*7
                        for p=1:3
                            d.jac_phi[indi+3+k,indd+p] += fac*s.m[j]*(d.dadq[k,i,p,di]-d.dadq[k,j,p,di])
                        end
                        for p=1:3
                            d.jac_phi[indj+3+k,indd+p] -= fac*s.m[i]*(d.dadq[k,i,p,di]-d.dadq[k,j,p,di])
                        end
                        # Don't forget mass-dependent term:
                        d.jac_phi[indi+3+k,indd+7] += fac*s.m[j]*(d.dadq[k,i,4,di]-d.dadq[k,j,4,di])
                        d.jac_phi[indj+3+k,indd+7] -= fac*s.m[i]*(d.dadq[k,i,4,di]-d.dadq[k,j,4,di])
                    end
                    # Now, for the final term:  (\delta a_ij . r_ij ) r_ij
                    fac = 3.0*fac1*s.rij[k]
                    @inbounds for di=1:n
                        indd = (di-1)*7
                        for p=1:3
                            d.jac_phi[indi+3+k,indd+p] += fac*s.m[j]*d.dotdadq[p,di]
                        end
                        for p=1:3
                            d.jac_phi[indj+3+k,indd+p] -= fac*s.m[i]*d.dotdadq[p,di]
                        end
                        d.jac_phi[indi+3+k,indd+7] += fac*s.m[j]*d.dotdadq[4,di]
                        d.jac_phi[indj+3+k,indd+7] -= fac*s.m[i]*d.dotdadq[4,di]
                    end
                end
            end
        end
    end
    return
end

"""

Carries out a Kepler step and reverse drift for bodies i & j, and computes Jacobian. Uses new version of the code with gamma in favor of s.
"""
function kepler_driftij_gamma!(s::State{T},d::AbstractDerivatives{T},i::Int64,j::Int64,h::T,drift_first::Bool) where {T <: Real}
    # Initial state:
    @inbounds for k=1:NDIM
        s.x0[k] = s.x[k,i] - s.x[k,j] # x0 = positions of body i relative to j
        s.v0[k] = s.v[k,i] - s.v[k,j] # v0 = velocities of body i relative to j
    end
    gm = GNEWT*(s.m[i]+s.m[j])
    if gm == 0; return; end
    # jac_ij should be the Jacobian for going from (x_{0,i},v_{0,i},m_i) &  (x_{0,j},v_{0,j},m_j)
    # to  (x_i,v_i,m_i) &  (x_j,v_j,m_j), a 14x14 matrix for the 3-dimensional case.:
    fill!(d.jac_ij,zero(T))
    s.delxv .= 0.0
    d.jac_kepler .= 0.0
    d.jac_mass .= 0.0
    params::NTuple{22,T} = jac_delxv_gamma!(s,gm,h,drift_first)
    compute_jacobian_gamma!(params...,s.x0,s.v0,d.jac_kepler,d.jac_mass,drift_first,false)
    mijinv::T = one(T)/(s.m[i] + s.m[j])
    mi::T = s.m[i]*mijinv # Normalize the masses
    mj::T = s.m[j]*mijinv
    @inbounds for k=1:3
        # Add kepler-drift differences, weighted by masses, to start of step:
        s.x[k,i],s.xerror[k,i] = comp_sum(s.x[k,i],s.xerror[k,i], mj*s.delxv[k])
        s.x[k,j],s.xerror[k,j] = comp_sum(s.x[k,j],s.xerror[k,j],-mi*s.delxv[k])
    end
    @inbounds for k=1:3
        s.v[k,i],s.verror[k,i] = comp_sum(s.v[k,i],s.verror[k,i], mj*s.delxv[3+k])
        s.v[k,j],s.verror[k,j] = comp_sum(s.v[k,j],s.verror[k,j],-mi*s.delxv[3+k])
    end
    # Compute Jacobian:
    @inbounds for l=1:6, k=1:6
        # Compute derivatives of x_i,v_i with respect to initial conditions:
        d.jac_ij[  k,  l] += mj*d.jac_kepler[k,l]
        d.jac_ij[  k,7+l] -= mj*d.jac_kepler[k,l]
        # Compute derivatives of x_j,v_j with respect to initial conditions:
        d.jac_ij[7+k,  l] -= mi*d.jac_kepler[k,l]
        d.jac_ij[7+k,7+l] += mi*d.jac_kepler[k,l]
    end
    @inbounds for k=1:6
        # Compute derivatives of x_i,v_i with respect to the masses:
        d.jac_ij[   k, 7] = d.jac_mass[k]*s.m[j]
        d.jac_ij[   k,14] =  mi*s.delxv[k]*mijinv + GNEWT*mj*d.jac_kepler[  k,7]
        # Compute derivatives of x_j,v_j with respect to the masses:
        d.jac_ij[ 7+k, 7] = -mj*s.delxv[k]*mijinv - GNEWT*mi*d.jac_kepler[  k,7]
        d.jac_ij[ 7+k,14] = -d.jac_mass[k]*s.m[i]
    end
    # The following lines are meant to compute dq/dt for kepler_driftij,
    # but they currently contain an error (test doesn't pass in test_ahl21.jl). [ ]
    @inbounds for k=1:6
        # Position/velocity derivative, body i:
        d.dqdt_ij[  k] =  mj*d.jac_kepler[k,8]
        # Position/velocity derivative, body j:
        d.dqdt_ij[7+k] = -mi*d.jac_kepler[k,8]
    end
    return
end

"""

Computes Jacobian of delx and delv with respect to x0, v0, k, and h.
"""
function jac_delxv_gamma!(s::State{T},k::T,h::T,drift_first::Bool;debug::Bool=false,grad::Bool=true) where {T <: Real}
    # Compute r0:
    r0 = zero(T)
    s.rtmp[1] = s.x0[1]-h*s.v0[1]
    s.rtmp[2] = s.x0[2]-h*s.v0[2]
    s.rtmp[3] = s.x0[3]-h*s.v0[3]
    drift_first ?  r0 = norm(s.rtmp) : r0 = norm(s.x0)
    # And its inverse:
    r0inv::T = inv(r0)
    # Compute beta_0:
    beta0::T = 2k*r0inv-dot_fast(s.v0,s.v0)
    beta0inv::T = inv(beta0)
    signb::T = sign(beta0)
    sqb::T = sqrt(signb*beta0)
    zeta::T = k-r0*beta0
    gamma_guess = zero(T)
    # Compute \eta_0 = x_0 . v_0:
    eta = zero(T)
    #drift_first ?  eta = dot(s.x0 .- h .* s.v0, s.v0) : eta = dot(s.x0,s.v0)
    if drift_first
        eta = dot_fast(s.rtmp,s.v0)
    else
        eta = dot_fast(s.x0,s.v0)
    end
    if zeta != zero(T)
        # Make sure we have a cubic in gamma (and don't divide by zero):
        zinv = 6/zeta
        gamma_guess = cubic1(0.5*eta*sqb*zinv,r0*signb*beta0*zinv,-h*signb*beta0*sqb*zinv)
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
    gamma1::T = 2*copy(gamma)
    gamma2::T = 3*copy(gamma)
    iter = 0
    ITMAX = 20
    # Compute coefficients: (8/28/19 notes)
    c1 = k; c2 = -2zeta; c3 = 2*eta*signb*sqb; c4 = -sqb*h*beta0; c5 = 2eta*signb*sqb
    # Solve for gamma:
    while iter < ITMAX
        gamma2 = gamma1
        gamma1 = gamma
        xx = 0.5*gamma
        if beta0 > 0
            sx,cx = sincos(xx);
        else
            sx = sinh(xx); cx = exp(-xx)+sx
        end
        gamma -= (k*gamma+c2*sx*cx+c3*sx^2+c4)/(2signb*zeta*sx^2+c5*sx*cx+r0*beta0)
        iter +=1
        if gamma == gamma2 || gamma == gamma1
            break
        end
    end
    # Set up a single output array for delx and delv:
    if debug
        s.delxv = zeros(T,12)
    else
        s.delxv .= zero(T)
    end
    # Since we updated gamma, need to recompute:
    xx = 0.5*gamma
    if beta0 > 0
        sx,cx = sincos(xx)
    else
        sx = sinh(xx); cx = exp(-xx)+sx
    end
    # Now, compute final values.  Compute Wisdom/Hernandez G_i^\beta(s) functions:
    g1bs = 2sx*cx/sqb
    g2bs = 2signb*sx^2*beta0inv
    g0bs = one(T)-beta0*g2bs
    g3bs = G3(gamma,beta0,sqb)
    h1 = zero(T); h2 = zero(T)
    # Compute r from equation (35):
    r = r0*g0bs+eta*g1bs+k*g2bs
    rinv = inv(r)
    dfdt = -k*g1bs*rinv*r0inv # [x]
    if drift_first
        # Drift backwards before Kepler step: (1/22/2018)
        fm1 = -k*r0inv*g2bs # [x]
        # This is given in 2/7/2018 notes: g-h*f
        gmh = k*r0inv*(h*g2bs-r0*g3bs)
    else
        # Drift backwards after Kepler step: (1/24/2018)
        # The following line is f-1-h fdot:
        h1= H1(gamma,beta0); h2= H2(gamma,beta0,sqb)
        fm1 =  k*rinv*(g2bs-k*r0inv*h1)
        # This is g-h*dgdt
        gmh = k*rinv*(r0*h2+eta*h1)
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
        s.delxv[  j] = fm1*s.x0[j]+gmh*s.v0[j]        # position x_ij(t+h)-x_ij(t) - h*v_ij(t) or -h*v_ij(t+h)
    end
    @inbounds for j=1:3
        s.delxv[3+j] = dfdt*s.x0[j]+dgdtm1*s.v0[j]    # velocity v_ij(t+h)-v_ij(t)
    end
    if debug
        s.delxv[7] = gamma
        s.delxv[8] = r
        s.delxv[9] = fm1
        s.delxv[10] = dfdt
        s.delxv[11] = gmh
        s.delxv[12] = dgdtm1
    end
    if grad
        return gamma,g0bs,g1bs,g2bs,g3bs,h1,h2,dfdt,fm1,gmh,dgdtm1,r0,r,r0inv,rinv,k,h,beta0,beta0inv,eta,sqb,zeta
    end
end

"""

Computes the gradient analytically.
"""
function compute_jacobian_gamma!(gamma::T,g0::T,g1::T,g2::T,g3::T,h1::T,h2::T,dfdt::T,fm1::T,gmh::T,dgdtm1::T,
    r0::T,r::T,r0inv::T,rinv::T,k::T,h::T,beta::T,betainv::T,eta::T,sqb::T,zeta::T,x0::Array{T,1},v0::Array{T,1},
    delxv_jac::Array{T,2},jac_mass::Array{T,1},drift_first::Bool,debug::Bool) where {T <: Real}
    # Computes Jacobian:
    r0inv2 = r0inv^2
    r0inv3 = r0inv2*r0inv
    rinv2 = rinv^2
    rinv3 = rinv2*rinv
    hsq = h^2
    ksq = k^2
    if drift_first
        # First, the diagonal terms:
        # Now the off-diagonal terms:
        d   = (h + eta*g2 + 2*k*g3)*betainv
        c1 = d-r0*g3
        c2 = eta*g0+g1*zeta
        c3  = d*k+g1*r0^2
        c4 = eta*g1+2*g0*r0
        c13 = g1*h-g2*r0
        c9 = 2*g2*h-3*g3*r0
        c10 = k*r0inv2^2*(-g2*r0*h+k*c9*betainv-c3*c13*rinv)
        c24 = r0inv3*(r0*(2*k*r0inv-beta)*betainv-g1*c3*rinv/g2)
        h6 = H6(gamma,beta)
        # Derivatives of \delta x with respect to x0, v0, k & h:
        dfm1dxx = fm1*c24
        dfm1dxv = -fm1*(g1*rinv+h*c24)
        dfm1dvx = dfm1dxv
        dfm1dvv = fm1*rinv*(-r0*g2 + k*h6*betainv/g2 + h*(2*g1+h*r*c24))
        dfm1dh  = fm1*(g1*rinv*(1/g2+2*k*r0inv-beta)-eta*c24)
        dfm1dk  = fm1*(1/k+g1*c1*rinv*r0inv/g2-2*betainv*r0inv)
        h4 = -H1(gamma,beta)*beta
        h5 = H5(gamma,beta,sqb)
        dfm1dk2  = (r0*h4+k*h6)
        dgmhdxx = c10
        dgmhdxv =  -g2*k*c13*rinv*r0inv-h*c10
        dgmhdvx =  dgmhdxv
        h3 = H3(gamma,beta,sqb)
        h8 = -2h3+3h5
        dgmhdvv =  2*g2*h*k*c13*rinv*r0inv+hsq*c10+
        k*betainv*rinv*r0inv*(r0^2*h8-beta*h*r0*g2^2 + (h*k+eta*r0)*h6)
        dgmhdh  =  g2*k*r0inv+k*c13*rinv*r0inv+g2*k*(2*k*r0inv-beta)*c13*rinv*r0inv-eta*c10
        dgmhdk  =  r0inv*(k*c1*c13*rinv*r0inv+g2*h-g3*r0-k*c9*betainv*r0inv)
        dgmhdk2 = (h6*g3*ksq+eta*r0*(h6+g2*h4)+r0^2*g0*h5+k*eta*g2*h6+(g1*h6+g3*h4)*k*r0)
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
            jac_mass[  j] = (GNEWT*r0inv)^2*betainv*rinv*(dfm1dk2*x0[j]-dgmhdk2*v0[j])
        end
        # Derivatives of \delta v with respect to x0, v0, k & h:
        c5 = (r0-k*g2)*rinv/g1
        c6 = (r0*g0-k*g2)*betainv
        c7 = g2*(1/g1+c2*rinv)
        c8 = (k*c6+r*r0+c3*c5)*r0inv3
        c12 = g0*h-g1*r0
        c17 = r0-r-g2*k
        c18 = eta*g1+2*g2*k
        c20 = k*(g2*k+r)-g0*r0*zeta
        c21 = (g2*k-r0)*(beta*c3-k*g1*r)*betainv*rinv2*r0inv3/g1+eta*g1*rinv*r0inv2-2r0inv2
        c22 = rinv*(-g1-g0*g2/g1+g2*c2*rinv)
        c25 = k*rinv*r0inv2*(-g2+k*(c13-g2*r0)*betainv*r0inv2-c13*r0inv-c12*c3*rinv*r0inv2+
                              c13*c2*c3*rinv2*r0inv2-c13*(k*(g2*k+r)-g0*r0*zeta)*betainv*rinv*r0inv2)
        c26 = k*rinv2*r0inv*(-g2*c12-g1*c13+g2*c13*c2*rinv)
        ddfdtdxx = dfdt*c21
        ddfdtdxv = dfdt*(c22-h*c21)
        ddfdtdvx = ddfdtdxv
        c34 = (-beta*eta^2*g2^2-eta*k*h8-h6*ksq-2beta*eta*r0*g1*g2+(g2^2-3*g1*g3)*beta*k*r0
               - beta*g1^2*r0^2)*betainv*rinv2+(eta*g2^2)*rinv/g1 + (k*h8)*betainv*rinv/g1
        ddfdtdvv = dfdt*(c34 - 2*h*c22 +hsq*c21)
        ddfdtdk  = dfdt*(1/k-betainv*r0inv-c17*betainv*rinv*r0inv-c1*(g1*c2-g0*r)*rinv2*r0inv/g1)
        ddfdtdk2 = -(g2*k-r0)*(beta*r0*(g3-g1*g2)-beta*eta*g2^2+k*h3)*betainv*rinv2*r0inv
        ddfdtdh  = dfdt*(g0*rinv/g1-c2*rinv2-(2*k*r0inv-beta)*c22-eta*c21)
        dgdtmhdfdtm1dxx = c25
        dgdtmhdfdtm1dxv = c26-h*c25
        dgdtmhdfdtm1dvx = c26-h*c25
        h2 = H2(gamma,beta,sqb)
        c33 = d*k*rinv3*r0inv*k*(h*g2- r0*g3)+k*(-eta*k*g1*g2^2-g1*g2*g3*ksq-r0*eta*beta*g1*g2^2-r0*k*g1*h2 - beta*g2^2*g0*r0^2)*betainv*rinv2*r0inv
        dgdtmhdfdtm1dvv = c33-2*h*c26+hsq*c25
        dgdtmhdfdtm1dk = rinv*r0inv*(-k*(c13-g2*r0)*betainv*r0inv+c13-k*c13*c17*betainv*rinv*r0inv+k*c1*c12*rinv*r0inv-k*c1*c2*c13*rinv2*r0inv)
        dgdtmhdfdtm1dk2 = k*betainv*rinv2*r0inv*(-beta*eta^2*g2^4+eta*g2*(g1*g2^2+g1^2*g3-5*g2*g3)*k+g2*g3*h3*ksq+
                                                  2eta*r0*beta*g2^2*(g3-g1*g2)+(4g3-g0*g3-g1*g2)*(g3-g1*g2)*r0*k+beta*(2g1*g3*g2-g1^2*g2^2-g3^2)*r0^2)
        dgdtmhdfdtm1dh = g1*k*rinv*r0inv+k*c12*rinv2*r0inv-k*c2*c13*rinv3*r0inv-(2*k*r0inv-beta)*c26-eta*c25
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
                delxv_jac[ 7,i] = -sqb*rinv*((g2-h*c3*r0inv3)*v0[i]+c3*x0[i]*r0inv3); delxv_jac[7,3+i] = sqb*rinv*((-d+2*g2*h-hsq*c3*r0inv3)*v0[i]+(-g2+h*c3*r0inv3)*x0[i])
                delxv_jac[ 8,i] = (c20*betainv-c2*c3*rinv)*r0inv3*x0[i]+((eta*g2+g1*r0)*rinv+h*r0inv3*(c2*c3*rinv-c20*betainv))*v0[i]
                drdv0x0 = (beta*g1*g2+((eta*g2+k*g3)*eta*g0*c3)*rinv*r0inv3 + (g1*g0*(2k*eta^2*g2+3eta*ksq*g3))*betainv*rinv*r0inv2-
                           k*betainv*r0inv3*(eta*g1*(eta*g2+k*g3)+g3*g0*r0^2*beta+2h*g2*k)+(g1*zeta)*rinv*((h*c3)*r0inv3 - g2) -
                           (eta*(beta*g2*g0*r0+k*g1^2)*(eta*g1+k*g2))*betainv*rinv*r0inv2)
                delxv_jac[8,3+i] = drdv0x0*x0[i] - (k*betainv*rinv*(eta*(beta*g2*g3-h8) - h6*k + (g2^2 - 2*g1*g3)*beta*r0) + h*drdv0x0)*v0[i]
                delxv_jac[ 9,i] = dfm1dxx*x0[i]+dfm1dxv*v0[i]; delxv_jac[ 9,3+i]=dfm1dvx*x0[i]+dfm1dvv*v0[i]
                delxv_jac[10,i] = ddfdtdxx*x0[i]+ddfdtdxv*v0[i]; delxv_jac[10,3+i]=ddfdtdvx*x0[i]+ddfdtdvv*v0[i]
                delxv_jac[11,i] = dgmhdxx*x0[i]+dgmhdxv*v0[i]; delxv_jac[11,3+i]=dgmhdvx*x0[i]+dgmhdvv*v0[i]
                delxv_jac[12,i] = dgdtmhdfdtm1dxx*x0[i]+dgdtmhdfdtm1dxv*v0[i]; delxv_jac[12,3+i]=dgdtmhdfdtm1dvx*x0[i]+dgdtmhdfdtm1dvv*v0[i]
            end
            delxv_jac[ 7,7] = sqb*c1*r0inv*rinv; delxv_jac[7,8] = sqb*rinv*(1+eta*c3*r0inv3+g2*(2k*r0inv-beta))
            delxv_jac[ 8,7] = betainv*r0inv*rinv*(-g2*r0^2*beta-eta*g1*g2*(k+beta*r0)+eta*g0*g3*(2*k+zeta)-
                                                  g2^2*(beta*eta^2+2*k*zeta)+g1*g3*zeta*(3*k-beta*r0)); delxv_jac[8,8] = c2*rinv
            delxv_jac[ 8,8] = ((r0*g1+eta*g2)*rinv)*(beta-2*k*r0inv)+c2*rinv+eta*r0inv3*(c2*c3*rinv-c20*betainv)
            delxv_jac[ 9,7] = dfm1dk; delxv_jac[ 9,8] = dfm1dh
            delxv_jac[10,7] = ddfdtdk; delxv_jac[10,8] = ddfdtdh
            delxv_jac[11,7] = dgmhdk; delxv_jac[11,8] = dgmhdh
            delxv_jac[12,7] = dgdtmhdfdtm1dk; delxv_jac[12,8] = dgdtmhdfdtm1dh
        end
    else
        # Now compute the Kepler-Drift Jacobian terms:
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
        h3 = H3(gamma,beta,sqb)
        h5 = H5(gamma,beta,sqb)
        h8 = -2h3+3h5
        # Derivatives of \delta x with respect to x0, v0, k & h:
        dfm1dxx = k*rinv3*betainv*r0inv^4*(k*h1*r^2*r0*(beta-2*k*r0inv)+beta*c3*(r*c23+c14*c2)+c14*r*(k*(r-g2*k)+g0*r0*zeta))
        dfm1dxv = k*rinv2*r0inv*(k*(g2*h2+g1*h1)-2g1*g2*r0+g2*c14*c2*rinv)
        dfm1dvx = dfm1dxv
        dfm1dvv = k*r0inv*rinv2*betainv*(2eta*k*(g2*g3-g1*h1)+(3g3*h2-4h1*g2)*ksq +
                                          beta*g2*r0*(3h1*k-g2*r0)+c14*rinv*(-beta*g2^2*eta^2+eta*k*(2g0*g3-h2)-
                                                                             h6*ksq+(-2eta*g1*g2+k*(h1-2g1*g3))*beta*r0-beta*g1^2*r0^2))
        dfm1dh  = (g1*k-h2*ksq*r0inv-k*c14*c2*rinv*r0inv)*rinv2
        dfm1dk  = rinv*r0inv*(4*h1*ksq*betainv*r0inv-k*h1-2*g2*k*betainv+c14-k*c14*c17*betainv*rinv*r0inv+
                              k*(g1*r0-k*h2)*c1*rinv*r0inv-k*c14*c1*c2*rinv2*r0inv)
        # New expression for d(f-1-h \dot f)/dk with cancellations of higher order terms in gamma is:
        dfm1dk2  = betainv*r0inv*rinv2*(r*(2eta*k*(g1*h1-g3*g2)+(4g2*h1-3g3*h2)*ksq-eta*r0*beta*g1*h1 + (g3*h2-4g2*h1)*beta*k*r0 + g2*h1*beta^2*r0^2) -
                                         # In the following line I need to replace 3g0*g3-g1*g2 by -H8:
                                         c14*(-eta^2*beta*g2^2 - k*eta*h8 - ksq*h6 - eta*r0*beta*(g1*g2 + g0*g3) + 2*(h1 - g1*g3)*beta*k*r0 - (g2 - beta*g1*g3)*beta*r0^2))
        dgmhdxx = k*rinv*r0inv*(h2+k*c19*betainv*r0inv2-c16*c3*rinv*r0inv2+c2*c3*c15*(rinv*r0inv)^2-c15*(k*(g2*k+r)-g0*r0*zeta)*betainv*rinv*r0inv2)
        dgmhdxv = k*rinv2*(h1*r-g2*c16-g1*c15+g2*c2*c15*rinv)
        dgmhdvx = dgmhdxv
        dgmhdvv = k*betainv*rinv2*(2*eta^2*(g1*h1-g2*g3)+eta*k*(4g2*h1-3h2*g3)+r0*eta*(4g0*h1-2g1*g3)+
                                    # In the following lines I need to replace g1*g2-3g0*g3-g1*g2 by H8:
                                    3r0*k*((g1+beta*g3)*h1-g3*g2)+(g0*h8-beta*g1*(g2^2+g1*g3))*r0^2 -
                                    c15*rinv*(beta*g2^2*eta^2+eta*k*h8+h6*ksq+(2eta*g1*g2-k*(g2^2-3g1*g3))*beta*r0+beta*g1^2*r0^2))
        dgmhdk  = rinv*(k*c1*c16*rinv*r0inv+c15-k*c15*c17*betainv*rinv*r0inv-k*c19*betainv*r0inv-k*c1*c2*c15*rinv2*r0inv)
        h7 = beta*g1*g2^2-g0*h8
        dgmhdk2 =  betainv*rinv2*(r*(2eta^2*(g3*g2-g1*h1) + eta*k*(3g3*h2 - 4g2*h1) +
                                      r0*eta*(beta*g3*(g1*g2 + g0*g3) - 2g0*h6) + (-h6*(g1 + beta*g3) + g2*(2g3 - h2))*r0*k +
                                      (h7 - beta^2*g1*g3^2)*r0^2)- c15*(-beta*eta^2*g2^2 + eta*k*(-h2 + 2g0*g3) - h6*ksq -
                                                                        r0*eta*beta*(h2 + 2g0*g3) + 2beta*(2*h1 - g2^2)*r0*k + beta*(beta*g1*g3 - g2)*r0^2))
        dgmhdh  = k*rinv3*(r*c16-c2*c15)
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
        c8 = (k*c6+r*r0+c3*c5)*r0inv3
        c12 = g0*h-g1*r0
        c20 = k*(g2*k+r)-g0*r0*zeta
        ddfdtdxx = dfdt*(eta*g1*rinv-2-g0*c3*rinv*r0inv/g1+c2*c3*r0inv*rinv2-k*(k*g2-r0)*betainv*rinv*r0inv)*r0inv2
        ddfdtdxv = -dfdt*(g0*g2/g1+(r0*g1+eta*g2)*rinv)*rinv
        ddfdtdvx = ddfdtdxv
        ddfdtdvv = -k*rinv3*r0inv*betainv*((beta*eta*g2^2+k*h8)*(r0*g0+k*g2)+
                                            g1*(- h6*ksq + (-2eta*g1*g2+(h1-2g1*g3)*k)*beta*r0 - beta*g1^2*r0^2))
        ddfdtdk  = dfdt*(1/k+c1*(r0-g2*k)*r0inv*rinv2/g1-betainv*r0inv*(1+c17*rinv))
        ddfdtdk2  = (r0-g2*k)*betainv*r0inv*rinv2*(-eta*beta*g2^2+h3*k+(g3-g1*g2)*beta*r0)
        ddfdtdh  = dfdt*(r0-g2*k)*rinv2/g1
        dgdotm1dxx = rinv2*r0inv3*((eta*g2+g1*r0)*k*c3*rinv+g2*k*(k*(g2*k-r)-g0*r0*zeta)*betainv)
        dgdotm1dxv = k*g2*rinv3*(r*g1+r0*g1+eta*g2)
        dgdotm1dvx = dgdotm1dxv
        dgdotm1dvv = k*betainv*rinv3*(eta^2*beta*g2^3-eta*k*g2*h3+3r0*eta*beta*g1*g2^2 +
                                       r0*k*(-g0*h6+3beta*g1*g2*g3)+beta*g2*(g0*g2+g1^2)*r0^2)
        dgdotm1dk = rinv*r0inv*(-r0*g2+g2*k*(r+r0-g2*k)*betainv*rinv-k*g1*c1*rinv+k*g2*c1*c2*rinv2)
        dgdotm1dk2 = betainv*rinv2*(-beta*eta^2*g2^3+eta*k*g2*h3+eta*r0*beta*g2*(g3-2g1*g2)+
                                     (h6-beta*g2^3)*r0*k + beta*g1*(g3 - g1*g2)*r0^2)
        dgdotm1dh = k*rinv3*(g2*c2-r*g1)
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
                delxv_jac[ 7,i] = -sqb*rinv*(g2*v0[i]+c3*x0[i]*r0inv3); delxv_jac[7,3+i] = -sqb*rinv*(d*v0[i]+g2*x0[i])
                delxv_jac[ 8,i] = (c20*betainv-c2*c3*rinv)*r0inv3*x0[i]+((r0*g1+eta*g2)*rinv)*v0[i]
                delxv_jac[8,3+i] = (c18*betainv-d*c2*rinv)*v0[i]+((r0*g1+eta*g2)*rinv)*x0[i]
                delxv_jac[ 9,i] = dfm1dxx*x0[i]+dfm1dxv*v0[i]; delxv_jac[ 9,3+i]=dfm1dvx*x0[i]+dfm1dvv*v0[i]
                delxv_jac[10,i] = ddfdtdxx*x0[i]+ddfdtdxv*v0[i]; delxv_jac[10,3+i]=ddfdtdvx*x0[i]+ddfdtdvv*v0[i]
                delxv_jac[11,i] = dgmhdxx*x0[i]+dgmhdxv*v0[i]; delxv_jac[11,3+i]=dgmhdvx*x0[i]+dgmhdvv*v0[i]
                delxv_jac[12,i] = dgdotm1dxx*x0[i]+dgdotm1dxv*v0[i]; delxv_jac[12,3+i]=dgdotm1dvx*x0[i]+dgdotm1dvv*v0[i]
            end
            delxv_jac[ 7,7] = sqb*c1*r0inv*rinv; delxv_jac[7,8] = sqb*rinv
            delxv_jac[ 8,7] = betainv*r0inv*rinv*(-g2*r0^2*beta-eta*g1*g2*(k+beta*r0)+eta*g0*g3*(2*k+zeta)-
                                                  g2^2*(beta*eta^2+2*k*zeta)+g1*g3*zeta*(3*k-beta*r0)); delxv_jac[8,8] = c2*rinv
            delxv_jac[ 9,7] = dfm1dk; delxv_jac[ 9,8] = dfm1dh
            delxv_jac[10,7] = ddfdtdk; delxv_jac[10,8] = ddfdtdh
            delxv_jac[11,7] = dgmhdk; delxv_jac[11,8] = dgmhdh
            delxv_jac[12,7] = dgdotm1dk; delxv_jac[12,8] = dgdotm1dh
        end
    end
    return
end
