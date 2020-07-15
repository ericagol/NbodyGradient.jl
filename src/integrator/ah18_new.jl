# For testing structures
"""

The AH18 integrator top level function. Carries out the AH18 mapping and computes the Jacobian.
"""
function ah18!(s::State{T},d::Derivatives{T},h::T,pair::Matrix{Bool}) where T<:AbstractFloat
    zilch = zero(T); uno = one(T); half = convert(T,0.5); two = convert(T,2.0); h2 = half*h; sevn = 7*s.n

    drift!(s.x,s.v,s.xerror,s.verror,h2,s.n,s.jac_step,s.jac_error)
    kickfast!(s.x,s.v,s.xerror,s.verror,h/6,s.m,s.n,d.jac_kick,d.dqdt_kick,pair)
    # Multiply Jacobian from kick step:
    if T == BigFloat
        d.jac_copy .= *(d.jac_kick,s.jac_step)
    else
        BLAS.gemm!('N','N',uno,d.jac_kick,s.jac_step,zilch,d.jac_copy)
    end
    # Add back in the identity portion of the Jacobian with compensated summation:
    comp_sum_matrix!(s.jac_step,s.jac_error,d.jac_copy)
    indi = 0; indj = 0
    @inbounds for i=1:s.n-1
        indi = (i-1)*7
        @inbounds for j=i+1:s.n
            indj = (j-1)*7
            if ~pair[i,j]  # Check to see if kicks have not been applied
                kepler_driftij_gamma!(s.m,s.x,s.v,s.xerror,s.verror,i,j,h2,d.jac_ij,d.dqdt_ij,true)
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
                    BLAS.gemm!('N','N',uno,d.jac_ij,d.jac_tmp1,zilch,d.jac_tmp2)
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
    phic!(s.x,s.v,s.xerror,s.verror,h,s.m,s.n,d.jac_phi,d.dqdt_phi,pair)
    phisalpha!(s.x,s.v,s.xerror,s.verror,h,s.m,two,s.n,d.jac_phi,d.dqdt_phi,pair) # 10%
    if T == BigFloat
        d.jac_copy .= *(d.jac_phi,s.jac_step)
    else
        BLAS.gemm!('N','N',uno,d.jac_phi,s.jac_step,zilch,d.jac_copy)
    end
    # Add back in the identity portion of the Jacobian with compensated summation:
    comp_sum_matrix!(s.jac_step,s.jac_error,d.jac_copy)
    indi=0; indj=0
    @inbounds for i=s.n-1:-1:1
        indi=(i-1)*7
        @inbounds for j=s.n:-1:i+1
            indj=(j-1)*7
            if ~pair[i,j]  # Check to see if kicks have not been applied
                kepler_driftij_gamma!(s.m,s.x,s.v,s.xerror,s.verror,i,j,h2,d.jac_ij,d.dqdt_ij,false)
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
                    BLAS.gemm!('N','N',uno,d.jac_ij,d.jac_tmp1,zilch,d.jac_tmp2)
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
    #kickfast!(x,v,h2,m,n,jac_kick,dqdt_kick,pair)
    kickfast!(s.x,s.v,s.xerror,s.verror,h/6,s.m,s.n,d.jac_kick,d.dqdt_kick,pair)
    # Multiply Jacobian from kick step:
    if T == BigFloat
        d.jac_copy .= *(d.jac_kick,s.jac_step)
    else
        BLAS.gemm!('N','N',uno,d.jac_kick,s.jac_step,zilch,d.jac_copy)
    end
    # Add back in the identity portion of the Jacobian with compensated summation:
    comp_sum_matrix!(s.jac_step,s.jac_error,d.jac_copy)
    # Edit this routine to do compensated summation for Jacobian [x]
    drift!(s.x,s.v,s.xerror,s.verror,h2,s.n,s.jac_step,s.jac_error)
    return
end

"""

Carries out AH18 mapping with compensated summation, WITHOUT derivatives
"""
function ah18!(s::State{T},h::T,pair::Matrix{Bool}) where T<:AbstractFloat
    h2 = 0.5*h; n = s.n
    drift!(s.x,s.v,s.xerror,s.verror,h2,n)
    kickfast!(s.x,s.v,s.xerror,s.verror,h/6,s.m,s.n,pair)
    @inbounds for i=1:n-1
        for j=i+1:n
            if ~pair[i,j]
                kepler_driftij_gamma!(s.m,s.x,s.v,s.xerror,s.verror,i,j,h2,true)
            end
        end
    end
    phic!(s.x,s.v,s.xerror,s.verror,h,s.m,n,pair)
    phisalpha!(s.x,s.v,s.xerror,s.verror,h,s.m,convert(T,2),n,pair)
    for i=n-1:-1:1
        for j=n:-1:i+1
            if ~pair[i,j]
                kepler_driftij_gamma!(s.m,s.x,s.v,s.xerror,s.verror,i,j,h2,false)
            end
        end
    end
    kickfast!(s.x,s.v,s.xerror,s.verror,h/6,s.m,n,pair)
    drift!(s.x,s.v,s.xerror,s.verror,h2,n)
    return
end

