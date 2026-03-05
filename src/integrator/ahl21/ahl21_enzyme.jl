using EnzymeCore
using EnzymeCore.EnzymeRules

# Pre-allocated workspace for Enzyme reverse passes
struct EnzymeWorkspace{T}
    # For phic!/phisalpha! VJPs
    jac_phi::Matrix{T}
    dadq::Array{T,4}
    a::Matrix{T}
    rij::Vector{T}
    aij::Vector{T}
    dotdadq::Matrix{T}
    grad_out::Vector{T}
    grad_in::Vector{T}

    # For drift_kepler!/kepler_drift! VJPs
    jac_kepler::Matrix{T}
    jac_mass::Vector{T}
    jac_ij::Matrix{T}
    grad_out_ij::Vector{T}
    grad_in_ij::Vector{T}
end

function EnzymeWorkspace(::Type{T}, n::Int) where T
    sevn = 7*n
    EnzymeWorkspace{T}(
        zeros(T, sevn, sevn),  # jac_phi
        zeros(T, 3, n, 4, n),  # dadq
        zeros(T, 3, n),        # a
        zeros(T, 3),           # rij
        zeros(T, 3),           # aij
        zeros(T, 4, n),        # dotdadq
        zeros(T, sevn),        # grad_out
        zeros(T, sevn),        # grad_in
        zeros(T, 6, 8),        # jac_kepler
        zeros(T, 6),           # jac_mass
        zeros(T, 14, 14),      # jac_ij
        zeros(T, 14),          # grad_out_ij
        zeros(T, 14),          # grad_in_ij
    )
end

# Get workspace from shadow state if available, otherwise allocate fresh.
# Using EnzymeShadow(s) pre-allocates workspace to avoid repeated allocations.
function get_enzyme_workspace(shadow::State{T, WS}) where {T, WS}
    if shadow.enzyme_workspace !== nothing
        return shadow.enzyme_workspace
    else
        return EnzymeWorkspace(T, shadow.n)
    end
end

function EnzymeRules.augmented_primal(
    config,
    func::EnzymeCore.Const{typeof(NbodyGradient.drift!)},
    ::Type{<:EnzymeCore.Const},
    s::EnzymeCore.Duplicated{State{T, EnzymeWorkspace{T}}},
    h::EnzymeCore.Const{T}
    ) where T

    # Do the foward pass
    NbodyGradient.drift!(s.val, h.val)

    # Return the step size in the tape
    return EnzymeRules.AugmentedReturn(nothing, nothing, h.val)
end

function EnzymeRules.reverse(
    config,
    func::EnzymeCore.Const{typeof(NbodyGradient.drift!)},
    ::Type{<:EnzymeCore.Const},
    tape,
    s::EnzymeCore.Duplicated{State{T, EnzymeWorkspace{T}}},
    h::EnzymeCore.Const{T}
    ) where T

    # Get the time step from the tape
    h_val = tape

    # Compute the jacobian of the drift step
    @inbounds for i in 1:s.val.n, j in 1:NDIM
        s.dval.v[j,i] += h_val * s.dval.x[j,i]
    end

    return (nothing, nothing)
end

function EnzymeRules.augmented_primal(
    config,
    func::EnzymeCore.Const{typeof(NbodyGradient.kickfast!)},
    ::Type{<:EnzymeCore.Const},
    s::EnzymeCore.Duplicated{State{T, EnzymeWorkspace{T}}},
    h::EnzymeCore.Const{T}
    ) where T

    # Save positions before kick for gradient computation later
    x_save = copy(s.val.x)

    # Do the kickfast
    NbodyGradient.kickfast!(s.val, h.val)

    # Return the saved positions and the time step
    return EnzymeRules.AugmentedReturn(nothing, nothing, (x_save, h.val))
end

function EnzymeRules.reverse(
    config,
    func::EnzymeCore.Const{typeof(NbodyGradient.kickfast!)},
    ::Type{<:EnzymeCore.Const},
    tape,
    s::EnzymeCore.Duplicated{State{T, EnzymeWorkspace{T}}},
    h::EnzymeCore.Const{T}
    ) where T

    n = s.val.n

    # Get the saved positions and time step
    x_saved, h_val = tape

    # Compute the jacobian
    @inbounds for i in 1:n-1
        for j in i+1:n
            if s.val.pair[i,j]
                # Unroll manually instead
                rij1 = x_saved[1,i] - x_saved[1,j]
                rij2 = x_saved[2,i] - x_saved[2,j]
                rij3 = x_saved[3,i] - x_saved[3,j]
                r2 = rij1*rij1 + rij2*rij2 + rij3*rij3
                r2inv = one(T) / r2
                r3inv = r2inv * sqrt(r2inv)
                fac2 = h_val * GNEWT * r3inv

                for k in 1:3
                    rij_k = (k==1 ? rij1 : (k==2 ? rij2 : rij3))
                    fac = fac2 * rij_k

                    # Get incoming gradients on velocities
                    gvi = s.dval.v[k,i]
                    gvj = s.dval.v[k,j]

                    # Do the VJP for mass derivatives
                    s.dval.m[j] -= fac*gvi
                    s.dval.m[i] += fac*gvj

                    # Now for the positions and velocities
                    fac3 = 3.0 * fac * r2inv
                    for p in 1:3
                        rij_p = (p==1 ? rij1 : (p==2 ? rij2 : rij3))

                        # positions
                        fac3rij = fac3 * rij_p
                        dxdvi = fac3rij * s.val.m[j] * gvi
                        dxdvj = fac3rij * s.val.m[i] * gvj
                        s.dval.x[p,i] += (dxdvi - dxdvj)
                        s.dval.x[p,j] += (dxdvj - dxdvi)
                    end

                    # Final term has no dot product, so just diagonal:
                    dxdvi = fac2 * s.val.m[j] * gvi
                    dxdvj = fac2 * s.val.m[i] * gvj
                    s.dval.x[k,i] += (dxdvj - dxdvi)
                    s.dval.x[k,j] += (dxdvi - dxdvj)
                end
            end
        end
    end
    return (nothing, nothing)
end

function EnzymeRules.augmented_primal(
    config,
    func::EnzymeCore.Const{typeof(NbodyGradient.phic!)},
    ::Type{<:EnzymeCore.Const},
    s::EnzymeCore.Duplicated{State{T, EnzymeWorkspace{T}}},
    h::EnzymeCore.Const{T}
    ) where T

    # Save positions before phic for gradient computation later
    x_save = copy(s.val.x)

    # Do the phic ("forward pass")
    NbodyGradient.phic!(s.val, h.val)

    return EnzymeRules.AugmentedReturn(nothing, nothing, (x_save, h.val))
end

function compute_jac_phi!(jac_phi::AbstractMatrix{T}, x::AbstractMatrix{T}, m::AbstractVector{T}, pair::Matrix{Bool}, h::T, n::Int, dadq, a, rij, aij, dotdadq) where T
    coeff = h^3/36*GNEWT

    @inbounds for i in 1:n-1
        indi = (i-1)*7
        for j in i+1:n
            if pair[i,j]
                indj = (j-1)*7
                for k in 1:3
                    rij[k] = x[k, i] - x[k, j]
                end
                r2inv = inv(dot_fast(rij))
                r3inv = r2inv * sqrt(r2inv)

                for k in 1:3
                    fac = GNEWT * rij[k] * r3inv
                    facv = fac * 2*h/3

                    # acceleration
                    a[k,i] -= m[j] * fac
                    a[k,j] += m[i] * fac

                    # Mass derivatives
                    jac_phi[indi+3+k, indj+7] -= facv
                    jac_phi[indj+3+k, indi+7] += facv

                    # position derivatives
                    # Dot product terms
                    facv *= 3.0 * r2inv
                    facvi = facv * m[i]
                    facvj = facv * m[j]
                    for p in 1:3
                        di = facvi * rij[p]
                        dj = facvj * rij[p]
                        jac_phi[indi+3+k, indi+p] += dj
                        jac_phi[indi+3+k, indj+p] -= dj
                        jac_phi[indj+3+k, indj+p] += di
                        jac_phi[indj+3+k, indi+p] -= di
                    end

                    # diagonal terms
                    facv = 2*h/3 * GNEWT * r3inv
                    facvi = facv * m[i]
                    facvj = facv * m[j]
                    jac_phi[indi+3+k, indi+k] -= facvj
                    jac_phi[indi+3+k, indj+k] += facvj
                    jac_phi[indj+3+k, indj+k] -= facvi
                    jac_phi[indj+3+k, indi+k] += facvi

                    # Now acceleration derivatives
                    dadq[k,i,4,j] -= fac
                    dadq[k,j,4,i] += fac
                    fac *= 3.0*r2inv
                    facmi = fac * m[i]
                    facmj = fac * m[j]
                    for p in 1:3
                        facmjp = facmj * rij[p]
                        facmip = facmi * rij[p]
                        dadq[k,i,p,i] += facmjp
                        dadq[k,i,p,j] -= facmjp
                        dadq[k,j,p,j] += facmip
                        dadq[k,j,p,i] -= facmip
                    end
                    fac = GNEWT * r3inv
                    facmi = fac * m[i]
                    facmj = fac * m[j]
                    dadq[k,i,k,i] -= facmj
                    dadq[k,i,k,j] += facmj
                    dadq[k,j,k,j] -= facmi
                    dadq[k,j,k,i] += facmi
                end
            end
        end
    end

    # Next, compute g_i acceleration vector
    @inbounds for i in 1:n-1
        indi = (i-1)*7
        for j in i+1:n
            if pair[i,j]
                indj = (j-1)*7
                @inbounds for k in 1:3
                    rij[k] = x[k,i] - x[k,j]
                    aij[k] = a[k,i] - a[k,j]
                end

                # compute dot product of r_ij with \delta a_ij
                fill!(dotdadq, 0.0)
                @inbounds for di in 1:n, p in 1:4, k in 1:3
                    dotdadq[p,di] += rij[k] * (dadq[k,i,p,di] - dadq[k,j,p,di])
                end

                r2 = dot_fast(rij)
                r1 = sqrt(r2)
                ardot = dot_fast(aij, rij)
                fac1 = coeff / (r2*r2*r1)
                fac2 = 3*ardot
                for k in 1:3
                    fac = fac1 * (rij[k]*fac2 - r2*aij[k])

                    # Mass derivative
                    jac_phi[indi+3+k, indj+7] += fac
                    jac_phi[indj+3+k, indi+7] -= fac

                    # position derivatives
                    fac *= 5.0 / r2
                    facmi = fac*m[i]
                    facmj = fac*m[j]
                    for p in 1:3
                        facmip = facmi * rij[p]
                        facmjp = facmj * rij[p]
                        jac_phi[indi+3+k, indi+p] -= facmjp
                        jac_phi[indi+3+k, indj+p] += facmjp
                        jac_phi[indj+3+k, indj+p] -= facmip
                        jac_phi[indj+3+k, indi+p] += facmip
                    end

                    # Diagonal position terms
                    fac = fac1 * fac2
                    facmi = fac*m[i]
                    facmj = fac*m[j]
                    jac_phi[indi+3+k, indi+k] += facmj
                    jac_phi[indi+3+k, indj+k] -= facmj
                    jac_phi[indj+3+k, indj+k] += facmi
                    jac_phi[indj+3+k, indi+k] -= facmi

                    # dot product \delta rij terms
                    fac = -2*fac1*aij[k]
                    for p in 1:3
                        fac3 = fac*rij[p] + fac1*3.0*rij[k]*aij[p]
                        fac3mi = fac3 * m[i]
                        fac3mj = fac3 * m[j]
                        jac_phi[indi+3+k, indi+p] += fac3mj
                        jac_phi[indi+3+k, indj+p] -= fac3mj
                        jac_phi[indj+3+k, indj+p] += fac3mi
                        jac_phi[indj+3+k, indi+p] -= fac3mi
                    end

                    # diagonal acceleration terms
                    fac = -fac1*r2
                    facmi = fac*m[i]
                    facmj = fac*m[j]
                    for di in 1:n
                        indd = (di-1)*7
                        for p in 1:3
                            dadq_p = (dadq[k,i,p,di] - dadq[k,j,p,di])
                            jac_phi[indi+3+k, indd+p] += facmj*dadq_p
                            jac_phi[indj+3+k, indd+p] -= facmi*dadq_p
                        end
                        dadq_4 = (dadq[k,i,4,di] - dadq[k,j,4,di])
                        jac_phi[indi+3+k, indd+7] += facmj*dadq_4
                        jac_phi[indj+3+k, indd+7] -= facmi*dadq_4
                    end

                    # Now, for the final term
                    fac = 3.0*fac1*rij[k]
                    facmi = fac*m[i]
                    facmj = fac*m[j]
                    @inbounds for di in 1:n
                        indd = (di-1)*7
                        for p in 1:3
                            jac_phi[indi+3+k, indd+p] += facmj * dotdadq[p,di]
                            jac_phi[indj+3+k, indd+p] -= facmi * dotdadq[p,di]
                        end
                        jac_phi[indi+3+k, indd+7] += facmj*dotdadq[4,di]
                        jac_phi[indj+3+k, indd+7] -= facmi*dotdadq[4,di]
                    end
                end
            end
        end
    end
    return
end

function EnzymeRules.reverse(
    config,
    func::EnzymeCore.Const{typeof(NbodyGradient.phic!)},
    ::Type{<:EnzymeCore.Const},
    tape,
    s::EnzymeCore.Duplicated{State{T, EnzymeWorkspace{T}}},
    h::EnzymeCore.Const{T}
    ) where T

    n = s.val.n

    # Get saved positions
    x_saved, h_val = tape

    # Get pre-allocated workspace (only allocates on first call)
    ws = get_enzyme_workspace(s.dval)

    # Reset workspace arrays
    fill!(ws.jac_phi, zero(T))
    fill!(ws.dadq, zero(T))
    fill!(ws.a, zero(T))

    # Compute Jacobian
    compute_jac_phi!(ws.jac_phi, x_saved, s.val.m, s.val.pair, h_val, n,
                     ws.dadq, ws.a, ws.rij, ws.aij, ws.dotdadq)

    # Build gradient vector from shadow (explicit indexing to avoid allocations)
    @inbounds for i in 1:n
        idx = (i-1)*7
        for k in 1:3
            ws.grad_out[idx+k] = s.dval.x[k,i]
            ws.grad_out[idx+3+k] = s.dval.v[k,i]
        end
        ws.grad_out[idx+7] = s.dval.m[i]
    end

    # Apply VJP
    mul!(ws.grad_in, ws.jac_phi', ws.grad_out)

    # Place back into the shadow state (explicit indexing)
    @inbounds for i in 1:n
        idx = (i-1)*7
        for k in 1:3
            s.dval.x[k,i] += ws.grad_in[idx+k]
            s.dval.v[k,i] += ws.grad_in[idx+3+k]
        end
        s.dval.m[i] += ws.grad_in[idx+7]
    end

    return (nothing, nothing)
end

function EnzymeRules.augmented_primal(
    config,
    func::EnzymeCore.Const{typeof(NbodyGradient.phisalpha!)},
    ::Type{<:EnzymeCore.Const},
    s::EnzymeCore.Duplicated{State{T, EnzymeWorkspace{T}}},
    h::EnzymeCore.Const{T},
    alpha::EnzymeCore.Const{T}
    ) where T

    # Save positions before phisalpha for gradient computation later
    x_save = copy(s.val.x)

    # Do the phisalpha step
    NbodyGradient.phisalpha!(s.val, h.val, alpha.val)

    return EnzymeRules.AugmentedReturn(nothing, nothing, (x_save, h.val, alpha.val))
end

function compute_jac_phisalpha!(jac_phi::AbstractMatrix{T}, x::AbstractMatrix{T}, m::AbstractVector{T}, pair::Matrix{Bool}, h::T, alpha::T, n::Int, dadq, a, rij, aij, dotdadq) where T
    coeff = alpha * h^3 / 96 * 2 * GNEWT

    # Compute accelerations its derivatives
    @inbounds for i in 1:n-1
        for j in i+1:n
            if ~pair[i,j] # correction for Kepler pairs
                for k in 1:3
                    rij[k] = x[k,i] - x[k,j]
                end
                r2 = dot_fast(rij)
                r3 = r2 * sqrt(r2)
                fac2 = GNEWT / r3

                for k in 1:3
                    fac = fac2 * rij[k]
                    a[k,i] -= m[j]*fac
                    a[k,j] += m[i]*fac

                    # Mass derivatives of acceleration vector
                    # No velocity dependence, so 4th parameter
                    dadq[k,i,4,j] -= fac
                    dadq[k,j,4,i] += fac

                    # position derivatives
                    fac *= 3.0 / r2
                    for p in 1:3
                        facmjp = fac * m[j] * rij[p]
                        facmip = fac * m[i] * rij[p]
                        dadq[k,i,p,i] += facmjp
                        dadq[k,i,p,j] -= facmjp
                        dadq[k,j,p,j] += facmip
                        dadq[k,j,p,i] -= facmip
                    end

                    # Diagonal terms
                    fac2j = fac2*m[j]
                    fac2i = fac2*m[i]
                    dadq[k,i,k,i] -= fac2j
                    dadq[k,i,k,j] += fac2j
                    dadq[k,j,k,j] -= fac2i
                    dadq[k,j,k,i] += fac2i
                end
            end
        end
    end

    # Now compute g_i acceleration vector and derivatives
    @inbounds for i in 1:n-1
        indi = (i-1)*7
        for j in i+1:n
            if ~pair[i,j] # correction for Kepler pairs
                indj = (j-1)*7
                for k in 1:3
                    rij[k] = x[k,i] - x[k,j]
                    aij[k] = a[k,i] - a[k,j]
                end

                # Compute dot product of r_ij with \delta a_ij
                fill!(dotdadq, 0.0)
                @inbounds for di in 1:n, p in 1:4, k in 1:3
                    dotdadq[p,di] += rij[k] * (dadq[k,i,p,di] - dadq[k,j,p,di])
                end

                r2 = dot_fast(rij)
                r1 = sqrt(r2)
                ardot = dot_fast(aij, rij)
                fac1 = coeff / (r2*r2*r1)
                fac2 = 2*GNEWT*(m[i] + m[j])/r1 + 3*ardot

                for k in 1:3
                    fac = fac1 * (rij[k]*fac2 - r2*aij[k])

                    # Mass derivatives (first)
                    jac_phi[indi+3+k, indj+7] += fac
                    jac_phi[indj+3+k, indi+7] -= fac

                    # position derivatives (dot product)
                    fac *= 5.0 / r2
                    @inbounds for p in 1:3
                        faci = fac * m[i] * rij[p]
                        facj = fac * m[j] * rij[p]
                        jac_phi[indi+3+k, indi+p] -= facj
                        jac_phi[indi+3+k, indj+p] += facj
                        jac_phi[indj+3+k, indj+p] -= faci
                        jac_phi[indj+3+k, indi+p] += faci
                    end

                    # Second mass derivaive
                    fac = 2*GNEWT*fac1*rij[k]/r1
                    faci = fac * m[i]
                    facj = fac * m[j]
                    jac_phi[indi+3+k, indi+7] += facj
                    jac_phi[indi+3+k, indj+7] += facj
                    jac_phi[indj+3+k, indj+7] -= faci
                    jac_phi[indj+3+k, indi+7] -= faci

                    # Diagonal position terms
                    fac = fac1 * fac2
                    faci = fac * m[i]
                    facj = fac * m[j]
                    jac_phi[indi+3+k, indi+k] += facj
                    jac_phi[indi+3+k, indj+k] -= facj
                    jac_phi[indj+3+k, indj+k] += faci
                    jac_phi[indj+3+k, indi+k] -= faci

                    # Dot product \delta rij terms
                    fac = -2*fac1*(rij[k]*GNEWT*(m[i]+m[j])/(r2*r1) + aij[k])
                    @inbounds for p in 1:3
                        fac3 = fac*rij[p] + fac1*3.0*rij[k]*aij[p]
                        fac3i = fac3 * m[i]
                        fac3j = fac3 * m[j]
                        jac_phi[indi+3+k, indi+p] += fac3j
                        jac_phi[indi+3+k, indj+p] -= fac3j
                        jac_phi[indj+3+k, indj+p] += fac3i
                        jac_phi[indj+3+k, indi+p] -= fac3i
                    end

                    # Diagonal acceleration terms
                    fac = -fac1 * r2
                    faci = fac * m[i]
                    facj = fac * m[j]
                    @inbounds for di in 1:n
                        indd = (di-1)*7
                        for p in 1:3
                            dadq_p = dadq[k,i,p,di] - dadq[k,j,p,di]
                            jac_phi[indi+3+k, indd+p] += facj * dadq_p
                            jac_phi[indj+3+k, indd+p] -= faci * dadq_p
                        end
                        dadq_4 = dadq[k,i,4,di] - dadq[k,j,4,di]
                        jac_phi[indi+3+k, indd+7] += facj * dadq_4
                        jac_phi[indj+3+k, indd+7] -= faci * dadq_4
                    end

                    # Now, for the final term: (\delta a_ij . r_ij) r_ij
                    fac = 3.0 * fac1 * rij[k]
                    faci = fac * m[i]
                    facj = fac * m[j]
                    @inbounds for di in 1:n
                        indd = (di-1)*7
                        for p in 1:3
                            jac_phi[indi+3+k, indd+p] += facj * dotdadq[p,di]
                            jac_phi[indj+3+k, indd+p] -= faci * dotdadq[p,di]
                        end
                        jac_phi[indi+3+k, indd+7] += facj * dotdadq[4,di]
                        jac_phi[indj+3+k, indd+7] -= faci * dotdadq[4,di]
                    end
                end
            end
        end
    end
    return
end

function EnzymeRules.reverse(
    config,
    func::EnzymeCore.Const{typeof(NbodyGradient.phisalpha!)},
    ::Type{<:EnzymeCore.Const},
    tape,
    s::EnzymeCore.Duplicated{State{T, EnzymeWorkspace{T}}},
    h::EnzymeCore.Const{T},
    alpha::EnzymeCore.Const{T}
    ) where T

    n = s.val.n

    # Get saved positions and parameters
    x_save, h_val, alpha_val = tape

    # Get pre-allocated workspace (only allocates on first call)
    ws = get_enzyme_workspace(s.dval)

    # Reset workspace arrays
    fill!(ws.jac_phi, zero(T))
    fill!(ws.dadq, zero(T))
    fill!(ws.a, zero(T))

    # Compute Jacobian
    compute_jac_phisalpha!(ws.jac_phi, x_save, s.val.m, s.val.pair, h_val, alpha_val, n,
                           ws.dadq, ws.a, ws.rij, ws.aij, ws.dotdadq)

    # Build gradient vector from shadow (explicit indexing to avoid allocations)
    @inbounds for i in 1:n
        idx = (i-1)*7
        for k in 1:3
            ws.grad_out[idx+k] = s.dval.x[k,i]
            ws.grad_out[idx+3+k] = s.dval.v[k,i]
        end
        ws.grad_out[idx+7] = s.dval.m[i]
    end

    # Apply VJP
    mul!(ws.grad_in, ws.jac_phi', ws.grad_out)

    # Place back into shadow (explicit indexing)
    @inbounds for i in 1:n
        idx = (i-1)*7
        for k in 1:3
            s.dval.x[k,i] += ws.grad_in[idx+k]
            s.dval.v[k,i] += ws.grad_in[idx+3+k]
        end
        s.dval.m[i] += ws.grad_in[idx+7]
    end

    return (nothing, nothing, nothing)
end

function compute_jac_ij!(s::State{T}, x_i, x_j, v_i, v_j, i, j, h, drift_first::Bool, jac_kepler, jac_mass, jac_ij) where T
    @inbounds for k = 1:3
        s.x0[k] = x_i[k] - x_j[k]
        s.v0[k] = v_i[k] - v_j[k]
    end

    gm = GNEWT*(s.m[i] + s.m[j])
    if gm == 0; return; end

    # Get 22 params and fill s.delvx
    fill!(jac_kepler, 0.0)
    fill!(jac_mass, 0.0)
    params::NTuple{22, T} = jac_delxv_gamma!(s,gm,h,drift_first)
    compute_jacobian_gamma!(params..., s.x0, s.v0, jac_kepler, jac_mass, drift_first, false)

    # Compute jacobian
    mijinv = 1.0 / (s.m[i] + s.m[j])
    mi = s.m[i] * mijinv
    mj = s.m[j] * mijinv
    # pos and vel derivatives first
    @inbounds for l in 1:6, k in 1:6
        jac_kep_kl = jac_kepler[k,l]
        jac_ij[k,l]     += mj*jac_kep_kl
        jac_ij[k,7+l]   -= mj*jac_kep_kl
        jac_ij[7+k,l]   -= mi*jac_kep_kl
        jac_ij[7+k,7+l] += mi*jac_kep_kl
    end
    # Now mass derivatives
    @inbounds for k in 1:6
        jac_ij[k,7] = jac_mass[k] * s.m[j]
        jac_ij[k,14] = mi*s.delxv[k]*mijinv + GNEWT*mj*jac_kepler[k,7]
        jac_ij[7+k,7] = -mj*s.delxv[k]*mijinv - GNEWT*mi*jac_kepler[k,7]
        jac_ij[7+k,14] = -jac_mass[k]*s.m[i]
    end
end

function EnzymeRules.augmented_primal(
    config,
    func::EnzymeCore.Const{typeof(NbodyGradient.drift_kepler!)},
    ::Type{<:EnzymeCore.Const},
    s::EnzymeCore.Duplicated{State{T, EnzymeWorkspace{T}}},
    h::EnzymeCore.Const{T},
    ) where T

    n = s.val.n

    # Count number of kepler pairs
    num_pairs = 0
    @inbounds for i in 1:n-1, j in i+1:n
        if ~s.val.pair[i,j]
            num_pairs += 1
        end
    end

    # Only save the TWO bodies involved in each pair (not all n bodies)
    x_save = Array{T, 3}(undef, 3, 2, num_pairs)
    v_save = Array{T, 3}(undef, 3, 2, num_pairs)
    pair_indices = Vector{Tuple{Int, Int}}(undef, num_pairs)

    # Do the drift+kepler step for each pair, and save the state after each pair step
    idx = 0
    @inbounds for i in 1:n-1, j in i+1:n
        if ~s.val.pair[i,j]
            idx += 1
            # Save only bodies i and j
            for k in 1:3
                x_save[k,1,idx] = s.val.x[k,i]
                x_save[k,2,idx] = s.val.x[k,j]
                v_save[k,1,idx] = s.val.v[k,i]
                v_save[k,2,idx] = s.val.v[k,j]
            end
            pair_indices[idx] = (i, j)
            kepler_driftij_gamma!(s.val, i, j, h.val, true)
        end
    end

    return EnzymeRules.AugmentedReturn(nothing, nothing, (x_save, v_save, pair_indices, h.val))
end

function EnzymeRules.reverse(
    config,
    func::EnzymeCore.Const{typeof(NbodyGradient.drift_kepler!)},
    ::Type{<:EnzymeCore.Const},
    tape,
    s::EnzymeCore.Duplicated{State{T, EnzymeWorkspace{T}}},
    h::EnzymeCore.Const{T},
    ) where T

    n = s.val.n

    # Get saved positions and parameters
    x_save, v_save, pair_indices, h_val = tape
    n_pairs = length(pair_indices)

    # Get pre-allocated workspace (only allocates on first call)
    ws = get_enzyme_workspace(s.dval)

    # Do loop in reverse order
    @inbounds for idx in n_pairs:-1:1
        i,j = pair_indices[idx]
        # Make the grad_out for this pair (explicit indexing to avoid allocations)
        for k in 1:3
            ws.grad_out_ij[k] = s.dval.x[k,i]
            ws.grad_out_ij[k+3] = s.dval.v[k,i]
            ws.grad_out_ij[k+7] = s.dval.x[k,j]
            ws.grad_out_ij[k+10] = s.dval.v[k,j]
        end
        ws.grad_out_ij[7] = s.dval.m[i]
        ws.grad_out_ij[14] = s.dval.m[j]

        # Compute jac_ij (x_save[:,1,idx] = body i, x_save[:,2,idx] = body j)
        fill!(ws.jac_ij, zero(T))
        compute_jac_ij!(s.val, @view(x_save[:,1,idx]), @view(x_save[:,2,idx]),
                        @view(v_save[:,1,idx]), @view(v_save[:,2,idx]),
                        i, j, h_val, true, ws.jac_kepler, ws.jac_mass, ws.jac_ij)

        # Compute the VJP
        mul!(ws.grad_in_ij, ws.jac_ij', ws.grad_out_ij)

        # Accumulate back (explicit indexing)
        for k in 1:3
            s.dval.x[k,i] += ws.grad_in_ij[k]
            s.dval.v[k,i] += ws.grad_in_ij[k+3]
            s.dval.x[k,j] += ws.grad_in_ij[k+7]
            s.dval.v[k,j] += ws.grad_in_ij[k+10]
        end
        s.dval.m[i] += ws.grad_in_ij[7]
        s.dval.m[j] += ws.grad_in_ij[14]
    end

    return (nothing, nothing)
end

function EnzymeRules.augmented_primal(
    config,
    func::EnzymeCore.Const{typeof(NbodyGradient.kepler_drift!)},
    ::Type{<:EnzymeCore.Const},
    s::EnzymeCore.Duplicated{State{T, EnzymeWorkspace{T}}},
    h::EnzymeCore.Const{T},
    ) where T

    n = s.val.n

    # Count number of kepler pairs
    num_pairs = 0
    @inbounds for i in n-1:-1:1, j in n:-1:i+1
        if ~s.val.pair[i,j]
            num_pairs += 1
        end
    end

    # Only save the TWO bodies involved in each pair (not all n bodies)
    x_save = Array{T, 3}(undef, 3, 2, num_pairs)
    v_save = Array{T, 3}(undef, 3, 2, num_pairs)
    pair_indices = Vector{Tuple{Int, Int}}(undef, num_pairs)

    # Do the kepler+drift step for each pair, and save the state after each pair step
    idx = 0
    @inbounds for i in n-1:-1:1, j in n:-1:i+1
        if ~s.val.pair[i,j]
            idx += 1
            # Save only bodies i and j
            for k in 1:3
                x_save[k,1,idx] = s.val.x[k,i]
                x_save[k,2,idx] = s.val.x[k,j]
                v_save[k,1,idx] = s.val.v[k,i]
                v_save[k,2,idx] = s.val.v[k,j]
            end
            pair_indices[idx] = (i, j)
            kepler_driftij_gamma!(s.val, i, j, h.val, false)
        end
    end

    return EnzymeRules.AugmentedReturn(nothing, nothing, (x_save, v_save, pair_indices, h.val))
end

function EnzymeRules.reverse(
    config,
    func::EnzymeCore.Const{typeof(NbodyGradient.kepler_drift!)},
    ::Type{<:EnzymeCore.Const},
    tape,
    s::EnzymeCore.Duplicated{State{T, EnzymeWorkspace{T}}},
    h::EnzymeCore.Const{T},
    ) where T

    n = s.val.n

    # Get saved positions and parameters
    x_save, v_save, pair_indices, h_val = tape
    n_pairs = length(pair_indices)

    # Get pre-allocated workspace (only allocates on first call)
    ws = get_enzyme_workspace(s.dval)

    # Do loop in reverse order
    @inbounds for idx in n_pairs:-1:1
        i,j = pair_indices[idx]
        # Make the grad_out for this pair (explicit indexing to avoid allocations)
        for k in 1:3
            ws.grad_out_ij[k] = s.dval.x[k,i]
            ws.grad_out_ij[k+3] = s.dval.v[k,i]
            ws.grad_out_ij[k+7] = s.dval.x[k,j]
            ws.grad_out_ij[k+10] = s.dval.v[k,j]
        end
        ws.grad_out_ij[7] = s.dval.m[i]
        ws.grad_out_ij[14] = s.dval.m[j]

        # Compute jac_ij (x_save[:,1,idx] = body i, x_save[:,2,idx] = body j)
        fill!(ws.jac_ij, zero(T))
        compute_jac_ij!(s.val, @view(x_save[:,1,idx]), @view(x_save[:,2,idx]),
                        @view(v_save[:,1,idx]), @view(v_save[:,2,idx]),
                        i, j, h_val, false, ws.jac_kepler, ws.jac_mass, ws.jac_ij)

        # Compute the VJP
        mul!(ws.grad_in_ij, ws.jac_ij', ws.grad_out_ij)

        # Accumulate back (explicit indexing)
        for k in 1:3
            s.dval.x[k,i] += ws.grad_in_ij[k]
            s.dval.v[k,i] += ws.grad_in_ij[k+3]
            s.dval.x[k,j] += ws.grad_in_ij[k+7]
            s.dval.v[k,j] += ws.grad_in_ij[k+10]
        end
        s.dval.m[i] += ws.grad_in_ij[7]
        s.dval.m[j] += ws.grad_in_ij[14]
    end

    return (nothing, nothing)
end
