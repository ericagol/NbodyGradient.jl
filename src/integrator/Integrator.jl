import NbodyGradient: InitialConditions

#========== Integrator ==========#
abstract type AbstractIntegrator end

"""

Integrator to be used, and relevant parameters.
"""
mutable struct Integrator{T<:AbstractFloat} <: AbstractIntegrator
    scheme::Function
    h::T
    t0::T
    tmax::T

    Integrator(scheme::Function,h::T,t0::T,tmax::T) where T<:AbstractFloat = new{T}(scheme,h,t0,tmax)
end

Integrator(h::T,t0::T) where T<:AbstractFloat = Integrator(ah18!,h,t0,t0+h)
Integrator(scheme::Function,h::Real,t0::Real,tmax::Real) = Integrator(scheme,promote(h,t0,tmax)...)

# Default to ah18!
Integrator(h::T,t0::T,tmax::T) where T<:AbstractFloat = Integrator(ah18!,h,t0,tmax)

#========== State ==========#
abstract type AbstractState end

"""

Current state of simulation.
"""
struct State{T<:AbstractFloat} <: AbstractState
    x::Matrix{T}
    v::Matrix{T}
    t::Vector{T}
    m::Vector{T}
    jac_step::Matrix{T}
    dqdt::Vector{T}
    jac_init::Matrix{T}
    xerror::Matrix{T}
    verror::Matrix{T}
    dqdt_error::Vector{T}
    jac_error::Matrix{T}
    n::Int64

    rij::Vector{T} ## kickfast!, phic!, phisalpha!
    a::Matrix{T} ## phic!, phisalpha!
    aij::Vector{T} ## phic!, phisalpha!
    x0::Vector{T} ## kepler_driftij_gamma!
    v0::Vector{T} ## kepler_driftij_gamma!
    input::Vector{T} ## jac_delxv_gamma!
    delxv::Vector{T} ##  jac_delxv_gamma!
    rtmp::Vector{T}
end

function State(ic::InitialConditions{T}) where T<:AbstractFloat
    x,v,jac_init = init_nbody(ic)
    n = ic.nbody
    xerror = zeros(T,size(x))
    verror = zeros(T,size(v))
    jac_step = Matrix{T}(I,7*n,7*n)
    dqdt = zeros(T,7*n)
    dqdt_error = zeros(T,size(dqdt))
    jac_error = zeros(T,size(jac_step))

    rij = zeros(T,3)
    a = zeros(T,3,n)
    aij = zeros(T,3)
    x0 = zeros(T,3)
    v0 = zeros(T,3)
    input = zeros(T,8)
    delxv = zeros(T,6)
    rtmp = zeros(T,3)
    return State(x,v,[ic.t0],ic.m,jac_step,dqdt,jac_init,xerror,verror,dqdt_error,jac_error,ic.nbody,
    rij,a,aij,x0,v0,input,delxv,rtmp)
end

function set_state!(s_old::State,s_new::State)
    fields = setdiff(fieldnames(State),[:m,:n])
    for fn in fields
        f_new = getfield(s_new,fn)
        f_old = getfield(s_old,fn)
        f_old .= f_new
    end
    return
end

"""Shows if the positions, velocities, and Jacobian are finite."""
Base.show(io::IO,::MIME"text/plain",s::State{T}) where {T} = begin
    println(io,"State{$T}:"); 
    println(io,"Positions  : ", all(isfinite.(s.x)) ? "finite" : "infinite!"); 
    println(io,"Velocities : ", all(isfinite.(s.v)) ? "finite" : "infinite!");
    println(io,"Jacobian   : ", all(isfinite.(s.jac_step)) ? "finite" : "infinite!");
    return
end 

#========== Running Methods ==========#
"""
    (::Integrator)(s, time; grad=true)

Callable `Integrator` method. Integrate to specific time.
"""
function (intr::Integrator)(s::State{T},time::T;grad::Bool=true) where T<:AbstractFloat 
    t0 = s.t[1]

    # Calculate number of steps
    nsteps = abs(round(Int64,(time - t0)/intr.h))

    # Step either forward or backward
    h = intr.h * check_step(t0,time)

    # Calculate last step (if needed)
    #while t0 + (h * nsteps) <= time; nsteps += 1; end
    tmax = t0 + (h * nsteps)

    # Preallocate struct of arrays for derivatives (and pair)
    if grad; d = Derivatives(T,s.n); end
    pair = zeros(Bool,s.n,s.n)

    for i in 1:nsteps
        # Take integration step and advance time
        if grad
            intr.scheme(s,d,h,pair)
        else
            intr.scheme(s,h,pair)
        end
    end

    # Do last step (if needed)
    if nsteps == 0; hf = time; end
    if tmax != time
        hf = time - tmax
        if grad
            intr.scheme(s,d,hf,pair)
        else
            intr.scheme(s,hf,pair)
        end
    end

    s.t[1] = time
    return 
end

"""

Take N steps.
"""
function (intr::Integrator)(s::State{T},N::Int64;grad::Bool=true) where T<:AbstractFloat
    s2 = zero(T) # For compensated summation

    # Preallocate struct of arrays for derivatives (and pair)
    if grad; d = Derivatives(T,s.n); end
    pair = zeros(Bool,s.n,s.n)

    # check to see if backward step
    if N < 0; intr.h *= -1; N *= -1; end

    for n in 1:N
        # Take integration step and advance time
        if grad
            intr.scheme(s,d,intr.h,pair)
        else
            intr.scheme(s,intr.h,pair)
        end
        s.t[1],s2 = comp_sum(s.t[1],s2,intr.h)
    end

    # Return time step to forward if needed
    if intr.h < 0; intr.h *= -1; end
    return
end

"""Integrates to `i.tmax`."""
(intr::Integrator)(s::State{T};grad::Bool=true) where T<:AbstractFloat = intr(s,intr.tmax,grad=grad)

"""

Integrate in the direction of tmax.
"""
function check_step(t0::T,tmax::T) where T<:AbstractFloat
    if abs(tmax) > abs(t0)
        return sign(tmax)
    else
        if sign(tmax) != sign(t0)
            return sign(tmax)
        else
            return -1 * sign(tmax)
        end
    end
end



#========== Includes  ==========#
const ints = ["ah18"]
for i in ints; include(joinpath(i,"$i.jl")); end

const ints_no_grad = ["ah18","dh17"]
for i in ints_no_grad; include(joinpath(i,"$(i)_no_grad.jl")); end

"""

Carry out AH18 mapping and calculates both the Jacobian and derivative with respect to the time step.
"""
function ah18!(s::State{T},d::Derivatives{T},h::T,pair::Matrix{Bool}) where T<:AbstractFloat
    zilch = zero(T); uno = one(T); half = convert(T,0.5); two = convert(T,2.0); h2 = half*h; sevn = 7*s.n
    n = s.n
    zero_out!(d)
    fill!(s.dqdt,zilch)

    drift!(s.x,s.v,s.xerror,s.verror,h2,s.n,s.jac_step,s.jac_error)
    # Compute time derivative of drift step:
    @inbounds for i=1:n, k=1:3
        s.dqdt[(i-1)*7+k] = half*s.v[k,i] + h2*s.dqdt[(i-1)*7+3+k]
    end
    kickfast!(s.x,s.v,s.xerror,s.verror,h/6,s.m,s.n,d.jac_kick,d.dqdt_kick,pair)
    d.dqdt_kick ./= 6 # Since step is h/6
    # Since I removed identity from kickfast, need to add in dqdt:
    s.dqdt .+= d.dqdt_kick .+ *(d.jac_kick,s.dqdt)
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
                # Copy current time derivatives for multiplication purposes:
                @inbounds for k1=1:7
                    d.dqdt_tmp1[  k1] = s.dqdt[indi+k1]
                    d.dqdt_tmp1[7+k1] = s.dqdt[indj+k1]
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
                # Add in partial derivatives with respect to time:
                # Need to multiply by 1/2 since we're taking 1/2 time step:
                #    BLAS.gemm!('N','N',uno,jac_ij,dqdt_tmp1,half,dqdt_ij)
                d.dqdt_ij .*= half
                d.dqdt_ij .+= *(d.jac_ij,d.dqdt_tmp1) .+ d.dqdt_tmp1
                # Copy back time derivatives:
                @inbounds for k1=1:7
                    s.dqdt[indi+k1] = d.dqdt_ij[  k1]
                    s.dqdt[indj+k1] = d.dqdt_ij[7+k1]
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
    # Add in time derivative with respect to prior parameters:
    #BLAS.gemm!('N','N',uno,jac_phi,dqdt,uno,dqdt_phi)
    # Copy result to dqdt:
    s.dqdt .+= d.dqdt_phi .+ *(d.jac_phi,s.dqdt)
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
                # Copy current time derivatives for multiplication purposes:
                @inbounds for k1=1:7
                    d.dqdt_tmp1[  k1] = s.dqdt[indi+k1]
                    d.dqdt_tmp1[7+k1] = s.dqdt[indj+k1]
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
                # Add in partial derivatives with respect to time:
                # Need to multiply by 1/2 since we're taking 1/2 time step:
                #BLAS.gemm!('N','N',uno,jac_ij,dqdt_tmp1,half,dqdt_ij)
                d.dqdt_ij .*= half
                d.dqdt_ij .+= *(d.jac_ij,d.dqdt_tmp1) .+ d.dqdt_tmp1
                # Copy back time derivatives:
                @inbounds for k1=1:7
                    s.dqdt[indi+k1] = d.dqdt_ij[  k1]
                    s.dqdt[indj+k1] = d.dqdt_ij[7+k1]
                end
            end
        end
    end
    fill!(d.dqdt_kick,zilch)
    #kickfast!(x,v,h2,m,n,jac_kick,dqdt_kick,pair)
    kickfast!(s.x,s.v,s.xerror,s.verror,h/6,s.m,s.n,d.jac_kick,d.dqdt_kick,pair)
    d.dqdt_kick ./= 6 # Since step is h/6
    # Copy result to dqdt:
    s.dqdt .+= d.dqdt_kick .+ *(d.jac_kick,s.dqdt)
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
    # Compute time derivative of drift step:
    @inbounds for i=1:n, k=1:3
        s.dqdt[(i-1)*7+k] += half*s.v[k,i] + h2*s.dqdt[(i-1)*7+3+k]
    end
    return
end
