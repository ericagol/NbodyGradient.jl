# TTV structures
import Base.@kwdef

"""

Holds the transit times and derivatives.
"""
struct TransitTiming{T<:AbstractFloat} <: AbstractOutput
    tt::Matrix{T}
    dtdq0::Array{T,4}
    dtdelements::Array{T,4}
    count::Vector{Int64}
    ntt::Int64
end

function TransitTiming(tmax,ic::ElementsIC{T}) where T<:AbstractFloat
    n = ic.nbody
    ind = isfinite.(tmax./ic.elements[:,2])
    ntt = maximum(ceil.(Int64,tmax./ic.elements[ind,2]).+3)
    tt = zeros(T,n,ntt)
    dtdq0 = zeros(T,n,ntt,7,n)
    dtdelements = zeros(T,n,ntt,7,n)
    count = zeros(Int64,n)
    return TransitTiming(tt,dtdq0,dtdelements,count,ntt)
end

"""

Integrator method for outputing `TransitTiming`.
"""
function (i::Integrator)(s::State{T},tt::TransitTiming) where T<:AbstractFloat 
    s2 = zero(T) # For compensated summation

    # Preallocate struct of arrays for derivatives (and pair)
    pair = zeros(Bool,s.n,s.n)

    # Run integrator and calculate transit times, with derivatives.
    rstar = 1e12 # Need to pass this in. 
    ttv!(s,i,tt,rstar,pair)
    dttv!(s,tt)
#=
    while s.t < (i.t0 + i.tmax)
        # Take integration step and advance time
        i.scheme(s,d,i.h,pair)
        s.t,s2 = comp_sum(s.t,s2,i.h)
    end
=#
    return
end

# Includes for source
files = ["ttv.jl","ttv_no_grad.jl"]
include.(files)

function ttv!(s::State{T},i::Integrator,tt::TransitTiming{T},rstar,pair) where T<:AbstractFloat
    n = s.n; ntt_max = tt.ntt; h = i.h
    x = s.x; v = s.v; jac_step = s.jac_step; m = s.m; t = s.t
    xerror = s.xerror; verror = s.verror; jac_error = s.jac_error
    count = tt.count; dtdq0 = tt.dtdq0
    d = Jacobian(T,s.n) 
    xprior = copy(x)
    vprior = copy(v)
    xtransit = copy(x)
    vtransit = copy(v)
    xerr_trans = zeros(T,size(x)); verr_trans =zeros(T,size(v))
    xerr_prior = zeros(T,size(x)); verr_prior =zeros(T,size(v))
    # Define error estimate based on Kahan (1965):
    s2 = zero(T)
    # Set step counter to zero:
    istep = 0
    # Jacobian for each step (7- 6 elements+mass, n_planets, 7 - 6 elements+mass, n planets):
    jac_prior = zeros(T,7*n,7*n)
    jac_error_prior = zeros(T,7*n,7*n)
    jac_transit = zeros(T,7*n,7*n)
    jac_trans_err= zeros(T,7*n,7*n)
    # Initialize matrix for derivatives of transit times with respect to the initial x,v,m:
    dtdq = zeros(T,1,7,n)

    # Save the g function, which computes the relative sky velocity dotted with relative position
    # between the planets and star:
    gsave = zeros(T,n)
    for i=2:n
        # Compute the relative sky velocity dotted with position:
        gsave[i]= g!(i,1,x,v)
    end
    # Loop over time steps:
    dt = zero(T)
    gi = zero(T)
    param_real = all(isfinite.(x)) && all(isfinite.(v)) && all(isfinite.(m)) && all(isfinite.(jac_step))
    while s.t < (i.t0+i.tmax) && param_real
        # Carry out a ah18 mapping step:
        #ah18!(x,v,xerror,verror,h,m,n,jac_step,jac_error,pair)
        i.scheme(s,d,h,pair)
        param_real = all(isfinite.(x)) && all(isfinite.(v)) && all(isfinite.(m)) && all(isfinite.(jac_step))
        # Check to see if a transit may have occured.  Sky is x-y plane; line of sight is z.
        # Star is body 1; planets are 2-nbody (note that this could be modified to see if
        # any body transits another body):
        for i=2:n
            # Compute the relative sky velocity dotted with position:
            gi = g!(i,1,x,v)
            ri = sqrt(x[1,i]^2+x[2,i]^2+x[3,i]^2)  # orbital distance
            # See if sign of g switches, and if planet is in front of star (by a good amount):
            # (I'm wondering if the direction condition means that z-coordinate is reversed? EA 12/11/2017)
            if gi > 0 && gsave[i] < 0 && x[3,i] > 0.25*ri && ri < rstar
                # A transit has occurred between the time steps - integrate ah18! between timesteps
                count[i] += 1
                if count[i] <= ntt_max
                    dt0 = -gsave[i]*h/(gi-gsave[i])  # Starting estimate
                    xtransit .= xprior; vtransit .= vprior; jac_transit .= jac_prior; jac_trans_err .= jac_error_prior
                    xerr_trans .= xerr_prior; verr_trans .= verr_prior
                    dt = findtransit3!(1,i,n,h,dt0,m,xtransit,vtransit,xerr_trans,verr_trans,jac_transit,jac_trans_err,dtdq,pair) # 20%
                    tt.tt[i,count[i]],stmp = comp_sum(s.t,s2,dt)
                    # Save for posterity:
                    for k=1:7, p=1:n
                        dtdq0[i,count[i],k,p] = dtdq[1,k,p]
                    end
                end
            end
            gsave[i] = gi
        end
        # Save the current state as prior state:
        xprior .= x
        vprior .= v
        jac_prior .= jac_step
        jac_error_prior .= jac_error
        xerr_prior .= xerror
        verr_prior .= verror
        # Increment time by the time step using compensated summation:
        s.t,s2 = comp_sum(s.t,s2,i.h)
        # Increment counter by one:
        istep +=1
    end
    return
end

function dttv!(s::State{T},tt::TransitTiming{T}) where T <: AbstractFloat
    for i=1:s.n, j=1:tt.count[i]
        if j <= tt.ntt
            # Now, multiply by the initial Jacobian to convert time derivatives to orbital elements:
            for k=1:s.n, l=1:7
                tt.dtdelements[i,j,l,k] = zero(T)
                for p=1:s.n, q=1:7
                    tt.dtdelements[i,j,l,k] += tt.dtdq0[i,j,q,p]*s.jac_init[(p-1)*7+q,(k-1)*7+l]
                end
            end
        end
    end
end