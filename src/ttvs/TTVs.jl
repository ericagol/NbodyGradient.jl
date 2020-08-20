include("brent.jl")

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
    #s2 = zero(T) # For compensated summation

    # Preallocate struct of arrays for derivatives (and pair)
    pair = zeros(Bool,s.n,s.n)

    # Run integrator and calculate transit times, with derivatives.
    rstar = 1e5 # Need to pass this in. 
    ttv!(s,i,tt,rstar,pair)
    dttv!(s,tt)

    return
end

# Includes for source
files = ["ttv.jl","ttv_no_grad.jl"]
include.(files)

function ttv!(s::State{T},intr::Integrator,tt::TransitTiming{T},rstar,pair) where T<:AbstractFloat
    n = s.n; ntt_max = tt.ntt;
    d = Jacobian(T,s.n) 
    dT = dTime(T,s.n)
    xprior = copy(s.x)
    vprior = copy(s.v)
    #xtransit = copy(x)
    #vtransit = copy(v)
    #xerr_trans = zeros(T,size(x)); verr_trans =zeros(T,size(v))
    xerr_prior = zeros(T,size(s.x)); verr_prior =zeros(T,size(s.v))
    # Define error estimate based on Kahan (1965):
    s2 = zero(T)
    # Set step counter to zero:
    istep = 0
    # Jacobian for each step (7- 6 elements+mass, n_planets, 7 - 6 elements+mass, n planets):
    jac_prior = zeros(T,7*s.n,7*s.n)
    jac_error_prior = zeros(T,7*s.n,7*s.n)
    #jac_transit = zeros(T,7*n,7*n)
    #jac_trans_err = zeros(T,7*n,7*n)
    # Initialize matrix for derivatives of transit times with respect to the initial x,v,m:
    dtdq = zeros(T,1,7,s.n)

    # Save the g function, which computes the relative sky velocity dotted with relative position
    # between the planets and star:
    gsave = zeros(T,s.n)
    for i=2:s.n
        # Compute the relative sky velocity dotted with position:
        gsave[i]= g!(i,1,s.x,s.v)
    end
    # Loop over time steps:
    dt = zero(T)
    gi = zero(T)
    param_real = all(isfinite.(s.x)) && all(isfinite.(s.v)) && all(isfinite.(s.m)) && all(isfinite.(s.jac_step))
    while s.t < (intr.t0+intr.tmax) && param_real
        # Carry out a ah18 mapping step:
        #ah18!(x,v,xerror,verror,h,m,n,jac_step,jac_error,pair)
        intr.scheme(s,d,intr.h,pair)
        param_real = all(isfinite.(s.x)) && all(isfinite.(s.v)) && all(isfinite.(s.m)) && all(isfinite.(s.jac_step))
        # Check to see if a transit may have occured.  Sky is x-y plane; line of sight is z.
        # Star is body 1; planets are 2-nbody (note that this could be modified to see if
        # any body transits another body):
        for i=2:s.n
            # Compute the relative sky velocity dotted with position:
            gi = g!(i,1,s.x,s.v)
            ri = sqrt(s.x[1,i]^2+s.x[2,i]^2+s.x[3,i]^2)  # orbital distance
            # See if sign of g switches, and if planet is in front of star (by a good amount):
            # (I'm wondering if the direction condition means that z-coordinate is reversed? EA 12/11/2017)
            if gi > 0 && gsave[i] < 0 && s.x[3,i] > 0.25*ri && ri < rstar
                # A transit has occurred between the time steps - integrate ah18! between timesteps
                tt.count[i] += 1
                if tt.count[i] <= ntt_max
                    dt0 = -gsave[i]*intr.h/(gi-gsave[i])  # Starting estimate
                    #xtransit .= xprior; vtransit .= vprior; jac_transit .= jac_prior; jac_trans_err .= jac_error_prior
                    #xerr_trans .= xerr_prior; verr_trans .= verr_prior
                    revert_state!(s,xprior,vprior,xerr_prior,verr_prior,jac_prior,jac_error_prior)
                    dt = findtransit3!(1,i,dt0,s,d,dT,intr,dtdq,pair)
                    #dt = findtransit3!(1,i,n,h,dt0,m,xtransit,vtransit,xerr_trans,verr_trans,jac_transit,jac_trans_err,dtdq,pair) # 20%
                    tt.tt[i,tt.count[i]],stmp = comp_sum(s.t,s2,dt)
                    # Save for posterity:
                    for k=1:7, p=1:n
                        tt.dtdq0[i,tt.count[i],k,p] = dtdq[1,k,p]
                    end
                end
            end
            gsave[i] = gi
        end
        # Save the current state as prior state:
        xprior .= s.x
        vprior .= s.v
        jac_prior .= s.jac_step
        jac_error_prior .= s.jac_error
        xerr_prior .= s.xerror
        verr_prior .= s.verror
        # Increment time by the time step using compensated summation:
        s,s2 = step_time(s,intr.h,s2)
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

function findtransit3!(i::Int64,j::Int64,dt0::T,s::State{T},d::Jacobian{T},dT::dTime{T},intr::Integrator,dtbvdq::Array{T,3},pair::Array{Bool,2}) where T<:AbstractFloat
    # Computes the transit time, approximating the motion as a fraction of a AH17 step forward in time.
    # Also computes the Jacobian of the transit time with respect to the initial parameters, dtbvdq[1-3,7,n].
    # This version is same as findtransit2, but uses the derivative of AH17 step with respect to time
    # rather than the instantaneous acceleration for finding transit time derivative (gdot).
    # Initial guess using linear interpolation:
    
    #x = s.x; v = s.v
    n = s.n;
    x1 = copy(s.x); v1 = copy(s.v);  
    xerror = copy(s.xerror); verror = copy(s.verror);
    jac_step = s.jac_step; jac_error = s.jac_error;
    s.dqdt .= 0.0

    dt = one(T)
    iter = 0
    r3 = zero(T)
    gdot = zero(T)
    gsky = zero(T)
    #x = copy(x1)
    #v = copy(v1)
    #xerr_trans = copy(xerror); verr_trans = copy(verror)
    TRANSIT_TOL = 10*eps(dt)
    tt1 = dt0 + 1
    tt2 = dt0 + 2
    ITMAX = 20
    while abs(dt) > TRANSIT_TOL && iter < 20
    #while true
        tt2 = tt1
        tt1 = dt0
        revert_state!(s,x1,v1,xerror,verror)
        #x .= x1; v .= v1; xerr_trans .= xerror; verr_trans .= verror
        # Advance planet state at start of step to estimated transit time:
        #intr.scheme(x,v,xerr_trans,verr_trans,dt0,m,n,dqdt,pair)
        intr.scheme(s,dT,dt0,pair)
        # Compute time offset:
        gsky = g!(i,j,s.x,s.v)
        #  # Compute derivative of g with respect to time:
        gdot = gd!(i,j,s.x,s.v,s.dqdt)
        # Refine estimate of transit time with Newton's method:
        dt = -gsky/gdot
        # Add refinement to estimated time:
        dt0 += dt
        iter += 1
        # Break out if we have reached maximum iterations, or if
        # current transit time estimate equals one of the prior two steps:
        if (iter >= ITMAX) || (dt0 == tt1) || (dt0 == tt2)
            break
        end
    end
    if iter >= 20
        #println("Exceeded iterations: planet ",j," iter ",iter," dt ",dt," gsky ",gsky," gdot ",gdot)
    end 
    # Compute time derivatives:
    #x .= x1; v .= v1; xerr_trans .= xerror; verr_trans .= verror
    revert_state!(s,x1,v1,xerror,verror)
    # Compute dgdt with the updated time step.
    #intr.scheme(x,v,xerr_trans,verr_trans,dt0,m,n,jac_step,jac_error,pair)
    intr.scheme(s,d,dt0,pair)
    # Need to reset to compute dqdt:
    #x .= x1; v .= v1; xerr_trans .= xerror; verr_trans .= verror
    revert_state!(s,x1,v1,xerror,verror)
    #intr.scheme(x,v,xerr_trans,verr_trans,dt0,m,n,dqdt,pair)
    intr.scheme(s,dT,dt0,pair)
    # Compute derivative of transit time, impact parameter, and sky velocity.
    dtbvdq!(i,j,s.x,s.v,s.jac_step,s.dqdt,dtbvdq)

    # Note: this is the time elapsed *after* the beginning of the timestep:
    ntbv = size(dtbvdq)[1]
    if ntbv == 3
        # Compute the impact parameter and sky velocity:
        vsky = sqrt((v[1,j]-v[1,i])^2 + (v[2,j]-v[2,i])^2)
        bsky2 = (x[1,j]-x[1,i])^2 + (x[2,j]-x[2,i])^2
        # partial derivative v_{sky} with respect to time:
        dvdt = ((v[1,j]-v[1,i])*(dqdt[(j-1)*7+4]-dqdt[(i-1)*7+4])+(v[2,j]-v[2,i])*(dqdt[(j-1)*7+5]-dqdt[(i-1)*7+5]))/vsky
        # (note that \partial b/\partial t = 0 at mid-transit since g_{sky} = 0 mid-transit).
        @inbounds for p=1:n
            indp = (p-1)*7
            @inbounds for k=1:7
                # Compute derivatives:
                #v_{sky} = sqrt((v[1,j]-v[1,i])^2+(v[2,j]-v[2,i])^2)
                dtbvdq[2,k,p] = ((jac_step[indj+3,indp+k]-jac_step[indi+3,indp+k])*(v[1,j]-v[1,i])+(jac_step[indj+4,indp+k]-jac_step[indi+4,indp+k])*(v[2,j]-v[2,i]))/vsky + dvdt*dtbvdq[1,k,p]
                #b_{sky}^2 = (x[1,j]-x[1,i])^2+(x[2,j]-x[2,i])^2
                dtbvdq[3,k,p] = 2*((jac_step[indj  ,indp+k]-jac_step[indi  ,indp+k])*(x[1,j]-x[1,i])+(jac_step[indj+1,indp+k]-jac_step[indi+1,indp+k])*(x[2,j]-x[2,i]))
            end
        end
        # return the transit time, impact parameter, and sky velocity:
        return dt0::T,vsky::T,bsky2::T
    else
        return dt0::T
    end
end

"""Used in computing transit time inside `findtransit3`."""
function g!(i::Int64,j::Int64,x::Array{T,2},v::Array{T,2}) where {T <: Real}
    # See equation 8-10 Fabrycky (2008) in Seager Exoplanets book
    g = (x[1,j]-x[1,i])*(v[1,j]-v[1,i])+(x[2,j]-x[2,i])*(v[2,j]-v[2,i])
    return g
end
function gd!(i::Int64,j::Int64,x::Matrix{T},v::Matrix{T},dqdt::Array{T}) where T<:AbstractFloat
    return ((x[1,j]-x[1,i])*(dqdt[(j-1)*7+4]-dqdt[(i-1)*7+4])+(x[2,j]-x[2,i])*(dqdt[(j-1)*7+5]-dqdt[(i-1)*7+5])
            +(v[1,j]-v[1,i])*(dqdt[(j-1)*7+1]-dqdt[(i-1)*7+1])+(v[2,j]-v[2,i])*(dqdt[(j-1)*7+2]-dqdt[(i-1)*7+2]))
end
function dtbvdq!(i,j,x,v,jac_step,dqdt,dtbvdq)
    n = size(x)[1]
    # Compute time offset:
    gsky = g!(i,j,x,v)
    # Compute derivative of g with respect to time:
    gdot = gd!(i,j,x,v,dqdt)

    fill!(dtbvdq,zero(typeof(x[1])))
    indj = (j-1)*7+1
    indi = (i-1)*7+1
    @inbounds for p=1:n
        indp = (p-1)*7
        @inbounds for k=1:7
            # Compute derivatives:
            dtbvdq[1,k,p] = -((jac_step[indj  ,indp+k]-jac_step[indi  ,indp+k])*(v[1,j]-v[1,i])+(jac_step[indj+1,indp+k]-jac_step[indi+1,indp+k])*(v[2,j]-v[2,i])+
                          (jac_step[indj+3,indp+k]-jac_step[indi+3,indp+k])*(x[1,j]-x[1,i])+(jac_step[indj+4,indp+k]-jac_step[indi+4,indp+k])*(x[2,j]-x[2,i]))/gdot
        end
    end
end
