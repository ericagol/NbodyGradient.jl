# Collection of functions to compute transit times, impact parameters, sky-velocity, and derivatives.

function detect_transits!(s::State{T},d::Derivatives{T},tt::TransitOutput{T},intr::Integrator{T}; grad::Bool=true) where T<:AbstractFloat
    rstar::T = 1e12 # Could this be removed?
    # Save current state as prior state
    set_state!(tt.s_prior, s)

    # Check to see if a transit may have occured before current state.
    # Sky is x-y plane; line of sight is z.
    # Body being transited is tt.ti, tt.occs is list of occultors:
    for i in tt.occs
        # Compute the relative sky velocity dotted with position:
        gi = g!(i,tt.ti,s.x,s.v)
        ri = sqrt(s.x[1,i]^2+s.x[2,i]^2+s.x[3,i]^2)  # orbital distance
        # See if sign of g switches, and if planet is in front of star (by a good amount):
        if gi > 0 && tt.gsave[i] < 0 && -s.x[3,i] > 0.25*ri && ri < rstar
            # A transit has occurred between the time steps - integrate ahl21!
            tt.count[i] += 1
            if tt.count[i] <= tt.ntt
                dt0 = tt.gsave[i]*intr.h/(gi-tt.gsave[i]) # Starting estimate
                set_state!(s,tt.s_prior)
                findtransit!(tt.ti,i,dt0,s,d,tt,intr;grad=grad) # Search for transit time (integrating 'backward')
            end
        end
        tt.gsave[i] = gi
    end
    set_state!(s,tt.s_prior)
    return
end

function findtransit!(i::Int64,j::Int64,dt0::T,s::State{T},d::Derivatives{T},tt::TransitOutput{T},intr::Integrator;grad::Bool=true) where T<:AbstractFloat
    # Computes the transit time approximating the motion as a fraction of a AHL21 step backward in time.
    # Computes the impact parameter and sky-velocity, if passed a TransitParameters.
    # Also computes the Jacobian of the transit time with respect to the initial parameters, dtbvdq[1-3,7,n].
    # Initial guess using linear interpolation:

    s.dqdt .= 0.0
    #set_state!(tt.s_transit,s)

    dt = one(T)
    iter = 0
    r3 = zero(T)
    gdot = zero(T)
    gsky = zero(T)
    stmp = zero(T)
    TRANSIT_TOL = 10*eps(dt)
    tt1 = dt0 + 1
    tt2 = dt0 + 2
    ITMAX = 20
    while abs(dt) > TRANSIT_TOL && iter < 20
        tt2 = tt1
        tt1 = dt0
        set_state!(s,tt.s_prior)
        # Advance planet state at start of step to estimated transit time:
        zero_out!(d)
        intr.scheme(s,d,dt0)
        # Compute time offset:
        gsky = g!(i,j,s.x,s.v)
        #  # Compute derivative of g with respect to time:
        gdot = gd!(i,j,s.x,s.v,s.dqdt)
        # Refine estimate of transit time with Newton's method:
        dt = -gsky/gdot
        # Add refinement to estimated time:
        #dt0 += dt
        dt0,stmp = comp_sum(dt0,stmp,dt)
        iter += 1
        # Break out if we have reached maximum iterations, or if
        # current transit time estimate equals one of the prior two steps:
        if (iter >= ITMAX) || (dt0 == tt1) || (dt0 == tt2)
            break
        end
    end

    if grad
        # Compute derivatives with updated time step
        set_state!(s,tt.s_prior)
        zero_out!(d)
        intr.scheme(s,d,dt0)
    end

    if tt isa TransitParameters
        # Compute the impact parameter and sky velocity, save to tt along with transit time.
        tt.ttbv[1,j,tt.count[j]] = s.t[1] + dt0

        if grad
            # Compute derivative of transit time, impact parameter, and sky velocity.
            vsky,bsky2 = dtbvdq!(i,j,s.x,s.v,s.jac_step,s.dqdt,tt.dtbvdq)
            tt.ttbv[2,j,tt.count[j]] = vsky
            tt.ttbv[3,j,tt.count[j]] = bsky2
            for itbv=1:3, k=1:7, p=1:s.n
                tt.dtbvdq0[itbv,j,tt.count[j],k,p] = tt.dtbvdq[itbv,k,p]
            end
            return
        end
        tt.ttbv[2,j,tt.count[j]] = calc_vsky(s.v,i,j)
        tt.ttbv[3,j,tt.count[j]] = calc_bsky2(s.x,i,j)
        return
    end

    tt.tt[j,tt.count[j]] = s.t[1] + dt0
    if grad
        # Compute derivative of transit time
        dtbvdq!(i,j,s.x,s.v,s.jac_step,s.dqdt,tt.dtdq)
        for k=1:7, p=1:s.n
            tt.dtdq0[j,tt.count[j],k,p] = tt.dtdq[1,k,p]
        end
    end
    return
end

function calc_dtdelements!(s::State{T},tt::TransitTiming{T}) where T <: AbstractFloat
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

function calc_dtdelements!(s::State{T},ttbv::TransitParameters{T}) where T <: AbstractFloat
    for itbv = 1:3, i=1:s.n, j = 1:ttbv.count[i]
        if j <= ttbv.ntt
            # Now, multiply by the initial Jacobian to convert time derivatives to orbital elements:
            for k=1:s.n, l=1:7
                ttbv.dtbvdelements[itbv,i,j,l,k] = zero(T)
                for p=1:s.n, q=1:7
                    ttbv.dtbvdelements[itbv,i,j,l,k] += ttbv.dtbvdq0[itbv,i,j,q,p]*s.jac_init[(p-1)*7+q,(k-1)*7+l]
                end
            end
        end
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

calc_bsky2(x::Matrix{T},i::Int64,j::Int64) where T<:AbstractFloat = (x[1,j]-x[1,i])^2 + (x[2,j]-x[2,i])^2
calc_vsky(v::Matrix{T},i::Int64,j::Int64) where T<:AbstractFloat = sqrt((v[1,j]-v[1,i])^2 + (v[2,j]-v[2,i])^2)

function dtbvdq!(i::Int64,j::Int64,x::Matrix{T},v::Matrix{T},jac_step::Matrix{T},dqdt::Vector{T},dtbvdq::Array{T}) where T<:AbstractFloat
    n = size(x)[2]
    # Compute time offset:
    gsky = g!(i,j,x,v)
    # Compute derivative of g with respect to time:
    gdot = gd!(i,j,x,v,dqdt)

    fill!(dtbvdq,zero(typeof(x[1])))
    indj = (j-1)*7+1
    indi = (i-1)*7+1
    for p=1:n
        indp = (p-1)*7
        for k=1:7
            # Compute derivatives:
            dtbvdq[1,k,p] = -((jac_step[indj  ,indp+k]-jac_step[indi  ,indp+k])*(v[1,j]-v[1,i])+(jac_step[indj+1,indp+k]-jac_step[indi+1,indp+k])*(v[2,j]-v[2,i])+
                          (jac_step[indj+3,indp+k]-jac_step[indi+3,indp+k])*(x[1,j]-x[1,i])+(jac_step[indj+4,indp+k]-jac_step[indi+4,indp+k])*(x[2,j]-x[2,i]))/gdot
        end
    end
    ntbv = size(dtbvdq)[1]
    if ntbv == 3
        # Compute the impact parameter and sky velocity:
        vsky = calc_vsky(v,i,j)
        bsky2 = calc_bsky2(x,i,j)
        # partial derivative v_{sky} with respect to time:
        dvdt = ((v[1,j]-v[1,i])*(dqdt[(j-1)*7+4]-dqdt[(i-1)*7+4])+(v[2,j]-v[2,i])*(dqdt[(j-1)*7+5]-dqdt[(i-1)*7+5]))/vsky
        # (note that \partial b/\partial t = 0 at mid-transit since g_{sky} = 0 mid-transit).
        for p=1:n
            indp = (p-1)*7
            for k=1:7
                # Compute derivatives:
                #v_{sky} = sqrt((v[1,j]-v[1,i])^2+(v[2,j]-v[2,i])^2)
                dtbvdq[2,k,p] = ((jac_step[indj+3,indp+k]-jac_step[indi+3,indp+k])*(v[1,j]-v[1,i])+(jac_step[indj+4,indp+k]-jac_step[indi+4,indp+k])*(v[2,j]-v[2,i]))/vsky + dvdt*dtbvdq[1,k,p]
                #b_{sky}^2 = (x[1,j]-x[1,i])^2+(x[2,j]-x[2,i])^2
                dtbvdq[3,k,p] = 2*((jac_step[indj  ,indp+k]-jac_step[indi  ,indp+k])*(x[1,j]-x[1,i])+(jac_step[indj+1,indp+k]-jac_step[indi+1,indp+k])*(x[2,j]-x[2,i]))
            end
        end
        return vsky,bsky2
    end
    return
end
