# Collection of function to compute transit times.

function calc_tt!(s::State{T},intr::Integrator,tt::TransitTiming{T},rstar::T;grad::Bool=true) where T<:AbstractFloat
    n = s.n; ntt_max = tt.ntt;
    d = Derivatives(T,s.n);
    s_prior = deepcopy(s)
    # Define error estimate based on Kahan (1965):
    s2 = zero(T)
    # Set step counter to zero:
    istep = 0
    # Initialize matrix for derivatives of transit times with respect to the initial x,v,m:
    if grad; dtdq = zeros(T,1,7,s.n); end

    # Initial time
    t0 = s.t[1]
    # Number of steps
    nsteps = abs(round(Int64,intr.tmax/intr.h))
#    nsteps = abs(round(Int64,(intr.tmax)/intr.h))
    # Time step
    h = intr.h * check_step(t0,intr.tmax+t0)
    # Save the g function, which computes the relative sky velocity dotted with relative position
    # between the planets and star:
    gsave = zeros(T,s.n)
    for i in tt.occs
        # Compute the relative sky velocity dotted with position:
        gsave[i] = g!(i,tt.ti,s.x,s.v)
    end
    # Loop over time steps:
    dt = zero(T)
    gi = zero(T)
    param_real = all(isfinite.(s.x)) && all(isfinite.(s.v)) && all(isfinite.(s.m)) && all(isfinite.(s.jac_step))
    for _ in 1:nsteps
    #while s.t[1] < (t0+intr.tmax) && param_real
        # Carry out a ahl21 mapping step and advance time:
        if grad
            intr.scheme(s,d,h)
        else
            intr.scheme(s,h)
        end
        istep += 1
        s.t[1] = t0 + (istep * h)
        param_real = all(isfinite.(s.x)) && all(isfinite.(s.v)) && all(isfinite.(s.m)) && all(isfinite.(s.jac_step))
        if ~param_real; break; end

        # Save current state as prior state.
        set_state!(s_prior,s)

        # Check to see if a transit may have occured before current state.
        # Sky is x-y plane; line of sight is z.
        # Body being transited is tt.ti, tt.occs is list of occultors:
        for i in tt.occs
            # Compute the relative sky velocity dotted with position:
            gi = g!(i,tt.ti,s.x,s.v)
            ri = sqrt(s.x[1,i]^2+s.x[2,i]^2+s.x[3,i]^2)  # orbital distance
            # See if sign of g switches, and if planet is in front of star (by a good amount):
            # (I'm wondering if the direction condition means that z-coordinate is reversed? EA 12/11/2017)
            if gi > 0 && gsave[i] < 0 && -s.x[3,i] > 0.25*ri && ri < rstar
                # A transit has occurred between the time steps - integrate ahl21! between timesteps
                tt.count[i] += 1
                if tt.count[i] <= ntt_max
                    dt0 = -gsave[i]*h/(gi-gsave[i])  # Starting estimate
                    set_state!(s,s_prior) # Set state to step after transit occured
                    if grad
                        dt = findtransit!(tt.ti,i,dt0,s,d,dtdq,intr) # Search for transit time (integrating 'backward')
                    else
                        dt = findtransit!(tt.ti,i,dt0,s,d,intr)
                    end
                    # Copy transit time and derivatives to TransitTiming structure
                    tt.tt[i,tt.count[i]] = s.t[1] + dt
                    if grad
                        for k=1:7, p=1:n
                            tt.dtdq0[i,tt.count[i],k,p] = dtdq[1,k,p]
                        end
                    end
                end
            end
            gsave[i] = gi
        end
        # Set state back to after transit
        set_state!(s,s_prior)
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

function findtransit!(i::Int64,j::Int64,dt0::T,s::State{T},d::Derivatives{T},dtbvdq::Array{T},intr::Integrator) where T<:AbstractFloat
    # Computes the transit time, approximating the motion as a fraction of a AH17 step backward in time.
    # Also computes the Jacobian of the transit time with respect to the initial parameters, dtbvdq[1-3,7,n].
    # Initial guess using linear interpolation:

    s.dqdt .= 0.0
    s_prior = deepcopy(s)

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
    #while true
        tt2 = tt1
        tt1 = dt0
        set_state!(s,s_prior)
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
    #if iter >= 20
    #    println("Exceeded iterations: planet ",j," iter ",iter," dt ",dt," gsky ",gsky," gdot ",gdot, "dt0 ", dt0)
    #end
    # Compute time derivatives:
    set_state!(s,s_prior)
    zero_out!(d)
    # Compute dgdt with the updated time step.
    intr.scheme(s,d,dt0)
    #s_prior.jac_step .= s.jac_step
    #s_prior.jac_error .= s.jac_error
    # Need to reset to compute dqdt:
    #set_state!(s,s_prior)
    #zero_out!(dT)
    #intr.scheme(s,dT,dt0)
    ntbv = size(dtbvdq)[1]
    # return the transit time, impact parameter, and sky velocity:
    if ntbv == 3
        vsky,bsky2 = dtbvdq!(i,j,s.x,s.v,s.jac_step,s.dqdt,dtbvdq)
        return dt0::T,vsky::T,bsky2::T
    else
        # Compute derivative of transit time, impact parameter, and sky velocity.
        dtbvdq!(i,j,s.x,s.v,s.jac_step,s.dqdt,dtbvdq)
        return dt0::T
    end
end

function findtransit!(i::Int64,j::Int64,dt0::T,s::State{T},d::Derivatives{T},intr::Integrator;bv::Bool=false) where T<:AbstractFloat
    # Computes the transit time, approximating the motion as a fraction of a AH17 step backward in time.
    # Initial guess using linear interpolation:

    s.dqdt .= 0.0
    s_prior = deepcopy(s)

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
    #while true
        tt2 = tt1
        tt1 = dt0
        set_state!(s,s_prior)
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
    #if iter >= 20
    #    println("Exceeded iterations: planet ",j," iter ",iter," dt ",dt," gsky ",gsky," gdot ",gdot, "dt0 ", dt0)
    #end
    # return the transit time, impact parameter, and sky velocity:
    if bv
        # Compute the sky velocity and impact parameter:
        vsky = sqrt((s.v[1,j]-s.v[1,i])^2 + (s.v[2,j]-s.v[2,i])^2)
        bsky2 = (s.x[1,j]-s.x[1,i])^2 + (s.x[2,j]-s.x[2,i])^2
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
        vsky = sqrt((v[1,j]-v[1,i])^2 + (v[2,j]-v[2,i])^2)
        bsky2 = (x[1,j]-x[1,i])^2 + (x[2,j]-x[2,i])^2
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
