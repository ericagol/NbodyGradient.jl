# Collection of functions to calculate transit parameters: timing, impact parameter, sky velocity

function calc_tt!(s::State{T},intr::Integrator,tt::TransitParameters{T},rstar::T,pair::Matrix{Bool};grad::Bool=true) where T<:AbstractFloat
    n = s.n; ntt_max = tt.ntt;
    grad ? d = Derivatives(T,s.n) : dT = dTime(T,s.n)
    s_prior = deepcopy(s)
    # Define error estimate based on Kahan (1965):
    s2 = zero(T)
    # Set step counter to zero:
    istep = 0
    # Initialize matrix for derivatives of transit times with respect to the initial x,v,m:
    if grad; dtbvdq = zeros(T,3,7,s.n); end

    # Initial time
    t0 = s.t[1]
    # Number of steps
    nsteps = abs(round(Int64,(intr.tmax - t0)/intr.h))
    # Time step
    h = intr.h * check_step(t0,intr.tmax)
    # Save the g function, which computes the relative sky velocity dotted with relative position
    # between the planets and star:
    gsave = zeros(T,s.n)
    for i in tt.occs
        # Compute the relative sky velocity dotted with position:
        gsave[i]= g!(i,tt.ti,s.x,s.v)
    end
    # Loop over time steps:
    dt = zero(T)
    gi = zero(T)
    param_real = all(isfinite.(s.x)) && all(isfinite.(s.v)) && all(isfinite.(s.m)) && all(isfinite.(s.jac_step))
    #while s.t[1] < (t0+intr.tmax) && param_real
    for _ in 1:nsteps
        # Carry out a ah18 mapping step and advance time:
        if grad
            intr.scheme(s,d,h,pair)
        else
            intr.scheme(s,h,pair)
        end
        istep += 1
        s.t[1] = t0 + (istep * h)
        param_real = all(isfinite.(s.x)) && all(isfinite.(s.v)) && all(isfinite.(s.m)) && all(isfinite.(s.jac_step))
        if ~param_real; break; end
        # Save current state as prior state.
        set_state!(s_prior,s)

        # Check to see if a transit may have occured before current state.
        # Sky is x-y plane; line of sight is z.
        for i in tt.occs
            # Compute the relative sky velocity dotted with position:
            gi = g!(i,tt.ti,s.x,s.v)
            ri = sqrt(s.x[1,i]^2+s.x[2,i]^2+s.x[3,i]^2)  # orbital distance
            # See if sign of g switches, and if planet is in front of star (by a good amount):
            # (I'm wondering if the direction condition means that z-coordinate is reversed? EA 12/11/2017)
            if gi > 0 && gsave[i] < 0 && -s.x[3,i] > 0.25*ri && ri < rstar
                # A transit has occurred between the time steps - integrate ah18! between timesteps
                tt.count[i] += 1
                if tt.count[i] <= ntt_max
                    dt0 = -gsave[i]*h/(gi-gsave[i])  # Starting estimate
                    set_state!(s,s_prior) # Set state to step after transit occured
                    if grad
                        dt,vsky,bsky2 = findtransit!(tt.ti,i,dt0,s,d,dtbvdq,intr,pair) # Search for transit time (integrating 'backward')
                    else
                        dt,vsky,bsky2 = findtransit!(tt.ti,i,dt0,s,dT,intr,pair,bv=true)
                    end
                    # Copy transit time, b, vsky and derivatives to TransitParameters structure
                    tt.ttbv[1,i,tt.count[i]] = s.t[1] + dt
                    tt.ttbv[2,i,tt.count[i]] = vsky
                    tt.ttbv[3,i,tt.count[i]] = bsky2
                    if grad
                        for itbv=1:3, k=1:7, p=1:s.n
                            tt.dtbvdq0[itbv,i,tt.count[i],k,p] = dtbvdq[itbv,k,p]
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