# Collection of functions to compute transit timing variations, using the AH18 integrator.

# wrapper for testing new ics.
@inline function ttv_elements!(el::ElementsIC{T},t0::T,h::T,tmax::T,tt::Array{T,2},count::Array{Int64,1},rstar::T) where T <: Real
    return ttv_elements!(el.H,t0,h,tmax,el.elements,tt,count,0.0,0,0,rstar)
end

"""

Computes Transit Timing Variations (TTVs) as a function of orbital elements, and computes Jacobian of transit times with respect to the initial orbital elements.
"""
function ttv_elements!(H::Union{Int64,Array{Int64,1}},t0::T,h::T,tmax::T,elements::Array{T,2},tt::Array{T,2},count::Array{Int64,1},dtdq0::Array{T,4},rstar::T;fout="",iout=-1,pair=zeros(Bool,H[1],H[1])) where {T <: Real}
    #
    # Input quantities:
    # n     = number of bodies
    # t0    = initial time of integration  [days]
    # h     = time step [days]
    # tmax  = duration of integration [days]
    # elements[i,j] = 2D n x 7 array of the masses & orbital elements of the bodies (currently first body's orbital elements are ignored)
    #            elements are ordered as: mass, period, t0, e*cos(omega), e*sin(omega), inclination, longitude of ascending node (Omega)
    # tt    = array of transit times of size [n x max(ntt)] (currently only compute transits of star, so first row is zero) [days]
    # count = array of the number of transits for each body
    # dtdq0 = derivative of transit times with respect to initial x,v,m [various units: day/length (3), day^2/length (3), day/mass]
    #         4D array  [n x max(ntt) ] x [n x 7] - derivatives of transits of each planet with respect to initial positions/velocities
    #             masses of *all* bodies.  Note: mass derivatives are *after* positions/velocities, even though they are at start
    #             of the elements[i,j] array.
    #
    # Output quantity:
    # dtdelements = 4D array  [n x max(ntt) ] x [n x 7] - derivatives of transits of each planet with respect to initial orbital
    #             elements/masses of *all* bodies.  Note: mass derivatives are *after* elements, even though they are at start
    #             of the elements[i,j] array
    #
    # Example: see test_ttv_elements.jl in test/ directory
    #
    # Define initial mass, position & velocity arrays:
    n = H[1]
    m=zeros(T,n)
    x=zeros(T,NDIM,n)
    v=zeros(T,NDIM,n)
    # Fill the transit-timing & jacobian arrays with zeros:
    fill!(tt,zero(T))
    fill!(dtdq0,zero(T))
    # Create an array for the derivatives with respect to the masses/orbital elements:
    dtdelements = copy(dtdq0)
    # Counter for transits of each planet:
    fill!(count,0)
    for i=1:n
        m[i] = elements[i,1]
    end
    # Initialize the N-body problem using nested hierarchy of Keplerians:
    init = ElementsIC(t0,H,elements)
    x,v,jac_init = init_nbody(init)
    
    #= Why is this here??
    elements_big=big.(elements); t0big = big(t0); #jac_init_big = zeros(BigFloat,7*n,7*n)
    init_big = ElementsIC(elements_big,H,t0big)
    xbig,vbig,jac_init_big = init_nbody(init_big)
    x = convert(Array{T,2},xbig); v = convert(Array{T,2},vbig); jac_init=convert(Array{T,2},jac_init_big)
    =#

    ttv!(n,t0,h,tmax,m,x,v,tt,count,dtdq0,rstar,pair;fout=fout,iout=iout)
    # Need to apply initial jacobian TBD - convert from
    # derivatives with respect to (x,v,m) to (elements,m):
    ntt_max = size(tt)[2]
    for i=1:n, j=1:count[i]
        if j <= ntt_max
            # Now, multiply by the initial Jacobian to convert time derivatives to orbital elements:
            for k=1:n, l=1:7
                dtdelements[i,j,l,k] = zero(T)
                for p=1:n, q=1:7
                    dtdelements[i,j,l,k] += dtdq0[i,j,q,p]*jac_init[(p-1)*7+q,(k-1)*7+l]
                end
            end
        end
    end
    return dtdelements
end

"""

Computes TTVs as a function of initial x,v,m, and derivatives (dtdq0).
"""
function ttv!(n::Int64,t0::T,h::T,tmax::T,m::Array{T,1},x::Array{T,2},v::Array{T,2},tt::Array{T,2},count::Array{Int64,1},dtdq0::Array{T,4},rstar::T,pair::Array{Bool,2};fout="",iout=-1) where {T <: Real}
    xprior = copy(x)
    vprior = copy(v)
    xtransit = copy(x)
    vtransit = copy(v)
    xerror = zeros(T,size(x)); verror=zeros(T,size(v))
    xerr_trans = zeros(T,size(x)); verr_trans =zeros(T,size(v))
    xerr_prior = zeros(T,size(x)); verr_prior =zeros(T,size(v))
    # Set the time to the initial time:
    t = t0
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
    # Initialize the Jacobian to the identity matrix:
    #jac_step = eye(T,7*n)
    jac_step = Matrix{T}(I,7*n,7*n)
    # Initialize Jacobian error array:
    jac_error = zeros(T,7*n,7*n)
    if fout != ""
        # Open file for output:
        file_handle =open(fout,"a")
    end
    # Output initial conditions:
    if mod(istep,iout) == 0 && iout > 0
        # Write to file:
        writedlm(file_handle,[convert(Float64,t);convert(Array{Float64,1},reshape(x,3n));convert(Array{Float64,1},reshape(v,3n))]') # Transpose to write each line
    end

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
    ntt_max = size(tt)[2]
    param_real = all(isfinite.(x)) && all(isfinite.(v)) && all(isfinite.(m)) && all(isfinite.(jac_step))
    while t < (t0+tmax) && param_real
        # Carry out a ah18 mapping step:
        ah18!(x,v,xerror,verror,h,m,n,jac_step,jac_error,pair)
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
                    tt[i,count[i]],stmp = comp_sum(t,s2,dt)
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
        t,s2 = comp_sum(t,s2,h)
        if mod(istep,iout) == 0 && iout > 0
            # Write to file:
            writedlm(file_handle,[convert(Float64,t);convert(Array{Float64,1},reshape(x,3n));convert(Array{Float64,1},reshape(v,3n));convert(Array{Float64,1},reshape(jac_step,49n^2))]') # Transpose to write each line
        end
        # t += h <- this leads to loss of precision
        # Increment counter by one:
        istep +=1
    end
    if fout != ""
        # Close output file:
        close(file_handle)
    end
    return
end

"""

Finds the transit by taking a partial ah18 step from prior time step, computes timing Jacobian, dtbvdq, with respect to initial cartesian coordinates and masses.
"""
function findtransit3!(i::Int64,j::Int64,n::Int64,h::T,tt::T,m::Array{T,1},x1::Array{T,2},v1::Array{T,2},xerror::Array{T,2},verror::Array{T,2},jac_step::Array{T,2},jac_error::Array{T,2},dtbvdq::Array{T,3},pair::Array{Bool,2}) where {T <: Real}
    # Computes the transit time, approximating the motion as a fraction of a AH17 step forward in time.
    # Also computes the Jacobian of the transit time with respect to the initial parameters, dtbvdq[1-3,7,n].
    # This version is same as findtransit2, but uses the derivative of AH17 step with respect to time
    # rather than the instantaneous acceleration for finding transit time derivative (gdot).
    # Initial guess using linear interpolation:
    dt = one(T)
    iter = 0
    r3 = zero(T)
    gdot = zero(T)
    gsky = zero(T)
    x = copy(x1)
    v = copy(v1)
    xerr_trans = copy(xerror); verr_trans = copy(verror)
    dqdt = zeros(T,7*n)
    TRANSIT_TOL = 10*eps(dt)
    tt1 = tt + 1
    tt2 = tt + 2
    ITMAX = 20
    #while abs(dt) > TRANSIT_TOL && iter < 20
    while true # while true is kind of a no-no...
        tt2 = tt1
        tt1 = tt
        x .= x1; v .= v1; xerr_trans .= xerror; verr_trans .= verror
        # Advance planet state at start of step to estimated transit time:
        ah18!(x,v,xerr_trans,verr_trans,tt,m,n,dqdt,pair)
        # Compute time offset:
        gsky = g!(i,j,x,v)
        #  # Compute derivative of g with respect to time:
        gdot = ((x[1,j]-x[1,i])*(dqdt[(j-1)*7+4]-dqdt[(i-1)*7+4])+(x[2,j]-x[2,i])*(dqdt[(j-1)*7+5]-dqdt[(i-1)*7+5])
                +  (v[1,j]-v[1,i])*(dqdt[(j-1)*7+1]-dqdt[(i-1)*7+1])+(v[2,j]-v[2,i])*(dqdt[(j-1)*7+2]-dqdt[(i-1)*7+2]))
        # Refine estimate of transit time with Newton's method:
        dt = -gsky/gdot
        # Add refinement to estimated time:
        tt += dt
        iter +=1
        # Break out if we have reached maximum iterations, or if
        # current transit time estimate equals one of the prior two steps:
        if (iter >= ITMAX) || (tt == tt1) || (tt == tt2)
            break
        end
    end
    if iter >= 20
        #  println("Exceeded iterations: planet ",j," iter ",iter," dt ",dt," gsky ",gsky," gdot ",gdot)
    end
    # Compute time derivatives:
    x .= x1; v .= v1; xerr_trans .= xerror; verr_trans .= verror
    # Compute dgdt with the updated time step.
    ah18!(x,v,xerr_trans,verr_trans,tt,m,n,jac_step,jac_error,pair)
    # Need to reset to compute dqdt:
    x .= x1; v .= v1; xerr_trans .= xerror; verr_trans .= verror
    ah18!(x,v,xerr_trans,verr_trans,tt,m,n,dqdt,pair)
    # Compute time offset:
    gsky = g!(i,j,x,v)
    # Compute derivative of g with respect to time:
    gdot  = ((x[1,j]-x[1,i])*(dqdt[(j-1)*7+4]-dqdt[(i-1)*7+4])+(x[2,j]-x[2,i])*(dqdt[(j-1)*7+5]-dqdt[(i-1)*7+5])
             +  (v[1,j]-v[1,i])*(dqdt[(j-1)*7+1]-dqdt[(i-1)*7+1])+(v[2,j]-v[2,i])*(dqdt[(j-1)*7+2]-dqdt[(i-1)*7+2]))
    # Set dtbvdq to zero:
    fill!(dtbvdq,zero(T))
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
        return tt::T,vsky::T,bsky2::T
    else
        return tt::T
    end
end

"""

Finds the transit by taking a partial ah18 step from prior times step, computes timing Jacobian, dtdq, wrt initial cartesian coordinates, masses:
"""
function findtransit3!(i::Int64,j::Int64,n::Int64,h::T,tt::T,m::Array{T,1},x1::Array{T,2},v1::Array{T,2},xerror::Array{T,2},verror::Array{T,2},pair::Array{Bool,2};calcbvsky=false) where {T <: Real}
    # Computes the transit time, approximating the motion as a fraction of a DH17 step forward in time.
    # Also computes the Jacobian of the transit time with respect to the initial parameters, dtdq[7,n].
    # This version is same as findtransit2, but uses the derivative of dh17 step with respect to time
    # rather than the instantaneous acceleration for finding transit time derivative (gdot).
    # Initial guess using linear interpolation:
    dt = one(T)
    iter = 0
    r3 = zero(T)
    gdot = zero(T)
    gsky = gdot
    x = copy(x1); v = copy(v1); xerr_trans = copy(xerror); verr_trans = copy(verror)
    dqdt = zeros(T,7*n)
    #TRANSIT_TOL = 10*sqrt(eps(dt)
    TRANSIT_TOL = 10*eps(dt)
    ITMAX = 20
    tt1 = tt + 1
    tt2 = tt + 2
    #while abs(dt) > TRANSIT_TOL && iter < 20
    while true
        tt2 = tt1
        tt1 = tt
        x .= x1; v .= v1; xerr_trans .= xerror; verr_trans .= verror
        # Advance planet state at start of step to estimated transit time:
        #  dh17!(x,v,tt,m,n,pair)
        #dh17!(x,v,xerror,verror,tt,m,n,dqdt,pair)
        ah18!(x,v,xerr_trans,verr_trans,tt,m,n,dqdt,pair)
        # Compute time offset:
        gsky = g!(i,j,x,v)
        #  # Compute derivative of g with respect to time:
        gdot = ((x[1,j]-x[1,i])*(dqdt[(j-1)*7+4]-dqdt[(i-1)*7+4])+(x[2,j]-x[2,i])*(dqdt[(j-1)*7+5]-dqdt[(i-1)*7+5])
                +  (v[1,j]-v[1,i])*(dqdt[(j-1)*7+1]-dqdt[(i-1)*7+1])+(v[2,j]-v[2,i])*(dqdt[(j-1)*7+2]-dqdt[(i-1)*7+2]))
        # Refine estimate of transit time with Newton's method:
        dt = -gsky/gdot
        # Add refinement to estimated time:
        tt += dt
        iter +=1
        # Break out if we have reached maximum iterations, or if
        # current transit time estimate equals one of the prior two steps:
        if (iter >= ITMAX) || (tt == tt1) || (tt == tt2)
            break
        end
    end
    if iter >= 20
        #  println("Exceeded iterations: planet ",j," iter ",iter," dt ",dt," gsky ",gsky," gdot ",gdot)
    end
    # Note: this is the time elapsed *after* the beginning of the timestep:
    if calcbvsky
        # Compute the sky velocity and impact parameter:
        vsky = sqrt((v[1,j]-v[1,i])^2 + (v[2,j]-v[2,i])^2)
        bsky2 = (x[1,j]-x[1,i])^2 + (x[2,j]-x[2,i])^2
        # return the transit time, impact parameter, and sky velocity:
        return tt::T,vsky::T,bsky2::T
    else
        return tt::T
    end
end

"""Used in computing transit time inside `findtransit3`."""
function g!(i::Int64,j::Int64,x::Array{T,2},v::Array{T,2}) where {T <: Real}
    # See equation 8-10 Fabrycky (2008) in Seager Exoplanets book
    g = (x[1,j]-x[1,i])*(v[1,j]-v[1,i])+(x[2,j]-x[2,i])*(v[2,j]-v[2,i])
    return g
end

"""

Computes TTVs as a function of orbital elements, allowing for a single log perturbation of dlnq for body jq and element iq. (NO GRADIENT)
"""
function ttv_elements!(H::Union{Int64,Array{Int64,1}},t0::T,h::T,tmax::T,elements::Array{T,2},tt::Array{T,2},count::Array{Int64,1},dlnq::T,iq::Int64,jq::Int64,rstar::T;fout="",iout=-1,pair = zeros(Bool,H[1],H[1])) where {T <: Real}
    #
    # Input quantities:
    # n     = number of bodies
    # t0    = initial time of integration  [days]
    # h     = time step [days]
    # tmax  = duration of integration [days]
    # elements[i,j] = 2D n x 7 array of the masses & orbital elements of the bodies (currently first body's orbital elements are ignored)
    #            elements are ordered as: mass, period, t0, e*cos(omega), e*sin(omega), inclination, longitude of ascending node (Omega)
    # tt    = pre-allocated array to hold transit times of size [n x max(ntt)] (currently only compute transits of star, so first row is zero) [days]
    #         upon output, set to transit times of planets.
    # count = pre-allocated array of the number of transits for each body upon output
    #
    # dlnq  = fractional variation in initial parameter jq of body iq for finite-difference calculation of
    #         derivatives [this is only needed for testing derivative code, below].
    #
    # Example: see test_ttv_elements.jl in test/ directory
    #
    #fcons = open("fcons.txt","w");
    # Set up mass, position & velocity arrays.  NDIM =3
    n = H[1]
    m=zeros(T,n)
    x=zeros(T,NDIM,n)
    v=zeros(T,NDIM,n)
    # Fill the transit-timing array with zeros:
    fill!(tt,0.0)
    # Counter for transits of each planet:
    fill!(count,0)
    # Insert masses from the elements array:
    for i=1:n
        m[i] = elements[i,1]
    end
    # Allow for perturbations to initial conditions: jq labels body; iq labels phase-space element (or mass)
    # iq labels phase-space element (1-3: x; 4-6: v; 7: m)
    dq = zero(T)
    if iq == 7 && dlnq != 0.0
        dq = m[jq]*dlnq
        m[jq] += dq
    end
    # Initialize the N-body problem using nested hierarchy of Keplerians:
    init = ElementsIC(t0,H,elements)
    x,v,_ = init_nbody(init)
    #elements_big=big.(elements); t0big = big(t0)
    #init_big = ElementsIC(elements_big,H,t0big)
    #xbig,vbig,_ = init_nbody(init_big)
    #if T != BigFloat
    #  println("Difference in x,v init: ",x-convert(Array{T,2},xbig)," ",v-convert(Array{T,2},vbig)," (dlnq version)")
    #end
    #x = convert(Array{T,2},xbig); v = convert(Array{T,2},vbig)
    # Perturb the initial condition by an amount dlnq (if it is non-zero):
    if dlnq != 0.0 && iq > 0 && iq < 7
        if iq < 4
            if x[iq,jq] != 0
                dq = x[iq,jq]*dlnq
            else
                dq = dlnq
            end
            x[iq,jq] += dq
        else
            # Same for v
            if v[iq-3,jq] != 0
                dq = v[iq-3,jq]*dlnq
            else
                dq = dlnq
            end
            v[iq-3,jq] += dq
        end
    end
    ttv!(n,t0,h,tmax,m,x,v,tt,count,fout,iout,rstar,pair)
    return dq
end

"""

Computes TTVs as a function of initial x,v,m. (NO GRADIENT).
"""
function ttv!(n::Int64,t0::T,h::T,tmax::T,m::Array{T,1},x::Array{T,2},v::Array{T,2},tt::Array{T,2},count::Array{Int64,1},fout::String,iout::Int64,rstar::T,pair::Array{Bool,2}) where {T <: Real}
    # Make some copies to allocate space for saving prior step and computing coordinates at the times of transit.
    xprior = copy(x)
    vprior = copy(v)
    xtransit = copy(x)
    vtransit = copy(v)
    #xerror = zeros(x); verror = zeros(v)
    xerror = copy(x); verror = copy(v)
    fill!(xerror,zero(T)); fill!(verror,zero(T))
    #xerr_prior = zeros(x); verr_prior = zeros(v)
    xerr_prior = copy(xerror); verr_prior = copy(verror)
    # Set the time to the initial time:
    t = t0
    # Define error estimate based on Kahan (1965):
    s2 = zero(T)
    # Set step counter to zero:
    istep = 0
    # Jacobian for each step (7 elements+mass, n_planets, 7 elements+mass, n planets):
    # Save the g function, which computes the relative sky velocity dotted with relative position
    # between the planets and star:
    gsave = zeros(T,n)
    gi  = 0.0
    dt::T = 0.0
    # Loop over time steps:
    ntt_max = size(tt)[2]
    param_real = all(isfinite.(x)) && all(isfinite.(v)) && all(isfinite.(m))
    if fout != ""
        # Open file for output:
        file_handle =open(fout,"a")
    end
    # Output initial conditions:
    if mod(istep,iout) == 0 && iout > 0
        # Write to file:
        writedlm(file_handle,[convert(Float64,t);convert(Array{Float64,1},reshape(x,3n));convert(Array{Float64,1},reshape(v,3n))]') # Transpose to write each line
    end
    while t < t0+tmax && param_real
        # Carry out a phi^2 mapping step:
        #  phi2!(x,v,h,m,n)
        ah18!(x,v,xerror,verror,h,m,n,pair)
        #  xbig = big.(x); vbig = big.(v); hbig = big(h); mbig = big.(m)
        #dh17!(x,v,xerror,verror,h,m,n,pair)
        #  dh17!(xbig,vbig,hbig,mbig,n,pair)
        #  x .= convert(Array{Float64,2},xbig); v .= convert(Array{Float64,2},vbig)
        param_real = all(isfinite.(x)) && all(isfinite.(v)) && all(isfinite.(m))
        # Check to see if a transit may have occured.  Sky is x-y plane; line of sight is z.
        # Star is body 1; planets are 2-nbody:
        for i=2:n
            # Compute the relative sky velocity dotted with position:
            gi = g!(i,1,x,v)
            ri = sqrt(x[1,i]^2+x[2,i]^2+x[3,i]^2)
            # See if sign switches, and if planet is in front of star (by a good amount):
            if gi > 0 && gsave[i] < 0 && x[3,i] > 0.25*ri && ri < rstar
                # A transit has occurred between the time steps.
                # Approximate the planet-star motion as a Keplerian, weighting over timestep:
                count[i] += 1
                #      tt[i,count[i]]=t+findtransit!(i,h,gi,gsave[i],m,xprior,vprior,x,v,pair)
                if count[i] <= ntt_max
                    dt0 = -gsave[i]*h/(gi-gsave[i])
                    xtransit .= xprior
                    vtransit .= vprior
                    #        dt = findtransit2!(1,i,n,h,dt0,m,xtransit,vtransit,pair)
                    #hbig = big(h); dt0big=big(dt0); mbig=big.(m); xtbig = big.(xtransit); vtbig = big.(vtransit)
                    #dtbig = findtransit2!(1,i,n,hbig,dt0big,mbig,xtbig,vtbig,pair)
                    #dt = convert(Float64,dtbig)
                    dt = findtransit3!(1,i,n,h,dt0,m,xtransit,vtransit,xerr_prior,verr_prior,pair)
                    #tt[i,count[i]]=t+dt
                    tt[i,count[i]],stmp = comp_sum(t,s2,dt)
                end
                #      tt[i,count[i]]=t+findtransit2!(1,i,n,h,gi,gsave[i],m,xprior,vprior,pair)
            end
            gsave[i] = gi
        end
        # Save the current state as prior state:
        xprior .= x
        vprior .= v
        xerr_prior .= xerror
        verr_prior .= verror
        # Increment time by the time step using compensated summation:
        #s2 += h; tmp = t + s2; s2 = (t - tmp) + s2
        #t = tmp
        t,s2 = comp_sum(t,s2,h)
        if mod(istep,iout) == 0 && iout > 0
            # Write to file:
            writedlm(file_handle,[convert(Float64,t);convert(Array{Float64,1},reshape(x,3n));convert(Array{Float64,1},reshape(v,3n))]') # Transpose to write each line
        end
        # t += h  <- this leads to loss of precision
        # Increment counter by one:
        istep +=1
        #  println("xerror: ",xerror)
        #  println("verror: ",verror)
    end
    #println("xerror: ",xerror)
    #println("verror: ",verror)
    if fout != ""
        # Close output file:
        close(file_handle)
    end
    return
end

