# TTV functions WITHOUT derivatives.

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
    init = ElementsIC(elements,H,t0)
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

