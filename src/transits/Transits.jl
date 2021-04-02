# Transit structures
abstract type TransitOutput{T} <: AbstractOutput{T} end

"""

Holds the transit times and derivatives.
"""
struct TransitTiming{T<:AbstractFloat} <: TransitOutput{T}
    tt::Matrix{T}
    dtdq0::Array{T,4}
    dtdelements::Array{T,4}
    count::Vector{Int64}
    ntt::Int64
    ti::Int64
    occs::Vector{Int64}
    dtdq::Array{T,3}
    gsave::Vector{T}
end

function TransitTiming(tmax,ic::ElementsIC{T},ti::Int64=1) where T<:AbstractFloat
    n = ic.nbody
    ind = isfinite.(tmax./ic.elements[:,2])
    ntt = maximum(ceil.(Int64,abs.(tmax./ic.elements[ind,2])).+3)
    tt = zeros(T,n,ntt)
    dtdq0 = zeros(T,n,ntt,7,n)
    dtdelements = zeros(T,n,ntt,7,n)
    count = zeros(Int64,n)
    occs = setdiff(collect(1:n),ti)
    dtdq = zeros(T,1,7,n)
    gsave = zeros(T,n)
    return TransitTiming(tt,dtdq0,dtdelements,count,ntt,ti,occs,dtdq,gsave)
end

"""

Structure for transit timing, impact parameter, and sky velocity
"""
struct TransitParameters{T<:AbstractFloat} <: TransitOutput{T}
    ttbv::Array{T,3}
    dtbvdq0::Array{T,5}
    dtbvdelements::Array{T,5}
    count::Vector{Int64}
    ntt::Int64
    ti::Int64
    occs::Vector{Int64}
    dtbvdq::Array{T,3}
    gsave::Vector{T}
end

function TransitParameters(tmax,ic::ElementsIC{T},ti::Int64=1) where T<:AbstractFloat
    n = ic.nbody
    ind = isfinite.(tmax./ic.elements[:,2])
    ntt = maximum(ceil.(Int64,abs.(tmax./ic.elements[ind,2])).+3)
    ttbv = zeros(T,3,n,ntt)
    dtbvdq0 = zeros(T,3,n,ntt,7,n)
    dtbvdelements = zeros(T,3,n,ntt,7,n)
    count = zeros(Int64,n)
    occs = setdiff(collect(1:n),ti)
    dtbvdq = zeros(T,3,7,n)
    gsave = zeros(T,n)
    return TransitParameters(ttbv,dtbvdq0,dtbvdelements,count,ntt,ti,occs,dtbvdq,gsave)
end

struct TransitSnapshot{T<:AbstractFloat} <: TransitOutput{T}
    nt::Int64
    times::Vector{T}
    bsky2::Matrix{T}
    vsky::Matrix{T}
    dbvdq0::Array{T,5}
    dbvdelements::Array{T,5}
end

function TransitSnapshot(times::Vector{T},ic::ElementsIC{T}) where T<:AbstractFloat
    n = ic.nbody
    nt = length(times)
    return TransitSeries(nt,times,zeros(T,n,nt),zeros(T,n,nt),zeros(T,2,n,nt,7,n),zeros(T,2,n,nt,7,n))
end

function Base.iterate(tt::AbstractOutput,state=1)
    fields = setdiff(fieldnames(typeof(tt)),[:times,:ti,:occs])
    if state > length(fields)
        return nothing
    end
    return (getfield(tt,fields[state]), state+1)
end

function zero_out!(tt::TransitOutput{T}) where T
    for i in tt
        if ~(typeof(i) <: Integer)
            i .= zero(T)
        end
    end
end

"""

Main integrator method for Transit calculations.
"""
function (intr::Integrator)(s::State{T}, tt::TransitOutput{T}; grad::Bool=true) where T<:AbstractFloat
    # Allocate structures
    d = Derivatives(T, s.n)
    s_prior = deepcopy(s)
    s_transit = deepcopy(s)

    t0 = s.t[1] # Initial time
    nsteps = abs(round(Int64,intr.tmax/intr.h))
    h = intr.h * check_step(t0, intr.tmax+t0) # get direction of integration

    for i in tt.occs
        # Compute the relative sky velocity dotted with position:
        tt.gsave[i] = g!(i,tt.ti,s.x,s.v)
    end

    istep = 0
    for _ in 1:nsteps

        # Take an integration step
        if grad
            intr.scheme(s,d,h)
        else
            intr.scheme(s,h)
        end
        istep += 1
        s.t[1] = t0 + (istep * h)

        # Check if a transit occured; record time.
        detect_transits!(s,s_prior,s_transit,d,tt,intr,grad=grad)
    end
    # Calculate derivatives
    if grad
        calc_dtdelements!(s,tt)
    end
end

function detect_transits!(s::State{T},s_prior::State{T},s_transit::State{T},d::Derivatives{T},tt::TransitOutput{T},intr::Integrator{T}; grad::Bool=true) where T<:AbstractFloat
    rstar::T = 1e12 # Could this be removed?
    # Save current state as prior state
    set_state!(s_prior, s)

    # Check to see if a transit may have occured before current state.
    # Sky is x-y plane; line of sight is z.
    # Body being transited is tt.ti, tt.occs is list of occultors:
    for i in tt.occs
        # Compute the relative sky velocity dotted with position:
        gi = g!(i,tt.ti,s.x,s.v)
        ri = sqrt(s.x[1,i]^2+s.x[2,i]^2+s.x[3,i]^2)  # orbital distance
        # See if sign of g switches, and if planet is in front of star (by a good amount):
        # (I'm wondering if the direction condition means that z-coordinate is reversed? EA 12/11/2017)
        if gi > 0 && tt.gsave[i] < 0 && -s.x[3,i] > 0.25*ri && ri < rstar
            # A transit has occurred between the time steps - integrate ahl21!
            tt.count[i] += 1
            if tt.count[i] <= tt.ntt
                dt0 = -tt.gsave[i]*intr.h/(gi-tt.gsave[i]) # Starting estimate
                set_state!(s,s_prior)
                if tt isa TransitTiming
                    dt = findtransit!(tt.ti,i,dt0,s,s_transit,d,tt,intr;grad=grad) # Search for transit time (integrating 'backward')
                    tt.tt[i,tt.count[i]] = s.t[1] + dt
                    if grad
                        for k=1:7, p=1:s.n
                            tt.dtdq0[i,tt.count[i],k,p] = tt.dtdq[1,k,p]
                        end
                    end
                else
                    dt, vsky, bsky2 = findtransit!(tt.ti,i,dt0,s,s_transit,d,tt,intr;grad=grad)
                    tt.ttbv[1,i,tt.count[i]] = s.t[1] + dt
                    tt.ttbv[2,i,tt.count[i]] = vsky
                    tt.ttbv[3,i,tt.count[i]] = bsky2
                    if grad
                        for itbv=1:3, k=1:7, p=1:s.n
                            tt.dtbvdq0[itbv,i,tt.count[i],k,p] = tt.dtbvdq[itbv,k,p]
                        end
                    end
                end
            end
        end
        tt.gsave[i] = gi
    end
    set_state!(s,s_prior)
    return
end

#=
"""

Integrator method for outputing `TransitTiming` or `TransitParameters`.
"""
function (i::Integrator)(s::State{T},tt::TransitOutput{T};grad::Bool=true) where T<:AbstractFloat

    # Run integrator and calculate transit times, with derivatives.
    rstar::T = 1e12 # Need to pass this in.
    calc_tt!(s,i,tt,rstar;grad=grad)
    if grad
        calc_dtdelements!(s,tt)
    end
    return
end

"""

Integrator method for outputting `TransitParameters`.
"""
function (i::Integrator)(s::State{T},ttbv::TransitParameters{T};grad::Bool=true) where T<:AbstractFloat

    # Run integrator and calculate transit times, with derivatives.
    rstar::T = 1e12 # Need to pass this in.
    calc_tt!(s,i,ttbv,rstar;grad=grad)
    if grad
        calc_dtdelements!(s,ttbv)
    end
    return
end
=#
"""

Integrator method for outputting `TransitSnapshot`.
"""
function (intr::Integrator)(s::State{T},ts::TransitSnapshot{T};grad::Bool=true) where T<:AbstractFloat
    # Integrate to, and output b and v_sky for each body, for a list of times.
    # Should be sorted times.
    # NOTE: Need to fix so that if initial time is a 0, s.dqdt isn't 0s.
    if grad; dbvdq = zeros(T,2,7,s.n); end

    # Integrate to each time, using intr.h, and output b and vsky (only want primary transits for now)
    t0 = s.t[1]
    for (j,ti) in enumerate(ts.times)
        intr(s,ti;grad=grad)
        for i in 2:s.n
            if grad
                ts.vsky[i,j],ts.bsky2[i,j] = calc_dbvdq!(s,dbvdq,1,i)
                for ibv=1:2, k=1:7, p=1:s.n
                    ts.dbvdq0[ibv,i,j,k,p] = dbvdq[ibv,k,p]
                end
            else
                ts.vsky[i,j],ts.bsky2[i,j] = calc_bv(s,1,i)
            end
        end
        # Step to the next full time step, as measured from t0.
        #steps = round(Int64 ,(s.t[1] - t0) / intr.h, RoundUp)
        #intr(s,t0 + (steps * intr.h); grad=grad)
    end
    calc_dbvdelements!(s,ts)
    return
end

# Includes for source
files = ["timing.jl","parameters.jl","snapshot.jl","ttv_no_grad.jl"]
include.(files)