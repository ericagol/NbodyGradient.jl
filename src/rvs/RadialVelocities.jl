struct RadialVelocities{T<:AbstractFloat}
    times::Vector{T}
    rvs::Vector{T}
    s_copy::State{T}
end

function RadialVelocities(times::Vector{T}, ic::ElementsIC{T}) where T<:AbstractFloat
    rvs = zeros(T, length(times))
    s_copy = State(ic)
    return RadialVelocities(times, rvs, s_copy)
end

function (intr::Integrator)(s::State{T}, rvs::RadialVelocities{T}) where T<:AbstractFloat

    t0 = s.t[1] #initial time
    nsteps = intr.tmax/intr.h
    h = intr.h

    s_copy = rvs.s_copy #shortcut to structure(s_copy)

    tc = t0
    for (i,t) in enumerate(rvs.times)

        dt = t-tc
        while dt > h
            intr.scheme(s,h)
            tc += h
            dt = t-tc
        end

        set_state!(s_copy, s) #copying data in 's' to 's_copy'
        intr.scheme(s_copy,dt)
        rvs.rvs[i] = s_copy.v[3,1]

    end
end