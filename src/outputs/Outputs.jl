abstract type AbstractOutput{T} end
"""

Preallocates and holds arrays for positions, velocities, and Jacobian at every integrator step
"""
# Should add option to choose out intervals, checkpointing, etc.
struct CartesianOutput{T<:AbstractFloat} <: AbstractOutput{T}
    states::Vector{State{T}}
    nstep::Int64
    filename::String
    file::Bool

    function CartesianOutput(::Type{T},nbody::Int64,nstep::Int64;filename="data.jld2",file=false) where T <: AbstractFloat
        states = Array{State,1}(undef,nstep)
        return new{T}(states,nstep,filename,file)
    end
end

# Allow user to not have to specify type. Defaults to Float64
CartesianOutput(nbody::T,nstep::T;filename="data.jld2") where T<:Int64 = CartesianOutput(Float64,nbody,nstep,filename=filename)

"""

Runs integrator like (*insert doc reference here*) and output positions, velocities, and Jacobian to a JLD2 file.
"""
function (intr::Integrator)(s::State{T},o::CartesianOutput{T}) where T<:AbstractFloat
    t0 = s.t[1] # Initial time
    time = intr.tmax
    nsteps = o.nstep

    # Integrate in proper direction
    h = intr.h * check_step(t0,time)
    tmax = t0 + (h * nsteps)

    # Preallocate struct of arrays for derivatives (and pair)
    d = Derivatives(T,s.n)

    for i in 1:nsteps
        # Save State from current step
        o.states[i] = deepcopy(s)
        # Take integration step and advance time
        intr.scheme(s,d,h)
        s.t[1] = t0 +  (h * i)
    end

    # Output to jld2 file if desired
    if o.file; save(o.filename,Dict("states" => o.states)); end
    return
end

# Includes
const outputs = ["elements"]
for i in outputs; include("$(i).jl"); end
