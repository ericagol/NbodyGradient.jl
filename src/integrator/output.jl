abstract type AbstractOutput end
abstract type Output{T} <: AbstractOutput end
"""

Preallocates and holds arrays for positions, velocities, and Jacobian at every integrator step
""" 
# Should add option to choose out intervals, checkpointing, etc.
mutable struct CartesianOutput{T<:AbstractFloat} <: Output{T}
    states::Vector{State{T}}
    nstep::Int64
    filename::String
    file::Bool

    function CartesianOutput(::Type{T},nbody::Int64,nstep::Int64;filename="data.jld2",file=false) where T <: AbstractFloat
        states = State[]
        return new{T}(states,nstep,filename,file)
    end
end

# Allow user to not have to specify type. Defaults to Float64
CartesianOutput(nbody::T,nstep::T;filename="data.jld2") where T<:Int64 = CartesianOutput(Float64,nbody,nstep,filename=filename)

"""

Preallocates and holds arrays for orbital elements at every integrator step
"""
mutable struct ElementsOutput{T<:AbstractFloat,M<:Matrix{T}} <: Output{T}
    m::M
    P::M
    t0::M
    e::M
    ϖ::M
    I::M
    Ω::M

    function ElementsOutput(::Type{T},nbody::Integer,nstep::Integer;filename="elems.out") where T<:AbstractFloat
        return new{T,Matrix{T}}([zeros(T,nstep,nbody) for i in 1:7]...)
    end
end

"""

Runs integrator like (*insert doc reference here*) and output positions, velocities, and Jacobian to a JLD2 file.
"""
function (i::Integrator)(s::State{T},o::CartesianOutput{T}) where T<:AbstractFloat
    s2 = zero(T) # For compensated summation
    
    # Preallocate struct of arrays for derivatives (and pair)
    d = Jacobian(T,s.n) 
    pair = zeros(Bool,s.n,s.n)
    
    # Push initial state
    push!(o.states,s)
    
    while s.t < (i.t0 + i.tmax)
        # Take integration step and advance time
        i.scheme(s,d,i.h,pair)
        s.t,s2 = comp_sum(s.t,s2,i.h)
        
        # Save coords to output structure
        push!(o.states,s)
    end
    if o.file; save(o.filename,Dict("states" => o.states)); end 
    return
end

