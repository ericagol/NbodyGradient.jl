import NbodyGradient: InitialConditions

#========== Integrator ==========#
abstract type AbstractIntegrator end

"""

Integrator to be used, and relevant parameters.
"""
mutable struct Integrator{T<:AbstractFloat} <: AbstractIntegrator
    scheme::Function
    h::T
    t0::T
    tmax::T

    Integrator(scheme::Function,h::T,t0::T,tmax::T) where T<:AbstractFloat = new{T}(scheme,h,t0,tmax)
end

Integrator(scheme::Function,h::Real,t0::Real,tmax::Real) = Integrator(scheme,promote(h,t0,tmax)...)

# Default to ah18!
Integrator(h::Real,t0::Real,tmax::Real) where T<:AbstractFloat = Integrator(ah18!,promote(h,t0,tmax)...)

#========== State ==========#
abstract type AbstractState end

"""

Current state of simulation.
"""
struct State{T<:AbstractFloat,V<:Vector{T},M<:Matrix{T}} <: AbstractState
    x::M
    v::M
    t::T
    m::V
    jac_step::M
    dqdt::V
    jac_init::M
    xerror::M # These might be put into an 'PreAllocArrays'
    verror::M
    jac_error::M
    n::Int64
end

function State(ic::InitialConditions{T}) where T<:AbstractFloat
    x,v,jac_init = init_nbody(ic)
    n = ic.nbody
    xerror = zeros(T,size(x))
    verror = zeros(T,size(v))
    jac_step = Matrix{T}(I,7*n,7*n)
    dqdt = zeros(T,7*n)
    jac_error = zeros(T,size(jac_step))
    return State(x,v,ic.t0,ic.m,jac_step,dqdt,jac_init,xerror,verror,jac_error,ic.nbody)
end

function step_time(s::State{T},h::T,s2::T) where T<:AbstractFloat
    t = zero(T)
    t,s2 = comp_sum(s.t,s2,h)
    return State(s.x,s.v,t,s.m,s.jac_step,s.dqdt,s.jac_init,s.xerror,s.verror,s.jac_error,s.n), s2
end

function revert_state!(s::State{T},x::Matrix{T},v::Matrix{T},xerror::Matrix{T},verror::Matrix{T}) where T<:AbstractFloat
    s.x .= x
    s.v .= v
    s.xerror .= xerror
    s.verror .= verror
    return
end 

function revert_state!(s::State{T},x::Matrix{T},v::Matrix{T},xerror::Matrix{T},verror::Matrix{T},jac_step,jac_error) where T<:AbstractFloat
    s.x .= x
    s.v .= v
    s.xerror .= xerror
    s.verror .= verror
    s.jac_step .= jac_step
    s.jac_error .= jac_error
    return
end 

"""Shows if the positions, velocities, and Jacobian are finite."""
Base.show(io::IO,::MIME"text/plain",s::State{T}) where {T} = begin
    println(io,"State{$T}:"); 
    println(io,"Positions  : ", all(isfinite.(s.x)) ? "finite" : "infinite!"); 
    println(io,"Velocities : ", all(isfinite.(s.v)) ? "finite" : "infinite!");
    println(io,"Jacobian   : ", all(isfinite.(s.jac_step)) ? "finite" : "infinite!");
    return
end 

#========== Running Methods ==========#
"""

Callable `Integrator` method. Integrates to `i.tmax`.
"""
function (i::Integrator)(s::State{T}) where T<:AbstractFloat 
    s2 = zero(T) # For compensated summation

    # Preallocate struct of arrays for derivatives (and pair)
    d = Jacobian(T,s.n) 
    pair = zeros(Bool,s.n,s.n)

    while s.t < (i.t0 + i.tmax)
        # Take integration step and advance time
        i.scheme(s,d,i.h,pair)
        s,s2 = step_time(s,i.h,s2)
    end
    return
end

"""

Take N steps.
"""
function (i::Integrator)(s::State{T},N::Int64) where T<:AbstractFloat
    s2 = zero(T) # For compensated summation

    # Preallocate struct of arrays for derivatives (and pair)
    d = Jacobian(T,s.n) 
    pair = zeros(Bool,s.n,s.n)

    # check to see if backward step
    if N < 0; i.h *= -1; N *= -1; end

    for n in 1:N
        # Take integration step and advance time
        i.scheme(s,d,i.h,pair)
        s,s2 = step_time(s,i.h,s2)
    end

    # Return time step to forward if needed
    if i.h < 0; i.h *= -1; end
    return
end

#========== Includes  ==========#
const ints = ["ah18"]
for i in ints; include(joinpath(i,"$i.jl")); end

const ints_no_grad = ["ah18","dh17"]
for i in ints_no_grad; include(joinpath(i,"$(i)_no_grad.jl")); end
