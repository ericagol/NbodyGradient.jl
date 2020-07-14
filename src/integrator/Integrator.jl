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
mutable struct State{T<:AbstractFloat,V<:Vector{T},M<:Matrix{T}} <: AbstractState
    x::M
    v::M
    t::T
    m::V
    jac_step::M
    xerror::M # These might be put into an 'PreAllocArrays'
    verror::M
    jac_error::M
    n::Int64

    function State(ic::InitialConditions{T}) where T<:AbstractFloat
        x,v,_ = init_nbody(ic)
        xerror = zeros(T,size(x))
        verror = zeros(T,size(v))
        jac_step = Matrix{T}(I,7*ic.nbody,7*ic.nbody)
        jac_error = zeros(T,size(jac_step))
        return new{T,Vector{T},Matrix{T}}(x,v,ic.t0,ic.m,jac_step,xerror,verror,jac_error,ic.nbody)
    end
end

#========== Running Methods ==========#
"""

Callable `Integrator` method. Integrates to `i.tmax`.
"""
function (i::Integrator)(s::State{T}) where T<:AbstractFloat 
    s2 = zero(T) # For compensated summation
    
    # Preallocate struct of arrays for derivatives (and pair)
    d = Derivatives(T,s.n) 
    pair = zeros(Bool,s.n,s.n)
    
    while s.t < (i.t0 + i.tmax)
        # Take integration step and advance time
        i.scheme(s,d,i.h,pair)
        s.t,s2 = comp_sum(s.t,s2,i.h)
    end
    return
end

#========== Includes  ==========#
const ints = ["ah18"]
for i in ints; include(joinpath(i,"$i.jl")); end

const ints_no_grad = ["ah18","dh17"]
for i in ints_no_grad; include(joinpath(i,"$(i)_no_grad.jl")); end
