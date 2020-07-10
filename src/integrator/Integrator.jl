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
mutable struct State{M<:AbstractMatrix} <: AbstractState
    x::M
    v::M
    jac_step::M
    xerror::M # These might be put into an 'PreAllocArrays'
    verror::M
    jac_error::M

    function State(ic::InitialConditions{T}) where T<:AbstractFloat
        x,v,_ = init_nbody(ic)
        xerror = zeros(T,size(x))
        verror = zeros(T,size(v))
        jac_step = Matrix{T}(I,7*ic.nbody,7*ic.nbody)
        jac_error = zeros(T,size(jac_step))
        return new{Matrix{T}}(x,v,jac_step,xerror,verror,jac_error)
    end
end

#========== Other Methods ==========#
"""Callable `Integrator` method. Takes one step."""
(i::Integrator)(ic::InitialConditions{T},s::State{Matrix{T}}) where T<:AbstractFloat = i.scheme(s.x,s.v,s.xerror,s.verror,i.h,ic.m,ic.nbody,s.jac_step,s.jac_error,zeros(Bool,ic.nbody,ic.nbody))

#========== Includes  ==========#
const ints = ["ah18"]
for i in ints; include(joinpath(i,"$i.jl")); end

const ints_no_grad = ["ah18","dh17"]
for i in ints_no_grad; include(joinpath(i,"$(i)_no_grad.jl")); end
