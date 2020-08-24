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
    t::V
    m::V
    jac_step::M
    dqdt::V
    jac_init::M
    xerror::M # These might be put into an 'PreAllocArrays'
    verror::M
    dqdt_error::V
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
    dqdt_error = zeros(T,size(dqdt))
    jac_error = zeros(T,size(jac_step))
    return State(x,v,[ic.t0],ic.m,jac_step,dqdt,jac_init,xerror,verror,dqdt_error,jac_error,ic.nbody)
end

function set_state!(s_old::State,s_new::State)
    fields = setdiff(fieldnames(State),[:m,:n])
    for fn in fields
        f_new = getfield(s_new,fn)
        f_old = getfield(s_old,fn)
        f_old .= f_new
    end
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
    t = s.t[1]
    # Preallocate struct of arrays for derivatives (and pair)
    d = Jacobian(T,s.n) 
    pair = zeros(Bool,s.n,s.n)

    while t < (i.t0 + i.tmax)
        # Take integration step and advance time
        i.scheme(s,d,i.h,pair)
        t,s2 = comp_sum(t,s2,i.h)
    end
    s.t[1] = t
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
        s.t[1],s2 = comp_sum(s.t[1],s2,i.h)
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
