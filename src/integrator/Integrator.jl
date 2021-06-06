import NbodyGradient: InitialConditions

#========== Integrator ==========#
abstract type AbstractIntegrator end

"""
    Integrator{T<:AbstractFloat}

Integrator. Used as a functor to integrate a [`State`](@ref).

# Fields
- `scheme::Function` : The integration scheme to use.
- `h::T` : Step size.
- `t0::T` : Initial time.
- `tmax::T` : Duration of simulation.
"""
mutable struct Integrator{T<:AbstractFloat, schemeT} <: AbstractIntegrator
    scheme::schemeT
    h::T
    t0::T
    tmax::T
end

Integrator(h::T,t0::T) where T<:AbstractFloat = Integrator(ahl21!,h,t0,t0+h)
Integrator(scheme,h::Real,t0::Real,tmax::Real) = Integrator(scheme,promote(h,t0,tmax)...)

# Default to ahl21!
Integrator(h::T,t0::T,tmax::T) where T<:AbstractFloat = Integrator(ahl21!,h,t0,tmax)

#========== State ==========#
abstract type AbstractState end

"""
    State{T<:AbstractFloat} <: AbstractState

Current state of simulation.

# Fields (relevant to the user)
- `x::Matrix{T}` : Positions of each body [dimension, body].
- `v::Matrix{T}` : Velocities of each body [dimension, body].
- `t::Vector{T}` : Current time of simulation.
- `m::Vector{T}` : Masses of each body.
- `jac_step::Matrix{T}` : Current Jacobian.
- `dqdt::Vector{T}` : Derivative with respect to time.
"""
struct State{T<:AbstractFloat} <: AbstractState
    x::Matrix{T}
    v::Matrix{T}
    t::Vector{T}
    m::Vector{T}
    jac_step::Matrix{T}
    dqdt::Vector{T}
    jac_init::Matrix{T}
    xerror::Matrix{T}
    verror::Matrix{T}
    dqdt_error::Vector{T}
    jac_error::Matrix{T}
    n::Int64
    pair::Matrix{Bool}

    rij::Vector{T}
    a::Matrix{T}
    aij::Vector{T}
    x0::Vector{T}
    v0::Vector{T}
    input::Vector{T}
    delxv::Vector{T}
    rtmp::Vector{T}
end

"""
    State(ic)

Constructor for [`State`](@ref) type.

# Arguments
- `ic::InitialConditions{T}` : Initial conditions for the system.
"""
function State(ic::InitialConditions{T}) where T<:AbstractFloat
    x,v,jac_init = init_nbody(ic)
    n = ic.nbody
    xerror = zeros(T,size(x))
    verror = zeros(T,size(v))
    jac_step = Matrix{T}(I,7*n,7*n)
    dqdt = zeros(T,7*n)
    dqdt_error = zeros(T,size(dqdt))
    jac_error = zeros(T,size(jac_step))
    pair = zeros(Bool,n,n)

    rij = zeros(T,3)
    a = zeros(T,3,n)
    aij = zeros(T,3)
    x0 = zeros(T,3)
    v0 = zeros(T,3)
    input = zeros(T,8)
    delxv = zeros(T,6)
    rtmp = zeros(T,3)
    return State(x,v,[ic.t0],ic.m,jac_step,dqdt,jac_init,xerror,verror,dqdt_error,jac_error,ic.nbody,
    pair,rij,a,aij,x0,v0,input,delxv,rtmp)
end

function initialize!(s::State{T}, ic::InitialConditions{T}) where T<:Real
    x,v,jac_init = init_nbody(ic)
    s.x .= x
    s.v .= v
    s.jac_init .= jac_init

    s.xerror .= 0.0
    s.verror .= 0.0
    return
end

function set_state!(s_old::State{T},s_new::State{T}) where T<:AbstractFloat
    s_old.t .= s_new.t
    s_old.x .= s_new.x
    s_old.v .= s_new.v
    s_old.jac_step .= s_new.jac_step
    s_old.xerror .= s_new.xerror
    s_old.verror .= s_new.verror
    s_old.jac_error .= s_new.jac_error
    s_old.dqdt .= s_new.dqdt
    s_old.dqdt_error .= s_new.dqdt_error
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
    (::Integrator)(s, time; grad=true)

Callable [`Integrator`](@ref) method. Integrate to specific time.

# Arguments
- `s::State{T}` : The current state of the simulation.
- `time::T` : Time to integrate to.

### Optional
- `grad::Bool` : Choose whether to calculate gradients. (Default = true)
"""
function (intr::Integrator)(s::State{T},time::T;grad::Bool=true) where T<:AbstractFloat
    t0 = s.t[1]

    # Calculate number of steps
    nsteps = abs(round(Int64,(time - t0)/intr.h))

    # Step either forward or backward
    h = intr.h * check_step(t0,time)

    # Calculate last step (if needed)
    #while t0 + (h * nsteps) <= time; nsteps += 1; end
    tmax = t0 + (h * nsteps)

    # Preallocate struct of arrays for derivatives
    if grad; d = Derivatives(T,s.n); end

    for i in 1:nsteps
        # Take integration step and advance time
        if grad
            intr.scheme(s,d,h)
        else
            intr.scheme(s,h)
        end
    end

    # Do last step (if needed)
    if nsteps == 0; hf = time; end
    if tmax != time
        hf = time - tmax
        if grad
            intr.scheme(s,d,hf)
        else
            intr.scheme(s,hf)
        end
    end

    s.t[1] = time
    return
end

"""
    (::Integrator)(s, N; grad=true)

Callable [`Integrator`](@ref) method. Integrate for N steps.

# Arguments
- `s::State{T}` : The current state of the simulation.
- `N::Int64` : Number of steps.

### Optional
- `grad::Bool` : Choose whether to calculate gradients. (Default = true)
"""
function (intr::Integrator)(s::State{T},N::Int64;grad::Bool=true) where T<:AbstractFloat
    s2 = zero(T) # For compensated summation

    # Preallocate struct of arrays for derivatives
    if grad; d = Derivatives(T,s.n); end

    # check to see if backward step
    if N < 0; intr.h *= -1; N *= -1; end
    h = intr.h

    for n in 1:N
        # Take integration step and advance time
        if grad
            intr.scheme(s,d,h)
        else
            intr.scheme(s,h)
        end
        s.t[1],s2 = comp_sum(s.t[1],s2,intr.h)
    end

    # Return time step to forward if needed
    if intr.h < 0; intr.h *= -1; end
    return
end

"""
    (::Integrator)(s; grad=true)

Callable [`Integrator`](@ref) method. Integrate from `s.t` to `tmax` -- specified in constructor.

# Arguments
- `s::State{T}` : The current state of the simulation.

### Optional
- `grad::Bool` : Choose whether to calculate gradients. (Default = true)
"""
(intr::Integrator)(s::State{T};grad::Bool=true) where T<:AbstractFloat = intr(s,s.t[1]+intr.tmax,grad=grad)

function check_step(t0::T,tmax::T) where T<:AbstractFloat
    if abs(tmax) > abs(t0)
        return sign(tmax)
    else
        if sign(tmax) != sign(t0)
            return sign(tmax)
        else
            return -1 * sign(tmax)
        end
    end
end

#========== Includes  ==========#
const ints = ["ahl21"]
for i in ints; include(joinpath(i,"$i.jl")); end

const ints_no_grad = ["ahl21","dh17"]
for i in ints_no_grad; include(joinpath(i,"$(i)_no_grad.jl")); end