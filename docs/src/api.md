# API

## Types

### Initial Conditions
```@docs
Elements
Elements(m::T,P::T,t0::T,ecosϖ::T,esinϖ::T,I::T,Ω::T) where T<:Real
ElementsIC
ElementsIC(t0::T,H::Union{Int64,Vector{Int64}},elems::Elements{T}...) where T <: AbstractFloat
```

### State
```@docs
State{T<:AbstractFloat}
State(ic::InitialConditions{T}) where T<:AbstractFloat
```

### Integrator
```@docs
Integrator{T<:AbstractFloat}
```