# API

## Types

### Initial Conditions
```@docs
CartesianIC
Elements
Elements(m::T,P::T,t0::T,ecosϖ::T,esinϖ::T,I::T,Ω::T) where T<:Real
ElementsIC
ElementsIC(t0::T, H::Matrix{<:Real}, elems::Elements{T}...) where T<:AbstractFloat
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

### TransitTiming
```@docs
TransitParameters{T<:AbstractFloat}
TransitParameters(tmax::T,ic::ElementsIC{T},ti::Int64=1) where T<:AbstractFloat
TransitTiming{T<:AbstractFloat}
TransitTiming(tmax::T,ic::ElementsIC{T},ti::Int64=1) where T<:AbstractFloat
```