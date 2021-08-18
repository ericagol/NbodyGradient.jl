# NbodyGradient
A fast, differentiable N-body integrator for modeling transiting exoplanets, and more.

This package provides a simple user-interface to carry out N-body simulations and compute the derivatives of the outputs with respect to the initial conditions. The current version of the code implements the following:
- Integrators:
    - AHL21 (4th-order Symplectic; [Agol, Hernandez, & Langford 2021](https://github.com/langfzac/nbg-papers))
- Initial Conditions:
    - Cartesian coordinates ([`CartesianIC`](@ref))
    - Orbital elements ([`ElementsIC`](@ref))
- Output Models:
    - Transit times ([`TransitTiming`](@ref))
    - Transit times, impact parameter, and sky-plane velocity ([`TransitParameters`](@ref))

## Getting Started
First, you'll need to add the package. Using the Julia REPL, run:
```julia
pkg> add NbodyGradient
```

If you'd like to use the developement version of the code, run:
```julia
pkg> add NbodyGradient#master
```

Then, use like any other Julia package
```julia
using NbodyGradient
```

See the [Tutorials](@ref) page for basic usage.