# Gradients
The main purpose of developing NbodyGradient.jl is to provide a differentiable N-body model for gradient-based optimization. Here, we walk though computing and accessing the jacobian of the transit times with respect to the initial orbital elements and masses.

We assume you've taken a look at the [Basic Usage](@ref) tutorial, and are familiar with the orbital elements and units used in NbodyGradient.jl.

!!! note "No gradient support for exactly circular orbits"
    This package supports specifying initial conditions for circular orbits (ie. eccentricity=0). However, the derivative computations contain 1/e terms -- requiring a non-zero eccentricity to be computed correctly. We expect the need for derivatives for exactly circular initial orbits to be minimal, and intend to implement them in the future.

Here, we will specify the orbital elements using the 'elements matrix' option (See [`ElementsIC`](@ref)).
```@example 2
using NbodyGradient

# Initial Conditions
elements = [
  # m    P       t0  ecosω esinω I   Ω
    1.0  0.0     0.0 0.0   0.0   0.0 0.0; # Star
    3e-6 365.256 0.0 0.01  0.0   π/2 0.0; # Earth analogue
]
ic = ElementsIC(0.0, 2, elements);
nothing # hide
```

Let's integrate for 5 periods of the planet and record the transit times.
```@example 2
s = State(ic)

P = elements[2,2] # Get the period from the elements matrix
tmax = 5 * P
tt = TransitTiming(5 * P, ic)

h = P/30.0
Integrator(h, tmax)(s, tt) # Run without creating an `Integrator` variable.

# Computed transit times
tt.tt
```

The derivatives of the transit times with respect to the initial orbitial elements and masses are held in the `dtdelements` field of the [`TransitTiming`](@ref) structure -- an `Array{T<:AbstractFloat,4}`. The indices correspond to the transiting body, transit number, orbital element, and body the orbital element is describing. For example, this is how to access the 'gradient' of the first transit time of the orbiting body:

```@example 2
tt.dtdelements[2,1,:,:]
```
Here, the first column is the derivative of the transit time with respect to the stars 'orbital elements', which are all zero except for the mass in the last row. The second column is the derivatives with respect to the orbital elements of the system.