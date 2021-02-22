# Examples

## Basic Usage
Here we'll walk through a simple example of integrating a 3-body system.
##### Units
A quick note on the units used throughout the code.
- Distance: AU
- Time: Days
- Mass: Solar Masses
- Angles: Radians

### Initial Conditions
First, we define the orbital elements of the system. This can be done by creating [`Elements`](@ref) for each body in the system.

Start with a 1 solar mass
```@example 1
using NbodyGradient # hide
a = Elements(m = 1.0);
nothing # hide
```

We then define the orbital elements for first body, say an Earth analogue.
```@example 1
b = Elements(
    m = 3e-6,
    P = 365.256,
    ecosϖ = 0.01
);
nothing # hide
```
The unspecified orbital elements default to zeros. (`ecosϖ` can be typed as `ecos\varpi` and then hitting tab)

Next we'll create a Jupiter analogue for the final body. Here the orbital elements are specified for the Keplerian ((a,b),c), or c orbiting the center of mass of a and b. (While this might not need to be stated explicitly here, this convention is useful for more complicated hierarchical systems).
```@example 1
c = Elements(
    m = 9.54e-4,
    P = 4332.59,
    ecosϖ = 0.05
);
nothing # hide
```

Now we need to pass our [`Elements`](@ref) to [`ElementsIC`](@ref).
```@example 1
t0 = 0.0 # Initial time
N = 3    # Number of bodies
ic = ElementsIC(t0,N,a,b,c)
```

Finally, we can pass the initial conditions to [`State`](@ref), which converts orbital elements to cartesian coordinates (and calculates the derivatives with respect to the initial conditions).

```@example 1
s = State(ic);
nothing # hide
```
The positions and velocities can be accessed by `s.x` and `s.v`, respectively. Each matrix contains the vector component (rows) for a particular body (columns).

```@example 1
s.x
```

### Integration
Now that we have initial conditions, we can construct and run the integrator. First, define an [`Integrator`](@ref), specifying the integration scheme, the time step, initial time, and final time. We'll use the `ah18!` mapping.
```@example 1
h = b.P/30.0 # We want at most 1/20th of the smallest period for a time step
t0 = 0.0
tmax = 5*c.P # Integrate for 5 orbital periods of the outer body
intr = Integrator(ah18!,h,t0,tmax);
nothing # hide
```

Finally, run the [`Integrator`](@ref) by passing it the [`State`](@ref).
```@example 1
intr(s)
s.x # Show final positions
```
This integrates from `t0` to `tmax`, steping by `h`. If you'd rather step a certain number of time steps:
```@example 1
N = 1000
intr(s,N)
```
**Note:** If you want to run the integration from the initial condtions again, you must 'reset' the [`State`](@ref). I.e. run `s = State(ic)`. Otherwise, the integration will begin from what ever `s.t` is currently equal to, and with those coordinates.

### Transit Timing
TBD. For a transit timing example notebook, see examples on GitHub: [NbodyGradient](https://github.com/ericagol/NbodyGradient).