# Basic Usage
Here we'll walk through an example of integrating a 3-body system.
##### Units
A quick note on the units used throughout the code.
- Distance: AU
- Time: Days
- Mass: Solar Masses
- Angles: Radians

### Initial Conditions
First, we define the orbital elements of the system. This can be done by creating [`Elements`](@ref) for each body in the system.

Start with a 1 solar mass star
```@example 1
using NbodyGradient

a = Elements(m = 1.0);
nothing # hide
```

We then define the orbital elements for second body, say an Earth analogue.
```@example 1
b = Elements(
    m = 3e-6,     # Mass [Solar masses]
    t0 = 0.0,     # Initial time of transit [days]
    P = 365.256,  # Period [days]
    ecosϖ = 0.01, # Eccentricity * cos(Argument of Periastron)
    esinϖ = 0.0,  # Eccentricity * sin(Argument of Periastron)
    I = π/2,      # Inclination [Radians]
    Ω = 0.0       # Longitude of Ascending Node [Radians]
);
nothing # hide
```
(`ecosϖ` can be typed as `ecos\varpi` and then hitting tab)

Next we'll create a Jupiter analogue for the final body. Here the orbital elements are specified for the Keplerian ((a,b),c), or c orbiting the center of mass of a and b. (While this might not need to be stated explicitly here, this convention is useful for more complicated hierarchical systems).
```@example 1
c = Elements(
    m = 9.54e-4,
    P = 4332.59,
    ecosϖ = 0.05,
    I = π/2
);
nothing # hide
```
Here, we can simply omit any orbital elements which are zero. Unspecified elements are set to zero by default.

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
Now that we have initial conditions, we can construct and run the integrator. First, define an [`Integrator`](@ref), specifying the integration scheme, the time step, and final time. We'll use the `ahl21!` mapping.
```@example 1
h = b.P/30.0 # We want at most 1/20th of the smallest period for a time step
tmax = 5*c.P # Integrate for 5 orbital periods of the outer body
intr = Integrator(ahl21!,h,tmax);
nothing # hide
```

Finally, run the [`Integrator`](@ref) by passing it the [`State`](@ref).
```@example 1
intr(s)
s.x # Show final positions
```
This integrates from `s.t` to `s.t+tmax`, steping by `h`. If you'd rather step a certain number of time steps:
```@example 1
N = 1000
intr(s,N)
```
!!! note "Re-running simulations"
    If you want to run the integration from the initial condtions again, you must 'reset' the [`State`](@ref). I.e. run `s = State(ic)`. Otherwise, the integration will begin from what ever `s.t` is currently equal to, and with those coordinates.

### Transit Timing
If we wish to compute transit times, we need only to pass a [`TransitTiming`](@ref)structure to the [`Integrator`](@ref).
```@example 1
s = State(ic) # Reset to the initial conditions
tt = TransitTiming(tmax, ic)
intr(s,tt)
```
To see the first 5 transit times of our second body about the first body, run:
```@example 1
tt.tt[2,1:5]
```