"""

To run interface check: `$ julia test_integrator.jl test`
To run benchmark: `$ julia test_integrator.jl benchmark`
"""
using NbodyGradient, LinearAlgebra, BenchmarkTools, Profile, Test
import NbodyGradient: init_nbody, comp_sum

function run_new()
    # Bodies in the system
    a = Elements(m=0.82)
    b = Elements(m=3.18e-4,P=228.774,e=0.0069,i=pi/2)
    c = Elements(m=3e-6,P=221.717,e=0.0054,i=pi/2,t0=-221.717/6)

    # Generate ICs at t=0.0 in a nested hierarchy
    ic = ElementsIC(0.0,a,b,c,H=[3,1,1])

    # Create the system state
    state = State(ic)

    # Now, set up integrator
    h = 10.0; t0 = 0.0; tmax = 9837.282
    AH18 = Integrator(h,t0,tmax)

    # Integrate!
    AH18(state)
    return state
end

function run_old()
    # Bodies in the system
    n = 3
    a = Elements(m=0.82)
    b = Elements(m=3.18e-4,P=228.774,e=0.0069,i=pi/2)
    c = Elements(m=3e-6,P=221.717,e=0.0054,i=pi/2,t0=-221.717/6)

    # Generate ICs at t=0.0 in a nested hierarchy
    ic = ElementsIC(0.0,a,b,c,H=[3,1,1])
    x,v,_ = init_nbody(ic)
    m = ic.elements[:,1]

    # Now, set up and run ah18!.
    h = 10.0; t0 = 0.0; tmax = 9837.283
    xerror = zeros(Float64,size(x)); verror = zeros(Float64,size(v))
    jac_step = Matrix{Float64}(I,7*n,7*n)
    jac_error = zeros(Float64,7*n,7*n)
    pair = zeros(Bool,n,n)

    t = t0; s2 = 0.0
    while t < (t0 + tmax)
        ah18!(x,v,xerror,verror,h,m,n,jac_step,jac_error,pair)
        t,s2 = comp_sum(t,s2,h)
    end
    return x,v,jac_step
end

function do_args()
    # Compare new structures to just passing arrays
    if ARGS[1] == "benchmark"
        print("New: ")
        @btime run_new();
        print("Old: ")
        @btime run_old();
    end

    # Profile new functions
    if ARGS[1] == "profile"
        Profile.clear()
        @profile run_new();
        Profile.print()
    end

    # Compare outputs
    if ARGS[1] == "test"
        @testset "ah18" begin
            state = run_new()
            x,v,jac = run_old()
            @test isapprox(state.x,x)
            @test isapprox(state.v,v)
            @test isapprox(state.jac_step,jac)
        end
    end
end

do_args()
