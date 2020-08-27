@testset "Timing Accuracy" begin
    n = 7
    t0 = 7257.0
    h = 0.01
    tmax = 10.0
    intr = Integrator(ah18!,h,t0,tmax)
    
    # Read in initial conditions:
    H = [n,ones(Int64, n - 1)...]
    elements = readdlm("elements.txt", ',')[1:n,:]
    ic = ElementsIC(t0,H,elements)
    s = State(ic)
    
    # Holds transit times, derivatives, etc.
    tt = TransitTiming(tmax,ic);

    # Run integrator and calculate times
    intr(s,tt)

    # Check that we get back the initial transit time (up to tolerance)
    tol = 1e-6 # Percent error
    for i in 1:n-1
        @test abs((ic.elements[i+1,3] - tt.tt[i+1,1])/ic.elements[i+1,3]) < tol
    end
end