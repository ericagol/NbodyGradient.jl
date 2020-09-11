import NbodyGradient: set_state!, zero_out!, amatrix

@testset "Photodynamics" begin
    N = 3
    t0 = 7257.93115525 - 7300.0 - 0.5 # Initialize IC before first transit 
    h = 0.04
    itime = 10.0
    tmax = itime + t0

    # Setup initial conditions:
    elements = readdlm("elements.txt", ',')[1:N,:]
    elements[2:end,3] .-= 7300.0
    elements[:,end] .= 0.0 # Set Ωs to 0
    elements[2,1] *= 10.0
    elements[3,1] *= 10.0
    ic = ElementsIC(t0, N, elements)

    # Calculate transit times and b,vsky
    function calc_times(h)
        intr = Integrator(ah18!, h, 0.0, tmax)
        s = State(ic)
        ttbv = TransitParameters(itime, ic)
        intr(s, ttbv;grad=false)
        return ttbv
    end

    # Calculate b,vsky for specified times
    function calc_pd(h,times)
        intr = Integrator(ah18!, h, 0.0, tmax)
        s = State(ic)
        pd = Photodynamics(times, ic)
        intr(s, pd;grad=false)
        return pd
    end

    ttbv = calc_times(h)
    mask = ttbv.ttbv[1,2,:] .!= 0.0
    pd = calc_pd(h,ttbv.ttbv[1,2,mask])

    tol = 1e-10

    @test all(abs.(ttbv.ttbv[2,2,mask] .- pd.vsky[2,:]) .< tol)
    @test all(abs.(ttbv.ttbv[3,2,mask] .- pd.bsky2[2,:]) .< tol)
end