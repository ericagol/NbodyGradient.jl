using FiniteDifferences
import NbodyGradient: set_state!, zero_out!, amatrix

@testset "Transit Series" begin
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
    function calc_times(h;grad=false)
        intr = Integrator(ahl21!, h, 0.0, tmax)
        s = State(ic)
        ttbv = TransitParameters(itime, ic)
        intr(s, ttbv;grad=grad)
        return ttbv
    end

    # Calculate b,vsky for specified times
    function calc_pd(h, times;grad=false)
        intr = Integrator(ahl21!, h, 0.0, tmax)
        s = State(ic)
        ts = TransitSeries(times, ic)
        intr(s, ts; grad=grad)
        return ts
    end

    ttbv = calc_times(h; grad=true)
    mask = ttbv.ttbv[1,2,:] .!= 0.0
    ts = calc_pd(h, ttbv.ttbv[1,2,mask]; grad=true)

    tol = 1e-10
    # Test whether TransitSeries can reproduce b,vsky at transit time from TransitParameters
    @test all(abs.(ttbv.ttbv[2,2,mask] .- ts.vsky[2,:]) .< tol)
    @test all(abs.(ttbv.ttbv[3,2,mask] .- ts.bsky2[2,:]) .< tol)

    # Now make set of times to calc b and vsky and compare derivatives
    times = [t0, t0 + 1.0]
    ts = calc_pd(h, times; grad=true)

    # method for finite diff
    function calc_b(θ,i=1;b=true,times=ttbv.ttbv[1,2,mask])
        elements = reshape(θ,3,7)
        intr = Integrator(ahl21!, h, 0.0, 10.0)
        ic = ElementsIC(t0,N,elements)
        s = State(ic)
        pd = Photodynamics(times, ic)
        intr(s, pd; grad=true)
        if b
            return pd.bsky2[i,:]
        else
            return pd.vsky[i,:]
        end
    end

    # Now test derivatives
    function calc_finite_diff(h, t0, times, tmax, elements)
        # Now do finite difference with big float
        dq0 = big(1e-10)
        ic_big = ElementsIC(big(t0), N, big.(elements))
        elements_big = copy(ic_big.elements)
        s_big = State(ic_big)
        pdp = TransitSeries(big.(times), ic_big);
        pdm = TransitSeries(big.(times), ic_big);
        dbvde_num = zeros(BigFloat, size(pdp.dbvdelements));
        intr_big = Integrator(big(h), zero(BigFloat), big(tmax))

        for jq in 1:N
            for iq in 1:7
                zero_out!(pdp); zero_out!(pdm);
                ic_big.elements .= elements_big
                ic_big.m .= elements_big[:,1]
                if iq == 7; ivary = 1; else; ivary = iq + 1; end
                ic_big.elements[jq,ivary] += dq0
                if ivary == 1; ic_big.m[jq] += dq0; amatrix(ic_big); end # Masses don't update with elements array
                sp = State(ic_big);
                intr_big(sp, pdp; grad=false)
                ic_big.elements[jq,ivary] -= 2dq0
                if ivary == 1; ic_big.m[jq] -= 2dq0; amatrix(ic_big); end
                sm = State(ic_big)
                intr_big(sm, pdm; grad=false)
                for i in 2:N
                    for k in 1:ts.nt
                        # Compute double-sided derivative for more accuracy:
                        dbvde_num[1,i,k,iq,jq] = (pdp.vsky[i,k] - pdm.vsky[i,k]) / (2dq0)
                        dbvde_num[2,i,k,iq,jq] = (pdp.bsky2[i,k] - pdm.bsky2[i,k]) / (2dq0)
                    end
                end
            end
        end
        return dbvde_num
    end

    mask = zeros(Bool, size(ts.dbvdelements));
    for jq = 1:N
        for iq = 1:7
            if iq == 7; ivary = 1; else; ivary = iq + 1; end  # Shift mass variation to end
            for i = 2:N
                for k = 1:ts.nt
                    for itbv = 1:2
                        # Ignore inclination & longitude of nodes variations:
                        if iq != 5 && iq != 6 && ~(jq == 1 && iq < 7) && ~(jq == i && iq == 7)
                            mask[2,i,k,iq,jq] = true
                        end
                    end
                end
            end
        end
    end

    dbvde_num = calc_finite_diff(h, t0, ts.times, tmax, elements)
    @test isapprox(asinh.(ts.dbvdelements[mask]), asinh.(dbvde_num[mask]); norm=maxabs)
    #mask_times = ttbv.ttbv[1,2,:] .!= 0.0
    #@test isapprox(asinh.(ts.dbvdelements[mask]), asinh.(ttbv.dtbvdelements[2:3,:,mask_times,:,:][mask]); norm=maxabs)
end