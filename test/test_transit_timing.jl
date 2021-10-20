import NbodyGradient: set_state!, zero_out!, amatrix, initialize!

@testset "Transit Timing" begin
    N = 3
    t0 = 7257.93115525 - 7300.0 - 0.5 # Initialize IC before first transit
    h = 0.04
    itime = 10.0 # Amount of time the integrator will run
    tmax = itime

    # Setup initial conditions:
    elements = readdlm("elements.txt", ',')[1:N,:]
    elements[2:end,3] .-= 7300.0
    elements[:,end] .= 0.0 # Set Ωs to 0
    elements[2,1] *= 10.0
    elements[3,1] *= 10.0
    ic = ElementsIC(t0, N, elements)

    function calc_times(h, ti)
        intr = Integrator(ahl21!, h, 0.0, tmax)
        s = State(ic)
        tt = TransitTiming(itime, ic, ti)
        intr(s, tt)
        return tt
    end

    function calc_finite_diff(h, t0, itime, tmax, elements, ti)
        # Now do finite difference with big float
        dq0 = big(1e-10)
        ic_big = ElementsIC(big(t0), N, big.(elements))
        elements_big = copy(ic_big.elements)
        sp, dp = dState(ic_big)
        sm, dm = dState(ic_big)
        ttp = TransitTiming(big(itime), ic_big, ti);
        ttm = TransitTiming(big(itime), ic_big, ti);
        dtde_num = zeros(BigFloat, size(ttp.dtdelements));
        intr_big = Integrator(big(h), zero(BigFloat), big(tmax))

        for jq in 1:N
            for iq in 1:7
                zero_out!(ttp); zero_out!(ttm); zero_out!(dp); zero_out!(dm);
                ic_big.elements .= elements_big
                ic_big.m .= elements_big[:,1]
                if iq == 7; ivary = 1; else; ivary = iq + 1; end
                ic_big.elements[jq,ivary] += dq0
                if ivary == 1; ic_big.m[jq] += dq0; amatrix(ic_big); end # Masses don't update with elements array
                initialize!(sp, ic_big)
                intr_big(sp, ttp, dp; grad=false)
                ic_big.elements[jq,ivary] -= 2dq0
                if ivary == 1; ic_big.m[jq] -= 2dq0; amatrix(ic_big); end
                initialize!(sm, ic_big)
                intr_big(sm, ttm, dm; grad=false)
                for i in ttm.occs
                    for k in 1:ttm.count[i]
                    # Compute double-sided derivative for more accuracy:
                        dtde_num[i,k,iq,jq] = (ttp.tt[i,k] - ttm.tt[i,k]) / (2dq0)
                    end
                end
            end
        end
        return dtde_num
    end

    for ti in 1:N
        # Calculate transit times
        hs = h ./ [1.0,2.0,4.0,8.0]
        tts = TransitTiming[]
        for h in hs
            push!(tts, calc_times(h, ti))
        end

        mask = zeros(Bool, size(tts[2].dtdq0));
        for jq = 1:N
            for iq = 1:7
                if iq == 7; ivary = 1; else; ivary = iq + 1; end  # Shift mass variation to end
                for i = tts[1].occs
                    for k = 1:tts[1].count[i]
                    # Ignore inclination & longitude of nodes variations:
                        if iq != 5 && iq != 6 && ~(jq == 1 && iq < 7) && ~(jq == i && iq == 7)
                            mask[i,k,iq,jq] = true
                        end
                    end
                end
            end
        end

        dtde_num = calc_finite_diff(h, t0, itime, tmax, elements, ti)
        @test isapprox(asinh.(tts[1].dtdelements[mask]), asinh.(dtde_num[mask]);norm=maxabs)
        @test isapprox(asinh.(tts[1].dtdelements), asinh.(dtde_num);norm=maxabs)
    end

    # Test that the transit times for the derivatives and non-derivatives match
    tmax = 2000.0
    s_nograd = State(ic); tt_nograd = TransitTiming(tmax, ic);
    s_grad = State(ic); tt_grad = TransitTiming(tmax, ic);
    Integrator(h, tmax)(s_nograd, tt_nograd, grad=false)
    Integrator(h, tmax)(s_grad, tt_grad, grad=true)
    @test tt_nograd.tt == tt_grad.tt
end