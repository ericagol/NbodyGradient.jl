using FiniteDifferences
import NbodyGradient: set_state!, zero_out!, amatrix

@testset "Photodynamics" begin
    N = 3
    t0 = 7257.93115525 - 7300.0 - 0.5 # Initialize IC before first transit 
    h = 0.04
    itime = 200.0
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
        intr = Integrator(ah18!, h, 0.0, tmax)
        s = State(ic)
        ttbv = TransitParameters(itime, ic)
        intr(s, ttbv;grad=grad)
        return ttbv
    end

    # Calculate b,vsky for specified times
    function calc_pd(h, times;grad=false)
        intr = Integrator(ah18!, h, 0.0, tmax)
        s = State(ic)
        pd = Photodynamics(times, ic)
        intr(s, pd; grad=grad)
        return pd
    end

    ttbv = calc_times(h; grad=true)
    mask = ttbv.ttbv[1,2,:] .!= 0.0
    pd = calc_pd(h, ttbv.ttbv[1,2,mask]; grad=true)

    tol = 1e-10
    #@test all(abs.(ttbv.ttbv[2,2,mask] .- pd.vsky[2,:]) .< tol)
    @test all(abs.(ttbv.ttbv[3,2,mask] .- pd.bsky2[2,:]) .< tol)

    # method for finite diff
    function calc_b(θ,i=1;b=true,times=ttbv.ttbv[1,2,mask]) 
        elements = reshape(θ,3,7)
        intr = Integrator(ah18!, h, 0.0, 10.0)
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
        pdp = Photodynamics(big.(times), ic_big); 
        pdm = Photodynamics(big.(times), ic_big);
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
                    for k in 1:length(times)-1
                        # Compute double-sided derivative for more accuracy:
                        dbvde_num[1,i,k,iq,jq] = (pdp.vsky[i,k] - pdm.vsky[i,k]) / (2dq0)
                        dbvde_num[2,i,k,iq,jq] = (pdp.bsky2[i,k] - pdm.bsky2[i,k]) / (2dq0)
                    end
                end
            end
        end
        return dbvde_num
    end

    dbvde_num = calc_finite_diff(h, t0, pd.times, tmax, elements)
    @test isapprox(asinh.(pd.dbvdelements[3,:,:,:,:]), asinh.(dbvde_num[2,:,:,:,:]);norm=maxabs)
end