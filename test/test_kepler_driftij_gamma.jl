# This code tests the function kepler_driftij_gamma
import NbodyGradient: kepler_driftij_gamma!, Derivatives

@testset "kepler_driftij_gamma" begin
    for drift_first in [true,false]
      # Next, try computing two-body Keplerian Jacobian:

        NDIM = 3
        n = 3
        H = [3,1,1]
        t0 = 7257.93115525
        h  = 0.25
        hbig  = big(h)
        tmax = 600.0
        dlnq = big(1e-15)

        elements = readdlm("elements.txt", ',')

        m = zeros(n)
        x0 = zeros(NDIM, n)
        v0 = zeros(NDIM, n)

        for k = 1:n
            m[k] = elements[k,1]
        end
        for iter = 1:2

            init = ElementsIC(t0, H, elements)
            ic_big = ElementsIC(big(t0), H, big.(elements))
            x0, v0, _ = init_nbody(init)
            s0 = State(init)
            sbig = State(ic_big)
            if iter == 2
                # Reduce masses to trigger hyperbolic routine:
                m[1:n] *= 1e-3
                s0.m[1:n] *= 1e-3
                sbig.m[1:n] *= 1e-3
                hbig = big(h)
            end
            # Tilt the orbits a bit:
            x0[2,1] = 5e-1 * sqrt(x0[1,1]^2 + x0[3,1]^2)
            x0[2,2] = -5e-1 * sqrt(x0[1,2]^2 + x0[3,2]^2)
            v0[2,1] = 5e-1 * sqrt(v0[1,1]^2 + v0[3,1]^2)
            v0[2,2] = -5e-1 * sqrt(v0[1,2]^2 + v0[3,2]^2)

            i = 1 ; j = 2
            x = copy(x0) ; v = copy(v0)
            s = deepcopy(State(init))
            s.x .= x; s.v .= v
            d = Derivatives(Float64, s.n)
            kepler_driftij_gamma!(s,d,i,j,h,drift_first)

            x0 = copy(s.x) ; v0 = copy(s.v)
            xerror = zeros(NDIM, n); verror = zeros(NDIM, n)
            xbig = big.(s.x) ; vbig = big.(s.v); mbig = big.(m)
            xerr_big = zeros(BigFloat, NDIM, n); verr_big = zeros(BigFloat, NDIM, n)
            kepler_driftij_gamma!(s,d,i,j,h,drift_first)
            d.jac_ij .+= Matrix{Float64}(I, size(d.jac_ij))

            # Now, compute the derivatives numerically:
            jac_ij_num = zeros(BigFloat, 14, 14)
            dqdt_num = zeros(BigFloat, 14)
            xsave = big.(x)
            vsave = big.(v)
            msave = big.(m)
            sbig.x .= copy(xsave)
            sbig.v .= copy(vsave)
            sbig.m .= copy(msave)
            # Compute the time derivatives:
            # Initial positions, velocities & masses:
            sm = deepcopy(State(ic_big))
            sm.x .= big.(x0)
            sm.v .= big.(v0)
            sm.m .= big.(msave)
            dq = dlnq * hbig
            hbig -= dq
            sm.xerror .= 0.0; sm.verror .= 0.0
            kepler_driftij_gamma!(sm,i,j,hbig,drift_first)
            sp = deepcopy(State(ic_big))
            sp.x .= big.(x0)
            sp.v .= big.(v0)
            sp.m .= big.(msave)
            hbig += 2dq
            sp.xerror .= 0.0; sp.verror .= 0.0
            kepler_driftij_gamma!(sp,i,j,hbig,drift_first)
            # Now x & v are final positions & velocities after time step
            for k = 1:3
                dqdt_num[   k] = 0.5 * (sp.x[k,i] - sm.x[k,i]) / dq
                dqdt_num[ 3 + k] = 0.5 * (sp.v[k,i] - sm.v[k,i]) / dq
                dqdt_num[ 7 + k] = 0.5 * (sp.x[k,j] - sm.x[k,j]) / dq
                dqdt_num[10 + k] = 0.5 * (sp.v[k,j] - sm.v[k,j]) / dq
            end
            hbig = big(h)

            # Compute position, velocity & mass derivatives:
            for jj = 1:3
                # Initial positions, velocities & masses:
                sm = deepcopy(State(ic_big))
                sm.x .= big.(x0)
                sm.v .= big.(v0)
                sm.m .= big.(msave)
                dq = dlnq * sm.x[jj,i]
                if sm.x[jj,i] != 0.0
                    sm.x[jj,i] -=  dq
                else
                    dq = dlnq
                    sm.x[jj,i] = -dq
                end
                sm.xerror .= 0.0; sm.verror .= 0.0
                kepler_driftij_gamma!(sm, i, j, hbig, drift_first)
                sp = deepcopy(State(ic_big))
                sp.x .= big.(x0)
                sp.v .= big.(v0)
                sp.m .= big.(msave)
                if sm.x[jj,i] != 0.0
                    sp.x[jj,i] +=  dq
                else
                    dq = dlnq
                    sp.x[jj,i] = dq
                end
                sp.xerror .= 0.0; sp.verror .= 0.0
                kepler_driftij_gamma!(sp, i, j, hbig, drift_first)
                # Now x & v are final positions & velocities after time step
                for k = 1:3
                    jac_ij_num[   k,  jj] = .5 * (sp.x[k,i] - sm.x[k,i]) / dq
                    jac_ij_num[ 3 + k,  jj] = .5 * (sp.v[k,i] - sm.v[k,i]) / dq
                    jac_ij_num[ 7 + k,  jj] = .5 * (sp.x[k,j] - sm.x[k,j]) / dq
                    jac_ij_num[10 + k,  jj] = .5 * (sp.v[k,j] - sm.v[k,j]) / dq
                end
                sm = deepcopy(State(ic_big))
                sm.x .= big.(x0)
                sm.v .= big.(v0)
                sm.m .= big.(msave)
                dq = dlnq * sm.v[jj,i]
                if sm.v[jj,i] != 0.0
                    sm.v[jj,i] -=  dq
                else
                    dq = dlnq
                    sm.v[jj,i] = -dq
                end
                sm.xerror .= 0.0; sm.verror .= 0.0
                kepler_driftij_gamma!(sm, i, j, hbig, drift_first)
                sp = deepcopy(State(ic_big))
                sp.x .= big.(x0)
                sp.v .= big.(v0)
                sp.m .= big.(msave)
                if sp.v[jj,i] != 0.0
                    sp.v[jj,i] +=  dq
                else
                    dq = dlnq
                    sp.v[jj,i] = dq
                end
                sp.xerror .= 0.0; sp.verror .= 0.0
                kepler_driftij_gamma!(sp, i, j, hbig, drift_first)
                for k = 1:3
                    jac_ij_num[   k,3 + jj] = .5 * (sp.x[k,i] - sm.x[k,i]) / dq
                    jac_ij_num[ 3 + k,3 + jj] = .5 * (sp.v[k,i] - sm.v[k,i]) / dq
                    jac_ij_num[ 7 + k,3 + jj] = .5 * (sp.x[k,j] - sm.x[k,j]) / dq
                    jac_ij_num[10 + k,3 + jj] = .5 * (sp.v[k,j] - sm.v[k,j]) / dq
                end
            end

            # Now vary mass of inner planet:
            sm = deepcopy(State(ic_big))
            sm.x .= big.(x0)
            sm.v .= big.(v0)
            sm.m .= big.(msave)
            dq = sm.m[i] * dlnq
            sm.m[i] -= dq
            sm.xerror .= 0.0; sm.verror .= 0.0
            kepler_driftij_gamma!(sm,i,j,hbig,drift_first)
            sp = deepcopy(State(ic_big))
            sp.x .= big.(x0)
            sp.v .= big.(v0)
            sp.m .= big.(msave)
            dq = sp.m[i] * dlnq
            sp.m[i] += dq
            sp.xerror .= 0.0; sp.verror .= 0.0
            kepler_driftij_gamma!(sp,i,j,hbig,drift_first)
            for k = 1:3
                jac_ij_num[   k,7] = .5 * (sp.x[k,i] - sm.x[k,i]) / dq
                jac_ij_num[ 3 + k,7] = .5 * (sp.v[k,i] - sm.v[k,i]) / dq
                jac_ij_num[ 7 + k,7] = .5 * (sp.x[k,j] - sm.x[k,j]) / dq
                jac_ij_num[10 + k,7] = .5 * (sp.v[k,j] - sm.v[k,j]) / dq
            end
            # The mass doesn't change:
            jac_ij_num[7,7] =  1.0

            for jj = 1:3
                # Now vary parameters of outer planet:
                sm = deepcopy(State(ic_big))
                sm.x .= big.(x0)
                sm.v .= big.(v0)
                sm.m .= big.(msave)
                dq = dlnq * sm.x[jj,j]
                if sm.x[jj,j] != 0.0
                    sm.x[jj,j] -=  dq
                else
                    dq = dlnq
                    sm.x[jj,j] = -dq
                end
                sm.xerror .= 0.0; sm.verror .= 0.0
                kepler_driftij_gamma!(sm, i, j, hbig, drift_first)
                sp = deepcopy(State(ic_big))
                sp.x .= big.(x0)
                sp.v .= big.(v0)
                sp.m .= big.(msave)
                if sp.x[jj,j] != 0.0
                    sp.x[jj,j] +=  dq
                else
                    dq = dlnq
                    sp.x[jj,j] = dq
                end
                sp.xerror .= 0.0; sp.verror .= 0.0
                kepler_driftij_gamma!(sp, i, j, hbig, drift_first)
                for k = 1:3
                    jac_ij_num[   k,7 + jj] = .5 * (sp.x[k,i] - sm.x[k,i]) / dq
                    jac_ij_num[ 3 + k,7 + jj] = .5 * (sp.v[k,i] - sm.v[k,i]) / dq
                    jac_ij_num[ 7 + k,7 + jj] = .5 * (sp.x[k,j] - sm.x[k,j]) / dq
                    jac_ij_num[10 + k,7 + jj] = .5 * (sp.v[k,j] - sm.v[k,j]) / dq
                end
                sm = deepcopy(State(ic_big))
                sm.x .= big.(x0)
                sm.v .= big.(v0)
                sm.m .= big.(msave)
                dq = dlnq * sm.v[jj,j]
                if sm.v[jj,j] != 0.0
                    sm.v[jj,j] -=  dq
                else
                    dq = dlnq
                    sm.v[jj,j] = -dq
                end
                sm.xerror .= 0.0; sm.verror .= 0.0
                kepler_driftij_gamma!(sm, i, j, hbig, drift_first)
                sp = deepcopy(State(ic_big))
                sp.x .= big.(x0)
                sp.v .= big.(v0)
                sp.m .= big.(msave)
                if sp.v[jj,j] != 0.0
                    sp.v[jj,j] +=  dq
                else
                    dq = dlnq
                    sp.v[jj,j] = dq
                end
                sp.xerror .= 0.0; sp.verror .= 0.0
                kepler_driftij_gamma!(sp, i, j, hbig, drift_first)
                for k = 1:3
                    jac_ij_num[   k,10 + jj] = .5 * (sp.x[k,i] - sm.x[k,i]) / dq
                    jac_ij_num[ 3 + k,10 + jj] = .5 * (sp.v[k,i] - sm.v[k,i]) / dq
                    jac_ij_num[ 7 + k,10 + jj] = .5 * (sp.x[k,j] - sm.x[k,j]) / dq
                    jac_ij_num[10 + k,10 + jj] = .5 * (sp.v[k,j] - sm.v[k,j]) / dq
                end
            end

            # Now vary mass of outer planet:
            sm = deepcopy(State(ic_big))
            sm.x .= big.(x0)
            sm.v .= big.(v0)
            sm.m .= big.(msave)
            dq = sm.m[j] * dlnq
            sm.m[j] -= dq
            sm.xerror .= 0.0; sm.verror .= 0.0
            kepler_driftij_gamma!(sm,i,j,hbig,drift_first)
            sp = deepcopy(State(ic_big))
            sp.x .= big.(x0)
            sp.v .= big.(v0)
            sp.m .= big.(msave)
            dq = sp.m[j] * dlnq
            sp.m[j] += dq
            sp.xerror .= 0.0; sp.verror .= 0.0
            kepler_driftij_gamma!(sp,i,j,hbig,drift_first)
            for k = 1:3
                jac_ij_num[   k,14] = .5 * (sp.x[k,i] - sm.x[k,i]) / dq
                jac_ij_num[ 3 + k,14] = .5 * (sp.v[k,i] - sm.v[k,i]) / dq
                jac_ij_num[ 7 + k,14] = .5 * (sp.x[k,j] - sm.x[k,j]) / dq
                jac_ij_num[10 + k,14] = .5 * (sp.v[k,j] - sm.v[k,j]) / dq
            end
            # The mass doesn't change:
            jac_ij_num[14,14] =  1.0

            @test isapprox(jac_ij_num, d.jac_ij ;norm=maxabs)
            @test isapprox(d.dqdt_ij, dqdt_num ;norm=maxabs)

        end
    end
end
