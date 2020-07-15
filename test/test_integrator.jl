@testset "Integrator" begin
    ###### Float64 ######
    # Initial Conditions for Float64 version
    H = [3,1,1]
    n = 3 
    t0 = 7257.93115525
    elements = readdlm("elements.txt",',')
    # Increase mass of inner planets:
    elements[2,1] *= 100.0
    elements[3,1] *= 100.0
    # Generate ICs
    ic = ElementsIC(elements[1:n,:],H,t0)

    # Setup State and tilt orbits
    function perturb!(s::State{<:AbstractFloat})
        s.x[2,1] = 5e-1*sqrt(s.x[1,1]^2+s.x[3,1]^2)
        s.x[2,2] = -5e-1*sqrt(s.x[1,2]^2+s.x[3,2]^2)
        s.x[2,3] = -5e-1*sqrt(s.x[1,2]^2+s.x[3,2]^2)
        s.v[2,1] = 5e-1*sqrt(s.v[1,1]^2+s.v[3,1]^2)
        s.v[2,2] = -5e-1*sqrt(s.v[1,2]^2+s.v[3,2]^2)
        s.v[2,3] = -5e-1*sqrt(s.v[1,2]^2+s.v[3,2]^2)
        return
    end

    s0 = State(ic)
    perturb!(s0)

    # Setup integrator
    h  = 0.05
    nstep = 100
    tmax = nstep*h
    AH18 = Integrator(ah18!,h,t0,tmax)

    # Integrate
    AH18(s0)
    ####################

    ##### BigFloat #####
    t0_big = big(t0)
    elements_big = big.(elements)
    ic_big = ElementsIC(elements_big,H,t0_big)

    s_big = State(ic_big)
    perturb!(s_big)

    h_big = big(h)
    tmax_big = nstep*h_big
    AH18_big = Integrator(ah18!,h_big,t0_big,tmax_big)

    AH18_big(s_big)
    ####################

    ## Numerical Derivatives ##
    # Vary the initial parameters of planet j:
    n = 3
    dlnq = big(1e-20)
    jac_step_num = zeros(BigFloat,7*n,7*n)
    for j=1:n
        # Vary the initial phase-space elements:
        for jj=1:3
            # Position derivatives:
            sm = deepcopy(State(ic_big))
            perturb!(sm)
            dq = dlnq * sm.x[jj,j]
            if sm.x[jj,j] != 0.0
                sm.x[jj,j] -=  dq
            else
                dq = dlnq
                sm.x[jj,j] = -dq
            end
            AH18_big(sm)

            sp = deepcopy(State(ic_big))
            perturb!(sp)
            dq = dlnq * sp.x[jj,j]
            if sp.x[jj,j] != 0.0
                sp.x[jj,j] +=  dq
            else
                dq = dlnq
                sp.x[jj,j] = dq
            end
            AH18_big(sp)

            # Now x & v are final positions & velocities after time step
            for i=1:n
                for k=1:3
                    jac_step_num[(i-1)*7+  k,(j-1)*7+ jj] = .5*(sp.x[k,i]-sm.x[k,i])/dq
                    jac_step_num[(i-1)*7+3+k,(j-1)*7+ jj] = .5*(sp.v[k,i]-sm.v[k,i])/dq
                end
            end

            # Next, velocity derivatives:
            sm = deepcopy(State(ic_big))
            perturb!(sm)
            dq = dlnq * sm.v[jj,j]
            if sm.v[jj,j] != 0.0
                sm.v[jj,j] -=  dq
            else
                dq = dlnq
                sm.v[jj,j] = -dq
            end
            AH18_big(sm)

            sp = deepcopy(State(ic_big))
            perturb!(sp)
            dq = dlnq * sp.v[jj,j]
            if sp.v[jj,j] != 0.0
                sp.v[jj,j] +=  dq
            else
                dq = dlnq
                sp.v[jj,j] = dq
            end
            AH18_big(sp)

            for i=1:n
                for k=1:3
                    jac_step_num[(i-1)*7+  k,(j-1)*7+3+jj] = .5*(sp.x[k,i]-sm.x[k,i])/dq
                    jac_step_num[(i-1)*7+3+k,(j-1)*7+3+jj] = .5*(sp.v[k,i]-sm.v[k,i])/dq
                end
            end
        end

        # Now vary mass of planet:
        sm = deepcopy(State(ic_big))
        perturb!(sm)
        dq = sm.m[j]*dlnq
        sm.m[j] -= dq
        AH18_big(sm)

        sp = deepcopy(State(ic_big))
        perturb!(sp)
        dq = sp.m[j]*dlnq
        sp.m[j] += dq
        AH18_big(sp)
        for i=1:n
            for k=1:3
                jac_step_num[(i-1)*7+  k,j*7] = .5*(sp.x[k,i]-sm.x[k,i])/dq
                jac_step_num[(i-1)*7+3+k,j*7] = .5*(sp.v[k,i]-sm.v[k,i])/dq
            end
        end
        # Mass unchanged -> identity
        jac_step_num[j*7,j*7] = 1.0
    end

    #= dqdt
    dqdt = zeros(7*n)
    dqdt_num = zeros(BigFloat,7*n)
    x = copy(x0)
    v = copy(v0)
    m = copy(m0)
    xerror = zeros(3,n); verror = zeros(3,n)
    ah18!(x,v,xerror,verror,h,m,n,dqdt,pair)
    xm= big.(x0)
    vm= big.(v0)
    mm= big.(m0)
    fill!(big_xerror,0.0)
    fill!(big_verror,0.0)
    dq = hbig*dlnq
    hbig -= dq
    ah18!(xm,vm,big_xerror,big_verror,hbig,mm,n,pair)
    xp= big.(x0)
    vp= big.(v0)
    mp= big.(m0)
    fill!(big_xerror,0.0)
    fill!(big_verror,0.0)
    hbig += 2dq
    ah18!(xp,vp,big_xerror,big_verror,hbig,mp,n,pair)
    for i=1:n, k=1:3
    dqdt_num[(i-1)*7+  k] = .5*(xp[k,i]-xm[k,i])/dq
    dqdt_num[(i-1)*7+3+k] = .5*(vp[k,i]-vm[k,i])/dq
    end
    dqdt_num = convert(Array{Float64,1},dqdt_num)
    =#
    ####################

    ###### Tests #######
    # Check analytic vs finite difference derivatives 
    @test isapprox(asinh.(s0.jac_step),asinh.(jac_step_num);norm=maxabs)
end