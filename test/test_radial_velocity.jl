#from https://github.com/jlustigy/ExoJulia/blob/master/ExoJulia/Orbit/keplerSolver.jl
function kepler_solver(M, ecc)
    Mred = mod(M,2pi)

    #Initial Guess:
    E = Mred + sign(sin(Mred))*0.85*ecc

    #Setting Initial Parameter
    di3 = 1.0

    #Setting the tolerance
    tol = 1e-12

    #Beginning the loop for quartic kepler_solver
    niter = 0
    while (abs(di3) > tol) & (niter < 30)
        SE = ecc*sin(E); CE = ecc*cos(E)
        f_of_E = E-SE-Mred; df_of_E=1-CE; d2f_of_E=SE; d3f_of_E=CE
        di1 = -f_of_E/df_of_E
        di2 = -f_of_E/(df_of_E+0.5*di1*d2f_of_E)
        di3 = -f_of_E/(df_of_E+0.5*di2*d2f_of_E+di2^2/6.0*d3f_of_E)

        E+=di3
        niter = niter + 1
    end

    E += (M-Mred)

    if(niter == 30)
        println("Error: Reached niter = 30")
    end

    E
end

#modified version of https://github.com/jlustigy/ExoJulia/blob/master/spring16/hw2/Agol_Agol/rv_model_one.jl
function analytic_rv(t, period, tp, ecc, m1, m2, a, w, inc)

    M = (t - tp)*2.0*pi/period

    eanom = kepler_solver(M, ecc)
    f = 2.0*atan(sqrt((1.0+ecc)/(1.0-ecc))*tan(0.5*eanom))

    G = NbodyGradient.GNEWT

    rvmod = ((sqrt(G/((m1+m2)*a*(1-ecc^2)))*m2*sin(inc))*(cos(w+f)+ecc*cos(w)))

end

function test_rvs()

    #defining the times of radial velocity (equivalent to how many data points)
    tmax = 40.0
    t = collect(0.0:1.0:tmax);

    #Parameters of Analytic Radial Velocity
    #period
    P = 17.5
    #time of periastron
    t_p = 0.0
    #eccentricity
    ecc = 0.001
    #star mass
    m_s = 1.034
    #planet mass
    m_p = 5.91e-5
    #semi-major axis
    a = 0.13340608724009007
    #argument of periastron
    ω = pi/2
    #inclination
    inc = pi/2

    #get radial velocities from analytic method
    rvs_analytic = analytic_rv.(t,P,t_p,ecc,m_s,m_p,a,ω,inc);


    #move to NbodyGradient method
    #Setting Initial Conditions for NbodyGradient Radial Velocity
    star = Elements(m = m_s)

    t0 = 0.0

    planet = Elements(
        m = m_p,
        P = P,
        t0 = t0,
        esinϖ = ecc*sin(pi+ω),
        ecosϖ = ecc*cos(pi+ω),
        Ω = float(pi/2),
        I = inc)

    N = 2 #number of bodies

    ic = ElementsIC(0.0,N,star,planet)
    s = State(ic)

    h = planet.P/30

    rvs = RadialVelocities(t,ic);

    #run Integrator
    intr = Integrator(h,tmax)
    intr(s,rvs)

    #get radial velocities from NbodyGradient method
    rvs_nbody = rvs.rvs;

    #test the accuracy between the analytic model and the NbodyGradient rv_model
    @test isapprox(rvs_analytic, rvs_nbody)
end