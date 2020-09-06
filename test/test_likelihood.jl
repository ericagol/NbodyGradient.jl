using NbodyGradient, LinearAlgebra, DelimitedFiles, Statistics,FileIO, JLD2
using Optim
using AdvancedHMC, MCMCChains, StatsPlots

function convert_elements(θ)
    θ_mod = length(θ) % 7
    N::Int64 = (length(θ) - θ_mod)/7
    
    if θ_mod == 0
        return reshape(θ,N,7), []
    else
        θ_error = θ[end-θ_mod:end]
        return reshape(θ[1:end-θ_mod],N,7), error
    end  
end
unpack_elements(el) = [el[:,i] for i in 1:7]

struct TimingData{T<:Real}
    N::Int64 # Number of data points
    nt::Vector{Int64} # Transit Number
    tt::Vector{T} # Transit times
    tt_err::Vector{T} # Uncertainty in transit times
end

# For Kepler-88 data set
function TimingData(filename::String;delim='\t',comments=true,T=Float64)
    data = readdlm(filename,delim,comments=comments)
    nt = Int64.(data[:,2])
    tn = T.(data[:,3])
    ttv = T.(data[:,4])
    N = length(nt)
    tt = tn .+ (ttv ./ 1440.0) # Minutes to days
    tt_err = T.(data[:,5]) ./ 1440.0 # Minutes to days
    return TimingData(N,nt,tt,tt_err)
end

function check_e(elements)
    _,_,_,ecosϖ,esinϖ,_,_ = unpack_elements(elements)
    e = sqrt.(ecosϖ.^2 + esinϖ.^2)
    if all(e .< 1.0)
        return false
    end
    return true
end

function check_bounds(θ,bounds)
    if all(θ .> bounds[1]) && all(θ .< bounds[2])
        return true
    end
    return false
end

# Assumes fully nested Keplerian
function logp(θ::Vector{T};data::TimingData{T},iplanet=2,mstar::T=0.990,h::T=0.27,t0::T=54.627020000014454,tmax::T=1700.0,grad=true) where T<:Real

    # Create elements arrays
    N = round(Int64,length(θ)/7) + 1
    elements = zeros(T,N,7)
    elements[1,1] = mstar
    elements[2:end,:],_ = convert_elements(θ)

    # Check that the e < 1
    if check_e(elements[2:end,:])
        if grad
            return 1e30, 1e30 .* ones(length(θ))
        else
            return 1e30
        end
    end

    # Get times for planet i
    tt = calc_transits(elements,h,t0,tmax;grad=grad)
    tti = tt.tt[iplanet,:]
    # Make sure we have at least the number of transits in the data
    if (length(tti) <= maximum(data.nt))
        return 1e30, 1e30 .* ones(length(θ))
    end
    # Mask points without data
    if data.nt[1] == 0
        mask = data.nt .+ 1
    else
        mask = data.nt
    end

    sig = data.tt_err
    res = (data.tt .- tti[mask]) ./ sig
    nll = dot(res,res)

    if grad
        dnll = zeros(T,length(θ))
        dtdel = tt.dtdelements[iplanet,mask,:,2:end]
        for i in 1:6
            for k in 1:N-1
                dnll[(N-1)*(i-1)+k+(N-1)] = -2.0*dot(res./sig,dtdel[:,i,k])
            end
        end
        for k in 1:N-1
            dnll[k] = -2.0*dot(res./sig,dtdel[:,7,k])
        end
        return nll,dnll
    end
    return nll
end

function dlogp(θ::Vector{T};data::TimingData{T},iplanet=2,mstar::T=1.0,h::T=1.0,t0::T=0.0,tmax::T=100.0) where T<:Real
    _,dnll = logp(θ;data=data,iplanet=iplanet,mstar=mstar,h=h,t0=t0,tmax=tmax,grad=true)
    return dnll
end

function calc_hessian(θ::Vector{T};data::TimingData{T},iplanet=2,mstar::T=0.990,h::T=0.27,t0::T=54.627020000014454,tmax::T=1700.0) where T<:Real
    N = length(θ)
    nbody = round(Int64,N/7) + 1
    elements = zeros(T,nbody,7)
    elements[1,1] = mstar
    elements[2:end,:] .= convert_elements(θ)[1]

    tt = calc_transits(elements,h,t0,tmax,grad=true)
    nt = tt.count[2]
    Jmass = reshape(tt.dtdelements[2,1:nt,end,2:end],nt,nbody-1)
    Jrest = reshape(tt.dtdelements[2,1:nt,1:6,2:end],nt,(nbody-1)*6)
    J = vcat([Jmass Jrest])
    hessian = zeros(T,N,N)
    for k in 1:N
        for l in 1:N
            hessian[k,l] = dot(J[:,k],J[:,l])
        end
    end
    return 2.0 * hessian ./ (data.tt_err.^2)
    return cov(J)
end

function calc_hessian(θ::Vector{T};data::TimingData{T},iplanet=2,mstar::T=0.990,h::T=0.27,t0::T=54.627020000014454,tmax::T=1700.0) where T<:Real
    N = length(θ)
    nbody = round(Int64,N/7) + 1
    m = nbody-1
    elements = zeros(T,nbody,7)
    elements[1,1] = mstar
    elements[2:end,:] .= convert_elements(θ)[1]

    tt = calc_transits(elements,h,t0,tmax,grad=true)
    # Mask points without data
    if data.nt[1] == 0
        mask = data.nt .+ 1
    else
        mask = data.nt
    end
    nt = tt.count[2]
    dtdel = tt.dtdelements[iplanet,mask,:,2:end]
    hessian = zeros(T,N,N)
    i = 0
    for ip in 1:m, ivary in 1:7
        i += 1
        # Mass is at the end
        if ivary == 1; iv = 7; else; iv = ivary-1; end
        j = 0
        for jp in 1:m, jvary in 1:7
            j += 1
            if jvary == 1; jv = 7; else; jv = jvary - 1; end
            hessian[i,j] = 2.0 * sum(dtdel[:,iv,ip] .* dtdel[:,jv,jp] ./ data.tt_err.^2)
        end
    end

    return hessian
    #return 2.0 * hessian ./ (data.tt_err.^2)
end

# Wrap with Optim.only_fg!()
function logp!(F,G,θ::Vector{T},data::TimingData{T},iplanet=2,mstar::T=1.0,h::T=1.0,t0::T=0.0,tmax::T=100.0) where T<:Real
    nll,dnll = logp(θ;data=data,iplanet=iplanet,mstar=mstar,h=h,t0=t0,tmax=tmax,grad=true)
    if G != nothing
        G .= dnll
    end
    if F != nothing
        return nll
    end
end

function calc_transits(elements::Matrix{T},h::T=1.0,t0::T=0.0,tmax::T=100.0;grad::Bool=true) where T<:AbstractFloat
    # Setup Simulation
    N = length(elements[:,1])
    ic = ElementsIC(t0,N,elements)
    s = State(ic)
    intr = Integrator(h,zero(T),tmax)
    tt = TransitTiming(tmax,ic)

    # Run Integrator
    intr(s,tt;grad=grad)

    # Return transit timing structure
    return tt
end

"""Bounds for optimizer"""
function get_upper_open(elements)
    jts = 9.547919e-4 # Jupiter to solar mass
    dtr = π/180 # degrees to radians
    m,P,t0,ecosϖ,esinϖ,I,Ω = unpack_elements(elements)
    N = length(m)
    ms = 2.0 .* ones(N)
    Ps = 1e4 .* ones(N)
    t0s = t0[:] .+ [0.00061,0.025,19.0]
    ecs = ones(N)
    ess = ones(N)
    Is = ones(N) .* π
    Is[1] = (π/2 + π/6)
    Ωs = ones(N) .* π
    return [ms...,Ps...,t0s...,ecs...,ess...,Is...,Ωs...]
end

function get_lower_open(elements)
    jts = 9.547919e-4 # Jupiter to solar mass
    dtr = π/180 # degrees to radians
    m,P,t0,ecosϖ,esinϖ,I,Ω = unpack_elements(elements)
    N = length(m)
    ms = ones(N) .* 1e-10
    Ps = ones(N) .* 0.5
    t0s = t0[:] .- [0.00061,0.025,19.0]
    ecs = -ones(N)
    ess = -ones(N)
    Is = zeros(N)
    Is[1] = (π/2 - π/6)
    Ωs = ones(N) .* -π
    return [ms...,Ps...,t0s...,ecs...,ess...,Is...,Ωs...]
end

function get_upper(elements)
    jts = 9.547919e-4 # Jupiter to solar mass
    dtr = π/180 # degrees to radians
    m,P,t0,ecosϖ,esinϖ,I,Ω = unpack_elements(elements)
    es = sqrt.(ecosϖ.^2 + esinϖ.^2)
    ms = m[:] .+ ([0.0036,0.0016,0.16] .* jts)
    Ps = P[:] .+ [0.00014,0.00067,14.0]
    t0s = t0[:] .+ [0.00061,0.025,19.0]
    ecs = ecosϖ[:] .+ ([0.00031,0.00095,0.03] .* es[:])
    ess = esinϖ[:] .+ ([0.0027,0.0033,0.05] .* es[:])
    Is = I[:] .+ ([0.12,0.68,1.0] .* dtr)
    Ωs = Ω[:] .+ ([0.1,0.19,0.1] .* dtr)
    return [ms...,Ps...,t0s...,ecs...,ess...,Is...,Ωs...]
end

function get_lower(elements)
    jts = 9.547919e-4 # Jupiter to solar mass
    dtr = π/180 # degrees to radians
    m,P,t0,ecosϖ,esinϖ,I,Ω = unpack_elements(elements)
    es = sqrt.(ecosϖ.^2 + esinϖ.^2)
    ms = m[:] .- ([0.0036,0.0016,0.16] .* jts)
    Ps = P[:] .- [0.00014,0.00067,14.0]
    t0s = t0[:] .- [0.00061,0.025,19.0]
    ecs = ecosϖ[:] .- ([0.00031,0.00095,0.03] .* es[:])
    ess = esinϖ[:] .- ([0.0027,0.0033,0.05] .* es[:])
    Is = I[:] .- ([0.12,0.68,1.0] .* dtr)
    Ωs = Ω[:] .- ([0.1,0.19,0.1] .* dtr)
    return [ms...,Ps...,t0s...,ecs...,ess...,Is...,Ωs...]
end

function test_fdiff(N)
    mstar = 1.0; h = 0.03
    t0 = 0.0; tmax = 100.0

    # Read elements
    elements = readdlm("elements.txt",',')[1:N,:]
    elements[:,end] .= 0.0
    θ = elements[2:end,:][:]
    pert = randn(length(θ)) .* 0.001
    θ .+= pert
    θ_save = copy(θ)

    # Make fake data
    ic = ElementsIC(t0,[N,ones(Int64,N-1)...],elements)
    s = State(ic); tt = TransitTiming(tmax,ic)
    intr = Integrator(h,0.0,tmax)
    intr(s,tt;grad=false)
    times = tt.tt[2,:]
    while times[end] == 0.0
        pop!(times)
    end

    nt = collect(1:1:tt.count[2])
    dataset = TimingData(tt.count[2],nt,times,[0.006])

    # Calculate gradient of model
    print("Calculating model gradient... ")
    dnll = dlogp(θ;data=dataset,iplanet=2,mstar=mstar,h=h,t0=t0,tmax=tmax)
    println("Done.")

    # Now do finite difference
    print("Calculating finite difference... ")
    newdiff = grad(central_fdm(5,1),x->logp(x,iplanet=2,data=dataset,mstar=mstar,h=h,t0=t0,tmax=tmax,grad=false),θ)[1]
    println(" done.")
    return dnll, newdiff
end

function test_fdiff(N)
    mstar = 0.990; h = 0.27
    t0 = 54.627020000014454; tmax = 1700.0

    # Read elements
    elements = readdlm("../../../KOI142/kepler88_elements.txt",',')[1:N,:]
    θ = elements[2:end,:][:]
    pert = randn(length(θ)) .* 0.001
    θ .+= pert

    # Read data
    dataset = TimingData("../../../KOI142/kep88.txt")

    # Calculate gradient of model
    print("Calculating model gradient... ")
    dnll = dlogp(θ;data=dataset,iplanet=2,mstar=mstar,h=h,t0=t0,tmax=tmax)
    println("Done.")

    # Now do finite difference
    print("Calculating finite difference... ")
    newdiff = grad(central_fdm(5,1),x->logp(x,iplanet=2,data=dataset,mstar=mstar,h=h,t0=t0,tmax=tmax,grad=false),θ)[1]
    println(" done.")
    return dnll, newdiff
end

function do_optimize(elements=nothing)
    N = 4
    mstar = 0.990; h = 0.27
    t0 = 54.627020000014454; tmax = 1700.0

    if elements == nothing
    # Read elements
    elements = readdlm("../../../KOI142/kepler88_elements.txt",',')[1:N,:]
    end
    elements = elements[1:N,:]
    θ_init = elements[2:end,:][:]

    # Datasets 
    dataset = TimingData("../../../KOI142/kep88.txt")

    # Likelihood function with gradients (wrapper)
    loglike!(F,G,θ) = logp!(F,G,θ,dataset,2,mstar,h,t0,tmax)

    # Bounds
    lower = get_lower_open(elements[2:end,:])
    upper = get_upper_open(elements[2:end,:])

    # Optimizer
    opt = LBFGS()

    return Optim.optimize(Optim.only_fg!(loglike!),lower,upper,θ_init,Fminbox(opt),
        Optim.Options(show_trace=true))
end

function do_hmc(elements=nothing)
    N = 4
    mstar = 0.990; h = 0.27
    t0 = 54.627020000014454; tmax = 1700.0

    if elements === nothing
    # Read elements
    elements = readdlm("../../../KOI142/kepler88_elements.txt",',')[1:N,:]
    end
    elements = elements[1:N,:]
    θ_init = elements[2:end,:][:]

    # Datasets 
    dataset = TimingData("../../../KOI142/kep88.txt")

    # Loglikelihood wrapper
    loglike(θ) = logp(θ,data=dataset,iplanet=2,mstar=mstar,h=h,t0=t0,tmax=tmax,grad=false)
    grad_loglike(θ) = logp(θ,data=dataset,iplanet=2,mstar=mstar,h=h,t0=t0,tmax=tmax,grad=true)

    # Hessian from covariance matrix of Jacobian of optim
    M = calc_hessian(θ_init,data=dataset,iplanet=2,mstar=mstar,h=h,t0=t0,tmax=tmax)

    # Advanced HMC Setup
    n_samples, n_adapts = 200,100
    metric = DenseEuclideanMetric(M)
    #metric = DenseEuclideanMetric(length(θ_init))
    H = Hamiltonian(metric, loglike, grad_loglike)
    #init_ϵ = find_good_stepsize(H, θ_init)
    init_ϵ = 0.2
    integrator = Leapfrog(init_ϵ)
    proposal = NUTS{MultinomialTS, ClassicNoUTurn}(integrator)
    #proposal = StaticTrajectory(integrator, 3)
    adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.65, integrator))
    samples, stats = sample(H, proposal, θ_init, n_samples, 
    adaptor, n_adapts, 
    progress=true)
end