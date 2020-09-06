using LinearAlgebra, Distributions, MCMCChains, Statistics, MCMCDiagnostics, ProgressMeter
using StatsPlots

Ham(x,p,M_inv,U) = 0.5 * (p' * M_inv * p) + U(x) 

function update_M(chains::Chains)
    N = length(chains[:,1,1].data)
    D = length(chains[1,:,1].value.data)
    q = reshape(chains[:,:,1].value.data, N, D)
    μ = mean.([q[:,i] for i in 1:D])
    M_inv = zeros(Float64,D,D)
    for i in 1:D
        for j in 1:D
            #M_inv[i,j] = 
        end
    end
    return q, μ
end

function update_M(chains::Matrix)
    D,N = size(chains)
    q = copy(chains)
    μ = mean.([q[i,:] for i in 1:D])
    E = zeros(Float64,D,D)
    for i in 1:D
        for j in 1:D
            E[i,j] = sum((q[i,:] .- μ[i]) .* (q[j,:] .- μ[j]))
        end
    end
    return E
end

function HMC(θ,M_inv,U::Function,L0,ϵ0,n_samples=10;∇U::Function=nothing,bounds=nothing,param_names=nothing)
    # Allocate arrays
    N = length(θ)
    x0 = copy(θ)
    x = copy(θ)
    xnew = copy(θ)
    p0 = zeros(N)
    p = copy(p0)
    phalf = zeros(size(p))
    chains = zeros(length(θ),n_samples)
    E = zeros(n_samples)
    n_accepted = 0

    # For Kinetic energy distribution
    Mtmp = 0.5 * (inv(M_inv) + inv(M_inv)')
    M_U = permutedims(cholesky(Mtmp).U)

    # Perform HMC
    @showprogress 1 "Running HMC..." for iter in 1:n_samples

        # Leapfrog integration
        #x0 .= x
        p0 .= M_U * randn(N)
        # Choose epsilon
        ϵ = ϵ0 #* (0.75 + 0.25*rand())
        L = L0 #round(Int64,L0*(0.8 + 0.2*rand()))
        for i in 1:L
            phalf .= p0 .- (0.5*ϵ .* ∇U(x))
            xnew .= x .+ (ϵ .* (M_inv * phalf))
            if bounds != nothing
                if ~check_bounds(xnew,bounds)
                    println("Out of bounds.")
                    break
                end
            end
            p .= phalf .- (0.5*ϵ .* ∇U(xnew))
            x .= xnew
            println(abs(Ham(x,p,M_inv,U) - Ham(x0,p0,M_inv,U)))
        end
        p .= -p # Reverse direction

        # Metropolis-Hastings step
        H0 = Ham(x0,p0,M_inv,U)
        Hnew = Ham(x,p,M_inv,U)
        condition = exp(H0 - Hnew)
        if rand() < exp(H0 - Hnew)
            println("Accepted. ", "Hnew ", Hnew, " H0 ", H0)
            E[iter] = Hnew
            n_accepted += 1
            x0 .= x
        else
            println("Rejected. ", "Hnew ", Hnew, " H0 ", H0)
            E[iter] = H0
            x .= x0
        end
        chains[:,iter] .= x 

    end
    println("Acceptance fraction: ", n_accepted/n_samples)
    println("Energy Bayesian fraction of missing information: ", EBFMI(E))
    if param_names != nothing
        return Chains(chains',param_names)
    end
    return Chains(chains')
end


function HMC_NUTS(θ,M,U::Function,∇U::Function,L,ϵ,n_samples=10)
    # Allocate arrays
    N = length(θ)
    x0 = copy(θ)
    x = copy(θ)
    p0 = rand(MvNormal(zeros(N),M))
    p = copy(p0)
    phalf = zeros(size(p))
    chains = zeros(length(θ),n_samples)
    E = zeros(n_samples)
    M_inv = inv(M)
    n_accepted = 0

    # Perform HMC
    iter = 0
    while iter < n_samples
        # NUTS condition
        H0 = Hamiltonian(x0,p0,M_inv,U)
        Un = rand(Uniform(0.0,exp(-H0)))
        traj = Trajectories(N)
        
        # Get initial positions
        traj.pp .= rand(MvNormal(zeros(N),M))
        traj.pm .= traj.pp
        traj.xp .= x0; traj.xm .= x0

        # Do Leapfrog integration
        for i in 1:L
            phalf .= p0 .- (0.5*ϵ .* ∇U(x0))
            x .= x0 .+ (ϵ .* (M_inv * phalf))
            p .= phalf .- (0.5*ϵ .* ∇U(x))
        end
        p = -p # Reverse direction

        # NUTS step
        Hnew = Hamiltonian(x,p,M_inv,U)

        condition = exp(H0 - Hnew)
        if rand() < condition
            #println("Accepted.")
            E[iter+1] = Hnew
            n_accepted += 1
            x0 .= x
        else
            #println("Rejected")
            E[iter+1] = H0
            x .= x0
        end
        iter += 1
        chains[:,iter] .= x0 
    end
    println("Acceptance fraction: ", n_accepted/n_samples)
    println("Energy Bayesian fraction of missing information: ", EBFMI(E))
    return Chains(chains')
end


# Just use ess() from MCMCChains (?)
function get_ess(chains)
    for i in eachindex(chains[1,:,1])
        println("Effective Sample Size for $i: ", effective_sample_size(chains[:,i,1]))
    end
end

function EBFMI(E::Vector{T}) where T <: Real
    Esum = 0.0
    Emean = mean(E)
    Evar = sum((E .- Emean).^2)
    for n in 2:length(E)
        Esum += (E[n] - E[n-1])^2
    end
    return Esum/Evar
end

using ForwardDiff
# Good integration time ~0.6
test_U(x) = -logpdf(MvNormal(zeros(size(x)),ones(size(x))),x)
test_∇U = x -> ForwardDiff.gradient(test_U,x)

function tt_hmc(elements=nothing,M=nothing)
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

    # Loglikelihood wrapped
    loglike(θ) = logp(θ,data=dataset,iplanet=2,mstar=mstar,h=h,t0=t0,tmax=tmax,grad=false)
    grad_loglike(θ) = logp(θ,data=dataset,iplanet=2,mstar=mstar,h=h,t0=t0,tmax=tmax,grad=true)[2]

    if M == nothing
        # Hessian from covariance matrix of Jacobian of optim
        #M = calc_hessian(θ_init,data=dataset,iplanet=2,mstar=mstar,h=h,t0=t0,tmax=tmax)
        #M = 0.5 * (M + M')
        # Start with unit metric
        M = Matrix{Float64}(I,length(θ_init),length(θ_init))
    end

    # Bounds on parameters
    lower = get_lower_open(elements[2:N,:])
    upper = get_upper_open(elements[2:N,:])
    bounds = (lower,upper)

    # Label the chains
    param_names = vcat([["m_","P_","t0_","ecosϖ_","esinϖ_","I_","Ω_"] .* p for p in ["b","c","d"]]...)[:]

    #return M
    return HMC(θ_init,M,loglike,20,1e-8,10;∇U=grad_loglike,bounds=bounds,param_names=param_names)
end