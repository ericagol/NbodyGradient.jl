abstract type PreAllocArrays end
abstract type AbstractDerivatives{T} <: PreAllocArrays end

struct Jacobian{T<:AbstractFloat} <: AbstractDerivatives{T}

    jac_phi::Matrix{T}
    jac_kick::Matrix{T}
    jac_copy::Matrix{T}
    jac_ij::Matrix{T}
    jac_tmp1::Matrix{T}
    jac_tmp2::Matrix{T}
    jac_err1::Matrix{T}
    dqdt_ij::Vector{T}
    dqdt_phi::Vector{T}
    dqdt_kick::Vector{T}

    function Jacobian(::Type{T},n::Integer) where T<:AbstractFloat
        sevn::Int64 = 7*n
        jac_phi = zeros(T,sevn,sevn)
        jac_kick = zeros(T,sevn,sevn)
        jac_copy = zeros(T,sevn,sevn)
        jac_ij = zeros(T,14,14)
        jac_tmp1 = zeros(T,14,sevn)
        jac_tmp2 = zeros(T,14,sevn)
        jac_err1 = zeros(T,14,sevn)
        dqdt_ij = zeros(T,14)
        dqdt_phi = zeros(T,sevn)
        dqdt_kick = zeros(T,sevn)

        return new{T}(jac_phi,jac_kick,jac_copy,jac_ij,jac_tmp1,jac_tmp2,jac_err1,
                    dqdt_ij,dqdt_phi,dqdt_kick)
    end
end

struct dTime{T<:AbstractFloat} <: AbstractDerivatives{T}

    jac_phi::Matrix{T}
    jac_kick::Matrix{T}
    jac_ij::Matrix{T}
    dqdt_phi::Vector{T}
    dqdt_kick::Vector{T}
    dqdt_ij::Vector{T}
    dqdt_tmp1::Vector{T}
    dqdt_tmp2::Vector{T}
    jac_kepler::Matrix{T}
    jac_mass::Vector{T}

    function dTime(::Type{T},n::Integer) where T<:AbstractFloat
        sevn::Int64 = 7*n
        jac_phi = zeros(T,sevn,sevn)
        jac_kick = zeros(T,sevn,sevn)
        jac_ij = zeros(T,14,14)
        dqdt_phi = zeros(T,sevn)
        dqdt_kick = zeros(T,sevn)
        dqdt_ij = zeros(T,14)
        dqdt_tmp1 = zeros(T,14)
        dqdt_tmp2 = zeros(T,14)
        jac_kepler = zeros(T,6,8)
        jac_mass = zeros(T,6)
        return new{T}(jac_phi,jac_kick,jac_ij,dqdt_phi,dqdt_kick,dqdt_ij,
            dqdt_tmp1,dqdt_tmp2,jac_kepler,jac_mass)
    end
end

struct Derivatives{T<:AbstractFloat} <: AbstractDerivatives{T}
    jac_phi::Matrix{T}
    jac_kick::Matrix{T}
    jac_copy::Matrix{T}
    jac_ij::Matrix{T}
    jac_tmp1::Matrix{T}
    jac_tmp2::Matrix{T}
    jac_err1::Matrix{T}
    dqdt_phi::Vector{T}
    dqdt_kick::Vector{T}
    dqdt_ij::Vector{T}
    dqdt_tmp1::Vector{T}
    dqdt_tmp2::Vector{T}
    jac_kepler::Matrix{T}
    jac_mass::Vector{T}
    dadq::Array{T,4}
    dotdadq::Matrix{T}
    tmp7n::Vector{T}
    tmp14::Vector{T}
    ctime::Vector{T}

    function Derivatives(::Type{T},n::Integer) where T<:AbstractFloat
        sevn::Int64 = 7*n
        jac_phi = zeros(T,sevn,sevn)
        jac_kick = zeros(T,sevn,sevn)
        jac_copy = zeros(T,sevn,sevn)
        jac_ij = zeros(T,14,14)
        jac_tmp1 = zeros(T,14,sevn)
        jac_tmp2 = zeros(T,14,sevn)
        jac_err1 = zeros(T,14,sevn)
        dqdt_phi = zeros(T,sevn)
        dqdt_kick = zeros(T,sevn)
        dqdt_ij = zeros(T,14)
        dqdt_tmp1 = zeros(T,14)
        dqdt_tmp2 = zeros(T,14)
        jac_kepler = zeros(T,6,8)
        jac_mass = zeros(T,6)
        dadq = zeros(T,3,n,4,n)
        dotdadq = zeros(T,4,n)

        tmp7n = zeros(T,sevn)
        tmp14 = zeros(T,14)
        ctime = zeros(T,1)

        return new{T}(jac_phi,jac_kick,jac_copy,jac_ij,jac_tmp1,jac_tmp2,jac_err1,
                      dqdt_phi,dqdt_kick,dqdt_ij,dqdt_tmp1,dqdt_tmp2,jac_kepler,jac_mass,
                      dadq,dotdadq,tmp7n,tmp14,ctime)
    end
end

function Base.iterate(d::AbstractDerivatives{T},state=1) where T<:AbstractFloat
    fields = fieldnames(typeof(d))
    if state > length(fields)
        return nothing
    end
    return (getfield(d,fields[state]), state+1)
end

function zero_out!(d::AbstractDerivatives{T}) where T<:AbstractFloat
    for i::Array{T} in d
        i .= zero(T)
    end
end

function zero_out!(d::Derivatives{T}) where T<:AbstractFloat
    d.jac_phi .= zero(T)
    d.jac_kick .= zero(T)
    d.jac_copy .= zero(T)
    d.jac_ij .= zero(T)
    d.jac_tmp1 .= zero(T)
    d.jac_tmp2 .= zero(T)
    d.jac_err1 .= zero(T)
    d.dqdt_phi .= zero(T)
    d.dqdt_kick .= zero(T)
    d.dqdt_ij .= zero(T)
    d.dqdt_tmp1 .= zero(T)
    d.dqdt_tmp2 .= zero(T)
    d.jac_kepler .= zero(T)
    d.jac_mass .= zero(T)
    d.dadq .= zero(T)
    d.dotdadq .= zero(T)
    d.tmp7n .= zero(T)
    d.tmp14 .= zero(T)
end
