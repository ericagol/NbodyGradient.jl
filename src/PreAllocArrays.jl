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
        return new{T}(jac_phi,jac_kick,jac_ij,dqdt_phi,dqdt_kick,dqdt_ij,dqdt_tmp1,dqdt_tmp2)
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

        return new{T}(jac_phi,jac_kick,jac_copy,jac_ij,jac_tmp1,jac_tmp2,jac_err1,
                      dqdt_phi,dqdt_kick,dqdt_ij,dqdt_tmp1,dqdt_tmp2)
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
    for i in d
        i .= zero(T)
    end
end