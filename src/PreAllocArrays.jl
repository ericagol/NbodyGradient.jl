abstract type PreAllocArrays end
abstract type AbstractDerivatives{T} <: PreAllocArrays end

mutable struct Jacobian{T<:AbstractFloat,V<:Vector{T},M<:Matrix{T}} <: AbstractDerivatives{T}
   
    jac_phi::M 
    jac_kick::M
    jac_copy::M
    jac_ij::M
    jac_tmp1::M 
    jac_tmp2::M
    jac_err1::M
    dqdt_ij::V
    dqdt_phi::V
    dqdt_kick::V

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

        return new{T,Vector{T},Matrix{T}}(jac_phi,jac_kick,jac_copy,jac_ij,
            jac_tmp1,jac_tmp2,jac_err1,dqdt_ij,dqdt_phi,dqdt_kick)
    end
end

mutable struct dTime{T<:AbstractFloat,V<:Vector{T},M<:Matrix{T}} <: AbstractDerivatives{T}
   
    jac_phi::M 
    jac_kick::M
    jac_ij::M
    dqdt_phi::V
    dqdt_kick::V
    dqdt_ij::V
    dqdt_tmp1::V
    dqdt_tmp2::V

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
        return new{T,Vector{T},Matrix{T}}(jac_phi,jac_kick,jac_ij,dqdt_phi,dqdt_kick,
            dqdt_ij,dqdt_tmp1,dqdt_tmp2)
    end
end