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
        return new{T,Vector{T},Matrix{T}}([zeros(T,sevn,sevn) for _ in 1:3]...,
                                        zeros(T,14,14),
                                        [zeros(14,sevn) for _ in 1:3]...,
                                        zeros(T,14),
                                        [zeros(T,sevn) for _ in 1:2]...)
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
        return new{T,Vector{T},Matrix{T}}([zeros(T,sevn,sevn) for _ in 1:2]...,
                                        zeros(T,14,14),
                                        [zeros(T,sevn) for _ in 1:2]...,
                                        zeros(T,14),
                                        [zeros(T,14) for _ in 1:2]...)
    end
end

