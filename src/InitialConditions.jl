include("setup_hierarchy.jl")
abstract type InitialConditions end

mutable struct ElementsIC{T<:AbstractFloat} <: InitialConditions
    elements::Array{T,2}
    ϵ::Array{T,2}
    amat::Array{T,2}
    NDIM::Int64
    nbody::Int64
    m::Array{T,2}
    t0::T
    der::Bool

    function ElementsIC(elems::Union{String,Array{T,2}},H::Array{Int64,1},t0::T;
            NDIM::Int64 = 3,der::Bool=true) where T <: AbstractFloat
        ϵ = convert(Array{T},hierarchy(H))
        if typeof(elems) == String
            elements = convert(Array{T},readdlm(elems,',',comments=true))
        else
            elements = elems
        end
        nbody = H[1]
        m = reshape(vcat(elements[:,1])[1:nbody],nbody,1)
        amat = amatrix(ϵ,m)
        return new{T}(elements,ϵ,amat,NDIM,nbody,m,t0,der);
    end
end

#function ElementsIC(sys::System,H::Array{Int64,1},t0::T;
#        NDIM::Int64 = 3,der::Bool=true) where T <: AbstractFloat
#    elements::Array{Float64,2} = make_elements_array(sys)
#    return ElementsIC(elements,H,t0;NDIM=3,der=true)
#end     

mutable struct CartesianIC{T<:AbstractFloat} <: InitialConditions
    x::Array{T,2}
    v::Array{T,2}
    jac_init::Array{T,2}
    m::Array{T,2}
    t0::T
    nbody::Int64
    NDIM::Int64
    
    function CartesianIC(filename::String,x,v,jac_init,t0;
            NDIM::Int64 = 3) where T <: AbstractFloat
        m = convert(Array{T},readdlm(filename,',',comments=true))
        nbody = length(m)
        return new{T}(x,v,jac_init,m,t0,nbody,NDIM)
    end
end
