abstract type InitialConditions end

struct Elements{T<:AbstractFloat} <: InitialConditions 
    m::T
    P::T
    t0::T
    ecosϖ::T
    esinϖ::T
    i::T
    Ω::T

    function Elements(;m::T=0.0,P::T=0.0,t0::T=0.0,e::T=0.0,ϖ::T=0.0,i::T=0.0,Ω::T=0.0) where T<:AbstractFloat
        esinϖ,ecosϖ = e .* sincos(ϖ)
        return new{T}(m,P,t0,ecosϖ,esinϖ,i,Ω)
    end
end

mutable struct ElementsIC{T<:AbstractFloat} <: InitialConditions
    elements::Array{T,2}
    H::Array{Int64,1}
    ϵ::Array{T,2}
    amat::Array{T,2}
    NDIM::Int64
    nbody::Int64
    m::Array{T,2}
    t0::T
    der::Bool

    function ElementsIC(elems::Union{String,Array{T,2}},H::Union{Array{Int64,1},Int64},
                        t0::T;NDIM::Int64 = 3,der::Bool=true) where T <: AbstractFloat
        # Check if only number of bodies was passed. Assumes fully nested.
        if typeof(H) == Int64
            H::Vector{Int64} = [H,ones(H-1)...]
        end
        ϵ = convert(Array{T},hierarchy(H))
        if typeof(elems) == String
            elements = convert(Array{T},readdlm(elems,',',comments=true))
        else
            elements = elems
        end
        nbody = H[1]
        m = reshape(vcat(elements[:,1])[1:nbody],nbody,1) # There's probably a better way to do this...
        amat = amatrix(ϵ,m)
        return new{T}(elements,H,ϵ,amat,NDIM,nbody,m,t0,der);
    end
end

function ElementsIC(t0::T,elems::Elements{T}...;H::Vector{Int64}) where T <: AbstractFloat
    
    # Not sure if this is a good spot for this...
    function parse_system(elems::Elements{T}...) where T <: AbstractFloat
        bodies = Dict{Symbol,Elements}()
        for i in 1:length(elems)
            key = Meta.parse("b$i")
            bodies[key] = elems[i]
        end
        key = sort(collect(keys(bodies)))
        field = fieldnames(Elements)
        elements = zeros(length(bodies),7)
        for i in 1:length(bodies), elm in enumerate(fieldnames(Elements))
            elements[i,elm[1]] = getfield(bodies[key[i]],elm[2])
        end
        return elements
    end     
            
    elements = parse_system(elems...)
    return ElementsIC(elements,H,t0)
end

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

# Include ics source files
const ics = ["kepler","kepler_init","setup_hierarchy","init_nbody"]
for i in ics; include("$(i).jl"); end
