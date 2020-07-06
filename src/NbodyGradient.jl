"""
    NbodyGradient

An N-body itegrator that computes derivatives with respect to initial conditions for TTVs, RV, Photodynamics, and more.
"""

module NbodyGradient

include("PreAllocArrays.jl")
include("integrator.jl")
include("InitialConditions.jl")
include("ttv.jl")

function ttv_elements!(elems::ElementsIC{T},t0::T,h::T,tmax::T,tt::Array{T,2},count::Array{Int64,1},rstar::T) where T <: AbstractFloat
    elements = elems.elements
    H = elems.H
    return ttv_elements!(H,t0,h,tmax,elements,tt,count,0.0,0,0,rstar)
end

# Output Methods
export ttv_elements!, ttvbv_elements!

# Integrator methods
export ah18!

# Types
export Elements, ElementsIC

end
