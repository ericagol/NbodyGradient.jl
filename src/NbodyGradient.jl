"""
    NbodyGradient

An N-body itegrator that computes derivatives with respect to initial conditions for TTVs, RV, Photodynamics, and more.
"""

module NbodyGradient

using LinearAlgebra
using DelimitedFiles

# Constants used by most functions
# Need to clean this up
const NDIM = 3
const YEAR = 365.242
const GNEWT = 39.4845/(YEAR*YEAR)
const third = 1.0/3.0
const alpha0 = 0.0

#include("PreAllocArrays.jl")
include("setup_hierarchy.jl")
include("utils.jl")
include("grad/integrator.jl")
include("nograd/integrator_no_grad.jl")
include("init_nbody.jl")
include("nograd/ttv.jl")
include("grad/ttv.jl")

# Included for tests
include("dh17/dh17.jl")

function ttv_elements!(elems::ElementsIC{T},t0::T,h::T,tmax::T,tt::Array{T,2},count::Array{Int64,1},rstar::T) where T <: AbstractFloat
    elements = elems.elements
    H = elems.H
    return ttv_elements!(H,t0,h,tmax,elements,tt,count,0.0,0,0,rstar)
end

# Output Methods
export ttv_elements!, ttvbv_elements!

# Integrator methods
export ah18!, dh17!

# Types
export Elements, ElementsIC, CartesianIC

end
