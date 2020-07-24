"""
    NbodyGradient

An N-body itegrator that computes derivatives with respect to initial conditions for TTVs, RV, Photodynamics, and more.
"""
module NbodyGradient

using LinearAlgebra, DelimitedFiles
using FileIO, JLD2

# Constants used by most functions
# Need to clean this up
const NDIM = 3
const YEAR = 365.242
const GNEWT = 39.4845/(YEAR*YEAR)
const third = 1.0/3.0
const alpha0 = 0.0

# Types
export Elements, ElementsIC, CartesianIC
export State
export Integrator
export Jacobian, dTime
export CartesianOutput, ElementsOutput

# Output Methods
export ttv_elements!#, ttvbv_elements!

# Integrator methods
export ah18!, dh17!

# Source code
include("utils.jl")
include("PreAllocArrays.jl")
include("ics/InitialConditions.jl")
include("integrator/Integrator.jl")
include("ttvs/TTVs.jl")

# For testing
include("integrator/ah18_new.jl")
include("integrator/output.jl")

# wrapper for testing new ics.
@inline function ttv_elements!(el::ElementsIC{T},t0::T,h::T,tmax::T,tt::Array{T,2},count::Array{Int64,1},rstar::T) where T <: Real
    return ttv_elements!(el.H,t0,h,tmax,el.elements,tt,count,0.0,0,0,rstar)
end

end
