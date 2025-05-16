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
export Elements, ElementsIC, CartesianIC, InitialConditions
export State, dState
export Integrator
export Jacobian, dTime
export CartesianOutput, ElementsOutput
export TransitTiming, TransitParameters, TransitSnapshot
export RadialVelocities

# Integrator methods
export ahl21!, dh17!

# Utility functions
export available_systems, get_default_ICs

# Source code
include("PreAllocArrays.jl")
include("ics/InitialConditions.jl")
include("integrator/Integrator.jl")
include("utils.jl")
include("outputs/Outputs.jl")
include("transits/Transits.jl")
include("rvs/RadialVelocities.jl")

# To be cleaned up
# include("integrator/ah18/ah18_old.jl")

end
