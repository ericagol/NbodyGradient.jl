 

include("loglinspace.jl")
include("../src/ttv.jl")
maxabs(x) = maximum(abs.(x))
if VERSION >= v"0.7"
  using Test
  using LinearAlgebra
  using Statistics
  using DelimitedFiles
else
  using Base.Test
end

include("test_kepler_init.jl")
include("test_init_nbody.jl")
include("test_elliptic_derivative.jl")
include("test_keplerij.jl")
include("test_kepler_drift.jl")
include("test_kepler_driftij.jl")
include("test_kepler_drift_gamma.jl")
include("test_kepler_driftij_gamma.jl")
include("test_kickfast.jl")
include("test_phisalpha.jl")
include("test_dh17.jl")
include("test_ah18.jl")
include("test_findtransit3.jl")
include("test_ttv_cartesian.jl")
include("test_ttvbv_cartesian.jl")
include("test_ttv_elements.jl")
include("test_ttvbv_elements.jl")
