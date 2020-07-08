using NbodyGradient
import NbodyGradient: kepler_init, init_nbody

maxabs(x) = maximum(abs.(x))

# For backwards compatability
include("loglinspace.jl")
if VERSION >= v"0.7"
  using Test
  using LinearAlgebra
  using Statistics
  using DelimitedFiles
else
  using Base.Test
end

@testset "NbodyGradient" begin
    print("Initial Conditions... ")
    @testset "Initial Conditions" begin
        include("test_kepler_init.jl")
        include("test_init_nbody.jl")
    end;
    println("Finished.")

    print("Integrator... ")
    @testset "Integrator" begin
        #include("test_elliptic_derivative.jl") #Deprecated functions
        #include("test_keplerij.jl") # Deprecated functions
        #include("test_kepler_drift.jl") # Deprecated functions (??)
        #include("test_kepler_driftij.jl") # Deprecated functions
        #include("test_kepler_drift_gamma.jl") # Only for auto-diff version, deprecated
        include("test_kepler_driftij_gamma.jl")
        include("test_kickfast.jl")
        include("test_phisalpha.jl")
        # Need to fix dh17.
        #include("test_dh17.jl")
        include("test_ah18.jl")
    end;
    println("Finished.")

    print("TTVs... ")
    @testset "TTVs" begin
        include("test_findtransit3.jl")
        include("test_ttv_cartesian.jl")
        include("test_ttv_elements.jl")
        # Add in later
        #include("test_ttvbv_cartesian.jl")
        #include("test_ttvbv_elements.jl")
    end;
    println("Finished.")
end
