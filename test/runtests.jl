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
        include("test_initial_conditions.jl")
    end;
    println("Finished.")

    print("Integrator... ")
    @testset "Integrator" begin
        include("test_kickfast.jl")
        include("test_kepler_driftij_gamma.jl")
        include("test_phisalpha.jl")
        include("test_integrator.jl")
    end;
    println("Finished.")

    print("Outputs... ")
    @testset "Outputs" begin
        include("test_cartesian_to_elements.jl")
    end
    println("Finished.")

    print("TTVs... ")
    @testset "TTVs" begin
        include("test_findtransit.jl")
        include("test_transit_timing.jl")
        include("test_transit_parameters.jl")
        #include("test_transit_series.jl") ## Not working...
    end;
    println("Finished.")
end
