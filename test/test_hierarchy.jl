# Check whether the test is being run as standalone
if ~@isdefined Test
    using Test
end

include("../src/setup_hierarchy.jl")


@testset "hierarchy matrix" begin
    # Makes sure the hierarchy matrix generator is working as expected for a select set of orbital systems.
    # Fully nested Keplerians
    ϵ_nested::Matrix{Float64} = [-1  1  0  0;
                                 -1 -1  1  0;
                                 -1 -1 -1  1;
                                 -1 -1 -1 -1;]
    @test hierarchy([4,1,1,1]) == ϵ_nested
    
    # Double Binary
    ϵ_double::Matrix{Float64} = [-1  1  0  0;
                                  0  0 -1  1;
                                 -1 -1  1  1;
                                 -1 -1 -1 -1;]
    @test hierarchy([4,2,1]) == ϵ_double
    
    # Double binary + top level body 
    ϵ_asym::Matrix{Float64} = [-1  1  0  0  0;
                                0  0 -1  1  0;
                               -1 -1  1  1  0;
                               -1 -1 -1 -1  1;
                               -1 -1 -1 -1 -1;]
    @test hierarchy([5,2,1,1]) == ϵ_asym
    
    # Test of specific ambiguous initialization case of a 'symmetric fully-nested' (top) and the double binary + tertiary binary (bottom)
    ϵ_sym_full::Matrix{Float64} = [-1  1  0  0  0  0;
                                    0  0  0 -1  1  0;
                                   -1 -1  1  0  0  0;
                                    0  0  0 -1 -1  1;
                                   -1 -1 -1  1  1  1;
                                   -1 -1 -1 -1 -1 -1;]
    
    ϵ_asym_multi::Matrix{Float64}=[-1  1  0  0  0  0;
                                    0  0 -1  1  0  0;
                                   -1 -1  1  1  0  0;
                                    0  0  0  0 -1  1;
                                   -1 -1 -1 -1  1  1;
                                   -1 -1 -1 -1 -1 -1;]
    h_sym_full,h_asym_multi = hierarchy.([[6,2,2,1,2],[6,2,2,1]])
    
    @test ϵ_sym_full == h_sym_full
    @test ϵ_asym_multi == h_asym_multi
end