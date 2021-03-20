# Test all IC specification methods
using DelimitedFiles

N = 3
t0 = 7257.93115525 - 7300.0
H = [N,ones(Int64, N - 1)...]

function get_elements_array_ic()
    fname = "elements.txt"
    elements = readdlm(fname, ',', comments=true)[1:N,:]
    ic = ElementsIC(t0, H, elements)
    return ic
end

function get_cartesian_array_ic()
    fname = "coordinates.txt"
    coords = readdlm(fname, ',', comments=true)[1:N,:]
    ic = CartesianIC(t0, H[1], coords)
    return ic
end

function get_elements_file_ic()
    fname = "elements.txt"
    ic = ElementsIC(t0, H, fname)
    return ic
end

function get_cartesian_file_ic()
    fname = "coordinates.txt"
    ic = CartesianIC(t0, H[1], fname)
    return ic
end

function get_elements_ic()
    fname = "elements.txt"
    elements = readdlm(fname, ',', comments=true)[1:N,:]
    elems = Elements{Float64}[]
    for i in 1:N
        push!(elems, Elements(elements[i,:]...))
    end
    ic = ElementsIC(t0, H, elems)
end

@testset "Initial Conditions" begin
    @testset "Elements" begin
        ic_file = get_elements_file_ic()
        ic_array = get_elements_array_ic()
        ic_elems = get_elements_ic()

        @test ic_file.elements == ic_array.elements
        @test ic_elems.elements == ic_array.elements

        @test ic_file.amat == ic_array.amat
        @test ic_elems.amat == ic_array.amat
    end

    @testset "Cartesian" begin
        ic_file = get_cartesian_file_ic()
        ic_array = get_cartesian_array_ic()

        @test ic_file.x == ic_array.x
        @test ic_file.v == ic_array.v
        @test ic_file.m == ic_array.m
    end
end