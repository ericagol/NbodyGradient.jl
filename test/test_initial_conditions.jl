# Make sure that each IC specification method yields the same orbital elements
# or Cartesian coordinates.
using DelimitedFiles

function get_elements_ic_array(t0, H, N)
    fname = "elements.txt"
    elements = readdlm(fname, ',', comments=true)[1:N,:]
    ic = ElementsIC(t0, H, elements)
    return ic
end

function get_elements_ic_file(t0, H, N)
    fname = "elements.txt"
    ic = ElementsIC(t0, H, fname)
    return ic
end

function get_elements_ic_elems(t0, H, N)
    fname = "elements.txt"
    elements = readdlm(fname, ',', comments=true)[1:N,:]
    elems = Elements{Float64}[]
    for i in 1:N
        push!(elems, Elements(elements[i,:]...))
    end
    ic = ElementsIC(t0, H, elems)
    return ic
end

function get_cartesian_ic_file(t0, N)
    fname = "coordinates.txt"
    ic = CartesianIC(t0, N, fname)
    return ic
end

function get_cartesian_ic_array(t0, N)
    fname = "coordinates.txt"
    coords = readdlm(fname, ',', comments=true)[1:N,:]
    ic = CartesianIC(t0, N, coords)
    return ic
end

# Run tests for given H type (int, vector, matrix)
@generated function run_elements_tests(t0, H, N)
    inputs = ["file", "array", "elems"]
    funcs = [Symbol("get_elements_ic_$(i)") for i in inputs]
    ics = [Symbol("ic_$(i)") for i in inputs]

    # Get ICs for each method
    setup = Vector{Expr}()
    for (ic, f) in zip(ics, funcs)
        ex = :($ic = $f(t0, H, N))
        push!(setup, ex)
    end

    # Compare outputs of each method
    tests = Vector{Expr}()
    fields = [:elements, :amat]
    for ic1 in ics, ic2 in ics, f in fields
        ex = :(@test $ic1.$f == $ic2.$f)
        push!(tests, ex)
    end
    return Expr(:block, setup..., tests...)
end

# Run tests for given H type (int, vector)
@generated function run_cartesian_tests(t0, H)
    inputs = ["file", "array"]
    funcs = [Symbol("get_cartesian_ic_$(i)") for i in inputs]
    ics = [Symbol("ic_$(i)") for i in inputs]

    # Get ICs for each method
    setup = Vector{Expr}()
    for (ic, f) in zip(ics, funcs)
        ex = :($ic = $f(t0, H))
        push!(setup, ex)
    end

    # Compare the outputs of each method
    tests = Vector{Expr}()
    fields = [:x, :v, :m]
    for ic1 in ics, ic2 in ics, f in fields
        ex = :(@test $ic1.$f == $ic2.$f)
        push!(tests, ex)
    end
    return Expr(:block, setup..., tests...)
end

@testset "Initial Conditions" begin
    @testset "Elements" begin
        N = 8
        t0 = 7257.93115525 - 7300.0
        H_vec = [N,ones(Int64, N - 1)...]
        H_mat = NbodyGradient.hierarchy(H_vec)

        run_elements_tests(t0, H_mat, N)
        run_elements_tests(t0, H_vec, N) # Test hierarchy vector specification
        run_elements_tests(t0, N, N)     # Test number-of-bodies specification
    end

    @testset "Cartesian" begin
        N = 8
        t0 = 7257.93115525 - 7300.0
        run_cartesian_tests(t0, N)
    end
end