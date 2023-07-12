# Make sure that each IC specification method yields the same orbital elements
# or Cartesian coordinates.
using DelimitedFiles

function compare_ics(ic1::T, ic2::T) where T<:InitialConditions
    # Compare the fields of the structs (https://stackoverflow.com/questions/62336686/struct-equality-with-arrays)
    fns = fieldnames(T)
    f1 = getfield.(Ref(ic1), fns)
    f2 = getfield.(Ref(ic2), fns)
    return f1 == f2 
end

function get_elements_ic_array(t0, H, N)
    fname = "elements.txt"
    elements = readdlm(fname, ',', comments=true)
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
    elements = readdlm(fname, ',', comments=true)
    elems = Elements{Float64}[]
    for i in 1:N
        push!(elems, Elements(elements[i,:]..., zeros(4)...))
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
    coords = readdlm(fname, ',', comments=true)
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
    for ic in ics
        ex = :(@test compare_ics($(ics[1]), $ic))
        push!(tests, ex)
    end
    return Expr(:block, setup..., tests...)
end

function run_elements_H_tests(t0, n)
    H_vec = [n, ones(Int64, n - 1)...]
    H_mat = NbodyGradient.hierarchy(H_vec)

    ic_n = get_elements_ic_file(t0, n, n)
    ic_vec = get_elements_ic_file(t0, H_vec, n)
    ic_mat = get_elements_ic_file(t0, H_mat, n)

    @test compare_ics(ic_n, ic_vec)
    @test compare_ics(ic_n, ic_mat)
    return
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

function test_default_trappist()
    # Check that default ics match those produced by the test file
    fname = "elements.txt"
    t0 = 0.0
    for n in 2:8
        ic_file = ElementsIC(t0, n, fname)
        ic_default = get_default_ICs("trappist-1", t0, n)
        @test compare_ics(ic_file, ic_default)
    end

    # Check that each input string works correctly
    sysnames = NbodyGradient._available_systems()[1]
    ic_true = ElementsIC(t0, 7, fname)
    for sn in sysnames
        @test compare_ics(ic_true, get_default_ICs(sn, t0, 7))
    end
end

@testset "Initial Conditions" begin
    @testset "Elements" begin
        N = 8
        t0 = 7257.93115525 - 7300.0

        for n in 2:N
            H_vec = [n,ones(Int64, n - 1)...]
            H_mat = NbodyGradient.hierarchy(H_vec)

            # Test each specification method with static hierarchy
            run_elements_tests(t0, H_mat, n)
            run_elements_tests(t0, H_vec, n) # Test hierarchy vector specification
            run_elements_tests(t0, n, n)     # Test number-of-bodies specification

            # Now test each hierarchy method with file method
            run_elements_H_tests(t0, n)
        end
    end

    @testset "Cartesian" begin
        N = 8
        t0 = 7257.93115525 - 7300.0

        for n in 2:N
            run_cartesian_tests(t0, n)
        end
    end

    @testset "Defaults" begin
        test_default_trappist()
    end

    @testset "Deprecated" begin
        el = Elements(m=1.0, P=1.0)
        @test_logs (:warn, "ecosϖ (\\varpi) will be removed, use ecosω (\\omega).") Base.getproperty(el, :ecosϖ)
        @test_logs (:warn, "esinϖ (\\varpi) will be removed, use esinω (\\omega).") Base.getproperty(el, :esinϖ)
        @test_logs (:warn, "ϖ (\\varpi) will be removed, use ω (\\omega).") Base.getproperty(el, :ϖ)
    end
end