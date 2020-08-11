import NbodyGradient: get_orbital_elements

function Base.isapprox(a::Elements,b::Elements;tol=1e-8)
    fields = setdiff(fieldnames(Elements),[:a,:e,:ϖ])
    for i in fields
        af = getfield(a,i)
        bf = getfield(b,i)
        if abs(af - bf) > tol
            return false
        end
    end
    return true
end


@testset "Cartesian to Elements" begin

    # Get known Elements
    fname = "elements.txt"
    t0 = 7257.93115525-7300.0
    H = [3,1,1]
    ic = ElementsIC(fname,H,t0)

    a = Elements(ic.elements[1,:]...)
    b = Elements(ic.elements[2,:]...)
    c = Elements(ic.elements[3,:]...)
    d = Elements(ic.elements[4,:]...)
    system = [a,b,c,d]

    # Convert to Cartesian coordinates
    s = State(ic)

    # Convert back to elements
    elems = get_orbital_elements(s,ic)

    for i in eachindex(system)
        @test isapprox(elems[1],system[1])
    end
    
end
