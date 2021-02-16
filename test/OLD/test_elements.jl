@testset "Elements" begin

    # Get known 'good' elements
    fname = "elements.txt"
    t0 = 7257.93115525-7300.0
    H = [3,1,1]
    init0 = ElementsIC(fname,H,t0)

    # Fill new methods with pre-calculated elements
    # See if we get back the proper array
    elems_arr = readdlm(fname,',',comments=true)

    # Convert ecosϖ/esinϖ to ϖ and e
    elems_a = copy(elems_arr[1,:])
    elems_b = copy(elems_arr[2,:])
    elems_c = copy(elems_arr[3,:])

    function convert_back!(elems::Vector{<:Real})
        e = sqrt(elems[4]^2 + elems[5]^2)
        ϖ = atan(elems[5],elems[4]) 
        elems .= [elems[1:3];e;ϖ;elems[6:end]]
        return
    end

    convert_back!.([elems_b,elems_c])
    a = Elements(elems_a...)
    b = Elements(elems_b...)
    c = Elements(elems_c...)
    init_test = ElementsIC(t0,a,b,c,H=H)
     
    @test isapprox(init0.elements[1:3,:],init_test.elements);

end;
