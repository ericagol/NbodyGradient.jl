using NbodyGradient, DelimitedFiles
include("../src/integrator/convert.jl")

fname = "elements.txt"
el = readdlm(fname,',')

ic = ElementsIC(el,8,0.0)
s = State(ic)

elements = get_orbital_elements(s,ic)

println(isapprox(elements,ic.elements))
