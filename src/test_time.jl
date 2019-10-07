
using PyPlot
# Tests different ways of keeping track of time.
# Looks like multiplication is just as accurate as
# compensated summation, yet is much faster.  Not
# that this is a limiting factor.
function test_time(nt)
#nt = 10000000
T = Float64
s2 = zero(T)
h = 0.04
#hbig = big(h)
t = zeros(T,nt)
#tb = zeros(BigFloat,nt)
tc = zeros(T,nt)
tm = zeros(T,nt)
for i=2:nt
  tc[i],s2 = comp_sum(tc[i-1],s2,h)
  t[i] = t[i-1] + h
  tm[i] = (i-1)*h
#  tb[i] = tb[i-1] + hbig
end

#tref = convert(Array{Float64,1},tb)
#semilogy(abs.(t-tref),label="Added")
#println("Maximum difference added: ",maximum(abs.(t-tref)))
#plot(abs.(tc-tref),":",label="Compensated")
#println("Maximum difference compensated: ",maximum(abs.(tc-tref)))
#plot(abs.(tm-tref),"--",label="Multiplied")
#println("Maximum difference multiplied: ",maximum(abs.(tm-tref)))
#legend()
return
end
