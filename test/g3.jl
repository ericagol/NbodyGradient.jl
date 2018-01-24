# Approximate G_3 for small \sqrt{|\beta|s}:

const cg3 = 1./[6.0,-120.0,5040.0,-362880.0,39916800.0,-6227020800.0]

function g3(x::T,sqb::T,beta::T) where {T <: Real}
if beta >= 0 
  return (x-sin(x))/(sqb*beta) 
else 
  return (x-sinh(x))/(sqb*beta)
end
end

function g3_series(s::Array{T,1},beta::T) where {T <: Real}
x2 = abs(beta)*s^2
pm = sign(beta)
return s*x2*(pm*cg3[1]+x2*(cg3[2]+x2*(pm*cg3[3]+x2*(cg3[4]+x2*(pm*cg3[5]+x2*cg3[6])))))/beta
end

nx = 10001
s = linspace(0.0,10.0,nx)

beta = -1.0
sqb = sqrt(abs(beta))
x = sqb*s
g31_bigm= g3.(convert(Array{BigFloat,1},x),sqrt(big(abs(beta))),big(beta))
g31m= g3.(x,sqb,beta)
@time g31m= g3.(x,sqb,beta)
e31m = convert(Array{Float64,1},1.0-g31m./g31_bigm)
@time g32m = g3_series.(s,beta)
e32m = convert(Array{Float64,1},1.0-g32m./g31_bigm)

beta = 1.0
g31_bigp= g3.(convert(Array{BigFloat,1},x),big(beta))
@time g31p= g3.(x,beta)
e31p = convert(Array{Float64,1},1.0-g31p./g31_bigp)
@time g32p = g3_series.(s,beta)
e32p = convert(Array{Float64,1},1.0-g32p./g31_bigp)

# It looks like for x < 0.5, the fractional error on g3_series is <epsilon for
# float64 numbers.

# So, for x > 0.5, we can use x-sin(x), while for x < 0.5, we can use the series.
# Also need to consider what happens for negative values.

using PyPlot
clf()
semilogy(x,abs.(e31m))
semilogy(x,abs.(e31p))
semilogy(x,abs.(e32m))
semilogy(x,abs.(e32p))
semilogy(x,0.*x+eps(1.0))

read(STDIN,Char)
clf()
semilogy(x,abs.(g31m))
semilogy(x,abs.(g32m))
semilogy(x,abs.(g31p))
semilogy(x,abs.(g32p))
