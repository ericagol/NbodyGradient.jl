# Approximate G_3 for small \sqrt{|\beta|s}:

const cg3 = 1./[6.0,-120.0,5040.0,-362880.0,39916800.0,-6227020800.0]
const cg1g2mg0g3 = 1./[3.0,-30.0,840.0,-45360.0,3991680.0,-518918400.0]
const cg2_2mg1g3 = 1./[12.0,-180.0,6720.0,-453600.0,47900160.0,-7264857600.0]
 
function g3(x::T,sqb::T,beta::T) where {T <: Real}
if beta >= 0 
  return (x-sin(x))/(sqb*beta) 
else 
  return (x-sinh(x))/(sqb*beta)
end
end

function g1g2mg0g3(x::T,sqb::T,beta::T) where {T <: Real}
if beta >= 0
  return (sin(x)-x*cos(x))/(sqb*beta) 
else 
  return (sinh(x)-x*cosh(x))/(sqb*beta)
end
end

function g2_2mg1g3(x::T,beta::T) where {T <: Real}
if beta >= 0
  return (2.0-2cos(x)-x*sin(x))/beta^2
else 
  return (2.0-2cosh(x)+x*sinh(x))/beta^2
end
end

#function g3_series(s::Array{T,1},beta::T) where {T <: Real}
function g3_series(s::T,beta::T) where {T <: Real}
x2 = abs(beta)*s^2
pm = sign(beta)
return s*x2*(pm*cg3[1]+x2*(cg3[2]+x2*(pm*cg3[3]+x2*(cg3[4]+x2*(pm*cg3[5]+x2*cg3[6])))))/beta
end

#function g1g2mg0g3_series(s::Array{T,1},beta::T) where {T <: Real}
function g1g2mg0g3_series(s::T,beta::T) where {T <: Real}
x2 = abs(beta)*s^2
pm = sign(beta)
return s^3*(cg1g2mg0g3[1]+x2*(pm*cg1g2mg0g3[2]+x2*(cg1g2mg0g3[3]+
   x2*(pm*cg1g2mg0g3[4]+x2*(cg1g2mg0g3[5]+x2*pm*cg1g2mg0g3[6])))))
end

#function g2_2mg1g3_series(s::Array{T,1},beta::T) where {T <: Real}
function g2_2mg1g3_series(s::T,beta::T) where {T <: Real}
x2 = abs(beta)*s^2
pm = sign(beta)
return x2*x2*(cg2_2mg1g3[1]+x2*(pm*cg2_2mg1g3[2]+x2*(cg2_2mg1g3[3]+
   x2*(pm*cg2_2mg1g3[4]+x2*(cg2_2mg1g3[5]+x2*pm*cg2_2mg1g3[6])))))/beta^2
end

nx = 10000
s = linspace(-2.0,2.0,nx)

beta = -1.0
sqb = sqrt(abs(beta))
x = sqb*s
g31_bigm= g3.(convert(Array{BigFloat,1},x),sqrt(big(abs(beta))),big(beta))
g31m= g3.(x,sqb,beta)
e31m = convert(Array{Float64,1},1.0-g31m./g31_bigm)
g32m = g3_series.(s,beta)
e32m = convert(Array{Float64,1},1.0-g32m./g31_bigm)

g1g2mg0g31_bigm= g1g2mg0g3.(convert(Array{BigFloat,1},x),sqrt(big(abs(beta))),big(beta))
g1g2mg0g31m= g1g2mg0g3.(x,sqb,beta)
eg1g2mg0g31m = convert(Array{Float64,1},1.0-g1g2mg0g31m./g1g2mg0g31_bigm)
g1g2mg0g32m = g1g2mg0g3_series.(s,beta)
eg1g2mg0g32m = convert(Array{Float64,1},1.0-g1g2mg0g32m./g1g2mg0g31_bigm)

g2_2mg1g31_bigm= g2_2mg1g3.(convert(Array{BigFloat,1},x),big(beta))
g2_2mg1g31m= g2_2mg1g3.(x,beta)
eg2_2mg1g31m = convert(Array{Float64,1},1.0-g2_2mg1g31m./g2_2mg1g31_bigm)
g2_2mg1g32m = g2_2mg1g3_series.(s,beta)
eg2_2mg1g32m = convert(Array{Float64,1},1.0-g2_2mg1g32m./g2_2mg1g31_bigm)

beta = 1.0
sqb = sqrt(abs(beta))
x = sqb*s
g31_bigp= g3.(convert(Array{BigFloat,1},x),sqrt(big(abs(beta))),big(beta))
g31p= g3.(x,sqb,beta)
e31p = convert(Array{Float64,1},1.0-g31p./g31_bigp)
g32p = g3_series.(s,beta)
e32p = convert(Array{Float64,1},1.0-g32p./g31_bigp)

g1g2mg0g31_bigp= g1g2mg0g3.(convert(Array{BigFloat,1},x),sqrt(big(abs(beta))),big(beta))
g1g2mg0g31p= g1g2mg0g3.(x,sqb,beta)
eg1g2mg0g31p = convert(Array{Float64,1},1.0-g1g2mg0g31p./g1g2mg0g31_bigp)
g1g2mg0g32p = g1g2mg0g3_series.(s,beta)
eg1g2mg0g32p = convert(Array{Float64,1},1.0-g1g2mg0g32p./g1g2mg0g31_bigp)

g2_2mg1g31_bigp= g2_2mg1g3.(convert(Array{BigFloat,1},x),big(beta))
g2_2mg1g31p= g2_2mg1g3.(x,beta)
eg2_2mg1g31p = convert(Array{Float64,1},1.0-g2_2mg1g31p./g2_2mg1g31_bigp)
g2_2mg1g32p = g2_2mg1g3_series.(s,beta)
eg2_2mg1g32p = convert(Array{Float64,1},1.0-g2_2mg1g32p./g2_2mg1g31_bigp)

# It looks like for x < 0.5, the fractional error on g3_series is <epsilon for
# float64 numbers.

# So, for x > 0.5, we can use x-sin(x), while for x < 0.5, we can use the series.
# Also need to consider what happens for negative values.

using PyPlot
clf()
semilogy(x,abs.(e31m),label="G3, exact, beta<0 ")
println("e31m: ",minimum(abs.(e31m)))
semilogy(x,abs.(e32m),label="G3, series, beta<0 ")
println("e32m: ",minimum(abs.(e32m)))
read(STDIN,Char)
clf()
semilogy(x,abs.(e31p),label="G3, exact, beta>0 ")
println("e31p: ",minimum(abs.(e31p)))
semilogy(x,abs.(e32p),label="G3, series, beta>0 ")
println("e32p: ",minimum(abs.(e32p)))
read(STDIN,Char)
clf()
semilogy(x,abs.(eg1g2mg0g31m),label="G1G2-G0G3, exact, beta<0 ")
println("eg1g2mg0g31m: ",minimum(abs.(eg1g2mg0g31m)))
semilogy(x,abs.(eg1g2mg0g32m),label="G1G2-G0G3, series, beta<0 ")
println("eg1g2mg0g32m: ",minimum(abs.(eg1g2mg0g32m)))
read(STDIN,Char)
clf()
semilogy(x,abs.(eg1g2mg0g31p),label="G1G2-G0G3, exact, beta>0 ")
println("eg1g2mg0g31p: ",minimum(abs.(eg1g2mg0g31p)))
semilogy(x,abs.(eg1g2mg0g32p),label="G1G2-G0G3, series, beta>0 ")
println("eg1g2mg0g32p: ",minimum(abs.(eg1g2mg0g32p)))
read(STDIN,Char)
clf()
semilogy(x,abs.(eg2_2mg1g31m),label="G2^2-G1G3, exact, beta<0 ")
println("eg2_2mg1g31m: ",minimum(abs.(eg2_2mg1g31m)))
semilogy(x,abs.(eg2_2mg1g32m),label="G2^2-G1G3, series, beta<0 ")
println("eg2_2mg1g32m: ",minimum(abs.(eg2_2mg1g31p)))
read(STDIN,Char)
clf()
semilogy(x,abs.(eg2_2mg1g31p),label="G2^2-G1G3, exact, beta>0 ")
println("eg2_2mg1g31p: ",minimum(abs.(eg2_2mg1g31p)))
semilogy(x,abs.(eg2_2mg1g32p),label="G2^2-G1G3, series, beta>0 ")
println("eg2_2mg1g32p: ",minimum(abs.(eg2_2mg1g32p)))
semilogy(x,0.*x+eps(1.0))
#legend(loc="lower left")

read(STDIN,Char)
clf()
plot(x,abs.(g31m))
plot(x,abs.(g32m))
plot(x,abs.(g31p))
plot(x,abs.(g32p))
plot(x,abs.(g1g2mg0g31m))
plot(x,abs.(g1g2mg0g32m))
plot(x,abs.(g1g2mg0g31p))
plot(x,abs.(g1g2mg0g32p))
plot(x,abs.(g2_2mg1g31m))
plot(x,abs.(g2_2mg1g32m))
plot(x,abs.(g2_2mg1g31p))
plot(x,abs.(g2_2mg1g32p))
