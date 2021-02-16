# Approximate G_3 for small \sqrt{|\beta|s}:

using ForwardDiff

#const cg3 = 1./[6.0,-120.0,5040.0,-362880.0,39916800.0,-6227020800.0]
#const cH2 = 1./[3.0,-30.0,840.0,-45360.0,3991680.0,-518918400.0]
#const cH1 = 1./[12.0,-180.0,6720.0,-453600.0,47900160.0,-7264857600.0]

# Define a dummy function for automatic differentiation: 
function G3(param::Array{T,1}) where {T <: Real}
return G3(param[1],param[2])
end

function G3(s::T,beta::T) where {T <: Real}
sqb = sqrt(abs(beta))
x = sqb*s
if beta >= 0 
  return (x-sin(x))/(sqb*beta) 
else 
  return (x-sinh(x))/(sqb*beta)
end
end

y = zeros(2)
dg3 = y -> ForwardDiff.gradient(g3,y);

function H2(s::T,beta::T) where {T <: Real}
sqb = sqrt(abs(beta))
x=sqb*s
if beta >= 0
  return (sin(x)-x*cos(x))/(sqb*beta) 
else 
  return (sinh(x)-x*cosh(x))/(sqb*beta)
end
end

function H1(s::T,beta::T) where {T <: Real}
x=sqrt(abs(beta))*s
if beta >= 0
  return (2.0-2cos(x)-x*sin(x))/beta^2
else 
  return (2.0-2cosh(x)+x*sinh(x))/beta^2
end
end

function g3_series(s::T,beta::T) where {T <: Real}
epsilon = eps(s)
# Computes G_3(\beta,s) using a series tailored to the precision of s.
x2 = -beta*s^2
term = one(s)
g3 = one(s)
n=0
# Terminate series when required precision reached:
while abs(term) > epsilon*abs(g3)
  n += 1
  term *= x2/((2n+3)*(2n+2))
  g3 += term
end
g3 *= s^3/6
return g3::T
end

dg3_series = y -> ForwardDiff.gradient(g3_series,y);

# Define a dummy function for automatic differentiation: 
function g3_series(param::Array{T,1}) where {T <: Real}
return g3_series(param[1],param[2])
end

#function H2_series(s::T,beta::T) where {T <: Real}
#x2 = abs(beta)*s^2
#pm = sign(beta)
#return s^3*(cH2[1]+x2*(pm*cH2[2]+x2*(cH2[3]+
#   x2*(pm*cH2[4]+x2*(cH2[5]+x2*pm*cH2[6])))))
#end

function H2_series(s::T,beta::T) where {T <: Real}
# Computes H_2(\beta,s) using a series tailored to the precision of s.
epsilon = eps(s)
x2 = -beta*s^2
term = one(s)
h2 = one(s)
n=0
# Terminate series when required precision reached:
while abs(term) > epsilon*abs(h2)
  n += 1
  term *= x2
  term /= (4n+6)*n
  h2 += term
end
h2 *= s^3/3
return h2::T
end

#function H1_series(s::T,beta::T) where {T <: Real}
#x2 = abs(beta)*s^2
#pm = sign(beta)
#return x2*x2*(cH1[1]+x2*(pm*cH1[2]+x2*(cH1[3]+
#   x2*(pm*cH1[4]+x2*(cH1[5]+x2*pm*cH1[6])))))/beta^2
#end

function H1_series(s::T,beta::T) where {T <: Real}
# Computes H_1(\beta,s) using a series tailored to the precision of s.
epsilon = eps(s)
x2 = -beta*s^2
term = one(s)
h1 = one(s)
n=0
# Terminate series when required precision reached:
while abs(term) > epsilon*abs(h1)
  n += 1
  term *= x2*(n+1)
  term /= (2n+4)*(2n+3)*n
  h1 += term
end
h1 *= s^4/12
return h1::T
end

nx = 10001
s = linearspace(-3.0,3.0,nx)
#s = linearspace(-0.5,0.5,nx)

beta = -rand(); epsilon = eps(beta)
sqb = sqrt(abs(beta))
x = sqb*s
g31_bigm= g3.(convert(Array{BigFloat,1},s),big(beta))
println(typeof(x))
g31m= g3.(s,beta)
@time g31m= g3.(s,beta)
e31m = convert(Array{Float64,1},1.0-g31m./g31_bigm)
e31m[isnan.(e31m)]=epsilon
g32m = g3_series.(s,beta)
@time g32m = g3_series.(s,beta)
e32m = convert(Array{Float64,1},1.0-g32m./g31_bigm)
e32m[isnan.(e32m)]=epsilon

H21_bigm= H2.(convert(Array{BigFloat,1},s),big(beta))
H21m= H2.(s,beta)
@time H21m= H2.(s,beta)
eH21m = convert(Array{Float64,1},1.0-H21m./H21_bigm)
eH21m[isnan.(eH21m)]=epsilon
H22m = H2_series.(s,beta)
@time H22m = H2_series.(s,beta)
eH22m = convert(Array{Float64,1},1.0-H22m./H21_bigm)
eH22m[isnan.(eH22m)]=epsilon

H11_bigm= H1.(convert(Array{BigFloat,1},s),big(beta))
H11m= H1.(s,beta)
@time H11m= H1.(s,beta)
eH11m = convert(Array{Float64,1},1.0-H11m./H11_bigm)
eH11m[isnan.(eH11m)]=epsilon
H12m = H1_series.(s,beta)
@time H12m = H1_series.(s,beta)
eH12m = convert(Array{Float64,1},1.0-H12m./H11_bigm)
eH12m[isnan.(eH12m)]=epsilon

beta = rand()
sqb = sqrt(abs(beta))
x = sqb*s
g31_bigp= g3.(convert(Array{BigFloat,1},s),big(beta))
println(typeof(x))
@time g31p= g3.(s,beta)
e31p = convert(Array{Float64,1},1.0-g31p./g31_bigp)
e31p[isnan.(e31p)]=epsilon
@time g32p = g3_series.(s,beta)
e32p = convert(Array{Float64,1},1.0-g32p./g31_bigp)
e32p[isnan.(e32p)]=epsilon

H21_bigp= H2.(convert(Array{BigFloat,1},s),big(beta))
@time H21p= H2.(s,beta)
eH21p = convert(Array{Float64,1},1.0-H21p./H21_bigp)
eH21p[isnan.(eH21p)]=epsilon
@time H22p = H2_series.(s,beta)
eH22p = convert(Array{Float64,1},1.0-H22p./H21_bigp)
eH22p[isnan.(eH22p)]=epsilon

H11_bigp= H1.(convert(Array{BigFloat,1},s),big(beta))
@time H11p= H1.(s,beta)
eH11p = convert(Array{Float64,1},1.0-H11p./H11_bigp)
eH11p[isnan.(eH11p)]=epsilon
@time H12p = H1_series.(s,beta)
eH12p = convert(Array{Float64,1},1.0-H12p./H11_bigp)
eH12p[isnan.(eH12p)]=epsilon

# It looks like for x < 0.5, the fractional error on g3_series is <epsilon for
# float64 numbers.

# So, for x > 0.5, we can use x-sin(x), while for x < 0.5, we can use the series.
# Also need to consider what happens for negative values.

using PyPlot
clf()
semilogy(x,abs.(e31m),label="G3, exact, beta<0 ")
println("e31m: ",minimum(abs.(e31m)))
semilogy(x,abs.(e32m),label="G3, series, beta<0 ")
#semilogy([.5,.5],[1e-16,maxabs(e31m)],linestyle="dashed")
println("e32m: ",minimum(abs.(e32m)))
#semilogy(-[.5,.5],[1e-16,maxabs(e31m)],linestyle="dashed")
semilogy(x,0.*x+eps(1.0))
read(STDIN,Char)
clf()
semilogy(x,abs.(e31p),label="G3, exact, beta>0 ")
println("e31p: ",minimum(abs.(e31p)))
#semilogy([.5,.5],[1e-16,1.],linestyle="dashed")
semilogy(x,abs.(e32p),label="G3, series, beta>0 ")
println("e32p: ",minimum(abs.(e32p)))
#semilogy(-[.5,.5],[1e-16,maxabs(e31p)],linestyle="dashed")
semilogy(x,0.*x+eps(1.0))
read(STDIN,Char)
clf()
semilogy(x,abs.(eH21m),label="G1G2-G0G3, exact, beta<0 ")
println("eH21m: ",minimum(abs.(eH21m)))
#semilogy( [.5,.5],[1e-16,1.],linestyle="dashed")
semilogy(x,abs.(eH22m),label="G1G2-G0G3, series, beta<0 ")
println("eH22m: ",minimum(abs.(eH22m)))
#semilogy(-[.5,.5],[1e-16,maxabs(eH21m)],linestyle="dashed")
semilogy(x,0.*x+eps(1.0))
read(STDIN,Char)
clf()
semilogy(x,abs.(eH21p),label="G1G2-G0G3, exact, beta>0 ")
println("eH21p: ",minimum(abs.(eH21p)))
#semilogy( [.5,.5],[1e-16,1.],linestyle="dashed")
semilogy(x,abs.(eH22p),label="G1G2-G0G3, series, beta>0 ")
println("eH22p: ",minimum(abs.(eH22p)))
#semilogy(-[.5,.5],[1e-16,maxabs(abs.(eH21p))],linestyle="dashed")
semilogy(x,0.*x+eps(1.0))
read(STDIN,Char)
clf()
semilogy(x,abs.(eH11m),label="G2^2-G1G3, exact, beta<0 ")
println("eH11m: ",minimum(abs.(eH11m)))
#semilogy( [.5,.5],[1e-16,1.],linestyle="dashed")
semilogy(x,abs.(eH12m),label="G2^2-G1G3, series, beta<0 ")
println("eH12m: ",minimum(abs.(eH11p)))
#semilogy(-[.5,.5],[1e-16,maxabs(eH11m)],linestyle="dashed")
semilogy(x,0.*x+eps(1.0))
read(STDIN,Char)
clf()
semilogy(x,abs.(eH11p),label="G2^2-G1G3, exact, beta>0 ")
println("eH11p: ",minimum(abs.(eH11p)))
#semilogy( [.5,.5],[1e-16,1.],linestyle="dashed")
semilogy(x,abs.(eH12p),label="G2^2-G1G3, series, beta>0 ")
println("eH12p: ",minimum(abs.(eH12p)))
semilogy(x,0.*x+eps(1.0))
#semilogy(-[.5,.5],[1e-16,maxabs(eH11p)],linestyle="dashed")
#legend(loc="lower left")

read(STDIN,Char)
clf()
plot(x,abs.(g31m))
plot(x,abs.(g32m))
plot(x,abs.(g31p))
plot(x,abs.(g32p))
plot(x,abs.(H21m))
plot(x,abs.(H22m))
plot(x,abs.(H21p))
plot(x,abs.(H22p))
plot(x,abs.(H11m))
plot(x,abs.(H12m))
plot(x,abs.(H11p))
plot(x,abs.(H12p))
