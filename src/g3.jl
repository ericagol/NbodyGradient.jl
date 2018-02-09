# Series expansions for G_3, H_1 & H_2 for small \sqrt{|\beta|s}:

# Define a dummy function for automatic differentiation: 
function g3(param::Array{T,1}) where {T <: Real}
return g3(param[1],param[2])
end

function g3(s::T,beta::T) where {T <: Real}
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
