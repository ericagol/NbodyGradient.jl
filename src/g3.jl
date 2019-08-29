# Series expansions for G_3, H_1 & H_2 for small \sqrt{|\beta|s}:

# Define a dummy function for automatic differentiation: 
function g3(param::Array{T,1}) where {T <: Real}
return g3(param[1],param[2])
end

function g3(gamma::T,beta::T) where {T <: Real}
sqb = sqrt(abs(beta))
#x = sqb*s
if gamma < 0.5
  return g3_series(gamma,beta)
else
  if beta >= 0 
    return (gamma-sin(gamma))/(sqb*beta) 
  else 
    return (gamma-sinh(gamma))/(sqb*beta)
  end
end
end

y = zeros(2)
dg3 = y -> ForwardDiff.gradient(g3,y);

function H2(gamma::T,beta::T) where {T <: Real}
sqb = sqrt(abs(beta))
#x=sqb*s
if gamma < 0.5
  return H2_series(gamma,beta)
else
  if beta >= 0
    return (sin(gamma)-gamma*cos(gamma))/(sqb*beta) 
  else 
    return (sinh(gamma)-gamma*cosh(gamma))/(sqb*beta)
  end
end
end

function H1(gamma::T,beta::T) where {T <: Real}
#x=sqrt(abs(beta))*s
if gamma < 0.5
  return H1_series(gamma,beta)
else
  if beta >= 0
    return (2.0-2cos(gamma)-gamma*sin(gamma))/beta^2
  else 
    return (2.0-2cosh(gamma)+gamma*sinh(gamma))/beta^2
  end
end
end

function g3_series(gamma::T,beta::T) where {T <: Real}
epsilon = eps(gamma)
# Computes G_3(\beta,s) using a series tailored to the precision of s.
#x2 = -beta*s^2
x2 = -sign(beta)*gamma^2
term = one(gamma)
g3 = one(gamma)
n=0
# Terminate series when required precision reached:
while abs(term) > epsilon*abs(g3)
  n += 1
  term *= x2/((2n+3)*(2n+2))
  g3 += term
end
g3 *= gamma^3/(6*sqrt(abs(beta^3)))
return g3::T
end

dg3_series = y -> ForwardDiff.gradient(g3_series,y);

# Define a dummy function for automatic differentiation: 
function g3_series(param::Array{T,1}) where {T <: Real}
return g3_series(param[1],param[2])
end

function H2_series(gamma::T,beta::T) where {T <: Real}
# Computes H_2(\beta,s) using a series tailored to the precision of s.
epsilon = eps(gamma)
#x2 = -beta*s^2
x2 = -sign(beta)*gamma^2
term = one(gamma)
h2 = one(gamma)
n=0
# Terminate series when required precision reached:
while abs(term) > epsilon*abs(h2)
  n += 1
  term *= x2
  term /= (4n+6)*n
  h2 += term
end
h2 *= gamma^3/(3*sqrt(abs(beta^3)))
return h2::T
end

function H1_series(gamma::T,beta::T) where {T <: Real}
# Computes H_1(\beta,s) using a series tailored to the precision of s.
epsilon = eps(gamma)
#x2 = -beta*s^2
x2 = -sign(beta)*gamma^2
term = one(gamma)
h1 = one(gamma)
n=0
# Terminate series when required precision reached:
while abs(term) > epsilon*abs(h1)
  n += 1
  term *= x2*(n+1)
  term /= (2n+4)*(2n+3)*n
  h1 += term
end
h1 *= gamma^4/(12*beta^2)
return h1::T
end
