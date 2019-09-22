

# Define a dummy function for automatic differentiation: 
function G3(param::Array{T,1}) where {T <: Real}
return G3(param[1],param[2])
end

function G3(gamma::T,beta::T) where {T <: Real}
sqb = sqrt(abs(beta))
#x = sqb*s
if gamma < 0.5
  return G3_series(gamma,beta)
else
  if beta >= 0 
    return (gamma-sin(gamma))/(sqb*beta) 
  else 
    return (gamma-sinh(gamma))/(sqb*beta)
  end
end
end

y = zeros(2)
dG3 = y -> ForwardDiff.gradient(G3,y);

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

function G3_series(gamma::T,beta::T) where {T <: Real}
epsilon = eps(gamma)
# Computes G_3(\beta,s) using a series tailored to the precision of s.
#x2 = -beta*s^2
x2 = -sign(beta)*gamma^2
term = one(T)
g3 = one(T)
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

dG3_series = y -> ForwardDiff.gradient(G3_series,y);

# Define a dummy function for automatic differentiation: 
function G3_series(param::Array{T,1}) where {T <: Real}
return G3_series(param[1],param[2])
end

function H2_series(gamma::T,beta::T) where {T <: Real}
# Computes H_2(\beta,s) using a series tailored to the precision of s.
epsilon = eps(gamma)
#x2 = -beta*s^2
x2 = -sign(beta)*gamma^2
term = one(T)
h2 = one(T)
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
term = one(T)
h1 = one(T)
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

function H5(gamma::T,beta::T) where {T <: Real}
# This is G_1 G_2 -(2+G_0) G_3 = H_2 - 2 G_3:
if gamma < 0.5
  return H5_series(gamma,beta)
else
  if beta >= 0
    return (3sin(gamma)-2gamma-gamma*cos(gamma))/(beta*sqrt(beta))
  else 
    return (3sinh(gamma)-2gamma-gamma*cosh(gamma))/(beta*sqrt(-beta))
  end
end
end

function H5_series(gamma::T,beta::T) where {T <: Real}
# Computes H_5(\beta,s) using a series tailored to the precision of gamma:
epsilon = eps(gamma)
#x2 = -beta*s^2
x2 = -sign(beta)*gamma^2
term = one(T)/60
h5 = copy(term)
n=0
# Terminate series when required precision reached:
while abs(term) > epsilon*abs(h5)
  n += 1
  term *= x2*(n+1)
  term /= (2n+5)*(2n+4)*n
  h5 += term
end
h5 *= -gamma^5/(beta*sqrt(abs(beta)))
return h5::T
end

function H6(gamma::T,beta::T) where {T <: Real}
# This is 2 G_2^2 -3 G_1 G_3:
if gamma < 0.5
  return H6_series(gamma,beta)
else
  if beta >= 0
    return (9-8cos(gamma) -cos(2gamma) -6gamma*sin(gamma))/(2beta^2)
  else 
    return (9-8cosh(gamma)-cosh(2gamma)+6gamma*sinh(gamma))/(2beta^2)
  end
end
end

function H6_series(gamma::T,beta::T) where {T <: Real}
# Computes H_6(\beta,s) using a series tailored to the precision of gamma:
epsilon = eps(gamma)
#x2 = -beta*s^2
x2 = -sign(beta)*gamma^2
term = convert(T,1//360)
h6 = convert(T,1//40)
n=0
# Terminate series when required precision reached:
while abs(term) > epsilon*abs(h6)
  n += 1
  term *= x2
  term /= (2n+5)*(2n+6)
  h6 += term*(4^(n+2)-3*n-7)
end
h6 *= gamma^6/(beta*abs(beta))
return h6::T
end

function H3(gamma::T,beta::T) where {T <: Real}
# This is H_3 = G_1 G_2 - 3 G_3:
if gamma < 0.5
  return H3_series(gamma,beta)
else
  if beta >= 0
    return (4*sin(gamma)-sin(gamma)*cos(gamma)-3*gamma)/(beta*sqrt(beta))
  else
    return (4*sinh(gamma)-sinh(gamma)*cosh(gamma)-3*gamma)/(beta*sqrt(-beta))
  end
end
end

function H3_series(gamma::T,beta::T) where {T <: Real}
# Computes H_7(\beta,s) using a series tailored to the precision of gamma:
epsilon = eps(gamma)
#x2 = -beta*s^2
x2 = -sign(beta)*gamma^2
term = convert(T,1//30)
h3 = convert(T,1//10)
n=0
# Terminate series when required precision reached:
while abs(term) > epsilon*abs(h3)
  n += 1
  term *= x2
  term /= (2n+4)*(2n+5)
  h3 += term*(4^(n+1)-1)
end
h3 *= -gamma^5/(beta*sqrt(abs(beta)))
return h3::T
end

function H7(gamma::T,beta::T) where {T <: Real}
# This is H_7 = G_1 G_2 (1-2 G_0) + 3 G_0^2 G_3:
if gamma < 0.5
  return H7_series(gamma,beta)
else
  if beta >= 0
    return (3*cos(gamma)*(gamma*cos(gamma)-sin(gamma))+sin(gamma)^3)/(beta*sqrt(beta))
  else
    return (3*cosh(gamma)*(gamma*cosh(gamma)-sinh(gamma))-sinh(gamma)^3)/(beta*sqrt(-beta))
  end
end
end

function H7_series(gamma::T,beta::T) where {T <: Real}
# Computes H_3(\beta,s) using a series tailored to the precision of gamma:
epsilon = eps(gamma)
#x2 = -beta*s^2
x2 = -sign(beta)*gamma^2
term = -convert(T,3//20160)*x2
h7 = convert(T,1//10-11//840*x2)
n=0
# Terminate series when required precision reached:
while abs(term) > epsilon*abs(h7)
  n += 1
  term *= x2
  term /= (2n+6)*(2n+7)
  h7 += term*(9^(3+n)-1-(5+2*n)*2^(7+2*n))
end
h7 *= gamma^5/(beta*sqrt(abs(beta)))
return h7::T
end

function H8(gamma::T,beta::T) where {T <: Real}
# This is H_8 = G_1 G_2 - 3 G_0 G_3:
if gamma < 0.5
  return H8_series(gamma,beta)
else
  if beta >= 0
    return (-3gamma*cos(gamma) +sin(gamma) +sin(2gamma))/(beta*sqrt(beta))
  else
    return (-3gamma*cosh(gamma)+sinh(gamma)+sinh(2gamma))/(beta*sqrt(-beta))
  end
end
end

function H8_series(gamma::T,beta::T) where {T <: Real}
# Computes H_3(\beta,s) using a series tailored to the precision of gamma:
epsilon = eps(gamma)
#x2 = -beta*s^2
x2 = -sign(beta)*gamma^2
term = convert(T,1//120)
h8 = convert(T,3//20)
n=0
# Terminate series when required precision reached:
while abs(term) > epsilon*abs(h8)
  n += 1
  term *= x2
  term /= (2n+4)*(2n+5)
  h8 += term*(2^(5+2n)-14-6n)
end
h8 *= gamma^5/(beta*sqrt(abs(beta)))
return h8::T
end
