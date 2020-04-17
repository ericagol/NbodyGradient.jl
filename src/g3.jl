

# Define a dummy function for automatic differentiation: 
function G3(param::Array{T,1}) where {T <: Real}
return G3(param[1],param[2])
end

function G3(gamma::T,beta::T;gc=convert(T,0.5)) where {T <: Real}
sqb = sqrt(abs(beta))
#x = sqb*s
if gamma < gc
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

function H2(gamma::T,beta::T;gc=convert(T,0.5)) where {T <: Real}
sqb = sqrt(abs(beta))
#x=sqb*s
if gamma < gc
  return H2_series(gamma,beta)
else
  if beta >= 0
    return (sin(gamma)-gamma*cos(gamma))/(sqb*beta) 
  else 
    return (sinh(gamma)-gamma*cosh(gamma))/(sqb*beta)
  end
end
end

function H1(gamma::T,beta::T;gc=convert(T,0.5)) where {T <: Real}
#x=sqrt(abs(beta))*s
if gamma < gc
  return H1_series(gamma,beta)
else
  if beta >= 0
    return (4sin(0.5*gamma)^2 -gamma*sin(gamma))/beta^2
  else 
    return (-4sinh(0.5*gamma)^2+gamma*sinh(gamma))/beta^2
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
g31 = 2g3
g32 = 2g3
n=0
iter = 0
ITMAX = 100
# Terminate series when required precision reached:
#while abs(term) > epsilon*abs(g3)
while true
  g32 = g31
  g31 = g3
  n += 1
  term *= x2/((2n+3)*(2n+2))
  g3 += term
  iter +=1
  if iter >= ITMAX || g3 == g32 || g3 == g31
    break
  end
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
h21 = 2h2
h22 = 2h2
n=0
iter = 0
ITMAX = 100
# Terminate series when required precision reached:
#while abs(term) > epsilon*abs(h2)
while true
  h22 = h21
  h21 = h2
  n += 1
  term *= x2
  term /= (4n+6)*n
  h2 += term
  iter += 1 
  if iter >= ITMAX || h2 == h22 || h2 == h21
    break
  end
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
h11 = 2h1
h12 = 2h1
n=0
iter = 0
ITMAX = 100
# Terminate series when required precision reached:
#while abs(term) > epsilon*abs(h1)
while true
  h12 = h11
  h11 = h1
  n += 1
  term *= x2*(n+1)
  term /= (2n+4)*(2n+3)*n
  h1 += term
  iter +=1
  if iter >= ITMAX || h1 == h12 || h1 == h11
    break
  end
end
h1 *= gamma^4/(12*beta^2)
return h1::T
end

function H5(gamma::T,beta::T;gc=convert(T,0.5)) where {T <: Real}
# This is G_1 G_2 -(2+G_0) G_3 = H_2 - 2 G_3:
if gamma < gc
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
h51 = 2h5
h52 = 2h5
n=0
iter = 0
ITMAX = 100
# Terminate series when required precision reached:
#while abs(term) > epsilon*abs(h5)
while true
  h52 = h51
  h51 = h5
  n += 1
  term *= x2*(n+1)
  term /= (2n+5)*(2n+4)*n
  h5 += term
  iter += 1
  if iter >= ITMAX || h5 == h52 || h5 == h51
    break
  end
end
h5 *= -gamma^5/(beta*sqrt(abs(beta)))
return h5::T
end

function H6(gamma::T,beta::T;gc=convert(T,0.5)) where {T <: Real}
# This is 2 G_2^2 -3 G_1 G_3:
if gamma < gc
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
h61 = 2h6
h62 = 2h6
n=0
iter = 0
ITMAX = 100
# Terminate series when required precision reached:
#while abs(term) > epsilon*abs(h6)
while true
  h62 = h61
  h61 = h6
  n += 1
  term *= x2
  term /= (2n+5)*(2n+6)
  h6 += term*(4^(n+2)-3*n-7)
  iter +=1
  if iter >= ITMAX || h6 == h62 || h6 == h61
    break
  end
end
h6 *= gamma^6/(beta*abs(beta))
return h6::T
end

function H3(gamma::T,beta::T;gc=convert(T,0.5)) where {T <: Real}
# This is H_3 = G_1 G_2 - 3 G_3:
if gamma < gc
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
# Computes H_3(\beta,s) using a series tailored to the precision of gamma:
epsilon = eps(gamma)
#x2 = -beta*s^2
x2 = -sign(beta)*gamma^2
term = convert(T,1//30)
h3 = convert(T,1//10)
h31 = 2h3
h32 = 2h3
n=0
iter = 0
ITMAX = 100
# Terminate series when required precision reached:
#while abs(term) > epsilon*abs(h3)
while true
  h32 = h31
  h31 = h3
  n += 1
  term *= x2
  term /= (2n+4)*(2n+5)
  h3 += term*(4^(n+1)-1)
  iter += 1
  if iter >= ITMAX || h3 == h32 || h3 == h31
    break
  end
end
h3 *= -gamma^5/(beta*sqrt(abs(beta)))
return h3::T
end

function H7(gamma::T,beta::T;gc=convert(T,0.5)) where {T <: Real}
# This is H_7 = G_1 G_2 (1-2 G_0) + 3 G_0^2 G_3:
if gamma < gc
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
# Computes H_7(\beta,s) using a series tailored to the precision of gamma:
epsilon = eps(gamma)
#x2 = -beta*s^2
x2 = -sign(beta)*gamma^2
term = -convert(T,3//20160)*x2
h7 = convert(T,1//10-11//840*x2)
h71 = 2h7
h72 = 2h7
n=0
iter = 0
ITMAX = 100
# Terminate series when required precision reached:
#while abs(term) > epsilon*abs(h7)
while true
  h72 = h71
  h71 = h7
  n += 1
  term *= x2
  term /= (2n+6)*(2n+7)
  h7 += term*(9^(3+n)-1-(5+2*n)*2^(7+2*n))
  iter += 1
  if iter >= ITMAX || h7 == h72 || h7 == h71
    break
  end
end
h7 *= gamma^5/(beta*sqrt(abs(beta)))
return h7::T
end

function H8(gamma::T,beta::T;gc=convert(T,0.5)) where {T <: Real}
# This is H_8 = G_1 G_2 - 3 G_0 G_3:
if gamma < gc
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
# Computes H_8(\beta,s) using a series tailored to the precision of gamma:
epsilon = eps(gamma)
#x2 = -beta*s^2
x2 = -sign(beta)*gamma^2
term = convert(T,1//120)
h8 = convert(T,3//20)
h81 = 2h8
h82 = 2h8
n=0
iter = 0
ITMAX = 100
# Terminate series when required precision reached:
#while abs(term) > epsilon*abs(h8)
while true
  h82 = h81
  h81 = h8
  n += 1
  term *= x2
  term /= (2n+4)*(2n+5)
  h8 += term*(2^(5+2n)-14-6n)
  iter +=1
  if iter >= ITMAX || h8 == h82 || h8 == h81
    break
  end
end
h8 *= gamma^5/(beta*sqrt(abs(beta)))
return h8::T
end
