
# Wisdom & Hernandez version of Kepler solver, with Rein & Tamayo convergence test.
# Now using \gamma = \sqrt{\abs{\beta}}s rather than s now to solve Kepler's equation.

using ForwardDiff

include("g3.jl")

function cubic1(a::T, b::T, c::T) where {T <: Real}
a3 = a*third
Q = a3^2 - b*third
R = a3^3 + 0.5*(-a3*b + c)
if R^2 < Q^3
# println("Error in cubic solver ",R^2," ",Q^3)
 return -c/b
else
 A = -sign(R)*cbrt(abs(R) + sqrt(R*R - Q*Q*Q))
 if A == 0.0
   B = 0.0
  else
   B = Q/A
 end
 x1 = A + B - a3
 return x1
end
return
end

function jac_delxv!(x0::Array{T,1},v0::Array{T,1},k::T,h::T,drift_first::Bool;grad::Bool=false,auto::Bool=true,dlnq::T=convert(T,0.0)) where {T <: Real}
# Using autodiff, computes Jacobian of delx & delv with respect to x0, v0, k & h.

# Autodiff requires a single-vector input, so create an array to hold the independent variables:
  input = zeros(typeof(h),8)
  input[1:3]=x0; input[4:6]=v0; input[7]=k; input[8]=h

# Create a closure so that the function knows value of drift_first:

  function delx_delv(input) # input = x0,v0,k,h,drift_first
  # Compute delx and delv from h, s, k, beta0, x0 and v0:
  x0 = input[1:3]; v0 = input[4:6]; k = input[7]; h = input[8]
  # Compute r0:
  drift_first ?  r0 = norm(x0-h*v0) : r0 = norm(x0)
  # And its inverse:
  r0inv = inv(r0)
  # Compute beta_0:
  beta0 = 2k*r0inv-dot(v0,v0)
  beta0inv = inv(beta0)
  signb = sign(beta0)
  sqb = sqrt(signb*beta0)
  zeta = k-r0*beta0
  # Compute \eta_0 = x_0 . v_0:
  drift_first ?  eta = dot(x0-h*v0,v0) : eta = dot(x0,v0)
  if zeta != zero
    # Make sure we have a cubic in gamma (and don't divide by zero):
    gamma_guess = cubic1(3eta*sqb/zeta,6r0*signb*beta0/zeta,-6h*signb*beta0*sqb/zeta)
  else
    # Check that we have a quadratic in gamma (and don't divide by zero):
    if eta != zero
      reta = r0/eta
      disc = reta^2+2h/eta
      disc > zero ?  gamma_guess = sqb*(-reta+sqrt(disc)) : gamma_guess = h*r0inv*sqb
    else
      gamma_guess = h*r0inv*sqb
    end
  end
  gamma  = copy(gamma_guess)
  # Make sure prior two steps differ:
  gamma1 = 2*copy(gamma)
  gamma2 = 3*copy(gamma)
  iter = 0
  ITMAX = 20
  # Compute coefficients: (8/28/19 notes)
  c1 = k; c2 = -zeta; c3 = -eta*sqb; c4 = sqb*(eta-h*beta0); c5 = eta*signb*sqb
  # Solve for gamma:
  while true 
    gamma2 = gamma1
    gamma1 = gamma
    if beta0 > 0 
      sx = sin(gamma); cx = cos(gamma) 
    else 
      cx = cosh(gamma); sx = exp(gamma)-cx
    end
    gamma -= (c1*gamma+c2*sx+c3*cx+c4)/(c2*cx+c5*sx+c1)
    iter +=1 
    if iter >= ITMAX || gamma == gamma2 || gamma == gamma1
      break
    end
  end
#  if typeof(gamma) ==  Float64
#    println("s: ",gamma/sqb)
#  end
  # Set up a single output array for delx and delv:
  delxv = zeros(typeof(h),6)
  # Since we updated gamma, need to recompute:
  xx = 0.5*gamma
  if beta0 > 0 
    sx = sin(xx); cx = cos(xx) 
  else
    cx = cosh(xx); sx = exp(xx)-cx
  end
  # Now, compute final values.  Compute Wisdom/Hernandez G_i^\beta(s) functions:
  g1bs = 2sx*cx/sqb
  g2bs = 2signb*sx^2*beta0inv
  g0bs = one(h)-beta0*g2bs
  g3bs = g3(gamma,beta0)
#  if typeof(g1bs) == Float64
#    println("g1: ",g1bs," g2: ",g2bs," g3: ",g3bs)
#  end
  # Compute r from equation (35):
  r = r0*g0bs+eta*g1bs+k*g2bs
#  if typeof(r) == Float64
#    println("r: ",r)
#  end
  rinv = inv(r)
  dfdt = -k*g1bs*rinv*r0inv
  if drift_first
    # Drift backwards before Kepler step: (1/22/2018)
    fm1 = -k*r0inv*g2bs
    # This is given in 2/7/2018 notes:
    gmh = k*r0inv*(r0*(g1bs*g2bs-g3bs)+eta*g2bs^2+k*g3bs*g2bs)
  else
    # Drift backwards after Kepler step: (1/24/2018)
    fm1 =  k*rinv*(g2bs-k*r0inv*H1(gamma,beta0)) 
    # This is g-h*dgdt
    gmh = k*rinv*(r0*H2(gamma,beta0)+eta*H1(gamma,beta0))
  end
  # Compute velocity component functions:
  if drift_first
    dgdtm1 = k*r0inv*rinv*(r0*g0bs*g2bs+eta*g1bs*g2bs+k*g1bs*g3bs)
  else
    dgdtm1 = -k*rinv*g2bs
  end
#  if typeof(fm1) == Float64
#    println("fm1: ",fm1," dfdt: ",dfdt," gmh: ",gmh," dgdt-1: ",dgdtm1)
#  end
  for j=1:3
  # Compute difference vectors (finish - start) of step:
    delxv[  j] = fm1*x0[j]+gmh*v0[j]        # position x_ij(t+h)-x_ij(t) - h*v_ij(t) or -h*v_ij(t+h)
    delxv[3+j] = dfdt*x0[j]+dgdtm1*v0[j]    # velocity v_ij(t+h)-v_ij(t)
  end
  return delxv
  end

# Use autodiff to compute Jacobian:
if grad
  if auto
    delxv_jac = ForwardDiff.jacobian(delx_delv,input)
  else
# Use finite differences to compute Jacobian:
    delxv_jac = zeros(typeof(h),6,8)
    delxv = delx_delv(input)
    for j=1:8
      # Difference the jth component:
      inputp = copy(input); dp = dlnq*inputp[j]; inputp[j] += dp
      delxvp = delx_delv(inputp)
      inputm = copy(input); inputm[j] -= dp
      delxvm = delx_delv(inputm)
      delxv_jac[:,j] = (delxvp-delxvm)/(2*dp)
    end  
  end
# Return Jacobian:
  return  delxv_jac
else
  return delx_delv(input)
end
end
