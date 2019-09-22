  

# Wisdom & Hernandez version of Kepler solver, with Rein & Tamayo convergence test.
# Now using \gamma = \sqrt{\abs{\beta}}s rather than s now to solve Kepler's equation.

using ForwardDiff, DiffResults

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

function jac_delxv_gamma!(x0::Array{T,1},v0::Array{T,1},k::T,h::T,drift_first::Bool;grad::Bool=false,auto::Bool=false,dlnq::T=convert(T,0.0),debug=false) where {T <: Real}
# Using autodiff, computes Jacobian of delx & delv with respect to x0, v0, k & h.

# Autodiff requires a single-vector input, so create an array to hold the independent variables:
  input = zeros(T,8)
  input[1:3]=x0; input[4:6]=v0; input[7]=k; input[8]=h
  if grad
    if debug 
      # Also output gamma, r, fm1, dfdt, gmh, dgdtm1, and for debugging:
      delxv_jac = zeros(T,12,8)
    else
      # Output \delta x & \delta v only:
      delxv_jac = zeros(T,6,8)
    end
  end

# Create a closure so that the function knows value of drift_first:

  function delx_delv(input::Array{T2,1}) where {T2 <: Real} # input = x0,v0,k,h,drift_first
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
  gamma_guess = zero(T2)
  # Compute \eta_0 = x_0 . v_0:
  drift_first ?  eta = dot(x0-h*v0,v0) : eta = dot(x0,v0)
  if zeta != zero(T2)
    # Make sure we have a cubic in gamma (and don't divide by zero):
    gamma_guess = cubic1(3eta*sqb/zeta,6r0*signb*beta0/zeta,-6h*signb*beta0*sqb/zeta)
  else
    # Check that we have a quadratic in gamma (and don't divide by zero):
    if eta != zero(T2)
      reta = r0/eta
      disc = reta^2+2h/eta
      disc > zero(T2) ?  gamma_guess = sqb*(-reta+sqrt(disc)) : gamma_guess = h*r0inv*sqb
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
  if debug
    delxv = zeros(T2,12)
  else
    delxv = zeros(T2,6)
  end
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
  g0bs = one(T2)-beta0*g2bs
  g3bs = G3(gamma,beta0)
  h1 = zero(T2); h2 = zero(T2)
#  if typeof(g1bs) == Float64
#    println("g1: ",g1bs," g2: ",g2bs," g3: ",g3bs)
#  end
  # Compute r from equation (35):
  r = r0*g0bs+eta*g1bs+k*g2bs
#  if typeof(r) == Float64
#    println("r: ",r)
#  end
  rinv = inv(r)
  dfdt = -k*g1bs*rinv*r0inv # [x]
  if drift_first
    # Drift backwards before Kepler step: (1/22/2018)
    fm1 = -k*r0inv*g2bs # [x]
    # This is given in 2/7/2018 notes: g-h*f
#    gmh = k*r0inv*(r0*(g1bs*g2bs-g3bs)+eta*g2bs^2+k*g3bs*g2bs)  # [x]
#    println("Old gmh: ",gmh," new gmh: ",k*r0inv*(h*g2bs-r0*g3bs))  # [x]
    gmh = k*r0inv*(h*g2bs-r0*g3bs)  # [x]
  else
    # Drift backwards after Kepler step: (1/24/2018)
    # The following line is f-1-h fdot:
    h1= H1(gamma,beta0); h2= H2(gamma,beta0)
#    fm1 =  k*rinv*(g2bs-k*r0inv*H1(gamma,beta0))  # [x]
    fm1 =  k*rinv*(g2bs-k*r0inv*h1)  # [x]
    # This is g-h*dgdt
#    gmh = k*rinv*(r0*H2(gamma,beta0)+eta*H1(gamma,beta0)) # [x]
    gmh = k*rinv*(r0*h2+eta*h1) # [x]
  end
  # Compute velocity component functions:
  if drift_first
    # This is gdot - h fdot - 1:
#    dgdtm1 = k*r0inv*rinv*(r0*g0bs*g2bs+eta*g1bs*g2bs+k*g1bs*g3bs) # [x]
#    println("gdot-h fdot-1: ",dgdtm1," alternative expression: ",k*r0inv*rinv*(h*g1bs-r0*g2bs))
    dgdtm1 = k*r0inv*rinv*(h*g1bs-r0*g2bs)
  else
    # This is gdot - 1:
    dgdtm1 = -k*rinv*g2bs # [x]
  end
#  if typeof(fm1) == Float64
#    println("fm1: ",fm1," dfdt: ",dfdt," gmh: ",gmh," dgdt-1: ",dgdtm1)
#  end
  @inbounds for j=1:3
  # Compute difference vectors (finish - start) of step:
    delxv[  j] = fm1*x0[j]+gmh*v0[j]        # position x_ij(t+h)-x_ij(t) - h*v_ij(t) or -h*v_ij(t+h)
  end
  @inbounds for j=1:3
    delxv[3+j] = dfdt*x0[j]+dgdtm1*v0[j]    # velocity v_ij(t+h)-v_ij(t)
  end
  if debug
    delxv[7] = gamma
    delxv[8] = r
    delxv[9] = fm1
    delxv[10] = dfdt
    delxv[11] = gmh
    delxv[12] = dgdtm1
  end
  if grad == true && auto == false && dlnq == 0.0
    # Compute gradient analytically:
    jac_mass = zeros(T,6)
    compute_jacobian_gamma!(gamma,g0bs,g1bs,g2bs,g3bs,h1,h2,dfdt,fm1,gmh,dgdtm1,r0,r,r0inv,rinv,k,h,beta0,beta0inv,eta,sqb,zeta,x0,v0,delxv_jac,jac_mass,drift_first,debug)
  end
  return delxv
  end

# Use autodiff to compute Jacobian:
if grad
  if auto
#    delxv_jac = ForwardDiff.jacobian(delx_delv,input)
    if debug
      delxv = zeros(T,12)
    else
      delxv = zeros(T,6)
    end
    out = DiffResults.JacobianResult(delxv,input)
    ForwardDiff.jacobian!(out,delx_delv,input)
    delxv_jac = DiffResults.jacobian(out)
    delxv = DiffResults.value(out)
  elseif dlnq != 0.0
# Use finite differences to compute Jacobian:
    if debug
      delxv_jac = zeros(T,12,8)
    else
      delxv_jac = zeros(T,6,8)
    end
    delxv = delx_delv(input)
    @inbounds for j=1:8
      # Difference the jth component:
      inputp = copy(input); dp = dlnq*inputp[j]; inputp[j] += dp
      delxvp = delx_delv(inputp)
      inputm = copy(input); inputm[j] -= dp
      delxvm = delx_delv(inputm)
      delxv_jac[:,j] = (delxvp-delxvm)/(2*dp)
    end
  else
# If grad = true and dlnq = 0.0, then the above routine will compute Jacobian analytically.
    delxv = delx_delv(input)
  end
# Return Jacobian:
  return  delxv::Array{T,1},delxv_jac::Array{T,2}
else
  return delx_delv(input)::Array{T,1}
end
end

function jac_delxv_gamma!(x0::Array{T,1},v0::Array{T,1},k::T,h::T,drift_first::Bool,delxv::Array{T,1},delxv_jac::Array{T,2},jac_mass::Array{T,1},debug::Bool) where {T <: Real}
# Analytically computes Jacobian of delx & delv with respect to x0, v0, k & h.

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
  gamma_guess = zero(T)
  # Compute \eta_0 = x_0 . v_0:
  drift_first ?  eta = dot(x0-h*v0,v0) : eta = dot(x0,v0)
  if zeta != zero(T)
    # Make sure we have a cubic in gamma (and don't divide by zero):
    gamma_guess = cubic1(3eta*sqb/zeta,6r0*signb*beta0/zeta,-6h*signb*beta0*sqb/zeta)
  else
    # Check that we have a quadratic in gamma (and don't divide by zero):
    if eta != zero(T)
      reta = r0/eta
      disc = reta^2+2h/eta
      disc > zero(T) ?  gamma_guess = sqb*(-reta+sqrt(disc)) : gamma_guess = h*r0inv*sqb
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
  # Set up a single output array for delx and delv:
  fill!(delxv,zero(T))
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
  g0bs = one(T)-beta0*g2bs
  g3bs = G3(gamma,beta0)
  h1 = zero(T); h2 = zero(T)
  # Compute r from equation (35):
  r = r0*g0bs+eta*g1bs+k*g2bs
  rinv = inv(r)
  dfdt = -k*g1bs*rinv*r0inv # [x]
  if drift_first
    # Drift backwards before Kepler step: (1/22/2018)
    fm1 = -k*r0inv*g2bs # [x]
    # This is given in 2/7/2018 notes: g-h*f
#    gmh = k*r0inv*(r0*(g1bs*g2bs-g3bs)+eta*g2bs^2+k*g3bs*g2bs)  # [x]
    gmh = k*r0inv*(h*g2bs-r0*g3bs)  # [x]
  else
    # Drift backwards after Kepler step: (1/24/2018)
    # The following line is f-1-h fdot:
    h1= H1(gamma,beta0); h2= H2(gamma,beta0)
#    fm1 =  k*rinv*(g2bs-k*r0inv*H1(gamma,beta0))  # [x]
    fm1 =  k*rinv*(g2bs-k*r0inv*h1)  # [x]
    # This is g-h*dgdt
#    gmh = k*rinv*(r0*H2(gamma,beta0)+eta*H1(gamma,beta0)) # [x]
    gmh = k*rinv*(r0*h2+eta*h1) # [x]
  end
  # Compute velocity component functions:
  if drift_first
    # This is gdot - h fdot - 1:
#    dgdtm1 = k*r0inv*rinv*(r0*g0bs*g2bs+eta*g1bs*g2bs+k*g1bs*g3bs) # [x]
    dgdtm1 = k*r0inv*rinv*(h*g1bs-r0*g2bs)
  else
    # This is gdot - 1:
    dgdtm1 = -k*rinv*g2bs # [x]
  end
  @inbounds for j=1:3
  # Compute difference vectors (finish - start) of step:
    delxv[  j] = fm1*x0[j]+gmh*v0[j]        # position x_ij(t+h)-x_ij(t) - h*v_ij(t) or -h*v_ij(t+h)
  end
  @inbounds for j=1:3
    delxv[3+j] = dfdt*x0[j]+dgdtm1*v0[j]    # velocity v_ij(t+h)-v_ij(t)
  end
  if debug
    delxv[7] = gamma
    delxv[8] = r
    delxv[9] = fm1
    delxv[10] = dfdt
    delxv[11] = gmh
    delxv[12] = dgdtm1
  end
  # Compute gradient analytically:
  compute_jacobian_gamma!(gamma,g0bs,g1bs,g2bs,g3bs,h1,h2,dfdt,fm1,gmh,dgdtm1,r0,r,r0inv,rinv,k,h,beta0,beta0inv,eta,sqb,zeta,x0,v0,delxv_jac,jac_mass,drift_first,debug)
end

function compute_jacobian_gamma!(gamma::T,g0::T,g1::T,g2::T,g3::T,h1::T,h2::T,dfdt::T,fm1::T,gmh::T,dgdtm1::T,r0::T,r::T,r0inv::T,rinv::T,k::T,h::T,beta::T,betainv::T,eta::T,
  sqb::T,zeta::T,x0::Array{T,1},v0::Array{T,1},delxv_jac::Array{T,2},jac_mass::Array{T,1},drift_first::Bool,debug::Bool) where {T <: Real}
# Computes Jacobian:
if drift_first
  # First, x0 derivatives:
  #  delxv[  j] = fm1*x0[j]+gmh*v0[j]        # position x_ij(t+h)-x_ij(t) - h*v_ij(t) or -h*v_ij(t+h)
  #  delxv[3+j] = dfdt*x0[j]+dgdtm1*v0[j]    # velocity v_ij(t+h)-v_ij(t)
  # First, the diagonal terms:
  # Now the off-diagonal terms:
  d   = (h + eta*g2 + 2*k*g3)*betainv
  c1 = d-r0*g3
  c2 = eta*g0+g1*zeta
  c3  = d*k+g1*r0^2
  c4 = eta*g1+2*g0*r0
  c13 = g1*h-g2*r0
#  c9  = g2*r-h1*k
#  c10 = c3*c9*rinv+g2*r0*h-k*(2*g2*h+3*g3*r0)*betainv
  c9 = 2*g2*h-3*g3*r0
  c10 = k*r0inv^4*(-g2*r0*h+k*c9*betainv-c3*c13*rinv)
  c24 = r0*(2*k*r0inv-beta)*betainv-g1*c3*rinv/g2
  h6 = H6(gamma,beta)
  # Derivatives of \delta x with respect to x0, v0, k & h:
  dfm1dxx = fm1*r0inv^3*c24
  dfm1dxv = -fm1*(g1*rinv+h*r0inv^3*c24)
  dfm1dvx = dfm1dxv
#  dfm1dvv = fm1*(2*betainv-g1*rinv*(d/g2-2*h)+h^2*r0inv^3*c24)
  dfm1dvv = fm1*rinv*(-r0*g2 + k*h6*betainv/g2 + h*(2*g1+h*r*c24*r0inv^3))
  dfm1dh  = fm1*(g1*rinv*(1/g2+2*k*r0inv-beta)-eta*r0inv^3*c24)
  dfm1dk  = fm1*(1/k+g1*c1*rinv*r0inv/g2-2*betainv*r0inv)
#  dfm1dk2  = 2g2*betainv-c1*g1*rinv
  h4 = -H1(gamma,beta)*beta
#  h5 = g1*g2-g3*(2+g0)
  h5 = H5(gamma,beta)
#  println("H5 : ",g1*g2-g3*(2+g0)," ",H2(gamma,beta)-2*G3(gamma,beta)," ",h5)
#  h6 = 2*g2^2-3*g1*g3
#  println("H6: ",2*g2^2-3*g1*g3," ",h6)
  dfm1dk2  = (r0*h4+k*h6)*betainv*rinv
#  println("dfm1dk2: ",dfm1dk2," ", (r0*h4+k*h6)*betainv*rinv)
  dgmhdxx = c10
  dgmhdxv =  -g2*k*c13*rinv*r0inv-h*c10
  dgmhdvx =  dgmhdxv
  # dgmhdvv =  -d*k*c13*rinv*r0inv+2*g2*h*k*c13*rinv*r0inv+k*c9*betainv*r0inv+h^2*c10
  h8 = H8(gamma,beta)
  dgmhdvv =  2*g2*h*k*c13*rinv*r0inv+h^2*c10+
              k*betainv*rinv*r0inv*(r0^2*h8-beta*h*r0*g2^2 + (h*k+eta*r0)*h6)
  dgmhdh  =  g2*k*r0inv+k*c13*rinv*r0inv+g2*k*(2*k*r0inv-beta)*c13*rinv*r0inv-eta*c10
  dgmhdk  =  r0inv*(k*c1*c13*rinv*r0inv+g2*h-g3*r0-k*c9*betainv*r0inv)
#  dgmhdk2  =  c1*c13*rinv-c9*betainv
  dgmhdk2 = -betainv*rinv*(h6*g3*k^2+eta*r0*(h6+g2*h4)+r0^2*g0*h5+k*eta*g2*h6+(g1*h6+g3*h4)*k*r0)
#  println("dgmhdk2: ",dgmhdk2," ",-betainv*rinv*(h6*g3*k^2+eta*r0*(h6+g2*h4)+r0^2*g0*h5+k*eta*g2*h6+(g1*h6+g3*h4)*k*r0))
  @inbounds for j=1:3
    # First, compute the \delta x-derivatives:
    delxv_jac[  j,  j] = fm1
    delxv_jac[  j,3+j] = gmh
    @inbounds for i=1:3
      delxv_jac[j,  i] += (dfm1dxx*x0[i]+dfm1dxv*v0[i])*x0[j] + (dgmhdxx*x0[i]+dgmhdxv*v0[i])*v0[j]
      delxv_jac[j,3+i] += (dfm1dvx*x0[i]+dfm1dvv*v0[i])*x0[j] + (dgmhdvx*x0[i]+dgmhdvv*v0[i])*v0[j]
    end
    delxv_jac[  j,  7] = dfm1dk*x0[j] + dgmhdk*v0[j]
    delxv_jac[  j,  8] = dfm1dh*x0[j] + dgmhdh*v0[j]
    # Compute the mass jacobian separately since otherwise cancellations happen in kepler_driftij_gamma:
    jac_mass[  j] = GNEWT^2*r0inv^2*(dfm1dk2*x0[j]+dgmhdk2*v0[j])
  end
  # Derivatives of \delta v with respect to x0, v0, k & h:
  c5 = (r0-k*g2)*rinv/g1
  c6 = (r0*g0-k*g2)*betainv
  c7 = g2*(1/g1+c2*rinv)
  c8 = (k*c6+r*r0+c3*c5)*r0inv^3
  c12 = g0*h-g1*r0
  c17 = r0-r-g2*k
  c18 = eta*g1+2*g2*k
  c20 = k*(g2*k+r)-g0*r0*zeta
  c21 = (g2*k-r0)*(beta*c3-k*g1*r)*betainv*rinv^2*r0inv^3/g1+eta*g1*rinv*r0inv^2-2r0inv^2
  c22 = rinv*(-g1-g0*g2/g1+g2*c2*rinv)
  c25 = k*rinv*r0inv^2*(-g2+k*(c13-g2*r0)*betainv*r0inv^2-c13*r0inv-c12*c3*rinv*r0inv^2+
                      c13*c2*c3*rinv^2*r0inv^2-c13*(k*(g2*k+r)-g0*r0*zeta)*betainv*rinv*r0inv^2)
  c26 = k*rinv^2*r0inv*(-g2*c12-g1*c13+g2*c13*c2*rinv)
  ddfdtdxx = dfdt*c21
  ddfdtdxv = dfdt*(c22-h*c21)
  ddfdtdvx = ddfdtdxv
#  ddfdtdvv = dfdt*(betainv*(1-c18*rinv)+d*(g1*c2-g0*r)*rinv^2/g1-h*(2c22-h*c21))
  ddfdtdvv = dfdt*((-beta*eta^2*g2^2-eta*k*h8-h6*k^2-2beta*eta*r0*g1*g2+(g2^2-3*g1*g3)*beta*k*r0- 
             beta*g1^2*r0^2)*betainv*rinv^2+(eta*g2^2)*rinv/g1 + (k*h8)*betainv*rinv/g1 - 2*h*c22 +h^2*c21)
  ddfdtdk  = dfdt*(1/k-betainv*r0inv-c17*betainv*rinv*r0inv-c1*(g1*c2-g0*r)*rinv^2*r0inv/g1)
#  ddfdtdk2  = -g1*(-betainv*r0inv-c17*betainv*rinv*r0inv-c1*(g1*c2-g0*r)*rinv^2*r0inv/g1)
  #ddfdtdk2  = -(g2*k-r0)*(g1*r-beta*c1)*betainv*rinv^2*r0inv
  ddfdtdk2 = -(g2*k-r0)*(beta*r0*(g3-g1*g2)-beta*eta*g2^2+k*H3(gamma,beta))*betainv*rinv^2*r0inv
  ddfdtdh  = dfdt*(g0*rinv/g1-c2*rinv^2-(2*k*r0inv-beta)*c22-eta*c21)
  dgdtmhdfdtm1dxx = c25
  dgdtmhdfdtm1dxv = c26-h*c25
  dgdtmhdfdtm1dvx = c26-h*c25
  dgdtmhdfdtm1dvv = d*k*rinv^3*r0inv*(c13*c2-r*c12)+k*(c13*(r0*g0-k*g2)-g2*r*r0)*betainv*rinv^2*r0inv-2*h*c26+h^2*c25
#  dgdtmhdfdtm1dvv = d*k*rinv^3*r0inv*k*(r0*(g1*g2-g3)+eta*g2^2+k*g2*g3)+k*(c13*(r0*g0-k*g2)-g2*r*r0)*betainv*rinv^2*r0inv-2*h*c26+h^2*c25
  dgdtmhdfdtm1dk = rinv*r0inv*(-k*(c13-g2*r0)*betainv*r0inv+c13-k*c13*c17*betainv*rinv*r0inv+k*c1*c12*rinv*r0inv-k*c1*c2*c13*rinv^2*r0inv)
  dgdtmhdfdtm1dk2 = -(c13-g2*r0)*betainv*r0inv-c13*c17*betainv*rinv*r0inv+c1*c12*rinv*r0inv-c1*c2*c13*rinv^2*r0inv
  #dgdtmhdfdtm1dk2 = g2*betainv+rinv*r0inv*(c1*c2+c13*((k*g2-r0)*betainv-c1*c2*rinv))
  dgdtmhdfdtm1dh = g1*k*rinv*r0inv+k*c12*rinv^2*r0inv-k*c2*c13*rinv^3*r0inv-(2*k*r0inv-beta)*c26-eta*c25
  @inbounds for j=1:3
    # Next, compute the \delta v-derivatives:
    delxv_jac[3+j,  j] = dfdt
    delxv_jac[3+j,3+j] = dgdtm1
    @inbounds for i=1:3
      delxv_jac[3+j,  i] += (ddfdtdxx*x0[i]+ddfdtdxv*v0[i])*x0[j] + (dgdtmhdfdtm1dxx*x0[i]+dgdtmhdfdtm1dxv*v0[i])*v0[j]
      delxv_jac[3+j,3+i] += (ddfdtdvx*x0[i]+ddfdtdvv*v0[i])*x0[j] + (dgdtmhdfdtm1dvx*x0[i]+dgdtmhdfdtm1dvv*v0[i])*v0[j]
    end
    delxv_jac[3+j,  7] = ddfdtdk*x0[j] + dgdtmhdfdtm1dk*v0[j]
    delxv_jac[3+j,  8] = ddfdtdh*x0[j] + dgdtmhdfdtm1dh*v0[j]
    # Compute the mass jacobian separately since otherwise cancellations happen in kepler_driftij_gamma:
    jac_mass[3+j] = GNEWT^2*r0inv*rinv*(ddfdtdk2*x0[j]+dgdtmhdfdtm1dk2*v0[j])
  end
  if debug
    # Now include derivatives of gamma, r, fm1, dfdt, gmh, and dgdtmhdfdtm1:
    @inbounds for i=1:3
      delxv_jac[ 7,i] = -sqb*rinv*((g2-h*c3*r0inv^3)*v0[i]+c3*x0[i]*r0inv^3); delxv_jac[7,3+i] = sqb*rinv*((-d+2*g2*h-h^2*c3*r0inv^3)*v0[i]+(-g2+h*c3*r0inv^3)*x0[i])
      delxv_jac[ 8,i] = (c20*betainv-c2*c3*rinv)*r0inv^3*x0[i]+((eta*g2+g1*r0)*rinv+h*r0inv^3*(c2*c3*rinv-c20*betainv))*v0[i]
#      delxv_jac[8,3+i] = (g1-g2*c2*rinv+h*r0inv^3*(c2*c3*rinv-c20*betainv))*x0[i]+(-2g1*h+c18*betainv+(2g2*h-d)*c2*rinv+h^2*r0inv^3*(c20*betainv-c2*c3*rinv))*v0[i]
#      delxv_jac[8,3+i] = ((g1*r0+eta*g2)*rinv+h*r0inv^3*(c2*c3*rinv-c20*betainv))*x0[i]+(-2g1*h+c18*betainv+(2g2*h-d)*c2*rinv+h^2*r0inv^3*(c20*betainv-c2*c3*rinv))*v0[i]
      drdv0x0 = (beta*g1*g2+((eta*g2+k*g3)*eta*g0*c3)*rinv*r0inv^3 + (g1*g0*(2k*eta^2*g2+3eta*k^2*g3))*betainv*rinv*r0inv^2- 
                          k*betainv*r0inv^3*(eta*g1*(eta*g2+k*g3)+g3*g0*r0^2*beta+2h*g2*k)+(g1*zeta)*rinv*((h*c3)*r0inv^3 - g2) - 
                         (eta*(beta*g2*g0*r0+k*g1^2)*(eta*g1+k*g2))*betainv*rinv*r0inv^2)
#      delxv_jac[8,3+i] = drdv0x0*x0[i]+ (k*betainv*rinv*(eta*(2g0*g3-g1*g2+g3) - h6*k^2 + (g2^2 - 2*g1*g3)*beta*k*r0) + h*drdv0x0)*v0[i]
      delxv_jac[8,3+i] = drdv0x0*x0[i]+ (k*betainv*rinv*(eta*(beta*g2*g3-h8) - h6*k^2 + (g2^2 - 2*g1*g3)*beta*k*r0) + h*drdv0x0)*v0[i]
#                         +(-2g1*h+c18*betainv+(2g2*h-d)*c2*rinv+h^2*r0inv^3*(c20*betainv-c2*c3*rinv))*v0[i]
#      delxv_jac[8,3+i] = ((eta*g2+g1*r0)*rinv+h*betainv*rinv*r0inv^3*((3g1*g3 - 2g2^2)*k^3 - k^2*eta*h8 + 
#            beta*k^2*(g2^2 - 3*g1*g3)*r0 - beta*r0^3 - k*g2*beta*(eta^2*g2 + 2eta*g1*r0 + g0*r0^2)))*x0[i]+(-2g1*h+c18*betainv+((g1*r0-g3*k-2*g0*h)*c2)*betainv*rinv + 
#         h^2*((2*g2^2 - 3*g1*g3)*k^3 + beta*r0^3 + k^2*(eta*(g1*g2 - 3*g0*g3) + beta*(3*g1*g3 - g2^2)*r0) + 
#         k*beta*g2*(eta*(eta*g2 + g1*r0) + r0*(eta*g1 + g0*r0)))*betainv*rinv*r0inv^3)*v0[i]
      delxv_jac[ 9,i] = dfm1dxx*x0[i]+dfm1dxv*v0[i]; delxv_jac[ 9,3+i]=dfm1dvx*x0[i]+dfm1dvv*v0[i]
      delxv_jac[10,i] = ddfdtdxx*x0[i]+ddfdtdxv*v0[i]; delxv_jac[10,3+i]=ddfdtdvx*x0[i]+ddfdtdvv*v0[i]
      delxv_jac[11,i] = dgmhdxx*x0[i]+dgmhdxv*v0[i]; delxv_jac[11,3+i]=dgmhdvx*x0[i]+dgmhdvv*v0[i]
      delxv_jac[12,i] = dgdtmhdfdtm1dxx*x0[i]+dgdtmhdfdtm1dxv*v0[i]; delxv_jac[12,3+i]=dgdtmhdfdtm1dvx*x0[i]+dgdtmhdfdtm1dvv*v0[i]
    end
    delxv_jac[ 7,7] = sqb*c1*r0inv*rinv; delxv_jac[7,8] = sqb*rinv*(1+eta*c3*r0inv^3+g2*(2k*r0inv-beta))
#    delxv_jac[ 8,7] = (c17*betainv+c1*c2*rinv)*r0inv
    delxv_jac[ 8,7] = betainv*r0inv*rinv*(-g2*r0^2*beta-eta*g1*g2*(k+beta*r0)+eta*g0*g3*(2*k+zeta)-
           g2^2*(beta*eta^2+2*k*zeta)+g1*g3*zeta*(3*k-beta*r0)); delxv_jac[8,8] = c2*rinv
    delxv_jac[ 8,8] = ((r0*g1+eta*g2)*rinv)*(beta-2*k*r0inv)+c2*rinv+eta*r0inv^3*(c2*c3*rinv-c20*betainv)
    delxv_jac[ 9,7] = dfm1dk; delxv_jac[ 9,8] = dfm1dh
    delxv_jac[10,7] = ddfdtdk; delxv_jac[10,8] = ddfdtdh
    delxv_jac[11,7] = dgmhdk; delxv_jac[11,8] = dgmhdh
    delxv_jac[12,7] = dgdtmhdfdtm1dk; delxv_jac[12,8] = dgdtmhdfdtm1dh
  end
else
  # Now compute the Kepler-Drift Jacobian terms:
  # First, x0 derivatives:
  #  delxv[  j] = fm1*x0[j]+gmh*v0[j]        # position x_ij(t+h)-x_ij(t) - h*v_ij(t) or -h*v_ij(t+h)
  #  delxv[3+j] = dfdt*x0[j]+dgdtm1*v0[j]    # velocity v_ij(t+h)-v_ij(t)
  # First, the diagonal terms:
  # Now the off-diagonal terms:
  d   = (h + eta*g2 + 2*k*g3)*betainv
  c1 = d-r0*g3
  c2 = eta*g0+g1*zeta
  c3  = d*k+g1*r0^2
  c4 = eta*g1+2*g0*r0
  c6 = (r0*g0-k*g2)*betainv
  c9  = g2*r-h1*k
  c14 = r0*g2-k*h1
  c15 = eta*h1+h2*r0
  c16 = eta*h2+g1*gamma*r0/sqb
  c17 = r0-r-g2*k
  c18 = eta*g1+2*g2*k
  c19 = 4*eta*h1+3*h2*r0
  c23 = h2*k-r0*g1
  c24 = r0*(2k*r0inv-beta)/beta-g1*c3*rinv/g2
  h6 = H6(gamma,beta)
  h8 = H8(gamma,beta)
  # Derivatives of \delta x with respect to x0, v0, k & h:
  dfm1dxx = k*rinv^3*betainv*r0inv^4*(k*h1*r^2*r0*(beta-2*k*r0inv)+beta*c3*(r*c23+c14*c2)+c14*r*(k*(r-g2*k)+g0*r0*zeta))
  dfm1dxv = k*rinv^2*r0inv*(k*(g2*h2+g1*h1)-2g1*g2*r0+g2*c14*c2*rinv)
  dfm1dvx = dfm1dxv
#  dfm1dvv = k*r0inv*rinv^2*betainv*(r*(2*g2*r0-4*h1*k)+d*beta*c23-c18*c14+d*beta*c14*c2*rinv)
  dfm1dvv = k*r0inv*rinv^2*betainv*(2eta*k*(g2*g3-g1*h1)+(3g3*h2-4h1*g2)*k^2 + 
    beta*g2*r0*(3h1*k-g2*r0)+c14*rinv*(-beta*g2^2*eta^2+eta*k*(2g0*g3-h2)-
     h6*k^2+(-2eta*g1*g2+k*(h1-2g1*g3))*beta*r0-beta*g1^2*r0^2))
  dfm1dh  = (g1*k-h2*k^2*r0inv-k*c14*c2*rinv*r0inv)*rinv^2
  dfm1dk  = rinv*r0inv*(4*h1*k^2*betainv*r0inv-k*h1-2*g2*k*betainv+c14-k*c14*c17*betainv*rinv*r0inv+
        k*(g1*r0-k*h2)*c1*rinv*r0inv-k*c14*c1*c2*rinv^2*r0inv)
#  dfm1dk2_old  = 4*h1*k*betainv*r0inv-h1-2*g2*betainv-c14*c17*betainv*rinv*r0inv+ (g1*r0-k*h2)*c1*rinv*r0inv-c14*c1*c2*rinv^2*r0inv
  # New expression for d(f-1-h \dot f)/dk with cancellations of higher order terms in gamma is:
  dfm1dk2  = betainv*r0inv*rinv^2*(r*(2eta*k*(g1*h1-g3*g2)+(4g2*h1-3g3*h2)*k^2-eta*r0*beta*g1*h1 + (g3*h2-4g2*h1)*beta*k*r0 + g2*h1*beta^2*r0^2) - 
  # In the following line I need to replace 3g0*g3-g1*g2 by -H8:
          c14*(-eta^2*beta*g2^2 - k*eta*h8 - k^2*h6 - eta*r0*beta*(g1*g2 + g0*g3) + 2*(h1 - g1*g3)*beta*k*r0 - (g2 - beta*g1*g3)*beta*r0^2))
#  println("dfm1dk2: old ",dfm1dk2_old," new: ",dfm1dk2)
  dgmhdxx = k*rinv*r0inv*(h2+k*c19*betainv*r0inv^2-c16*c3*rinv*r0inv^2+c2*c3*c15*(rinv*r0inv)^2-c15*(k*(g2*k+r)-g0*r0*zeta)*betainv*rinv*r0inv^2)
  dgmhdxv = k*rinv^2*(h1*r-g2*c16-g1*c15+g2*c2*c15*rinv)
  dgmhdvx = dgmhdxv
#  dgmhdvv = k*rinv^2*(-d*c16-c15*c18*betainv+r*c19*betainv+d*c2*c15*rinv)
  dgmhdvv = k*betainv*rinv^2*(((2*eta^2*(g1*h1-g2*g3)+eta*k*(4g2*h1-3h2*g3)+r0*eta*(4g0*h1-2g1*g3)+ 
  # In the following lines I need to replace g1*g2-3g0*g3-g1*g2 by H8:
           3r0*k*((g1+beta*g3)*h1-g3*g2)+(g0*h8-beta*g1*(g2^2+g1*g3))*r0^2)) + 
           c15*rinv*(-beta*g2^2*eta^2-eta*k*h8-h6*k^2+(-2eta*g1*g2+k*(g2^2-3g1*g3))*beta*r0-beta*g1^2*r0^2))
  dgmhdk  = rinv*(k*c1*c16*rinv*r0inv+c15-k*c15*c17*betainv*rinv*r0inv-k*c19*betainv*r0inv-k*c1*c2*c15*rinv^2*r0inv)
#  dgmhdk2_old  = c1*c16*rinv-c15*c17*betainv*rinv-c19*betainv-c1*c2*c15*rinv^2
  h7 = H7(gamma,beta)
  dgmhdk2 =  betainv*rinv^2*(r*(2eta^2*(g3*g2-g1*h1) + eta*k*(3g3*h2 - 4g2*h1) +
    r0*eta*(beta*g3*(g1*g2 + g0*g3) - 2g0*h6) + (-h6*(g1 + beta*g3) + g2*(2g3 - h2))*r0*k + 
    (h7 - beta^2*g1*g3^2)*r0^2)- c15*(-beta*eta^2*g2^2 + eta*k*(-h2 + 2g0*g3) - h6*k^2 - 
    r0*eta*beta*(h2 + 2g0*g3) + 2beta*(2*h1 - g2^2)*r0*k + beta*(beta*g1*g3 - g2)*r0^2))
#  println("gmhgdot: old ",dgmhdk2_old," new: ",dgmhdk2)
  dgmhdh  = k*rinv^3*(r*c16-c2*c15)
  @inbounds for j=1:3
    # First, compute the \delta x-derivatives:
    delxv_jac[  j,  j] = fm1
    delxv_jac[  j,3+j] = gmh
    @inbounds for i=1:3
      delxv_jac[j,  i] += (dfm1dxx*x0[i]+dfm1dxv*v0[i])*x0[j] + (dgmhdxx*x0[i]+dgmhdxv*v0[i])*v0[j]
      delxv_jac[j,3+i] += (dfm1dvx*x0[i]+dfm1dvv*v0[i])*x0[j] + (dgmhdvx*x0[i]+dgmhdvv*v0[i])*v0[j]
    end
    delxv_jac[  j,  7] = dfm1dk*x0[j] + dgmhdk*v0[j]
    delxv_jac[  j,  8] = dfm1dh*x0[j] + dgmhdh*v0[j] 
    jac_mass[  j] = GNEWT^2*rinv*r0inv*(dfm1dk2*x0[j]+dgmhdk2*v0[j])
  end
  # Derivatives of \delta v with respect to x0, v0, k & h:
  c5 = (r0-k*g2)*rinv/g1
  c7 = g2*(1/g1+c2*rinv)
  c8 = (k*c6+r*r0+c3*c5)*r0inv^3
  c12 = g0*h-g1*r0
  c20 = k*(g2*k+r)-g0*r0*zeta
  ddfdtdxx = dfdt*(eta*g1*rinv-2-g0*c3*rinv*r0inv/g1+c2*c3*r0inv*rinv^2-k*(k*g2-r0)*betainv*rinv*r0inv)*r0inv^2
  ddfdtdxv = -dfdt*(g0*g2/g1+(r0*g1+eta*g2)*rinv)*rinv
  ddfdtdvx = ddfdtdxv
  ddfdtdvv = dfdt*(betainv-d*g0*rinv/g1-c18*betainv*rinv+d*c2*rinv^2)
  ddfdtdvv = -k*rinv*r0inv*((eta*g2^2)*rinv+k*h8*betainv*rinv+ 
       g1*(-beta*eta^2*g2^2-eta*k*h8- h6*k^2 + (-2eta*g1*g2+(h1-2g1*g3)*k)*beta*r0 - 
       beta*g1^2*r0^2)*betainv*rinv^2)
  ddfdtdk  = dfdt*(1/k+c1*(r0-g2*k)*r0inv*rinv^2/g1-betainv*r0inv*(1+c17*rinv))
  ddfdtdk2  = -g1*(c1*(r0-g2*k)*r0inv*rinv^2/g1-betainv*r0inv*(1+c17*rinv))
#  ddfdtdk2  = r0inv*(g1*c17*betainv*rinv+g1*betainv-g1*c1*c2*rinv^2-c1*g0*rinv)
  ddfdtdk2  = (r0-g2*k)*betainv*r0inv*rinv^2*(-eta*beta*g2^2+(g1*g2-3g3)*k+(g3-g1*g2)*beta*r0)
  ddfdtdh  = dfdt*(r0-g2*k)*rinv^2/g1
  dgdotm1dxx = rinv^2*r0inv^3*((eta*g2+g1*r0)*k*c3*rinv+g2*k*(k*(g2*k-r)-g0*r0*zeta)*betainv)
  dgdotm1dxv = k*g2*rinv^3*(r*g1+r0*g1+eta*g2)
  dgdotm1dvx = dgdotm1dxv
#  dgdotm1dvv = k*rinv^2*(d*g1+g2*c18*betainv-2*r*g2*betainv-d*g2*c2*rinv)
  dgdotm1dvv = k*betainv*rinv^3*(eta^2*beta*g2^3+eta*k*(3g2*g3-g1*g2^2)+3r0*eta*beta*g1*g2^2 +
         r0*k*(-g0*h6+3beta*g1*g2*g3)+beta*g2*(g0*g2+g1^2)*r0^2)
  dgdotm1dk = rinv*r0inv*(-r0*g2+g2*k*(r+r0-g2*k)*betainv*rinv-k*g1*c1*rinv+k*g2*c1*c2*rinv^2)
  dgdotm1dk2 = rinv*(g2*(r+r0-g2*k)*betainv-g1*c1+g2*c1*c2*rinv)
  dgdotm1dk2 = betainv*rinv^2*(-beta*eta^2*g2^3+eta*k*g2*(g1*g2-3*g3)+eta*r0*beta*g2*(-2g1*g2+g3)+ 
       (g2^2*(1+g0)-3*g1*g3)*r0*k + beta*(g1*g3 - g1^2*g2)*r0^2)
  dgdotm1dh = k*rinv^3*(g2*c2-r*g1)
  @inbounds for j=1:3
    # Next, compute the \delta v-derivatives:
    delxv_jac[3+j,  j] = dfdt
    delxv_jac[3+j,3+j] = dgdtm1
    @inbounds for i=1:3
      delxv_jac[3+j,  i] += (ddfdtdxx*x0[i]+ddfdtdxv*v0[i])*x0[j] + (dgdotm1dxx*x0[i]+dgdotm1dxv*v0[i])*v0[j]
      delxv_jac[3+j,3+i] += (ddfdtdvx*x0[i]+ddfdtdvv*v0[i])*x0[j] + (dgdotm1dvx*x0[i]+dgdotm1dvv*v0[i])*v0[j]
    end
    delxv_jac[3+j,  7] = ddfdtdk*x0[j] + dgdotm1dk*v0[j]
    delxv_jac[3+j,  8] = ddfdtdh*x0[j] + dgdotm1dh*v0[j]
    jac_mass[3+j] = GNEWT^2*rinv*r0inv*(ddfdtdk2*x0[j]+dgdotm1dk2*v0[j])
  end
  if debug
    # Now include derivatives of gamma, r, fm1, gmh, dfdt, and dgdtmhdfdtm1:
    @inbounds for i=1:3
      delxv_jac[ 7,i] = -sqb*rinv*(g2*v0[i]+c3*x0[i]*r0inv^3); delxv_jac[7,3+i] = -sqb*rinv*(d*v0[i]+g2*x0[i])
      delxv_jac[ 8,i] = (c20*betainv-c2*c3*rinv)*r0inv^3*x0[i]+((r0*g1+eta*g2)*rinv)*v0[i]
      delxv_jac[8,3+i] = (c18*betainv-d*c2*rinv)*v0[i]+((r0*g1+eta*g2)*rinv)*x0[i]
#      delxv_jac[8,3+i] = (g2*(2k*(r-r0)+beta*(r0^2+eta^2*g2)-zeta*eta*g1) + c2*g3*(2k+zeta))*betainv*rinv*r0inv*v0[i]+((r0*g1+eta*g2)*rinv)*x0[i]
      delxv_jac[ 9,i] = dfm1dxx*x0[i]+dfm1dxv*v0[i]; delxv_jac[ 9,3+i]=dfm1dvx*x0[i]+dfm1dvv*v0[i]
      delxv_jac[10,i] = ddfdtdxx*x0[i]+ddfdtdxv*v0[i]; delxv_jac[10,3+i]=ddfdtdvx*x0[i]+ddfdtdvv*v0[i]
      delxv_jac[11,i] = dgmhdxx*x0[i]+dgmhdxv*v0[i]; delxv_jac[11,3+i]=dgmhdvx*x0[i]+dgmhdvv*v0[i]
      delxv_jac[12,i] = dgdotm1dxx*x0[i]+dgdotm1dxv*v0[i]; delxv_jac[12,3+i]=dgdotm1dvx*x0[i]+dgdotm1dvv*v0[i]
    end
    delxv_jac[ 7,7] = sqb*c1*r0inv*rinv; delxv_jac[7,8] = sqb*rinv
#    delxv_jac[ 8,7] = (c17*betainv+c1*c2*rinv)*r0inv; delxv_jac[8,8] = c2*rinv
    delxv_jac[ 8,7] = betainv*r0inv*rinv*(-g2*r0^2*beta-eta*g1*g2*(k+beta*r0)+eta*g0*g3*(2*k+zeta)-
           g2^2*(beta*eta^2+2*k*zeta)+g1*g3*zeta*(3*k-beta*r0)); delxv_jac[8,8] = c2*rinv
    delxv_jac[ 9,7] = dfm1dk; delxv_jac[ 9,8] = dfm1dh
    delxv_jac[10,7] = ddfdtdk; delxv_jac[10,8] = ddfdtdh
    delxv_jac[11,7] = dgmhdk; delxv_jac[11,8] = dgmhdh
    delxv_jac[12,7] = dgdotm1dk; delxv_jac[12,8] = dgdotm1dh
  end
end
#return delxv_jac::Array{T,2}
return
end
