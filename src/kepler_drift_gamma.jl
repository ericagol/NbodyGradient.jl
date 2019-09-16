  

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

  function delx_delv(input::Array{T,1}) # input = x0,v0,k,h,drift_first
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
#  if typeof(gamma) ==  Float64
#    println("s: ",gamma/sqb)
#  end
  # Set up a single output array for delx and delv:
  if debug
    delxv = zeros(T,12)
  else
    delxv = zeros(T,6)
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
  g0bs = one(T)-beta0*g2bs
  g3bs = g3(gamma,beta0)
  h1 = zero(T); h2 = zero(T)
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
  if grad == true && dlnq == 0.0
    # Compute gradient analytically:
    delxv_jac = compute_jacobian_gamma(gamma,g0bs,g1bs,g2bs,g3bs,h1,h2,dfdt,fm1,gmh,dgdtm1,r0,r,r0inv,rinv,k,h,beta0,beta0inv,eta,sqb,zeta,x0,v0,drift_first,debug)
  end
  return delxv::Array{T,1}
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

function compute_jacobian_gamma(gamma::T,g0::T,g1::T,g2::T,g3::T,h1::T,h2::T,dfdt::T,fm1::T,gmh::T,dgdtm1::T,r0::T,r::T,r0inv::T,rinv::T,k::T,h::T,beta::T,betainv::T,eta::T,
  sqb::T,zeta::T,x0::Array{T,1},v0::Array{T,1},drift_first::Bool,debug::Bool) where {T <: Real}
# Computes Jacobian:
if debug
  delxv_jac = zeros(T,12,8)
else
  delxv_jac = zeros(T,6,8)
end
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
  # Derivatives of \delta x with respect to x0, v0, k & h:
  dfm1dxx = fm1*r0inv^3*c24
  dfm1dxv = -fm1*(g1*rinv+h*r0inv^3*c24)
  dfm1dvx = dfm1dxv
  dfm1dvv = fm1*(2*betainv-g1*rinv*(d/g2-2*h)+h^2*r0inv^3*c24)
  dfm1dh  = fm1*(g1*rinv*(1/g2+2*k*r0inv-beta)-eta*r0inv^3*c24)
  dfm1dk  = fm1*(1/k+g1*c1*rinv*r0inv/g2-2*betainv*r0inv)
  dgmhdxx = c10
  dgmhdxv =  -g2*k*c13*rinv*r0inv-h*c10
  dgmhdvx =  dgmhdxv
  dgmhdvv =  -d*k*c13*rinv*r0inv+2*g2*h*k*c13*rinv*r0inv+k*c9*betainv*r0inv+h^2*c10
  dgmhdh  =  g2*k*r0inv+k*c13*rinv*r0inv+g2*k*(2*k*r0inv-beta)*c13*rinv*r0inv-eta*c10
  dgmhdk  =  r0inv*(k*c1*c13*rinv*r0inv+g2*h-g3*r0-k*c9*betainv*r0inv)
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
  ddfdtdvv = dfdt*(betainv*(1-c18*rinv)+d*(g1*c2-g0*r)*rinv^2/g1-h*(2c22-h*c21))
  ddfdtdk  = dfdt*(1/k-betainv*r0inv-c17*betainv*rinv*r0inv-c1*(g1*c2-g0*r)*rinv^2*r0inv/g1)
  ddfdtdh  = dfdt*(g0*rinv/g1-c2*rinv^2-(2*k*r0inv-beta)*c22-eta*c21)
  dgdtmhdfdtm1dxx = c25
  dgdtmhdfdtm1dxv = c26-h*c25
  dgdtmhdfdtm1dvx = c26-h*c25
  dgdtmhdfdtm1dvv = d*k*rinv^3*r0inv*(c13*c2-r*c12)+k*(c13*(r0*g0-k*g2)-g2*r*r0)*betainv*rinv^2*r0inv-2*h*c26+h^2*c25
  dgdtmhdfdtm1dk = rinv*r0inv*(-k*(c13-g2*r0)*betainv*r0inv+c13-k*c13*c17*betainv*rinv*r0inv+k*c1*c12*rinv*r0inv-k*c1*c2*c13*rinv^2*r0inv)
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
  end
  if debug
    # Now include derivatives of gamma, r, fm1, dfdt, gmh, and dgdtmhdfdtm1:
    @inbounds for i=1:3
      delxv_jac[ 7,i] = -sqb*rinv*((g2-h*c3*r0inv^3)*v0[i]+c3*x0[i]*r0inv^3); delxv_jac[7,3+i] = sqb*rinv*((-d+2*g2*h-h^2*c3*r0inv^3)*v0[i]+(-g2+h*c3*r0inv^3)*x0[i])
      delxv_jac[ 8,i] = (c20*betainv-c2*c3*rinv)*r0inv^3*x0[i]+(g1-c2*g2*rinv+h*r0inv^3*(c2*c3*rinv-c20*betainv))*v0[i]
      delxv_jac[8,3+i] = (g1-g2*c2*rinv+h*r0inv^3*(c2*c3*rinv-c20*betainv))*x0[i]+(-2g1*h+c18*betainv+(2g2*h-d)*c2*rinv+h^2*r0inv^3*(c20*betainv-c2*c3*rinv))*v0[i]
      delxv_jac[ 9,i] = dfm1dxx*x0[i]+dfm1dxv*v0[i]; delxv_jac[ 9,3+i]=dfm1dvx*x0[i]+dfm1dvv*v0[i]
      delxv_jac[10,i] = ddfdtdxx*x0[i]+ddfdtdxv*v0[i]; delxv_jac[10,3+i]=ddfdtdvx*x0[i]+ddfdtdvv*v0[i]
      delxv_jac[11,i] = dgmhdxx*x0[i]+dgmhdxv*v0[i]; delxv_jac[11,3+i]=dgmhdvx*x0[i]+dgmhdvv*v0[i]
      delxv_jac[12,i] = dgdtmhdfdtm1dxx*x0[i]+dgdtmhdfdtm1dxv*v0[i]; delxv_jac[12,3+i]=dgdtmhdfdtm1dvx*x0[i]+dgdtmhdfdtm1dvv*v0[i]
    end
    delxv_jac[ 7,7] = sqb*c1*r0inv*rinv; delxv_jac[7,8] = sqb*rinv*(1+eta*c3*r0inv^3+g2*(2k*r0inv-beta))
    delxv_jac[ 8,7] = (c17*betainv+c1*c2*rinv)*r0inv
    delxv_jac[ 8,8] = (g1-g2*c2*rinv)*(beta-2*k*r0inv)+c2*rinv+eta*r0inv^3*(c2*c3*rinv-c20*betainv)
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
  # Derivatives of \delta x with respect to x0, v0, k & h:
  dfm1dxx = k*rinv^3*betainv*r0inv^4*(k*h1*r^2*r0*(beta-2*k*r0inv)+beta*c3*(r*c23+c14*c2)+c14*r*(k*(r-g2*k)+g0*r0*zeta))
  dfm1dxv = k*rinv^2*r0inv*(k*(g2*h2+g1*h1)-2g1*g2*r0+g2*c14*c2*rinv)
  dfm1dvx = dfm1dxv
  dfm1dvv = k*r0inv*rinv^2*betainv*(r*(2*g2*r0-4*h1*k)+d*beta*c23-c18*c14+d*beta*c14*c2*rinv)
  dfm1dh  = (g1*k-h2*k^2*r0inv-k*c14*c2*rinv*r0inv)*rinv^2
  dfm1dk  = rinv*r0inv*(4*h1*k^2*betainv*r0inv-k*h1-2*g2*k*betainv+c14-k*c14*c17*betainv*rinv*r0inv+
        k*(g1*r0-k*h2)*c1*rinv*r0inv-k*c14*c1*c2*rinv^2*r0inv)
  dgmhdxx = k*rinv*r0inv*(h2+k*c19*betainv*r0inv^2-c16*c3*rinv*r0inv^2+c2*c3*c15*(rinv*r0inv)^2-c15*(k*(g2*k+r)-g0*r0*zeta)*betainv*rinv*r0inv^2)
  dgmhdxv = k*rinv^2*(h1*r-g2*c16-g1*c15+g2*c2*c15*rinv)
  dgmhdvx = dgmhdxv
  dgmhdvv = k*rinv^2*(-d*c16-c15*c18*betainv+r*c19*betainv+d*c2*c15*rinv)
  dgmhdk  = rinv*(k*c1*c16*rinv*r0inv+c15-k*c15*c17*betainv*rinv*r0inv-k*c19*betainv*r0inv-k*c1*c2*c15*rinv^2*r0inv)
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
  end
  # Derivatives of \delta v with respect to x0, v0, k & h:
  c5 = (r0-k*g2)*rinv/g1
  c7 = g2*(1/g1+c2*rinv)
  c8 = (k*c6+r*r0+c3*c5)*r0inv^3
  c12 = g0*h-g1*r0
  c20 = k*(g2*k+r)-g0*r0*zeta
  ddfdtdxx = dfdt*(eta*g1*rinv-2-g0*c3*rinv*r0inv/g1+c2*c3*r0inv*rinv^2-k*(k*g2-r0)*betainv*rinv*r0inv)*r0inv^2
  ddfdtdxv = -dfdt*(g0*g2/g1+g1-g2*c2*rinv)*rinv
  ddfdtdvx = ddfdtdxv
  ddfdtdvv = dfdt*(betainv-d*g0*rinv/g1-c18*betainv*rinv+d*c2*rinv^2)
  ddfdtdk  = dfdt*(1/k+c1*(g0*r-g1*c2)*r0inv*rinv^2/g1-betainv*r0inv*(1+c17*rinv))
  ddfdtdh  = dfdt*(g0/g1-c2*rinv)*rinv
  dgdotm1dxx = rinv^2*r0inv^3*((eta*g2+g1*r0)*k*c3*rinv+g2*k*(k*(g2*k-r)-g0*r0*zeta)*betainv)
  dgdotm1dxv = k*g2*rinv^3*(2*r*g1-g2*c2)
  dgdotm1dvx = dgdotm1dxv
  dgdotm1dvv = k*rinv^2*(d*g1+g2*c18*betainv-2*r*g2*betainv-d*g2*c2*rinv)
  dgdotm1dk = rinv*r0inv*(-r0*g2+g2*k*(r+r0-g2*k)*betainv*rinv-k*g1*c1*rinv+k*g2*c1*c2*rinv^2)
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
  end
  if debug
    # Now include derivatives of gamma, r, fm1, gmh, dfdt, and dgdtmhdfdtm1:
    @inbounds for i=1:3
      delxv_jac[ 7,i] = -sqb*rinv*(g2*v0[i]+c3*x0[i]*r0inv^3); delxv_jac[7,3+i] = -sqb*rinv*(d*v0[i]+g2*x0[i])
      delxv_jac[ 8,i] = (c20*betainv-c2*c3*rinv)*r0inv^3*x0[i]+(g1-c2*g2*rinv)*v0[i]
      delxv_jac[8,3+i] = (c18*betainv-d*c2*rinv)*v0[i]+(g1-g2*c2*rinv)*x0[i]
      delxv_jac[ 9,i] = dfm1dxx*x0[i]+dfm1dxv*v0[i]; delxv_jac[ 9,3+i]=dfm1dvx*x0[i]+dfm1dvv*v0[i]
      delxv_jac[10,i] = ddfdtdxx*x0[i]+ddfdtdxv*v0[i]; delxv_jac[10,3+i]=ddfdtdvx*x0[i]+ddfdtdvv*v0[i]
      delxv_jac[11,i] = dgmhdxx*x0[i]+dgmhdxv*v0[i]; delxv_jac[11,3+i]=dgmhdvx*x0[i]+dgmhdvv*v0[i]
      delxv_jac[12,i] = dgdotm1dxx*x0[i]+dgdotm1dxv*v0[i]; delxv_jac[12,3+i]=dgdotm1dvx*x0[i]+dgdotm1dvv*v0[i]
    end
    delxv_jac[ 7,7] = sqb*c1*r0inv*rinv; delxv_jac[7,8] = sqb*rinv
    delxv_jac[ 8,7] = (c17*betainv+c1*c2*rinv)*r0inv; delxv_jac[8,8] = c2*rinv
    delxv_jac[ 9,7] = dfm1dk; delxv_jac[ 9,8] = dfm1dh
    delxv_jac[10,7] = ddfdtdk; delxv_jac[10,8] = ddfdtdh
    delxv_jac[11,7] = dgmhdk; delxv_jac[11,8] = dgmhdh
    delxv_jac[12,7] = dgdotm1dk; delxv_jac[12,8] = dgdotm1dh
  end
end
return delxv_jac
end
