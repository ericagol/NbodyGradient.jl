

struct GHcoeff{T<:AbstractFloat}
  G3_coeff::Vector{T}
  H1_coeff::Vector{T}
  H2_coeff::Vector{T}
  comp_coeff::Bool
  ITMAX::Int64
  function GHcoeff(::Type{T},ITMAX::Integer) where T<:AbstractFloat
     G3_coeff = zeros(T,ITMAX+1)
     H1_coeff = zeros(T,ITMAX+1)
     H2_coeff = zeros(T,ITMAX+1)
     G3_coeff[1] = one(T); H1_coeff[1] =one(T); H2_coeff[1] =one(T)
     for n=1:ITMAX
       G3_coeff[n+1] = G3_coeff[n]/((2n+3)*(2n+2))
       H1_coeff[n+1] = H1_coeff[n]*(n+1)/((2n+4)*(2n+3)*n)
       H2_coeff[n+1] = H2_coeff[n]/(n*(4n+6))
     end
     return new{T}(G3_coeff,H1_coeff,H2_coeff,true,ITMAX)
  end
end

function GHcomp!(g::GHcoeff{T},gamma::T,beta::T,sqb::T,gc::T,g3h1h2::Vector{T}) where T<:AbstractFloat
  if !g.comp_coeff
    g=GHcoeff(T,100)
  end
  x2 = -sign(beta)*gamma^2
  xn = one(T)
  G3 = g.G3_coeff[1]; G31 = 2G3; G32 = 2G3
  H1 = g.H1_coeff[1]; H11 = 2H1; H12 = 2H1
  H2 = g.H2_coeff[1]; H21 = 2H2; H22 = 2H2
  n = 0
  if abs(gamma) < gc
    while true
      xn *= x2
      G32 = G31; G31 = G3; H12 = H11; H11 = H1; H22 = H21; H21 = H2
      n += 1
      G3 += g.G3_coeff[n+1]*xn
      H1 += g.H1_coeff[n+1]*xn
      H2 += g.H2_coeff[n+1]*xn
      if n >= g.ITMAX || ((G3 == G31 || G3 == G32) && (H1 == H11 ||
                            H1 == H12) && (H2 == H21 || H2 == H22))
        break
      end
    end
    G3 *= gamma^3/(6*abs(beta)*sqb)
    H1 *= gamma^4/(12*beta^2)
    H2 *= gamma^3/(3*abs(beta)*sqb)
  else
    if beta >= 0
      # Change this to pass through sin(gamma) and cos(gamma) if already computed:
      G3 = (gamma-sin(gamma))/(sqb*beta)
      H1 = (4sin(0.5*gamma)^2 -gamma*sin(gamma))/beta^2
      H2 = (sin(gamma)-gamma*cos(gamma))/(sqb*beta)
    else
      # Change this to pass through sinh(gamma) and cosh(gamma) if already computed:
      G3 = (gamma-sinh(gamma))/(sqb*beta)
      H1 = (-4sinh(0.5*gamma)^2+gamma*sinh(gamma))/beta^2
      H2 = (sinh(gamma)-gamma*cosh(gamma))/(sqb*beta)
    end
  end
  g3h1h2[1]=G3; g3h1h2[2]=H1; g3h1h2[3]=H2
  return
end

# To find T, use eltype(typeof(g.G3_coeff)) where g is an instance of GHcoeff

function test_GHcoeff() where T<: AbstractFloat
   ghcb = GHcoeff(BigFloat,50)
   gamma_list = [-big(5.0),big(5.0),-big(0.5),big(0.5),-big(0.005),big(0.005)]
   beta_list  = [-big(5.0),big(5.0),-big(0.5),big(0.5),-big(0.005),big(0.005)]
#   gamma = -big(0.5); beta = big(0.5); sqb = sqrt(beta)
   # First compute using trig functions:
   for i=1:6, j=1:6
     gamma = gamma_list[i]; beta = beta_list[j]; sqb = sqrt(abs(beta))
     G3t,H1t,H2t = GHcomp(ghcb,gamma,beta,sqb,big(0.0)) 
     # Next compute using series:
     G3s,H1s,H2s = GHcomp(ghcb,gamma,beta,sqb,big(10.0))
     println("Differences, G3: ",convert(Float64,G3t-G3s),
             " H1: ",convert(Float64,H1t-H1s)," H2: ",convert(Float64,H2t-H2s))
   end
   return
end
