# Tests the routine phisalpha jacobian.  This routine
# computes the force gradient correction after Dehnen & Hernandez (2017).

#include("../src/ttv.jl")

#function phisalpha!(x::Array{Float64,2},v::Array{Float64,2},h::Float64,m::Array{Float64,1},alpha::Float64,n::Int64,jac_step::Array{Float64,4})
#function phisalpha!(x,v,h,m,alpha,n,jac_step)


# Next, try computing three-body Keplerian Jacobian:

@testset "phisalpha" begin
n = 3
t0 = 7257.93115525
h  = 0.05
tmax = 600.0
dlnq = big(1e-15)

elements = readdlm("elements.txt",',')
#elements[2,1] = 1.0
#elements[3,1] = 1.0

m =zeros(n)
x0=zeros(3,n)
v0=zeros(3,n)
alpha = 2.0

# Define which pairs will have impulse rather than -drift+Kepler:
pair = zeros(Bool,n,n)
# We want Keplerian between star & planets, and impulses between
# planets.  Impulse is indicated with 'true', -drift+Kepler with 'false':
#for i=2:n
#  pair[1,i] = false
#  # We don't need to define this, but let's anyways:
#  pair[i,1] = false
#end
println("pair: ",pair)

jac_step = Matrix{Float64}(I,7*n,7*n)

for k=1:n
  m[k] = elements[k,1]
end
m0 = copy(m)

x0,v0 = init_nbody(elements,t0,n)

# Tilt the orbits a bit:
x0[2,1] = 5e-1*sqrt(x0[1,1]^2+x0[3,1]^2)
x0[2,2] = -5e-1*sqrt(x0[1,2]^2+x0[3,2]^2)
x0[2,3] = -5e-1*sqrt(x0[1,2]^2+x0[3,2]^2)
v0[2,1] = 5e-1*sqrt(v0[1,1]^2+v0[3,1]^2)
v0[2,2] = -5e-1*sqrt(v0[1,2]^2+v0[3,2]^2)
v0[2,3] = -5e-1*sqrt(v0[1,2]^2+v0[3,2]^2)

# Take a step:
dh17!(x0,v0,h,m,n,pair)

# Now, copy these to compute Jacobian (so that I don't step
# x0 & v0 forward in time):
x = copy(x0); v = copy(v0); m = copy(m0)
# Compute jacobian exactly:
dqdt_phi = zeros(7*n)
xerror = zeros(3,n); verror = zeros(3,n)
phisalpha!(x,v,xerror,verror,h,m,alpha,n,jac_step,dqdt_phi,pair)


# Now compute numerical derivatives, using BigFloat to avoid
# round-off errors:
jac_step_num = zeros(BigFloat,7*n,7*n)
# Save these so that I can compute derivatives numerically:
xsave = big.(x0)
vsave = big.(v0)
msave = big.(m0)
hbig = big(h)
abig = big(alpha)
# Carry out step using BigFloat for extra precision:
phisalpha!(xsave,vsave,hbig,msave,abig,n,pair)
xbig = big.(x0)
vbig = big.(v0)
mbig = big.(m0)
# Compute numerical derivatives wrt time:
dqdt_num = zeros(BigFloat,7*n)
# Vary time:
phisalpha!(xbig,vbig,hbig,mbig,abig,n,pair)
# Initial positions, velocities & masses:
xbig .= big.(x0)
vbig .= big.(v0)
mbig .= big.(m0)
hbig = big(h)
dq = dlnq * hbig
hbig += dq
phisalpha!(xbig,vbig,hbig,mbig,abig,n,pair)
# Now x & v are final positions & velocities after time step
for i=1:n, k=1:3
  dqdt_num[(i-1)*7+  k] = (xbig[k,i]-xsave[k,i])/dq
  dqdt_num[(i-1)*7+3+k] = (vbig[k,i]-vsave[k,i])/dq
end
hbig = big(h)
# Vary the initial parameters of planet j:
for j=1:n
  # Vary the initial phase-space elements:
  for jj=1:3
  # Initial positions, velocities & masses:
    xbig .= big.(x0)
    vbig .= big.(v0)
    mbig .= big.(m0)
    dq = dlnq * xbig[jj,j]
    if xbig[jj,j] != 0.0
      xbig[jj,j] +=  dq
    else
      dq = dlnq
      xbig[jj,j] = dq
    end
    phisalpha!(xbig,vbig,hbig,mbig,abig,n,pair)
  # Now x & v are final positions & velocities after time step
    for i=1:n
      for k=1:3
        jac_step_num[(i-1)*7+  k,(j-1)*7+jj] = (xbig[k,i]-xsave[k,i])/dq
        jac_step_num[(i-1)*7+3+k,(j-1)*7+jj] = (vbig[k,i]-vsave[k,i])/dq
      end
    end
    xbig .= big.(x0)
    vbig .= big.(v0)
    mbig .= big.(m0)
    dq = dlnq * vbig[jj,j]
    if vbig[jj,j] != 0.0
      vbig[jj,j] +=  dq
    else
      dq = dlnq
      vbig[jj,j] = dq
    end
    phisalpha!(xbig,vbig,hbig,mbig,abig,n,pair)
    for i=1:n
      for k=1:3
        jac_step_num[(i-1)*7+  k,(j-1)*7+3+jj] = (xbig[k,i]-xsave[k,i])/dq
        jac_step_num[(i-1)*7+3+k,(j-1)*7+3+jj] = (vbig[k,i]-vsave[k,i])/dq
      end
    end
  end
# Now vary mass of planet:
  xbig .= big.(x0)
  vbig .= big.(v0)
  mbig .= big.(m0)
  dq = mbig[j]*dlnq
  mbig[j] += dq
  phisalpha!(xbig,vbig,hbig,mbig,abig,n,pair)
  for i=1:n
    for k=1:3
      jac_step_num[(i-1)*7+  k,j*7] = (xbig[k,i]-xsave[k,i])/dq
      jac_step_num[(i-1)*7+3+k,j*7] = (vbig[k,i]-vsave[k,i])/dq
    end
    # Mass unchanged -> identity
    jac_step_num[7*i,7*i] = big(1.0)
  end
end

# Now, compare the results:
#println(jac_step)
#println(convert(Array{Float64,2},jac_step_num))

#for j=1:3
#  for i=1:7
#    for k=1:3
#      println(jac_step[(j-1)*7+i,(k-1)*7+1:7*k]," ",jac_step_num[(j-1)*7+i,(k-1)*7+1:7*k]," ",jac_step_num[(j-1)*7+i,(k-1)*7+1:7*k]./jac_step[(j-1)*7+i,(k-1)*7+1:7*k]-1.)
#    end
#  end
#end

jacmax = 0.0
for i=1:7, j=1:3, k=1:7, l=1:3
  if jac_step[(j-1)*7+i,(l-1)*7+k] != 0 && jac_step_num[(j-1)*7+i,(l-1)*7+k] != 0
    # Compute the fractional error and absolute error, and take the minimum of the two:
#    diff = minimum([abs(float(jac_step_num[(j-1)*7+i,(l-1)*7+k])/jac_step[(j-1)*7+i,(l-1)*7+k]-1.0);float(jac_step_num[(j-1)*7+i,(l-1)*7+k])-jac_step[(j-1)*7+i,(l-1)*7+k]])
    diff = abs(float(jac_step_num[(j-1)*7+i,(l-1)*7+k])/jac_step[(j-1)*7+i,(l-1)*7+k]-1.0)
    if diff > jacmax
      jacmax = diff
      println(i," ",j," ",k," ",l," ",jacmax," ",convert(Float64,jac_step_num[(j-1)*7+i,(l-1)*7+k])," ",jac_step[(j-1)*7+i,(l-1)*7+k])
    end
  end
end

println("Maximum jac_step phisalpha error: ",convert(Float64,jacmax))
dqdt_num = convert(Array{Float64,1},dqdt_num)
println("Maximum dqdt_phi phisalpha error: ",maximum(abs.(dqdt_phi-dqdt_num)))

@test isapprox(jac_step,jac_step_num;norm=maxabs)
@test isapprox(dqdt_phi,dqdt_num;norm=maxabs)
end
