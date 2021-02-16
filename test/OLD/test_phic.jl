# Tests the routine phic jacobian.  This routine
# computes the force gradient correction after Dehnen & Hernandez (2017).

# Next, try computing three-body Keplerian Jacobian:

#@testset "phic" begin
n = 3
t0 = 7257.93115525
h  = 0.05
tmax = 600.0
dlnq = big(1e-15)

elements = readdlm("elements.txt",',')
elements[2,1] = 1.0
elements[3,1] = 1.0

m =zeros(n)
x0=zeros(3,n)
v0=zeros(3,n)

# Define which pairs will have impulse rather than -drift+Kepler:
pair = ones(Bool,n,n)
# We want Keplerian between star & planets, and impulses between
# planets.  Impulse is indicated with 'true', -drift+Kepler with 'false':
for i=2:n
  pair[1,i] = false
  # We don't need to define this, but let's anyways:
  pair[i,1] = false
end
println("pair: ",pair)

jac_step = zeros(7*n,7*n)

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
phic!(x,v,h,m,n,jac_step,pair)


# Now compute numerical derivatives, using BigFloat to avoid
# round-off errors:
jac_step_num = zeros(BigFloat,7*n,7*n)
# Save these so that I can compute derivatives numerically:
xsave = big.(x0)
vsave = big.(v0)
msave = big.(m0)
hbig = big(h)
# Carry out step using BigFloat for extra precision:
phic!(xsave,vsave,hbig,msave,n,pair)
xbig = big.(x0)
vbig = big.(v0)
mbig = big.(m0)
# Vary time:
phic!(xbig,vbig,hbig,mbig,n,pair)
# Initial positions, velocities & masses:
xbig .= big.(x0)
vbig .= big.(v0)
mbig .= big.(m0)
hbig = big(h)
dq = dlnq * hbig
hbig += dq
phic!(xbig,vbig,hbig,mbig,n,pair)
# Now x & v are final positions & velocities after time step
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
    phic!(xbig,vbig,hbig,mbig,n,pair)
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
    phic!(xbig,vbig,hbig,mbig,n,pair)
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
  phic!(xbig,vbig,hbig,mbig,n,pair)
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
  if jac_step[(j-1)*7+i,(l-1)*7+k] != 0
    # Compute the fractional error and absolute error, and take the minimum of the two:
    diff = minimum([abs(float(jac_step_num[(j-1)*7+i,(l-1)*7+k])/jac_step[(j-1)*7+i,(l-1)*7+k]-1.0);float(jac_step_num[(j-1)*7+i,(l-1)*7+k])-jac_step[(j-1)*7+i,(l-1)*7+k]])
    if diff > jacmax
      jacmax = diff
    end
  end
end
println(jac_step)
println(convert(Array{Float64,2},jac_step_num))
println("Maximum jac_step phic error: ",convert(Float64,jacmax))

@test isapprox(jac_step,jac_step_num;norm=maxabs)
#end
