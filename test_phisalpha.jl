# Tests the routine phisalpha jacobian:

include("ttv.jl")

#function phisalpha!(x::Array{Float64,2},v::Array{Float64,2},h::Float64,m::Array{Float64,1},alpha::Float64,n::Int64,jac_step::Array{Float64,4})
#function phisalpha!(x,v,h,m,alpha,n,jac_step)


# Next, try computing three-body Keplerian Jacobian:

n = 3
t0 = 7257.93115525
h  = 0.05
tmax = 600.0
dlnq = 1e-4

elements = readdlm("elements.txt",',')
elements[2,1] = 1.0
elements[3,1] = 1.0

m =zeros(n)
x0=zeros(3,n)
v0=zeros(3,n)
alpha = 0.25

ssave = zeros(Float64,n,n,2)
# Predict values of s:
spred = zeros(Float64,n,n)

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
dh17!(x0,v0,h,m,n)

# Now, copy these to compute Jacobian (so that I don't step
# x0 & v0 forward in time):
x = copy(x0)
v = copy(v0)
m = copy(m0)
# Compute jacobian exactly:
phisalpha!(x,v,h,m,alpha,n,jac_step)
# Save these so that I can compute derivatives numerically:
xsave = copy(x)
vsave = copy(v)
msave = copy(m)

# Now compute numerical derivatives:
jac_step_num = zeros(7*n,7*n)
# Vary the initial parameters of planet j:
for j=1:n
  # Vary the initial phase-space elements:
  for jj=1:3
  # Initial positions, velocities & masses:
    x = copy(x0)
    v = copy(v0)
    m = copy(m0)
    dq = dlnq * x[jj,j]
    if x[jj,j] != 0.0
      x[jj,j] +=  dq
    else
      dq = dlnq
      x[jj,j] = dq
    end
    phisalpha!(x,v,h,m,alpha,n)
  # Now x & v are final positions & velocities after time step
    for i=1:n
      for k=1:3
        jac_step_num[(i-1)*7+  k,(j-1)*7+jj] = (x[k,i]-xsave[k,i])/dq
        jac_step_num[(i-1)*7+3+k,(j-1)*7+jj] = (v[k,i]-vsave[k,i])/dq
      end
    end
    x=copy(x0)
    v=copy(v0)
    m=copy(m0)
    dq = dlnq * v[jj,j]
    if v[jj,j] != 0.0
      v[jj,j] +=  dq
    else
      dq = dlnq
      v[jj,j] = dq
    end
    phisalpha!(x,v,h,m,alpha,n)
    for i=1:n
      for k=1:3
        jac_step_num[(i-1)*7+  k,(j-1)*7+3+jj] = (x[k,i]-xsave[k,i])/dq
        jac_step_num[(i-1)*7+3+k,(j-1)*7+3+jj] = (v[k,i]-vsave[k,i])/dq
      end
    end
  end
# Now vary mass of planet:
  x=copy(x0)
  v=copy(v0)
  m=copy(m0)
  dq = m[j]*dlnq
  m[j] += dq
  phisalpha!(x,v,h,m,alpha,n)
  for i=1:n
    for k=1:3
      jac_step_num[(i-1)*7+  k,j*7] = (x[k,i]-xsave[k,i])/dq
      jac_step_num[(i-1)*7+3+k,j*7] = (v[k,i]-vsave[k,i])/dq
    end
    # Mass unchanged -> identity
    jac_step_num[7*i,7*i] = 1.0
  end
end

# Now, compare the results:
#println(jac_step)
#println(jac_step_num)

for j=1:3
  for i=1:7
    for k=1:3
      println(jac_step[(j-1)*7+i,(k-1)*7+1:7*k]," ",jac_step_num[(j-1)*7+i,(k-1)*7+1:7*k]," ",jac_step_num[(j-1)*7+i,(k-1)*7+1:7*k]./jac_step[(j-1)*7+i,(k-1)*7+1:7*k]-1.)
    end
  end
end

jacmax = 0.0
for i=1:7, j=1:3, k=1:7, l=1:3
  if jac_step[(j-1)*7+i,(l-1)*7+k] != 0
    diff = minimum([abs(jac_step_num[(j-1)*7+i,(l-1)*7+k]/jac_step[(j-1)*7+i,(l-1)*7+k]-1.0);jac_step_num[(j-1)*7+i,(l-1)*7+k]-jac_step[(j-1)*7+i,(l-1)*7+k]])
    if diff > jacmax
      jacmax = diff
    end
  end
end

println("Maximum error: ",jacmax)
