# Tests the routine dh17 jacobian:

include("ttv.jl")

#function dh17!(x::Array{Float64,2},v::Array{Float64,2},h::Float64,m::Array{Float64,1},n::Int64,jac_step::Array{Float64,2})


# Next, try computing three-body Keplerian Jacobian:

n = 8
#n = 3
#n = 2
t0 = 7257.93115525
#h  = 0.05
h  = 0.075
tmax = 600.0
dlnq = 3e-6

nstep = 8000

elements = readdlm("elements.txt",',')
#elements[2,1] = 1.0
#elements[3,1] = 1.0

m =zeros(n)
x0=zeros(3,n)
v0=zeros(3,n)

# Predict values of s:

# Initialize with identity matrix:
jac_step = eye(Float64,7*n)

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

# Take a single step (so that we aren't at initial coordinates):
dh17!(x0,v0,h,m,n)

# Now, copy these to compute Jacobian (so that I don't step
# x0 & v0 forward in time):
x = copy(x0)
v = copy(v0)
m = copy(m0)
# Compute jacobian exactly:
for istep=1:nstep
  dh17!(x,v,h,m,n,jac_step)
end
# Save these so that I can compute derivatives numerically:
xsave = copy(x)
vsave = copy(v)
msave = copy(m)
## Check that we have agreement:
#xtest = copy(x0)
#vtest = copy(v0)
#m = copy(m0)
#dh17!(xtest,vtest,h,m,n)
#println("x/v difference: ",x-xtest,v-vtest)

# Now compute numerical derivatives:
jac_step_num = zeros(7*n,7*n)
# Vary the initial parameters of planet j:
for j=1:n
  # Vary the initial phase-space elements:
  for jj=1:3
  # Initial positions, velocities & masses:
    xm = copy(x0)
    vm = copy(v0)
    mm = copy(m0)
    dq = dlnq * xm[jj,j]
    if xm[jj,j] != 0.0
      xm[jj,j] -=  dq
    else
      dq = dlnq
      xm[jj,j] = -dq
    end
    for istep=1:nstep
      dh17!(xm,vm,h,mm,n)
    end
    xp = copy(x0)
    vp = copy(v0)
    mp = copy(m0)
    dq = dlnq * xp[jj,j]
    if xp[jj,j] != 0.0
      xp[jj,j] +=  dq
    else
      dq = dlnq
      xp[jj,j] = dq
    end
    for istep=1:nstep
      dh17!(xp,vp,h,mp,n)
    end
  # Now x & v are final positions & velocities after time step
    for i=1:n
      for k=1:3
        jac_step_num[(i-1)*7+  k,(j-1)*7+ jj] = .5*(xp[k,i]-xm[k,i])/dq
        jac_step_num[(i-1)*7+3+k,(j-1)*7+ jj] = .5*(vp[k,i]-vm[k,i])/dq
      end
    end
  # Next velocity derivatives:
    xm=copy(x0)
    vm=copy(v0)
    mm=copy(m0)
    dq = dlnq * vm[jj,j]
    if vm[jj,j] != 0.0
      vm[jj,j] -=  dq
    else
      dq = dlnq
      vm[jj,j] = -dq
    end
    for istep=1:nstep
      dh17!(xm,vm,h,mm,n)
    end
    xp=copy(x0)
    vp=copy(v0)
    mp=copy(m0)
    dq = dlnq * vp[jj,j]
    if vp[jj,j] != 0.0
      vp[jj,j] +=  dq
    else
      dq = dlnq
      vp[jj,j] = dq
    end
    for istep=1:nstep
      dh17!(xp,vp,h,mp,n)
    end
    for i=1:n
      for k=1:3
        jac_step_num[(i-1)*7+  k,(j-1)*7+3+jj] = .5*(xp[k,i]-xm[k,i])/dq
        jac_step_num[(i-1)*7+3+k,(j-1)*7+3+jj] = .5*(vp[k,i]-vm[k,i])/dq
      end
    end
  end
# Now vary mass of planet:
  xm=copy(x0)
  vm=copy(v0)
  mm=copy(m0)
  dq = mm[j]*dlnq
  mm[j] -= dq
  for istep=1:nstep
    dh17!(xm,vm,h,mm,n)
  end
  xp=copy(x0)
  vp=copy(v0)
  mp=copy(m0)
  dq = mp[j]*dlnq
  mp[j] += dq
  for istep=1:nstep
    dh17!(xp,vp,h,mp,n)
  end
  for i=1:n
    for k=1:3
      jac_step_num[(i-1)*7+  k,j*7] = .5*(xp[k,i]-xm[k,i])/dq
      jac_step_num[(i-1)*7+3+k,j*7] = .5*(vp[k,i]-vm[k,i])/dq
    end
  end
  # Mass unchanged -> identity
  jac_step_num[j*7,j*7] = 1.0
end

# Now, compare the results:
#println(jac_step)
#println(jac_step_num)

for j=1:n
  for i=1:7
    for k=1:n
      println(i," ",j," ",k," ",jac_step[(j-1)*7+i,(k-1)*7+1:k*7]," ",jac_step_num[(j-1)*7+i,(k-1)*7+1:k*7]," ",jac_step[(j-1)*7+i,(k-1)*7+1:7*k]./jac_step_num[(j-1)*7+i,(k-1)*7+1:7*k]-1.)
    end
  end
end

jacmax = 0.0

for i=1:7, j=1:3, k=1:7, l=1:3
  if jac_step[(j-1)*7+i,(l-1)*7+k] != 0
    diff = abs(jac_step_num[(j-1)*7+i,(l-1)*7+k]/jac_step[(j-1)*7+i,(l-1)*7+k]-1.0)
    if diff > jacmax
      jacmax = diff
    end
  end
end

println("Maximum fractional error: ",jacmax)

