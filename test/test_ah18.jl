
# Tests the routine ah18 jacobian:

#include("../src/ttv.jl")

# Next, try computing three-body Keplerian Jacobian:

@testset "ah18" begin


#n = 8
n = 3
H = [3,1,1]
#n = 2
t0 = 7257.93115525
h  = 0.05
hbig = big(h)
tmax = 600.0
dlnq = big(1e-20)

#nstep = 8000
#nstep = 5000
nstep = 100
#nstep = 1

elements = readdlm("elements.txt",',')
# Increase mass of inner planets:
elements[2,1] *= 100.
elements[3,1] *= 100.

m =zeros(n)
x0=zeros(3,n)
v0=zeros(3,n)

# Define which pairs will have impulse rather than -drift+Kepler:
pair = zeros(Bool,n,n)

# Initialize with identity matrix:
jac_step = Matrix{Float64}(I,7*n,7*n)

for k=1:n
  m[k] = elements[k,1]
end
m0 = copy(m)

init = ElementsIC(elements,H,t0)
x0,v0,_ = init_nbody(init)

# Tilt the orbits a bit:
x0[2,1] = 5e-1*sqrt(x0[1,1]^2+x0[3,1]^2)
x0[2,2] = -5e-1*sqrt(x0[1,2]^2+x0[3,2]^2)
x0[2,3] = -5e-1*sqrt(x0[1,2]^2+x0[3,2]^2)
v0[2,1] = 5e-1*sqrt(v0[1,1]^2+v0[3,1]^2)
v0[2,2] = -5e-1*sqrt(v0[1,2]^2+v0[3,2]^2)
v0[2,3] = -5e-1*sqrt(v0[1,2]^2+v0[3,2]^2)
xbig = big.(x0); vbig = big.(v0)
xtest = copy(x0); vtest=copy(v0)
# Take a single step (so that we aren't at initial coordinates):
x = copy(x0); v = copy(v0)
xerror = zeros(3,n); verror = zeros(3,n)
big_xerror = copy(big.(xerror))
big_verror = copy(big.(verror))
for i=1:nstep; ah18!(x,v,xerror,verror,h,m,n,pair); end
# Take a step with big precision:
ah18!(xbig,vbig,big_xerror,big_verror,big(h),big.(m),n,pair)
#= Take a single DH17 step:
xerror = zeros(3,n); verror = zeros(3,n)
@time for i=1:nstep; dh17!(xtest,vtest,xerror,verror,h,m,n,pair);end
println("AH18 vs. DH17 x/v difference: ",x-xtest,v-vtest)
=#
# Compute x & v in BigFloat precision:
xbig = big.(x0)
vbig = big.(v0)
mbig = big.(m0)
xerr_big = zeros(BigFloat,3,n); verr_big = zeros(BigFloat,3,n)
jac_step_big = Matrix{BigFloat}(I,7*n,7*n)
jac_err_big = zeros(BigFloat,7*n,7*n)
for i=1:nstep; ah18!(xbig,vbig,xerr_big,verr_big,hbig,mbig,n,jac_step_big,jac_err_big,pair);end
#println("AH18 vs. AH18 BigFloat x/v difference: ",x-convert(Array{Float64,2},xbig),v-convert(Array{Float64,2},vbig))

# Now, copy these to compute Jacobian (so that I don't step
# x0 & v0 forward in time):
x = copy(x0)
v = copy(v0)
m = copy(m0)
xerror = zeros(3,n); verror = zeros(3,n); jac_error = zeros(7*n,7*n)
# Compute jacobian exactly over nstep steps:
for istep=1:nstep
  ah18!(x,v,xerror,verror,h,m,n,jac_step,jac_error,pair)
end

#println("AH18 vs. AH18 BigFloat jac_step difference: ",jac_step-convert(Array{Float64,2},jac_step_big))
#println(typeof(h)," ",jac_step)
#read(STDIN,Char)


# The following lines have a Julia bug - jac_big gets
# misaligned or shifted when returned.  If I modify ah18! to output
# jac_step, then it works.
# Initialize with identity matrix:
## Compute jacobian exactly over nstep steps:
#for istep=1:nstep
#  jac_copy = ah18!(xbig,vbig,hbig,mbig,n,jac_big,pair)
#end
#println(typeof(hbig)," ",convert(Array{Float64,2},jac_copy))
#println("Comparing x & xbig: ",maximum(abs,x-xbig))
#println("Comparing v & vbig: ",maximum(abs,v-vbig))
#println("Comparing jac_step and jac_big: ",maxabs(jac_step-jac_copy))
#println("Comparing jac_step and jac_big: ",jac_step./convert(Array{Float64,2},jac_big))

# Test that both versions of ah18 give the same answer:
#xtest = copy(x0)
#vtest = copy(v0)
#m = copy(m0)
#for istep=1:nstep
#  ah18!(xtest,vtest,h,m,n,pair)
#end
#println("x/v difference: ",x-xtest,v-vtest)

# Now compute numerical derivatives:
jac_step_num = zeros(BigFloat,7*n,7*n)
# Vary the initial parameters of planet j:
for j=1:n
  # Vary the initial phase-space elements:
  for jj=1:3
  # Initial positions, velocities & masses:
    xm = big.(x0)
    vm = big.(v0)
    mm = big.(m0)
    fill!(big_xerror,0.0)
    fill!(big_verror,0.0)
    dq = dlnq * xm[jj,j]
    if xm[jj,j] != 0.0
      xm[jj,j] -=  dq
    else
      dq = dlnq
      xm[jj,j] = -dq
    end
    for istep=1:nstep
      ah18!(xm,vm,big_xerror,big_verror,hbig,mm,n,pair)
    end
    xp = big.(x0)
    vp = big.(v0)
    mp = big.(m0)
    fill!(big_xerror,0.0)
    fill!(big_verror,0.0)
    dq = dlnq * xp[jj,j]
    if xp[jj,j] != 0.0
      xp[jj,j] +=  dq
    else
      dq = dlnq
      xp[jj,j] = dq
    end
    for istep=1:nstep
      ah18!(xp,vp,big_xerror,big_verror,hbig,mp,n,pair)
    end
  # Now x & v are final positions & velocities after time step
    for i=1:n
      for k=1:3
        jac_step_num[(i-1)*7+  k,(j-1)*7+ jj] = .5*(xp[k,i]-xm[k,i])/dq
        jac_step_num[(i-1)*7+3+k,(j-1)*7+ jj] = .5*(vp[k,i]-vm[k,i])/dq
      end
    end
  # Next velocity derivatives:
    xm= big.(x0)
    vm= big.(v0)
    mm= big.(m0)
    fill!(big_xerror,0.0)
    fill!(big_verror,0.0)
    dq = dlnq * vm[jj,j]
    if vm[jj,j] != 0.0
      vm[jj,j] -=  dq
    else
      dq = dlnq
      vm[jj,j] = -dq
    end
    for istep=1:nstep
      ah18!(xm,vm,big_xerror,big_verror,hbig,mm,n,pair)
    end
    xp= big.(x0)
    vp= big.(v0)
    mp= big.(m0)
    fill!(big_xerror,0.0)
    fill!(big_verror,0.0)
    dq = dlnq * vp[jj,j]
    if vp[jj,j] != 0.0
      vp[jj,j] +=  dq
    else
      dq = dlnq
      vp[jj,j] = dq
    end
    for istep=1:nstep
      ah18!(xp,vp,big_xerror,big_verror,hbig,mp,n,pair)
    end
    for i=1:n
      for k=1:3
        jac_step_num[(i-1)*7+  k,(j-1)*7+3+jj] = .5*(xp[k,i]-xm[k,i])/dq
        jac_step_num[(i-1)*7+3+k,(j-1)*7+3+jj] = .5*(vp[k,i]-vm[k,i])/dq
      end
    end
  end
# Now vary mass of planet:
  xm= big.(x0)
  vm= big.(v0)
  mm= big.(m0)
  fill!(big_xerror,0.0)
  fill!(big_verror,0.0)
  dq = mm[j]*dlnq
  mm[j] -= dq
  for istep=1:nstep
    ah18!(xm,vm,big_xerror,big_verror,hbig,mm,n,pair)
  end
  xp= big.(x0)
  vp= big.(v0)
  mp= big.(m0)
  fill!(big_xerror,0.0)
  fill!(big_verror,0.0)
  dq = mp[j]*dlnq
  mp[j] += dq
  for istep=1:nstep
    ah18!(xp,vp,big_xerror,big_verror,hbig,mp,n,pair)
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

#for j=1:n
#  for i=1:7
#    for k=1:n
#      println(i," ",j," ",k," ",jac_step[(j-1)*7+i,(k-1)*7+1:k*7]," ",jac_step_num[(j-1)*7+i,(k-1)*7+1:k*7]," ",jac_step[(j-1)*7+i,(k-1)*7+1:7*k]./jac_step_num[(j-1)*7+i,(k-1)*7+1:7*k]-1.)
#    end
#  end
#end

jacmax = 0.0; jac_diff = 0.0
imax = 0; jmax = 0; kmax = 0; lmax = 0
for i=1:7, j=1:3, k=1:7, l=1:3
  if jac_step[(j-1)*7+i,(l-1)*7+k] != 0
    diff = abs(jac_step_num[(j-1)*7+i,(l-1)*7+k]/jac_step[(j-1)*7+i,(l-1)*7+k]-1.0)
    if diff > jacmax
      jac_diff = diff; imax = i; jmax = j; kmax = k; lmax = l; jacmax = jac_step[(j-1)*7+i,(l-1)*7+k]
    end
  end
end

#println("Maximum fractional error: ",jac_diff," ",imax," ",jmax," ",kmax," ",lmax," ",jacmax)
#println(jac_step./jac_step_num)
#println("Maximum error jac_step:   ",maximum(abs.(jac_step-jac_step_num)))
#println("Maximum error jac_step_big vs. jac_step_num:   ",maximum(abs.(asinh.(jac_step_big)-asinh.(jac_step_num))))
#println("Maximum diff asinh(jac_step):   ",maximum(abs.(asinh.(jac_step)-asinh.(jac_step_num))))

# Compute dqdt:

dqdt = zeros(7*n)
dqdt_num = zeros(BigFloat,7*n)
x = copy(x0)
v = copy(v0)
m = copy(m0)
xerror = zeros(3,n); verror = zeros(3,n)
ah18!(x,v,xerror,verror,h,m,n,dqdt,pair)
xm= big.(x0)
vm= big.(v0)
mm= big.(m0)
fill!(big_xerror,0.0)
fill!(big_verror,0.0)
dq = hbig*dlnq
hbig -= dq
ah18!(xm,vm,big_xerror,big_verror,hbig,mm,n,pair)
xp= big.(x0)
vp= big.(v0)
mp= big.(m0)
fill!(big_xerror,0.0)
fill!(big_verror,0.0)
hbig += 2dq
ah18!(xp,vp,big_xerror,big_verror,hbig,mp,n,pair)
for i=1:n, k=1:3
  dqdt_num[(i-1)*7+  k] = .5*(xp[k,i]-xm[k,i])/dq
  dqdt_num[(i-1)*7+3+k] = .5*(vp[k,i]-vm[k,i])/dq
end
dqdt_num = convert(Array{Float64,1},dqdt_num)
#println("dqdt:     ",dqdt); println("dqdt_num: ",dqdt_num); println(" diff: ",dqdt-dqdt_num)
#println("dqdt-dqdt_num: ",maxabs(dqdt-convert(Array{Float64,1},dqdt_num)))

#@test isapprox(jac_step,jac_step_num)
#@test isapprox(jac_step,jac_step_num;norm=maxabs)
@test isapprox(asinh.(jac_step),asinh.(jac_step_num);norm=maxabs)
@test isapprox(dqdt,dqdt_num;norm=maxabs)
end
