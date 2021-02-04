# This code tests the function kepler_driftij_gamma
import NbodyGradient: kepler_driftij_gamma!

@testset "kepler_driftij_gamma" begin
for drift_first in [true,false]
# Next, try computing two-body Keplerian Jacobian:

NDIM = 3
n = 3
H = [3,1,1]
t0 = 7257.93115525
#t0 = -300.0
#h  = 0.0000005
#h  = 0.05
h  = 0.25
hbig  = big(h)
tmax = 600.0
#dlnq = 1e-8
dlnq = big(1e-15)

elements = readdlm("elements.txt",',')
#elements[2,1] = 0.75
#elements[2,1] = 1.0
#elements[3,1] = 1.0

m =zeros(n)
x0=zeros(NDIM,n)
v0=zeros(NDIM,n)

for k=1:n
  m[k] = elements[k,1]
end
for iter = 1:2

init = ElementsIC(t0,H,elements)
ic_big = ElementsIC(big(t0),H,big.(elements))
x0,v0,_ = init_nbody(init)
s0 = State(init)
sbig = State(ic_big)
 if iter == 2
   # Reduce masses to trigger hyperbolic routine:
    m[1:n] *= 1e-3
    s0.m[1:n] *= 1e-3
    sbig.m[1:n] *= 1e-3
    hbig = big(h)
 end
# Tilt the orbits a bit:
x0[2,1] = 5e-1*sqrt(x0[1,1]^2+x0[3,1]^2)
x0[2,2] = -5e-1*sqrt(x0[1,2]^2+x0[3,2]^2)
v0[2,1] = 5e-1*sqrt(v0[1,1]^2+v0[3,1]^2)
v0[2,2] = -5e-1*sqrt(v0[1,2]^2+v0[3,2]^2)

jac_ij = zeros(Float64,14,14)
dqdt_ij = zeros(Float64,14)
dqdt_num = zeros(BigFloat,14)

#println("Initial values: ",x0,v0)
#println("masses: ",m)
i=1 ; j=2
x = copy(x0) ; v=copy(v0)
xerror = zeros(NDIM,n); verror = zeros(NDIM,n)
# Predict values of s:
kepler_driftij_gamma!(m,x,v,xerror,verror,i,j,h,jac_ij,dqdt_ij,drift_first)
x0 = copy(x) ; v0 = copy(v)
xerror = zeros(NDIM,n); verror = zeros(NDIM,n)
xbig = big.(x) ; vbig=big.(v); mbig = big.(m)
xerr_big = zeros(BigFloat,NDIM,n); verr_big = zeros(BigFloat,NDIM,n)
kepler_driftij_gamma!(m,x,v,xerror,verror,i,j,h,jac_ij,dqdt_ij,drift_first)
# Now compute Jacobian with BigFloat precision:
jac_ij_big = zeros(BigFloat,14,14)
dqdt_ij_big = zeros(BigFloat,14)
KEPLER_TOL = sqrt(eps(big(1.0)))
kepler_driftij_gamma!(mbig,xbig,vbig,xerr_big,verr_big,i,j,hbig,jac_ij_big,dqdt_ij_big,drift_first)
#println("jac_ij: ",convert(Array{Float64,2},jac_ij_big))
#println("jac_ij - jac_ij_big: ",convert(Array{Float64,2},jac_ij_big)-jac_ij)
#println("max(jac_ij - jac_ij_big): ",maxabs(convert(Array{Float64,2},jac_ij_big)-jac_ij))
#s0.x .= copy(x0); s0.v .= copy(v0); s0.m .= copy(m)

# Now, compute the derivatives numerically:
jac_ij_num = zeros(BigFloat,14,14)
xsave = big.(x)
vsave = big.(v)
msave = big.(m)
sbig.x .= copy(xsave)
sbig.v .= copy(vsave)
sbig.m .= copy(msave)
# Compute the time derivatives:
# Initial positions, velocities & masses:
sm = deepcopy(State(ic_big))
sm.x .= big.(x0)
sm.v .= big.(v0)
sm.m .= big.(msave)
dq = dlnq * hbig
hbig -= dq
#fill!(xerr_big,0.0) ; fill!(verr_big,0.0)
sm.xerror .= 0.0; sm.verror .= 0.0
kepler_driftij_gamma!(sm,i,j,hbig,drift_first)
#xp = big.(x0)
#vp = big.(v0)
sp = deepcopy(State(ic_big))
sp.x .= big.(x0)
sp.v .= big.(v0)
sp.m .= big.(msave)
hbig += 2dq
#fill!(xerr_big,0.0) ; fill!(verr_big,0.0)
sp.xerror .= 0.0; sp.verror .= 0.0
kepler_driftij_gamma!(sp,i,j,hbig,drift_first)
# Now x & v are final positions & velocities after time step
for k=1:3
  dqdt_num[   k] = 0.5*(sp.x[k,i]-sm.x[k,i])/dq
  dqdt_num[ 3+k] = 0.5*(sp.v[k,i]-sm.v[k,i])/dq
  dqdt_num[ 7+k] = 0.5*(sp.x[k,j]-sm.x[k,j])/dq
  dqdt_num[10+k] = 0.5*(sp.v[k,j]-sm.v[k,j])/dq
end
hbig = big(h)

# Compute position, velocity & mass derivatives:
for jj=1:3
  # Initial positions, velocities & masses:
  #xm = big.(x0)
  #vm = big.(v0)
  #mm = big.(msave)
  sm = deepcopy(State(ic_big))
  sm.x .= big.(x0)
  sm.v .= big.(v0)
  sm.m .= big.(msave)
  dq = dlnq * sm.x[jj,i]
  if sm.x[jj,i] != 0.0
    sm.x[jj,i] -=  dq
  else
    dq = dlnq
    sm.x[jj,i] = -dq
  end
  #fill!(xerr_big,0.0) ; fill!(verr_big,0.0)
  sm.xerror .= 0.0; sm.verror .= 0.0
  kepler_driftij_gamma!(sm,i,j,hbig,drift_first)
  #xp = big.(x0)
  #vp = big.(v0)
  sp = deepcopy(State(ic_big))
  sp.x .= big.(x0)
  sp.v .= big.(v0)
  sp.m .= big.(msave)
  if sm.x[jj,i] != 0.0
    sp.x[jj,i] +=  dq
  else
    dq = dlnq
    sp.x[jj,i] = dq
  end
  #fill!(xerr_big,0.0) ; fill!(verr_big,0.0)
  sp.xerror .= 0.0; sp.verror .= 0.0
  kepler_driftij_gamma!(sp,i,j,hbig,drift_first)
  # Now x & v are final positions & velocities after time step
  for k=1:3
    jac_ij_num[   k,  jj] = .5*(sp.x[k,i]-sm.x[k,i])/dq
    jac_ij_num[ 3+k,  jj] = .5*(sp.v[k,i]-sm.v[k,i])/dq
    jac_ij_num[ 7+k,  jj] = .5*(sp.x[k,j]-sm.x[k,j])/dq
    jac_ij_num[10+k,  jj] = .5*(sp.v[k,j]-sm.v[k,j])/dq
  end
  #xm = big.(x0)
  #vm = big.(v0)
  #mm  = big.(msave)
  sm = deepcopy(State(ic_big))
  sm.x .= big.(x0)
  sm.v .= big.(v0)
  sm.m .= big.(msave)
  dq = dlnq * sm.v[jj,i]
  if sm.v[jj,i] != 0.0
    sm.v[jj,i] -=  dq
  else
    dq = dlnq
    sm.v[jj,i] = -dq
  end
  #fill!(xerr_big,0.0) ; fill!(verr_big,0.0)
  sm.xerror .= 0.0; sm.verror .= 0.0
  kepler_driftij_gamma!(sm,i,j,hbig,drift_first)
  #p = big.(x0)
  #vp = big.(v0)
  #mm  = big.(msave)
  sp = deepcopy(State(ic_big))
  sp.x .= big.(x0)
  sp.v .= big.(v0)
  sp.m .= big.(msave)
  if sp.v[jj,i] != 0.0
    sp.v[jj,i] +=  dq
  else
    dq = dlnq
    sp.v[jj,i] = dq
  end
  #fill!(xerr_big,0.0) ; fill!(verr_big,0.0)
  sp.xerror .= 0.0; sp.verror .= 0.0
  kepler_driftij_gamma!(sp,i,j,hbig,drift_first)
  for k=1:3
    jac_ij_num[   k,3+jj] = .5*(sp.x[k,i]-sm.x[k,i])/dq
    jac_ij_num[ 3+k,3+jj] = .5*(sp.v[k,i]-sm.v[k,i])/dq
    jac_ij_num[ 7+k,3+jj] = .5*(sp.x[k,j]-sm.x[k,j])/dq
    jac_ij_num[10+k,3+jj] = .5*(sp.v[k,j]-sm.v[k,j])/dq
  end
end

# Now vary mass of inner planet:
#xm= big.(x0)
#vm= big.(v0)
#mm= big.(msave)
sm = deepcopy(State(ic_big))
sm.x .= big.(x0)
sm.v .= big.(v0)
sm.m .= big.(msave)
dq = sm.m[i]*dlnq
sm.m[i] -= dq
#fill!(xerr_big,0.0) ; fill!(verr_big,0.0)
sm.xerror .= 0.0; sm.verror .= 0.0
kepler_driftij_gamma!(sm,i,j,hbig,drift_first)
#xp= big.(x0)
#vp= big.(v0)
#mp= big.(msave)
sp = deepcopy(State(ic_big))
sp.x .= big.(x0)
sp.v .= big.(v0)
sp.m .= big.(msave)
dq = sp.m[i]*dlnq
sp.m[i] += dq
#fill!(xerr_big,0.0) ; fill!(verr_big,0.0)
sp.xerror .= 0.0; sp.verror .= 0.0
kepler_driftij_gamma!(sp,i,j,hbig,drift_first)
for k=1:3
  jac_ij_num[   k,7] = .5*(sp.x[k,i]-sm.x[k,i])/dq
  jac_ij_num[ 3+k,7] = .5*(sp.v[k,i]-sm.v[k,i])/dq
  jac_ij_num[ 7+k,7] = .5*(sp.x[k,j]-sm.x[k,j])/dq
  jac_ij_num[10+k,7] = .5*(sp.v[k,j]-sm.v[k,j])/dq
end
# The mass doesn't change:
jac_ij_num[7,7] =  1.0

for jj=1:3
  # Now vary parameters of outer planet:
  #xm = big.(x0)
  #vm = big.(v0)
  #mm = big.(msave)
  sm = deepcopy(State(ic_big))
  sm.x .= big.(x0)
  sm.v .= big.(v0)
  sm.m .= big.(msave)
  dq = dlnq * sm.x[jj,j]
  if sm.x[jj,j] != 0.0
    sm.x[jj,j] -=  dq
  else
    dq = dlnq
    sm.x[jj,j] = -dq
  end
  #fill!(xerr_big,0.0) ; fill!(verr_big,0.0)
  sm.xerror .= 0.0; sm.verror .= 0.0
  kepler_driftij_gamma!(sm,i,j,hbig,drift_first)
  #xp = big.(x0)
  #vp = big.(v0)
  sp = deepcopy(State(ic_big))
  sp.x .= big.(x0)
  sp.v .= big.(v0)
  sp.m .= big.(msave)
  if sp.x[jj,j] != 0.0
    sp.x[jj,j] +=  dq
  else
    dq = dlnq
    sp.x[jj,j] = dq
  end
  #fill!(xerr_big,0.0) ; fill!(verr_big,0.0)
  sp.xerror .= 0.0; sp.verror .= 0.0
  kepler_driftij_gamma!(sp,i,j,hbig,drift_first)
  for k=1:3
    jac_ij_num[   k,7+jj] = .5*(sp.x[k,i]-sm.x[k,i])/dq
    jac_ij_num[ 3+k,7+jj] = .5*(sp.v[k,i]-sm.v[k,i])/dq
    jac_ij_num[ 7+k,7+jj] = .5*(sp.x[k,j]-sm.x[k,j])/dq
    jac_ij_num[10+k,7+jj] = .5*(sp.v[k,j]-sm.v[k,j])/dq
  end
  #xm= big.(x0)
  #vm= big.(v0)
  #mm = big.(msave)
  sm = deepcopy(State(ic_big))
  sm.x .= big.(x0)
  sm.v .= big.(v0)
  sm.m .= big.(msave)
  dq = dlnq * sm.v[jj,j]
  if sm.v[jj,j] != 0.0
    sm.v[jj,j] -=  dq
  else
    dq = dlnq
    sm.v[jj,j] = -dq
  end
  #fill!(xerr_big,0.0) ; fill!(verr_big,0.0)
  sm.xerror .= 0.0; sm.verror .= 0.0
  kepler_driftij_gamma!(sm,i,j,hbig,drift_first)
  #xp= big.(x0)
  #vp= big.(v0)
  sp = deepcopy(State(ic_big))
  sp.x .= big.(x0)
  sp.v .= big.(v0)
  sp.m .= big.(msave)
  if sp.v[jj,j] != 0.0
    sp.v[jj,j] +=  dq
  else
    dq = dlnq
    sp.v[jj,j] = dq
  end
  #fill!(xerr_big,0.0) ; fill!(verr_big,0.0)
  sp.xerror .= 0.0; sp.verror .= 0.0
  kepler_driftij_gamma!(sp,i,j,hbig,drift_first)
  for k=1:3
    jac_ij_num[   k,10+jj] = .5*(sp.x[k,i]-sm.x[k,i])/dq
    jac_ij_num[ 3+k,10+jj] = .5*(sp.v[k,i]-sm.v[k,i])/dq
    jac_ij_num[ 7+k,10+jj] = .5*(sp.x[k,j]-sm.x[k,j])/dq
    jac_ij_num[10+k,10+jj] = .5*(sp.v[k,j]-sm.v[k,j])/dq
  end
end

# Now vary mass of outer planet:
#xm = big.(x0)
#vm = big.(v0)
#mm = big.(msave)
sm = deepcopy(State(ic_big))
sm.x .= big.(x0)
sm.v .= big.(v0)
sm.m .= big.(msave)
dq = sm.m[j]*dlnq
sm.m[j] -= dq
#fill!(xerr_big,0.0) ; fill!(verr_big,0.0)
sm.xerror .= 0.0; sm.verror .= 0.0
kepler_driftij_gamma!(sm,i,j,hbig,drift_first)
#xp = big.(x0)
#vp = big.(v0)
#mp = big.(msave)
sp = deepcopy(State(ic_big))
sp.x .= big.(x0)
sp.v .= big.(v0)
sp.m .= big.(msave)
dq = sp.m[j]*dlnq
sp.m[j] += dq
#fill!(xerr_big,0.0) ; fill!(verr_big,0.0)
sp.xerror .= 0.0; sp.verror .= 0.0
kepler_driftij_gamma!(sp,i,j,hbig,drift_first)
for k=1:3
  jac_ij_num[   k,14] = .5*(sp.x[k,i]-sm.x[k,i])/dq
  jac_ij_num[ 3+k,14] = .5*(sp.v[k,i]-sm.v[k,i])/dq
  jac_ij_num[ 7+k,14] = .5*(sp.x[k,j]-sm.x[k,j])/dq
  jac_ij_num[10+k,14] = .5*(sp.v[k,j]-sm.v[k,j])/dq
end
# The mass doesn't change:
jac_ij_num[14,14] =  1.0

#println(jac_ij)
#println(jac_ij_num)
#println(jac_ij./jac_ij_num)
emax = 0.0; imax = 0; jmax = 0
emax_big = big(0.0); imax_big = 0; jmax_big = 0
for i=1:14
  jac_ij[i,i] += 1
  jac_ij_big[i,i] += 1
end
for i=1:14, j=1:14
  if jac_ij[i,j] != 0.0
    diff = abs(convert(Float64,jac_ij_num[i,j])/jac_ij[i,j]-1.0)
    if  diff > emax
      emax = diff; imax = i; jmax = j
    end
    diff_big = abs(convert(Float64,jac_ij_num[i,j])/jac_ij_big[i,j]-1.0)
    if  diff_big > emax_big
      emax_big = diff; imax_big = i; jmax_big = j
    end
  end
end
# Dont need for CI
#=println("Maximum fractional error: ",emax," ",imax," ",jmax)
for i=1:14
  println("jac_ij: i      ",i," ",jac_ij[i,:])
  println("jac_ij_num: i  ",i," ",convert(Array{Float64,1},jac_ij_num[i,:]))
  println("difference: i  ",i," ",jac_ij[i,:].-convert(Array{Float64,1},jac_ij_num[i,:]))
  if i != 7 && i != 14
    println("frac diff : i  ",i," ",jac_ij[i,:]./convert(Array{Float64,1},jac_ij_num[i,:]).-1.0)
  end
end
println("Maximum fractional error big: ",emax_big," ",imax_big," ",jmax_big)
#println(jac_ij)
#println(convert(Array{Float64,2},jac_ij_num))
println("Maximum jac_ij error:   ",maxabs(convert(Array{Float64,2},asinh.(jac_ij_num))-asinh.(jac_ij)))
println("Maximum jac_ij_big-jac_ij_num:   ",maxabs(convert(Array{Float64,2},asinh.(jac_ij_num)-asinh.(jac_ij_big))))
println("Max dqdt error: ",maxabs(dqdt_ij-convert(Array{Float64,1},dqdt_num)))
=#
@test isapprox(jac_ij_num,jac_ij;norm=maxabs)
@test isapprox(dqdt_ij,convert(Array{Float64,1},dqdt_num);norm=maxabs)

end
end
end
