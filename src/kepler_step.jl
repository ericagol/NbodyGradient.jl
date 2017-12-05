include("kepler_solver_derivative.jl")

# Takes a single kepler step, calling Wisdom & Hernandez solver
# 
function kepler_step!(gm::Float64,h::Float64,state0::Array{Float64,1},state::Array{Float64,1})
# compute beta, r0, dr0dt, get x/v from state vector & call correct subroutine
x0 = zeros(Float64,3)
v0 = zeros(Float64,3)
for k=1:3
  x0[k]=state0[k+1]
  v0[k]=state0[k+4]
end
#  x0=state0[2:4]
  r0 = sqrt(x0[1]*x0[1]+x0[2]*x0[2]+x0[3]*x0[3])
#  v0 = state0[5:7]
  dr0dt = (x0[1]*v0[1]+x0[2]*v0[2]+x0[3]*v0[3])/r0
  beta0 = 2.0*gm/r0-(v0[1]*v0[1]+v0[2]*v0[2]+v0[3]*v0[3])
  s0=state0[11]
  if beta0 > 0.0
    iter = kep_elliptic!(x0,v0,r0,dr0dt,gm,h,beta0,s0,state)
  else
    iter = kep_hyperbolic!(x0,v0,r0,dr0dt,gm,h,beta0,s0,state)
  end
return
end

function kepler_step!(gm::Float64,h::Float64,state0::Array{Float64,1},state::Array{Float64,1},jacobian::Array{Float64,2})
# compute beta, r0, dr0dt, get x/v from state vector & call correct subroutine
x0 = zeros(Float64,3)
v0 = zeros(Float64,3)
for k=1:3
  x0[k]=state0[k+1]
  v0[k]=state0[k+4]
end
#  x0=state0[2:4]
  r0 = sqrt(x0[1]*x0[1]+x0[2]*x0[2]+x0[3]*x0[3])
#  v0 = state0[5:7]
  dr0dt = (x0[1]*v0[1]+x0[2]*v0[2]+x0[3]*v0[3])/r0
  beta0 = 2.0*gm/r0-(v0[1]*v0[1]+v0[2]*v0[2]+v0[3]*v0[3])
  s0=state0[11]
  if beta0 > 0.0
    iter = kep_elliptic!(x0,v0,r0,dr0dt,gm,h,beta0,s0,state,jacobian)
  else
    iter = kep_hyperbolic!(x0,v0,r0,dr0dt,gm,h,beta0,s0,state,jacobian)
  end
return
end
