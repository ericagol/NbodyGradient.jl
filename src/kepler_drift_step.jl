
include("kepler_drift_solver.jl")

# Takes a single kepler step, with reverse drift, calling Wisdom & Hernandez solver
# 
function kepler_drift_step!(gm::T,h::T,state0::Array{T,1},state::Array{T,1},drift_first::Bool) where {T <: Real}
# compute beta, r0,  get x/v from state vector & call correct subroutine
x0 = zeros(eltype(state0),3)
v0 = zeros(eltype(state0),3)
zero = 0.0*h
for k=1:3
  x0[k]=state0[k+1]
  v0[k]=state0[k+4]
end
s0=state0[11]
iter = kep_drift_ell_hyp!(x0,v0,gm,h,s0,state,drift_first)
return
end

function kepler_drift_step!(gm::T,h::T,state0::Array{T,1},state::Array{T,1},jacobian::Array{T,2},drift_first::Bool) where {T <: Real}
# compute beta, r0,  get x/v from state vector & call correct subroutine
x0 = zeros(eltype(state0),3)
v0 = zeros(eltype(state0),3)
zero = 0.0*h
for k=1:3
  x0[k]=state0[k+1]
  v0[k]=state0[k+4]
end
s0=state0[11]
iter = kep_drift_ell_hyp!(x0,v0,gm,h,s0,state,jacobian,drift_first)
return
end
