# Translation of David Hernandez's nbody.c for integrating hiercharical
# system with BH15 integrator.  Please cite Hernandez & Bertschinger (2015)
# if using this in a paper.

const YEAR  = 365.242
const GNEWT = 39.4845/YEAR^2
const NDIM  = 3
const third = 1./3.
include("kepler_step.jl")
include("init_nbody.jl")

function nbody!(t0::Float64,h::Float64,tmax::Float64,elements::Array{Float64,2},t_out::Array{Float64,1},state_out::Array{Float64,2},nstep::Int64)
fcons = open("fcons.txt","w");
n=8
m=zeros(n)
x=zeros(NDIM,n)
v=zeros(NDIM,n)
L0=zeros(NDIM)
p0=zeros(NDIM)
xcm0=zeros(NDIM)
ssave = zeros(n,n,3)
# Read in the initial orbital elements from a file (this
# should be moved to an input parameter):
elements = readdlm("elements.txt",',')
for i=1:n
  m[i] = elements[i,1]
end
# Initialize the N-body problem using nested hierarchy of Keplerians:
x,v = init_nbody(elements,t0,n)
#    infig1!(m,x,v,n)
# Compute the conserved quantities:
E0=consq!(m,x,v,n,L0,p0,xcm0)
#println("E0: ",E0," L0: ",L0," p0: ",p0," xcm0: ",xcm0)
#read(STDIN,Char)
# Set the time to the initial time:
t = t0
# Set step counter to zero:
i=0
#nstep = 1
# Loop over time steps:
while t < t0+tmax
  # Increment time by the time step:
  t += h
  # Increment counter by one:
  i +=1
  # Carry out a phi^2 mapping step:
  phi2!(x,v,h,m,n)
  # Every nstep steps, output the state of the system:
  if mod(i,nstep) == 0
    state_string = @sprintf("%18.12f",t)
    for j=1:n
      state_string = string(state_string,@sprintf("%18.12f",x[1,j]),@sprintf("%18.12f",x[3,j]),@sprintf("%18.12f",v[1,j]),@sprintf("%18.12f",v[3,j]))
    end
    state_string = state_string*"\n"
    print(fcons,state_string)
  end
  # 
end
# Compute the final values of the conserved quantities:
L=zeros(NDIM)
p=zeros(NDIM)
xcm=zeros(NDIM)
E=consq!(m,x,v,n,L,p,xcm)
println("dE/E=",(E0-E)/E0)
# Close the output file:
close(fcons)
return 
end

# Advances the center of mass of a binary
function centerm!(m::Array{Float64,1},x::Array{Float64,2},v::Array{Float64,2},delx::Array{Float64,1},delv::Array{Float64,1},i::Int64,j::Int64,h::Float64)
vcm=zeros(NDIM)
mij =m[i] + m[j]
if mij == 0 
  for k=1:NDIM
    vcm[k] = (v[k,i]+v[k,j])/2
    x[k,i] += h*vcm[k]
    x[k,j] += h*vcm[k]
  end
else
  for k=1:NDIM
    vcm[k] = (m[i]*v[k,i] + m[j]*v[k,j])/mij
    x[k,i] +=  m[j]/mij*delx[k] + h*vcm[k]
    x[k,j] += -m[i]/mij*delx[k] + h*vcm[k]
    v[k,i] +=  m[j]/mij*delv[k]
    v[k,j] += -m[i]/mij*delv[k]
  end
end
return
end

# Drifts bodies i & j
function driftij!(x::Array{Float64,2},v::Array{Float64,2},i::Int64,j::Int64,h::Float64)
for k=1:NDIM
  x[k,i] += h*v[k,i]
  x[k,j] += h*v[k,j]
end
return
end

# Carries out a Kepler step for bodies i & j
function keplerij!(m::Array{Float64,1},x::Array{Float64,2},v::Array{Float64,2},i::Int64,j::Int64,h::Float64)
# The state vector has: 1 time; 2-4 position; 5-7 velocity; 8 r0; 9 dr0dt; 10 beta; 11 s; 12 ds
# Initial state:
s0 = zeros(Float64,12)
# Final state (after a step):
s = zeros(Float64,12)
delx = zeros(NDIM)
delv = zeros(NDIM)
for k=1:NDIM
  s0[1+k     ] = x[k,i] - x[k,j]
  s0[1+k+NDIM] = v[k,i] - v[k,j]
end
gm = GNEWT*(m[i]+m[j])
if gm == 0
  for k=1:NDIM
    x[k,i] += h*v[k,i]
    x[k,j] += h*v[k,j]
  end
else
  kepler_step!(gm, h, s0, s)
  for k=1:NDIM
    delx[k] = s[1+k] - s0[1+k]
    delv[k] = s[1+NDIM+k] - s0[1+NDIM+k]
  end
# Advance center of mass:
  centerm!(m,x,v,delx,delv,i,j,h)
end
return
end

# Drifts all particles:
function drift!(x::Array{Float64,2},v::Array{Float64,2},h::Float64,n::Int64)
for i=1:n
  for j=1:NDIM
    x[j,i] += h*v[j,i]
  end
end
return
end

# Carries out the phi^2 mapping
function phi2!(x::Array{Float64,2},v::Array{Float64,2},h::Float64,m::Array{Float64,1},n::Int64)
drift!(x,v,h/2,n)
for i=1:n-1
  for j=i+1:n
    driftij!(x,v,i,j,-h/2)
    keplerij!(m,x,v,i,j,h/2)
  end
end
for i=n-1:-1:1
  for j=n:-1:i+1
    keplerij!(m,x,v,i,j,h/2)
    driftij!(x,v,i,j,-h/2)
  end
end
drift!(x,v,h/2,n)
return
end


# Initializes the system for a specific dynamical problem: eccentric binary orbiting
# a larger body.  Units in ~AU & yr.
function infig1!(m::Array{Float64,1},x::Array{Float64,2},v::Array{Float64,2},n::Int64)
a0 = 0.0125
a1 = 1.0
e0 = 0.6
e1 = 0
m[1] = 1e-3
m[2] = 1e-3
m[3] = 1.0
mu0 = (m[1]*m[2])/(m[1]+m[2])
mt0 = m[1]+m[2]
mu1 = (m[3]*mt0)/(m[3]+mt0)
mt1 = m[3]+mt0
x0 = a0*(1+e0)
x1 = a1*(1+e1)
v0 = sqrt(GNEWT*mt0/a0*(1-e0)/(1+e0))
v1 = sqrt(GNEWT*mt1/a1*(1-e1)/(1+e1))
fa0 = m[1]/mt0
fb0 = m[2]/mt0
fa1 = mt0/mt1
fb1 = m[3]/mt1
x[1,1] = -fb1*x1-fb0*x0
x[1,2] = -fb1*x1+fa0*x0
x[1,3] = fa1*x1
v[2,1] = -fb1*v1-fb0*v0
v[2,2] = -fb1*v1+fa0*v0
v[2,3] = fa1*v1
return
end

# Computes conserved quantities:
function consq!(m::Array{Float64,1},x::Array{Float64,2},v::Array{Float64,2},n::Int64,
              L::Array{Float64,1},p::Array{Float64,1},xcm::Array{Float64,1})
fill!(p,0.0) # =zeros(NDIM)
fill!(xcm,0.0) # =zeros(NDIM)
fill!(L,0.0) #  =zeros(NDIM)
E = 0.0
for i=1:n
  if m[i] != 0
    for j=1:NDIM
      p[j] += m[i]*v[j,i];
      xcm[j] += m[i]*x[j,i];
    end
    L[1] += m[i]*(x[2,i]*v[3,i]-x[3,i]*v[2,i]);
    L[2] += m[i]*(x[3,i]*v[1,i]-x[1,i]*v[3,i]);
    L[3] += m[i]*(x[1,i]*v[2,i]-x[2,i]*v[1,i]);
  end
  for j=1:NDIM
    E += 1.0/2.0*m[i]*v[j,i]*v[j,i];
  end
  for j=i+1:n
    if m[j] != 0
      rij = 0.;
      for k=1:NDIM
        rij += (x[k,i]-x[k,j])*(x[k,i]-x[k,j]);
      end
      rij = sqrt(rij);
      E -= GNEWT*m[i]*m[j]/rij;
    end
  end
end
mt=sum(m)
if mt != 0
  for i=1:NDIM
    xcm[i] /= mt;
    p[i] /= mt;
  end
end
println("L: ",L)
println("p: ",p)
println("xcm: ",xcm)
return E
end
