# Writing a test for a new Hamiltonian formulation:

n = 4
x = zeros(3,n)
v = zeros(3,n)
vcom = zeros(3)
m = zeros(n)
h_cartesian = 0.0

# Compute Hamiltonian in Cartesian coordinates:
for i=1:n
  x[:,i] = randn(3)
  v[:,i] = randn(3)
  m[i] = exp(rand())
  h_cartesian += 0.5*m[i]*dot(v[:,i],v[:,i])
  vcom .+= m[i]*v[:,i]
end

# Center of mass velocity:
mtot = sum(m)
vcom ./= mtot

# Kinetic energy of the system:
h_sakura = h_cartesian
h_keplerian = 0.5*mtot*dot(vcom,vcom)

for i=1:n-1
  for j=i+1:n
    xij = x[:,j].-x[:,i]
    vij = v[:,j].-v[:,i]
    rij = norm(xij)
    h_cartesian -= m[i]*m[j]/rij
    mij = m[i]+m[j]
    mred = m[i]*m[j]/mij
    v2 = dot(vij,vij)
    h_sakura += mred*(0.5*v2-mij/rij) - 0.5*mred*v2
    h_keplerian += mred*(0.5*v2-mij/rij) - 0.5*(1.0-mij/mtot)*mred*v2
  end
end

println("h_cartesian: ",h_cartesian)
println("h_sakura:    ",h_sakura)
println("h_keplerian:    ",h_keplerian," ",h_keplerian/h_cartesian-1.)
