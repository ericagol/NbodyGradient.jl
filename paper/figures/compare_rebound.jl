# Compare the times of simulation
using NbodyGradient, DelimitedFiles

elements = readdlm("elements_rebound.txt", ',')
elements[:,end] .= 0.0

for n = 2:11
  t0 = 0.0
  h = 0.05
  # Rebound integration is for 5000 code units; inner planet has
  # a period of 2π, while in our case 1-day, so differs by 5000/2π
  tmax = 2500/π

  ic = ElementsIC(t0, n, elements[1:n,:])
  s = State(ic)
  intr = Integrator(h, 0.0, tmax)
  
  # Run integrator once, so it's compiled
  @elapsed intr(s)
  s = State(ic)
  @elapsed intr(s, grad=false)

  # Time without derivatives
  s = State(ic)
  t = @elapsed intr(s, grad=false)

  # Time with derivatives
  s = State(ic)
  grad_t = @elapsed intr(s)

  println("Nplanet: ", n-1, " No gradient: ", t, " gradient: ", grad_t, " ratio: ", grad_t/t)
end