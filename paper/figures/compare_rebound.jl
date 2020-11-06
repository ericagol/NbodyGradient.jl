# Compare the times of simulation
using NbodyGradient, DelimitedFiles, BenchmarkTools

elements = readdlm("elements_rebound.txt", ',')
elements[:,end] .= 0.0

try
  rm("nbg_times.txt")
catch
end

for n = 2:11
  t0 = 0.0
  h = 0.05
  # Rebound integration is for 5000 code units; inner planet has
  # a period of 2π, while in our case 1-day, so differs by 5000/2π
  tmax = 2500/π

  ic = ElementsIC(t0, n, elements[1:n,:])
  s = State(ic)
  intr = Integrator(h, 0.0, tmax)

  # Time without derivatives
  t = @belapsed intr(s, grad=false) setup=(s = State($ic))

  # Time with derivatives
  grad_t = @belapsed intr(s) setup=(s = State($ic))

  #println("Nplanet: ", n-1, " No gradient: ", t, " gradient: ", grad_t, " ratio: ", grad_t/t)
  println(n-1, ",", t,",",grad_t,",", grad_t/t)
  open("nbg_times.txt", "a") do io
    writedlm(io, [n-1 t grad_t grad_t/t], ',')
  end
end