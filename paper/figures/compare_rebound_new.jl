# Compare the times of simulation
using NbodyGradient, DelimitedFiles, BenchmarkTools

elements = readdlm("elements_rebound.txt", ',')
elements[:,end] .= 0.0

# Remove the old times file, if exists
try
  rm("nbg_times.txt")
catch
end

# Set up integrator
t0 = 0.0
h = 0.05
# Rebound integration is for 5000 code units; inner planet has
# a period of 2π, while in our case 1-day, so differs by 5000/2π
tmax = 2500/π

intr = Integrator(h, 0.0, tmax)

for n = 2:11

  ic = ElementsIC(t0, n, elements[1:n,:])
  s = State(ic)
  tt = TransitTiming(intr.tmax, ic)

  # Time without derivatives
  t = @belapsed intr(s, grad=false) setup=(s = State($ic))

  # Time with derivatives
  grad_t = @belapsed intr(s) setup=(s = State($ic))

  # Now while calculating transit times and derivatives
  grad_tt = @belapsed intr(s, tt) setup=(s = State($ic); tt = TransitTiming($intr.tmax, $ic))

  println("Nplanet: ", n-1, " No gradient: ", t, " gradient: ", grad_t, " grad w tts: ", grad_tt," ratio: ", grad_t/t)
  open("nbg_times.txt", "a") do io
    writedlm(io, [n-1 t grad_t grad_tt grad_t/t], ',')
  end
end
