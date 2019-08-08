
# Working on figuring out how to prevent rounding error in transit
# times.

# Compute in BigFloat:

#t0 =  4000.0
t0 =  -200.0
h  = 0.07
tmax = 400.0

time = t0
time_big = big(t0)
# Next use compensated summation:
time_comp = t0
# Utilize the notation from Kahan (1965):
s2 = 0.0

times = [time]
times_big = [time_big]
times_comp = [time_comp]
hbig = big(h)

while time < t0+tmax
  # Float64 times:
  time += h/16
  push!(times,time)
  # BigFloat times:
  time_big += hbig/16
  push!(times_big,time_big)
  # Kahan (1965) compensated summation algorithm:
  s2 += h/16
  tmp = time_comp + s2
  s2 = (time_comp - tmp) + s2
  time_comp = tmp
  push!(times_comp,time_comp)
end

using PyPlot

plot(times,times-convert(Array{Float64,1},times_big))
diff_double = maximum(abs.(times-convert(Array{Float64,1},times_big)))
plot(times,times_comp-convert(Array{Float64,1},times_big))
diff_comp = maximum(abs.(times_comp-convert(Array{Float64,1},times_big)))
println("Maximum difference with Float64 precision: ",diff_double)
println("Maximum difference with compensated summation: ",diff_comp)
