
using DelimitedFiles #, Plots
include("nbg_tts.jl")
include("ttvfast_tts.jl")

#ntransits = 500 # Number of transits for each planet
tmax = 4533. # Should give ~3000 transits of planet b
denom = 1000 # P of planet b / denom = time step

#println("Running NBG...")
#output_NBG_times(setup_sim(tmax,denom)...) # NbodyGradient transit times

# Run rebound and record times
println("Running Python...")
run(`python reb_tts.py`) # run from command line

# Run ttvfast and record times
cd("ttvfast/")
#run(`gcc -o ttvfast/run_TTVFast ttvfast/run_TTVFast.c ttvfast/TTVFast.c -lm -O3`)
run(`gcc -o run_TTVFast run_TTVFast.c TTVFast.c -lm -O3`)
#run(`ttvfast/run_TTVFast ttvfast/setup_trappist ttvfast/Times`)
println("Running TTVFast...")
run(`./run_TTVFast setup_trappist Times`)
cd("../")
#run(`cd ttvfast '&&' ./run_TTVFast setup_trappist Times '&&' cd ..`)
output_ttvfast_times()
println("Plotting...")
include("plot_times_new.jl")
