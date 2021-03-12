using DelimitedFiles, Plots
include("nbg_tts.jl")
include("ttvfast_tts.jl")

#ntransits = 500 # Number of transits for each planet
tmax = 4533. # Should give ~3000 transits of planet b
denom = 100 # P of planet b / denom = time step

output_NBG_times(setup_sim(tmax,denom)...) # NbodyGradient transit times

# Run rebound and record times
run(`python3 reb_tts.py`) # run from command line

# Run ttvfast and record times
run(`gcc -o ttvfast/run_TTVFast ttvfast/run_TTVFast.c ttvfast/TTVFast.c -lm -O3`)
run(`ttvfast/run_TTVFast setup_trappist Times`)
output_ttvfast_times()