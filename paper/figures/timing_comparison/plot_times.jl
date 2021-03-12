using DelimitedFiles, Plots
pyplot()

labels = ["Planet b","c","d","e","f","g","h"]
nbg = readdlm("nbg_tts.txt", ',')[:,1:3000]
reb = readdlm("reb_tts.txt", ',')[:,1:3000]
fast = readdlm("ttvfast_tts.txt", ',')[:,1:3000]

reb_diff = nbg .- reb
plot()
for i in 1:7
    ind = findfirst(isequal(0.0),reb_diff[i,:])
    ind = ind === nothing ? ind = 3000 : ind = ind - 5
    plot!(nbg[i,1:ind], reb_diff[i,1:ind] .* 86400, label=labels[i])
end
ylabel!("Difference in transit times [sec]")
xlabel!("Time [Days]")
savefig("nbg_vs_reb_transit_times.pdf")

fast_diff = nbg .- fast
plot()
for i in 1:7
    ind = findfirst(isequal(0.0),fast_diff[i,:])
    ind = ind === nothing ? ind = 3000 : ind = ind - 5
    plot!(nbg[i,1:ind], fast_diff[i,1:ind] .* 86400, label=labels[i])
end
ylabel!("Difference in transit times [sec]")
xlabel!("Time [Days]")
savefig("nbg_vs_ttvfast_transit_times.pdf")