using DelimitedFiles, PyPlot, Statistics
#pyplot()

labels = ["Planet b","c","d","e","f","g","h"]
color = ["C0","C1","C2","C3","C4","C5","C6"]
nbg = readdlm("nbg_tts.txt", ',')[:,1:3000]
reb = readdlm("reb_tts.txt", ',')[:,1:3000]
fast = readdlm("ttvfast_tts.txt", ',')[:,1:3000]

fig,axes = subplots(2,2,figsize=(10,5))
reb_diff = nbg .- reb
#plot()
ax = axes[1]
for i in 1:3
    ind = findfirst(isequal(0.0),reb_diff[i,:])
    ind = ind === nothing ? ind = 3000 : ind = ind - 5
    tt = nbg[i,1:ind] ; pmean = mean(tt[2:ind] .- tt[1:ind-1]);
    iref = collect(0:1:ind-1)
    tref = mean(tt .- pmean .* iref) .+ pmean .* iref ; ttv = (tt .- tref) .* 24*60
    ax.plot(nbg[i,1:ind], ttv, label=labels[i],color=color[i])
    ttv = (reb[i,1:ind] .- tref) .* 24*60
    ax.plot(nbg[i,1:ind], ttv, color="k",linestyle=":")
end
ax.legend()
ax.set_title("TTV of inner 3 planets")
ax.set_xlabel("Time [Days]")
ax.set_ylabel("TTV [minutes]")
ax.axis([0,4000,-800,800])
ax = axes[2]
for i in 4:7
    ind = findfirst(isequal(0.0),reb_diff[i,:])
    ind = ind === nothing ? ind = 3000 : ind = ind - 5
    tt = nbg[i,1:ind] ; pmean = mean(tt[2:ind] .- tt[1:ind-1]);
    iref = collect(0:1:ind-1)
    tref = mean(tt .- pmean .* iref) .+ pmean .* iref ; ttv = (tt .- tref) .* 24*60
    ax.plot(nbg[i,1:ind], ttv, label=labels[i],color=color[i])
    ttv = (fast[i,1:ind] .- tref) .* 24*60
    ax.plot(nbg[i,1:ind], ttv, color="k",linestyle=":")
end
ax.set_title("TTV of outer 4 planets")
ax.set_xlabel("Time [Days]")
ax.set_ylabel("TTV [minutes]")
ax.legend(ncol=2)
ax.axis([0,4000,-800,800])

ax = axes[3]
for i in 1:7
    ind = findfirst(isequal(0.0),reb_diff[i,:])
    ind = ind === nothing ? ind = 3000 : ind = ind - 5
    ax.plot(nbg[i,1:ind], reb_diff[i,1:ind] .* 24*60*60, label=labels[i],color=color[i])
end

ax.set_title("Time difference with REBOUND")
ax.set_ylabel("Transit time diff [sec]")
ax.set_xlabel("Time [Days]")
ax.axis([0,4000,-0.02,0.02])
#ax.legend(ncol=2,loc="upper left")
#read(stdin,Char)
#savefig("nbg_vs_reb_transit_times.pdf")

fast_diff = nbg .- fast
#plot()
ax = axes[4]
for i in 1:7
    ind = findfirst(isequal(0.0),fast_diff[i,:])
    ind = ind === nothing ? ind = 3000 : ind = ind - 5
    ax.plot(nbg[i,1:ind], fast_diff[i,1:ind] .* 24*60*60, label=labels[i],color=color[i])
end
ax.set_title("Time difference with TTVFast")
ax.set_ylabel("Transit time diff [sec]")
ax.set_xlabel("Time [Days]")
ax.axis([0,4000,-4,4])
#ax.legend(ncol=2,loc="upper left")
#savefig("nbg_vs_ttvfast_transit_times.pdf")
fig.tight_layout()
savefig("nbg_vs_ttvfast_vs_rebound_transit_times.pdf")
