


using DelimitedFiles
using PyPlot

#rebound = readdlm("rebound_times.txt",',',comments=true)
#nbodygrad = readdlm("nbodygradient_times.txt",',')
#nbodygrad_pair = readdlm("nbodygradient_times_pair.txt",',')
using DelimitedFiles
#rebound = readdlm("reb_times.txt",',')
rebound_ias15 = readdlm("rebound_times_ias15.txt",',',comments=true)
rebound_whfast = readdlm("rebound_times_whfast.txt",',',comments=true)
nbodygrad = readdlm("nbg_times.txt",',',comments=true)


using PyPlot

clf()
semilogy(rebound_ias15[:,1],rebound_ias15[:,3],label="IAS15, gradient",c="r",linewidth=2)
semilogy(rebound_ias15[:,1],rebound_ias15[:,3],".",c="r",markersize=15)
semilogy(nbodygrad[:,1],nbodygrad[:,6],label="NBG, tt w/ gradient",c="b",linewidth=2,linestyle=":")
semilogy(nbodygrad[:,1],nbodygrad[:,6],".",c="b",markersize=15)
semilogy(nbodygrad[:,1],nbodygrad[:,3],label="NBG, gradient",c="b",linewidth=2)
semilogy(nbodygrad[:,1],nbodygrad[:,3],".",c="b",markersize=15)
semilogy(nbodygrad[:,1],nbodygrad[:,5],label="NBG, gradient, adjacent Kepler",c="g",linewidth=2)
semilogy(nbodygrad[:,1],nbodygrad[:,5],".",c="g",markersize=15)
semilogy(rebound_whfast[:,1],rebound_whfast[:,3],label="WHFAST, gradient",c="m",linewidth=2)
semilogy(rebound_whfast[:,1],rebound_whfast[:,3],".",c="m",markersize=15)

semilogy(rebound_ias15[:,1],rebound_ias15[:,2],label="IAS15, no gradient",linestyle="dashed",c="r",linewidth=2)
semilogy(rebound_ias15[:,1],rebound_ias15[:,2],".",c="r",markersize=15)
semilogy(nbodygrad[:,1],nbodygrad[:,2],label="NBG, no gradient",linestyle="dashed",c="b",linewidth=2)
semilogy(nbodygrad[:,1],nbodygrad[:,2],".",c="b",markersize=15)
semilogy(nbodygrad[:,1],nbodygrad[:,4],label="NBG, no gradient, adjacent Kepler",linestyle="dashed",c="g",linewidth=2)
semilogy(nbodygrad[:,1],nbodygrad[:,4],".",c="g",markersize=15)
semilogy(rebound_whfast[:,1],rebound_whfast[:,2],label="WHFAST, no gradient",linestyle="dashed",c="m",linewidth=2)
semilogy(rebound_whfast[:,1],rebound_whfast[:,2],".",c="m",markersize=15)
grid(linestyle=":")
legend(loc="upper_left",fontsize=8)
axis([0,11,3e-3,1e2])
xlabel("Number of planets",fontsize=20)
ylabel("Integration time [sec]",fontsize=20)
#minorticks_on()
#tick_params(labeltop="true",labelright="true")
tick_params(axis="both",direction="inout",which="both",right="true",top="true",labeltop="true",labelright="true")
#secondary_yaxis("right")
#PyPlot.savefig("rebound_vs_nbodygradient.pdf",bbox_inches="tight")
PyPlot.savefig("reb_vs_nbg.pdf", bbox_inches="tight")
