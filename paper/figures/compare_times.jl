
using DelimitedFiles
using PyPlot

#rebound = readdlm("rebound_times.txt",',',comments=true)
#nbodygrad = readdlm("nbodygradient_times.txt",',')
#nbodygrad_pair = readdlm("nbodygradient_times_pair.txt",',')
using DelimitedFiles
#rebound = readdlm("reb_times.txt",',')
rebound = readdlm("rebound_times_updated.txt",',',comments=true)
nbodygrad = readdlm("nbg_times.txt",',',comments=true)


using PyPlot

clf()
semilogy(nbodygrad[:,1],nbodygrad[:,2],label="NbodyGrad, no gradient",linestyle="dashed",c="b",linewidth=2)
semilogy(nbodygrad[:,1],nbodygrad[:,2],".",c="b",markersize=15)
semilogy(nbodygrad[:,1],nbodygrad[:,3],label="NbodyGrad, gradient",c="b",linewidth=2)
semilogy(nbodygrad[:,1],nbodygrad[:,3],".",c="b",markersize=15)
semilogy(nbodygrad[:,1],nbodygrad[:,4],label="NbodyGrad, no gradient, adjacent Kepler",linestyle="dashed",c="g",linewidth=2)
semilogy(nbodygrad[:,1],nbodygrad[:,4],".",c="g",markersize=15)
semilogy(nbodygrad[:,1],nbodygrad[:,5],label="NbodyGrad, gradient, adjacent Kepler",c="g",linewidth=2)
semilogy(nbodygrad[:,1],nbodygrad[:,5],".",c="g",markersize=15)
semilogy(nbodygrad[:,1],nbodygrad[:,6],label="NbodyGrad, tt w/ gradient",c="m",linewidth=2)
semilogy(nbodygrad[:,1],nbodygrad[:,6],".",c="m",markersize=15)
semilogy(rebound[:,1],rebound[:,2],label="REBOUND, no gradient",linestyle="dashed",c="r",linewidth=2)
semilogy(rebound[:,1],rebound[:,2],".",c="r",markersize=15)
semilogy(rebound[:,1],rebound[:,3],label="REBOUND, gradient",c="r",linewidth=2)
semilogy(rebound[:,1],rebound[:,3],".",c="r",markersize=15)
legend(loc="lower right",fontsize=8)
xlabel("Number of planets",fontsize=20)
ylabel("Integration time [sec]",fontsize=20)
#PyPlot.savefig("rebound_vs_nbodygradient.pdf",bbox_inches="tight")
PyPlot.savefig("reb_vs_nbg.pdf", bbox_inches="tight")
