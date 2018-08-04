include("nbody.jl")

t0 = 7257.93115525
h  = 0.015
tmax = 600.0
nbody(t0,h,tmax)

using PyPlot
data = readdlm("fcons.txt")
#for i=1:8
#  plot(data[:,(i-1)*4+2],data[:,(i-1)*4+3],".")
#end
plot(data[:,2],data[:,3],".")
nt = 40000

using PyCall

@pyimport matplotlib.animation as anim

#Construct Figure and Plot Data
fig = figure("TRAPPIST1A_motion",figsize=(10,10))
#ax = axes(xlim = (0,10),ylim=(0,10))
#fig = figure()
xmin = minimum(data[:,2]) ; xmax = maximum(data[:,2])
ymin = minimum(data[:,3]) ; ymax = maximum(data[:,3])
ax = axes(xlim = (xmin,xmax),ylim=(ymin,ymax))
global line1 = ax[:plot]([],[],"r-")[1]
global p1 = ax[:plot]([],[],"or")[1]

function init()
    global line1
    global p1
    line1[:set_data]([],[])
    p1[:set_data]([],[])
    return (line1,p1,Union{})
end

step = 50
function animate(i)
    k = i + 1
    global line1
    global p1
    line1[:set_data](data[max(1,step*(k-50)):(step*k),2],data[max(1,step*(k-50)):(step*k),3])
#    line1[:set_data](data[max(1,step*(k-10)):(step*k),2],data[max(1,step*(k-10)):(step*k),3])
    p1[:set_data]([data[step*k,2]],data[step*k,3])
    return (line1,Union{})
end

#Call the animator.
myanim = anim.FuncAnimation(fig, animate, init_func=init, frames=floor(Int64,nt/step), interval=50)

#This will require ffmpeg or equivalent.
#myanim[:save]("T1A_motion.mp4", bitrate=-1, extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
myanim[:save]("T1A_motion.mp4", bitrate=-1)
#myanim[:save]("T1A_motion.mp4")
