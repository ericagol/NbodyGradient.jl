module Orbits
"""
The Orbits module allows user to create figures a la rebound.
Photodynamics.jl uses masses in solar masses 𝑀⊙, time in days, distance in AU, and angles in radians.
The right-handed coordinate system is set up so that the positive z-axis is pointing away from the observer, 
and the positive x-axis points to the right along the horizontal. 

In NbodyGradient the orientation is such that the x-y plane is the sky plane.
For inclinations close to 90 degrees this means that the x-z plane represents the top-down view.
Therefore we choose to take the z coordinates for plotting.
"""
using NbodyGradient,PyPlot,PyCall 
@pyimport matplotlib.animation as anim
"""
#Example for Kepler-16:
stara=Elements(m=0.6897, P=0.0  ,    
    t0=0.0   ,
    ecosω=0.0  ,  
    esinω=0.0  , 
    I=0.0   ,
    Ω=0.0)
starb=Elements(m=0.20255,
   P=41.079220,I=pi/2,e=0.15944)#,t0=212.12316)
pl=Elements(m=0.333*0.00095,P=229.0,I=pi/2,e=0.0069 ) # pl orbiting starb
t0=0.0; tmax=1000.0;h=5.0
ic=ElementsIC(t0,[-1 1 0;-1 -1 1; -1 -1 -1],[stara;starb;pl])
intr=Integrator(h,t0,tmax)
nsteps=100 #number of time steps that we want to simulate and plot, optional
r=Orbits.KeplerOrbits(intr,ic,nsteps) 
Orbits.make_plot(r)
Orbits.animate_plot(r,save=true,filename="Kepler-16.mp4") 
"""
abstract type AbstractOrbit end
struct KeplerOrbits{T <: AbstractFloat} <: AbstractOrbit
    # Cartesian coordinate positions and velocities of each body [dimension, body, step]
    xs::Array{T,3} # in AU
    vs::Array{T,3} # in AU/
    
    m::Vector{T} # Masses of all bodies in solar masses
    P::Vector{T} # Periods of secondary bodies (i.e. everything but primary) in days
    intr::Integrator # Integration scheme (with t0, tmax, and h defined)
    nsteps::Int64  # The number of steps to integrate
    save_interval::Real     # Number of steps to bin (i.e. binsize)
    s::State{T}  # Current state of simulation
    ic::InitialConditions{T}  # InitialConditions for system
end
"""
    KeplerOrbits{T <: AbstractFloat} <: AbstractOrbit

Main KeplerOrbits structure takes integration scheme and InitialConditions.
# Arguments
- nsteps::Int64 : the number of points to save for plotting purposes, tmax = t0 + (h * nsteps).
        If nsteps is not provided, we set the value to the period of outer planet/h

### Optional
- save_interval::Integer    : optional argument for binsize (i.e. save every Nth step)
        Can use save_interval for large N, where you want to plot really long orbits.
        If using save_intergral, increase h to do less steps and make orbits more smooth.
"""
function KeplerOrbits(intr::Integrator,ic::InitialConditions{T})  where (T <: AbstractFloat)
    s=State(ic)
    nbody=s.n
    m=s.m ;
    P=ic.elements[2:nbody,2] 
  """
  If number of integrations not provided, set the value to P_out/h.
  """
   # println("Number of integrations to plot (nsteps) not provided, setting value to outer planet period/h.")
    nsteps=Integer(round(maximum(P)/intr.h) )
    xs=zeros(3,nbody,nsteps);
    vs=zeros(3,nbody,nsteps);
    KeplerOrbits(xs,vs,m,P,intr,nsteps,0,s,ic)
end
function KeplerOrbits(intr::Integrator,ic::InitialConditions{T},nsteps::Int64)  where (T <: AbstractFloat)
        s=State(ic)
        nbody=s.n 
        xs=zeros(3,nbody,nsteps);
        vs=zeros(3,nbody,nsteps);
        m=s.m ;
        P=ic.elements[2:nbody,2]
        KeplerOrbits(xs,vs,m,P,intr,nsteps,0,s,ic)
    end
function KeplerOrbits(intr::Integrator,ic::InitialConditions{T},nsteps::Int64,save_interval::Int64)  where (T <: AbstractFloat)
        @assert (nsteps >= save_interval)
         s=State(ic)
        nbody=s.n 
        nsaves=Integer(round(nsteps/save_interval))
        xs=zeros(3,nbody,nsaves);
        vs=zeros(3,nbody,nsaves);
        m=s.m ;
        P=ic.elements[2:nbody,2]
        KeplerOrbits(xs,vs,m,P,intr,nsteps,save_interval,s,ic)
    end
# KeplerOrbits(intr,ic;nsteps::Int64,save_interval::Int64)=KeplerOrbits(intr,ic,nsteps,save_interval)
""" Allow keywords for nsteps."""
KeplerOrbits(intr,ic;nsteps::Int64)=KeplerOrbits(intr,ic,nsteps)
"""
    SimOrbits!(o::KeplerOrbits{T};grad=false)

Does integration for nsteps.
### Optional
- grad::Bool : Choose whether to calculate gradients. (Default = false)
"""
function SimOrbits!(o;grad::Bool=false) where T <: AbstractFloat
    xs=o.xs
    vs=o.vs
    nsteps=o.nsteps ;   save_interval=o.save_interval 
    ic=o.ic ;   intr=o.intr 
    s=State(ic)
    nbody=s.n
    init_state=deepcopy(s)
    t0 = s.t[1] # Initial time
    # Integrate in proper direction
    h = intr.h * NbodyGradient.check_step(t0,intr.tmax)
    tmax = t0 + (h * nsteps) 
#     @show t0,intr.tmax,h,tmax,nsteps
    # Preallocate struct of arrays for derivatives (and pair)
    if grad; d = Derivatives(T,s.n); pair = zeros(Bool,s.n,s.n) ;end
    if !isinf(nsteps/save_interval)
    nsaves=Integer(round(nsteps/save_interval))
    for j in 1:nsaves    
        for i=1:nsteps
            if nsteps%save_interval==0
            # Take integration step and advance time
            if grad
                intr.scheme(s,d,h)
            else
                intr.scheme(s,h)
            end
            s.t[1] = t0 +  (h * i)
            # Save State from current step
            xs[:,:,j] = s.x[:,:]
            vs[:,:,j] = s.v[:,:]
        end
        end
        end
    else
    for i=1:nsteps
        # Take integration step and advance time
        if grad
            intr.scheme(s,d,h)
        else
            intr.scheme(s,h)
        end
        s.t[1] = t0 +  (h * i)
        # Save State from current step
        xs[:,:,i] = s.x[:,:]
        vs[:,:,i] = s.v[:,:]
    end
    end
    return 
end
export KeplerOrbits, SimOrbits! 
# Set up figure for orbits
function setup()#,fig=Figure,figsize=(4,4),projection="xz",show_primary=true)
    show_primary=true
    fig,ax=plt.subplots(figsize=(4,4),dpi=100)
    ax.set_xlabel("x [au]")
    ax.set_ylabel("z [au]")
    if show_primary
    pc=ax.scatter([],[],s=25,color="black")
    end
    pc=ax.scatter([],[],s=25,color="black")
    ax=gca();fig=gcf()
    return ax,fig
end
julia_colors=["#000000","#389826","#CB3C33","#9558B2","#4063D8"]
"""
    make_plot(o::KeplerOrbits{T}) 

Plot static orbits, with orbital path as faded line and location of bodies at the initial state.
# Optional Arguments
- show_primary::Bool    : plots initial position of the primary star. Never plots the path. 
- legend::Bool : Choose whether to show legend. (Default = false)
- lw::Real : Linewidth
- colors::Vector{String} : user provided list of colors to use for orbits.
"""
function make_plot(o,save=false,filename::String="../test/test_orbits.png",
show_primary=true,legend=false,lw=1.5;colors::Vector{String}=julia_colors) where T<:AbstractFloat
    nbody=o.ic.nbody ; nsteps=o.nsteps ;
    SimOrbits!(o)
    ax,fig=setup();
    fig.set_figheight(4);fig.set_figwidth(4)
    if show_primary
        ax.scatter(o.s.x0[1,1],o.s.x0[3,1],marker="*",color="black",s=35*lw)
    end
    for body in 2:nbody
        ax.scatter(o.s.x[1,body],o.s.x[3,body],s=25*lw,color=colors[body],zorder=3)
        ax.plot(o.xs[1,body,:],o.xs[3,body,:],alpha=0.5,lw=lw,color=colors[body])
    end
    if legend
        fig.legend()
    end
    fig.tight_layout()
    if save
    fig.savefig(filename)
    end
    return  fig
end
"""
    animate_plot(o::KeplerOrbits{T}) 

Animates orbits, with orbital path as transparent line that is same color as the body.
# Optional Arguments
- colors::Vector{String} : user provided list of colors to use for orbits.
"""
function animate_plot(o,save=false,file_name::String="../test/test2.mp4";colors::Vector{String}=julia_colors) where T<:AbstractFloat
    nbody=o.ic.nbody;nsteps=o.nsteps;h=o.intr.h;save_interval=o.save_interval
    if h < 1.0 && nsteps < 100
    """
     This will create an animation which may not cover a full orbit for long orbits.
    """
        println("Your intergration timestep ($h) is low, and the animation may not cover a full orbit.")
    end
    fig=make_plot(o)
    ax=fig.gca()
    limits_x=maximum(abs.(o.xs[1,:,:])) *1.1;limits_y=maximum(abs.(o.xs[3,:,:]))*1.1
    xlabel=ax.get_xlabel();ylabel=ax.get_ylabel()
    show_primary=true
    for body in 2:nbody
        ax.plot(o.xs[1,body,:],o.xs[3,body,:],alpha=0.5,color=colors[body])
    end
    function update_plot(i)
        frame_mult=1 # can change to make faster animation. may have unexpected results
        if i*frame_mult < nsteps
        ax.clear();ax.set_ylim(-limits_y,limits_y);ax.set_xlim(-limits_x,limits_x)
        ax.set_ylabel(ylabel);ax.set_xlabel(xlabel);fig.tight_layout()
        if show_primary
            ax.scatter(o.xs[1,1,i*frame_mult],o.xs[3,1,i*frame_mult],s=35,marker="*",color="black")
        end
        for body in 2:nbody
        ax.plot(o.xs[1,body,:],o.xs[3,body,:],alpha=0.5,color=colors[body])
        ax.scatter(o.xs[1,body,i*frame_mult],o.xs[3,body,i*frame_mult],color=colors[body])
        end
        end
    return ax.get_lines()
    end
    if isinf(nsteps/save_interval)
        """
        If save_interval or nsteps is not defined, save 200 steps. 
        """
        nsaves=200
    else
        nsaves=nsteps
    end
    frames=[1:nsaves;]
    movie=anim.FuncAnimation(fig,update_plot,frames=frames,repeat=true,blit=true,interval=40)
    if save
     anim.FuncAnimation.save(movie,file_name)
    end
    return 
end
animate_plot(o;save::Bool=false,filename::String="test.mp4")=animate_plot(o,save,filename)
isnanall(x...) = all(isnan.(x))
function pos_wrt_primary(r0,r2,P,t) # not in 3D, only for simple_plot()
    x0=r0[1];y0=r0[3]
    x=r2[1];y=r2[3]
    d=sqrt(abs((x-x0).^2 .- (y-y0) .^2))
    n=2*π/P
    new_x= x0 + (d *cos(n*t))
    new_y= y0 + (d * sin(n*t))
    return [new_x new_y]
end
function new_pos(o,primary_indx,secondary_indx)
    P=o.ic.elements[secondary_indx,2] # mean motion of satellite about barycenter with host
    pos_about_parent=zeros(o.nsteps,2);
    for i=1:o.nsteps-1
       pos_about_parent[i,:]=pos_wrt_primary(o.s.x[:,primary_indx] ,o.s.x[:,secondary_indx],P,i)
    end
    return pos_about_parent
end
"""
    simple_plot(o::KeplerOrbits{T},parent::Vector{T}) where T <: Real

Plotting objects with respect to their parent (e.g. satellite around planet)
# Argument 
- parent::Vector{T} : indeces of parent of each body, where the main central star has parent=0
Example: if one planet orbiting a star has a satellite, the parent vector would be =[0, 1, 2]
"""
# needs work because broken 
function simple_plot(o,parent::Vector{Real};colors::Bool=true,
        xlim::Tuple{Real,Real}=(NaN,NaN),zlim::Tuple{Real,Real}=(NaN,NaN),lw::Real=NaN) where T<:AbstractFloat
    if isnan(lw);  lw=1.5    end
    nbody=o.ic.nbody;     nsteps= o.nsteps ;    
    if in(false,colors)
    color_list=[]
        for i=1:nbody
           push!(color_list,"black")
           end
    elseif in(true,colors)
     color_list=julia_colors
    end
    X=NbodyGradient.get_relative_positions(o.s.x,o.s.v,o.ic)[1]
    rel_pos=zeros(nbody,3)
    for i=1:nbody
        rel_pos[i,:]=[NbodyGradient.unpack.(X)[i]...]
    end
    factor=1.15
    fig=make_plot(o)
    ax=fig.gca()
    fig.set_figheight(4);fig.set_figwidth(4)
    if  isnanall(xlim[1],xlim[2],zlim[1],zlim[2])
        extent=maximum(abs.(rel_pos))
        xlim=(-extent*factor,extent*factor) ; zlim=(-extent*factor,extent*factor)
        ax.set_xlim(xlim);ax.set_ylim(zlim)
    else
        ax.set_xlim(xlim);ax.set_ylim(zlim)
    end
    for body in 1:nbody
        primary=parent[body]
        if primary !=1 && !iszero(primary) #&& body>1
           pos=zeros(nsteps,2)
            for i=1:nsteps
             pos[i,:]=new_pos(o,primary,body)
            ax.scatter(pos[i,1].*factor,pos[i,2].*factor,alpha=0.05,color=color_list[body],s=lw)
            end
        return pos
        end
    end
    fig.tight_layout()
end
# VisualOrbit(o,parent;colors::Bool,xlim::Tuple{T,T},zlim::Tuple{T,T},lw::T) where T<:Real =VisualOrbit(o,parent,colors,xlim,zlim,lw)
# TO DO: display output of animation as html in jupyter notebook cell or .gif
# TO DO: make trail for orbit path with line collections
# TO DO: 3D cartesian coord transformation for satellite   
end # module
#using .Orbits
