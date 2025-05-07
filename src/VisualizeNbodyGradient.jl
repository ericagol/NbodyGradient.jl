module VisualizeNbodyGradient
    # """
    #     This module allows user to create figures a la rebound from NbodyGradient, which uses masses in solar masses 𝑀⊙, time in days, distance in AU, and angles in radians.
    #     The right-handed coordinate system is set up so that the positive z-axis is pointing away from the observer, 
    #     and the positive x-axis points to the right along the horizontal. 

    #     In NbodyGradient the orientation is such that the x-y plane is the sky plane.
    #     For inclinations close to 90 degrees this means that the x-z plane represents the top-down view.
    #     Therefore we choose to take the z coordinates for plotting.
    # """
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
    r=Keplerians(intr,ic,nsteps) 
    make_plot(r)
    animate_plot(r,save=true,filename="Kepler-16.mp4") 
"""
abstract type AbstractOrbit end
struct Keplerians{T <: AbstractFloat} <: AbstractOrbit
    # Cartesian coordinate positions and velocities of each body [dimension, body, step]
    xs::Array{T,3} # in AU
    vs::Array{T,3} # in AU/day
    intr::Integrator # Integration scheme (with t0, tmax, and h defined)
    nsteps::Int64  # The number of steps to integrate
    save_interval::Real     # Number of steps to bin (i.e. binsize)
    s::State{T}  # Current state of simulation
    ic::InitialConditions{T}  # InitialConditions for system
    # primary::Vector{T}
    names::Vector{String}
end
"""
    Keplerians{T <: AbstractFloat} <: AbstractOrbit

Main Keplerians structure must be built with an InitialConditions instance.
        Currently assumes that the hierarchy in the InitialConditions is correctly setup for the desired system. 
    # TODO: 
    # -figure out how to extract primary/secondary indices of objects based on heirarchy.
# Arguments
- Integrator
 OR
- t0::Real : start time for integration
- tmax::Real : end time for integration 
- h::Real : integrator step size

### Optional
- nsteps::Int64 : the number of points to save for plotting purposes, tmax = t0 + (h * nsteps).
        If nsteps is not provided, we set the value to the period of outer planet
- save_interval::Integer    : optional argument for binsize (i.e. save every Nth step)
        Can use save_interval for large N, where you want to plot really long orbits.
        If using save_intergral, increase h to take less integration steps and make orbit paths more smooth.
- names:: Vector{String} : names of simulated object orbits, listed as designated in the heirarchy.
"""
function Keplerians(ic::InitialConditions{T},t0::T,tmax::T,h::T;names::Vector{String}=[]) where (T <: AbstractFloat)
    intr=Integrator(h,t0,tmax)
    nsteps=tmax-t0
    xs=zeros(3,nbody,nsteps);
    vs=zeros(3,nbody,nsteps);
    if isempty(names)
        names = ["body_$i" for i = 1:nbody]
    end
    Keplerians(xs,vs,intr,nsteps,0,s,ic,names)
end
# If user wants to provide existing Integrator
function Keplerians(intr::Integrator,ic::InitialConditions{T};names::Vector{String}=[])  where (T <: AbstractFloat)
    s=State(ic)
    nbody=s.n
    if intr.h < 1 # update intr to a more sensible value
        intr.h = 1
    end
    P =  ic.elements[:,2]
     # If the number of integrations (i.e. steps) to plot is not provided, find an appropriate value.
    nsteps=Integer(clamp(round(maximum(P)),50,200))
    xs=zeros(3,nbody,nsteps);
    vs=zeros(3,nbody,nsteps);

    if isempty(names)
        names = ["body_$i" for i = 1:nbody]
    end
    Keplerians(xs,vs,intr,nsteps,0,s,ic,names)
end
""" Allow keywords for nsteps."""
function Keplerians(intr::Integrator,ic::InitialConditions{T};nsteps::Int64,names::Vector{String}=[])  where (T <: AbstractFloat)
        s=State(ic)
        nbody=s.n 
        xs=zeros(3,nbody,nsteps);
        vs=zeros(3,nbody,nsteps);
    if isempty(names)
        names = ["body_$i" for i = 1:nbody]
    end
        Keplerians(xs,vs,intr,nsteps,0,s,ic,names)
end
function Keplerians(intr::Integrator,ic::InitialConditions{T};nsteps::Int64,save_interval::Int64,names::Vector{String}=[])  where (T <: AbstractFloat)
        @assert (nsteps >= save_interval)
         s=State(ic)
        nbody=s.n 
        nsaves=Integer(round(nsteps/save_interval))
        xs=zeros(3,nbody,nsaves);
        vs=zeros(3,nbody,nsaves);
        if isempty(names)
            names = ["body_$i" for i = 1:nbody]
        end
        Keplerians(xs,vs,intr,nsteps,save_interval,s,ic,names)
end

# Get masses (in solar masses) and periods (in days) of all bodies 
get_masses(o) = o.s.m
get_periods(o) = o.ic.elements[:,2]
_get_heirarchy(o) = o.ic.ϵ
    # nsteps=Integer(round(maximum(P)/intr.h) )
"""
    SimOrbits!(o::Keplerians{T})

Does integration for nsteps
"""
function SimOrbits!(o) where T <: AbstractFloat
    xs=o.xs
    vs=o.vs
    nsteps=o.nsteps ;   
    save_interval=o.save_interval 
    ic=o.ic ;   
    intr=o.intr 
    s=State(ic)
    nbody=s.n
    init_state=deepcopy(s)
    t0 = s.t[1] # Initial time
    # Integrate in proper direction
    h = intr.h * NbodyGradient.check_step(t0,intr.tmax)
    tmax = t0 + (h * nsteps) 

    if !isinf(nsteps/save_interval)
        nsaves=Integer(round(nsteps/save_interval))
        for j in 1:nsaves    
            for i=1:nsteps
                if nsteps%save_interval==0
                # Take integration step and advance time
                    intr.scheme(s,h)
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
            intr.scheme(s,h)
            s.t[1] = t0 +  (h * i)
            # Save State from current step
            xs[:,:,i] = s.x[:,:]
            vs[:,:,i] = s.v[:,:]
        end
    end
    return 
end

# Set up figure for static plot and animation
function _setup(show_primary::Bool)
    fig,ax=plt.subplots(figsize=(4,4),dpi=100)
    ax.set_xlabel("x [au]")
    ax.set_ylabel("z [au]")
    if show_primary
        pc=ax.scatter([],[],s=25,color="black")
    end
    pc=ax.scatter([],[],s=25,color="black")
    return fig,ax
end

function _setup_set(show_primary::Bool)
    fig,ax=plt.subplots(2,2,figsize=(4,4),dpi=100)
    ax[2].set_xlabel("x [au]")
    ax[2].set_ylabel("y [au]")
    ax[1].set_ylabel("z [au]")
    ax[4].set_xlabel("z [au]")
    ax[1].xaxis.set_visible(false)
    ax[4].yaxis.set_visible(false)
    ax[3].set_frame_on(false)
    ax[3].set_xticks([]);ax[3].set_yticks([])
    if show_primary
        pc=ax[2].scatter([],[],s=25,color="black")
    end
        pc=ax[2].scatter([],[],s=25,color="black")
    return fig,ax
end
julia_colors=["#000000","#389826","#CB3C33","#9558B2","#4063D8"] # starts with black
"""
    make_plot(o::Keplerians{T}) 

Plot static orbits, with orbital path as faded line and location of bodies at the initial state.
# Optional Arguments
- show_primary::Bool        : plots initial position of the primary star. Never plots the path. 
- lw::Real                  : Linewidth
- ms::Real                  : Markersize
- colors::Vector{String}    : user provided list of colors to use for orbits.
- legend::Bool              : Choose whether to show legend. (Default = false)

"""
function make_plot(o,save::Bool=false,filename::String="../test/test_orbits.png",
show_primary=true,lw::Real=1.5,ms::Real=25;colors::Vector{String}=[],legend::Bool=false,figsize::Tuple{T,T}=(4,4),dpi=100) where T<:Real
    nbody=o.ic.nbody ; 
    nsteps=o.nsteps ;
    SimOrbits!(o)
    fig,ax=_setup(show_primary);
    fig.set_figheight(figsize[1]);
    fig.set_figwidth(figsize[2])
    if show_primary
        ax.scatter(o.s.x0[1,1],o.s.x0[3,1],marker="*",color="black",s=35*lw)
    end
    # Update colors.
    if isempty(colors)
        colors=julia_colors
    elseif  colors==["black"]
        colors = repeat(["black"],nbody)
    end
    for body in 2:nbody
        ax.scatter(o.s.x[1,body],o.s.x[3,body],s=ms*lw,color=colors[body],zorder=3,label=o.names[body])
        ax.plot(o.xs[1,body,:],o.xs[3,body,:],alpha=0.5,lw=lw,color=colors[body])
    end
    if legend
        fig.legend()
    end
    fig.tight_layout()
    if save
        fig.savefig(filename,dpi)
    end
    return  fig,ax
end

function make_plot_set(o,save::Bool=false,filename::String="../test/test_orbits.png",show_primary=true,lw::Real=1.5,ms::Real=25;colors::Vector{String}=["black"],legend::Bool=false,figsize::Tuple{T,T}=(4,4)) where T <:Real
    nbody=o.ic.nbody ; 
    nsteps=o.nsteps ;
    SimOrbits!(o)
    fig,ax=_setup_set(show_primary);
    fig.set_figheight(figsize[1]);
    fig.set_figwidth(figsize[2])    
    if show_primary
        ax[1].scatter(o.s.x0[1,1],o.s.x0[3,1],marker="*",color="black",s=35*lw)
        ax[2].scatter(o.s.x0[1,1],o.s.x0[2,1],marker="*",color="black",s=35*lw)
        ax[4].scatter(o.s.x0[3,1],o.s.x0[1,1],marker="*",color="black",s=35*lw)
    end
    # Update colors.
    if isempty(colors)
        colors=julia_colors
    elseif  colors==["black"]
        colors = repeat(["black"],nbody)
    end
    for body in 2:nbody
        ax[1].scatter(o.s.x[1,body],o.s.x[3,body],s=ms*lw,color=colors[body],zorder=3)
        ax[1].plot(o.xs[1,body,:],o.xs[3,body,:],alpha=0.5,lw=lw,color=colors[body])
        ax[2].scatter(o.s.x[1,body],o.s.x[2,body],s=ms*lw,color=colors[body],zorder=3,label=o.names[body])
        ax[2].plot(o.xs[1,body,:],o.xs[2,body,:],alpha=0.5,lw=lw,color=colors[body])
        ax[4].scatter(o.s.x[3,body],o.s.x[2,body],s=ms*lw,color=colors[body],zorder=3)
        ax[4].plot(o.xs[3,body,:],o.xs[2,body,:],alpha=0.5,lw=lw,color=colors[body])
    end
    if legend
        fig.legend()
    end
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.0,hspace=0.0,right=0.98,top=0.98)

    if save
    fig.savefig(filename)
    end
    return  fig,ax
end
"""
    animate_plot(o::Keplerians{T}) 

Animates orbits, with orbital path as transparent line that is same color as the body.
# Optional Arguments
- show_primary::Bool        : plots initial position of the primary star. Never plots the path. 
- lw::Real                  : Linewidth
- ms::Real                  : Markersize
- colors::Vector{String}    : user provided list of colors to use for orbits.
- legend::Bool              : Choose whether to show legend. (Default = false)
- figsize::Tuple            
"""
function animate_plot(o,save::Bool=false,file_name::String="../test/test2.mp4",show_primary::Bool=true,lw::Real=1.5,ms::Real=25;colors::Vector{String}=[],legend::Bool=false,figsize::Tuple{T,T}=(4,4)) where T<:Real
    nbody=o.ic.nbody;
    nsteps=o.nsteps;
    h=o.intr.h;
    save_interval=o.save_interval
    # if h < 1.0 && nsteps < 100
     # This will create an animation which may not cover a full orbit for long period objects.
    # end
    fig,ax=make_plot(o;colors,legend,figsize)
    limits_x=maximum(abs.(o.xs[1,:,:]))*1.1
    limits_y=maximum(abs.(o.xs[3,:,:]))*1.1
    xlabel=ax.get_xlabel()
    ylabel=ax.get_ylabel()
    for body in 2:nbody
        ax.plot(o.xs[1,body,:],o.xs[3,body,:],alpha=0.5,color=colors[body],lw=lw)
    end
    function update_plot(i)
        frame_mult=1 # can change to make faster animation. may have unexpected results
        if i*frame_mult < nsteps
        ax.clear();
        ax.set_ylim(-limits_y,limits_y);
        ax.set_xlim(-limits_x,limits_x)
        ax.set_ylabel(ylabel);
        ax.set_xlabel(xlabel);
        if show_primary
            ax.scatter(o.xs[1,1,i*frame_mult],o.xs[3,1,i*frame_mult],s=35*lw,marker="*",color="black")
        end
        # Update colors.
        if isempty(colors)
            colors=julia_colors
        elseif  colors==["black"]
            colors = repeat(["black"],nbody)
        end
        for body in 2:nbody
        ax.plot(o.xs[1,body,:],o.xs[3,body,:],alpha=0.5,color=colors[body],lw=lw)
        ax.scatter(o.xs[1,body,i*frame_mult],o.xs[3,body,i*frame_mult],color=colors[body],s=ms*lw)
        end
        end
    return ax.get_lines()
    end
    if isinf(nsteps/save_interval) 
        # If save_interval or nsteps is not defined, save 200 steps. 
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

export Keplerians, SimOrbits! 
export animate_plot,make_plot,make_plot_set

function _pos_wrt_primary(r0,r2,P,t) # not in 3D, only for simple_plot()
    x0=r0[1];y0=r0[3]
    x=r2[1];y=r2[3]
    d=sqrt(abs((x-x0).^2 .- (y-y0) .^2))
    n=2*π/P
    new_x= x0 + (d *cos(n*t))
    new_y= y0 + (d * sin(n*t))
    return [new_x new_y]
end

function _new_pos(o,primary_indx,secondary_indx)
    P=o.ic.elements[secondary_indx,2] # mean motion of satellite about barycenter with host
    pos_about_parent=zeros(o.nsteps,2);
    for i=1:o.nsteps-1
       pos_about_parent[i,:]=_pos_wrt_primary(o.s.x[:,primary_indx] ,o.s.x[:,secondary_indx],P,i)
    end
    return pos_about_parent
end

function _transform_coord_system(Ω,ω,I,r)
    x,y,z=r
    X = x * (cos(Ω)* cos(ω) - sin(Ω)*sin(ω)*cos(I)) - y *(cos(Ω)* sin(ω) - sin(Ω)*cos(ω)*cos(I))
    Y = x * (sin(Ω)* cos(ω) - cos(Ω)*sin(ω)*cos(I)) - y *(sin(Ω)* sin(ω) - cos(Ω)*cos(ω)*cos(I))
    Z = x * sin(ω) * sin(I) + y * cos(ω)*sin(I)
    return X,Y,Z
end

function _update_3D_cartesian(o)
    intr= o.intr 
    nbody = o.ic.nbody
    elems = o.ic.elements
    ω = atan.(elems[:,4]./elems[:,5])
    I = elems[:,6]
    Ω = elems[:,7]
    new_xs=zeros(3,nbody,o.nsteps);
    # transform all objects to a common coordinate system
    for body=2:nbody
        for i=1:o.nsteps
            new_xs[:,body,i] .= _transform_coord_system(Ω[body],ω[body],I[body],o.xs[1:3,body,i])
        end
    end
    return new_xs
end
end # module