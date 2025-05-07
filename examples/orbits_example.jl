# examples
include("../src/VisualizaNbodyGradient.jl")
 #Example for Kepler-16 with additional planet orbiting star B:
    stara=Elements(m=1.0) # Star A
    starb=Elements(m=0.20255,P=0.1591*365.2,I=pi/2) # Star B
    plb=Elements(m=0.333*0.00095,P=164.3753 ,I=pi/2) # Circumbinary planet ABb
    plbb = Elements(m=0.333*0.00095,P=5.1980,I=pi/2)# Planet Bb, orbiting star B 
    end
    t0=0.0; tmax=500.0;h=0.5
    H = [-1 1 1 0;0 -1 1 0; -1 -1 -1 1; -1 -1 -1 -1]
    ic=ElementsIC(t0,H,[stara;starb;plb;plbb])
    nsteps=340
    r=Keplerians(intr,ic,nsteps;names=["A","B","Bb","ABb"]) # nsteps to visualize
    # r=Keplerians(ic,t0,tmax,h;names=["A","B","Bb","ABb"])  
    make_plot(r)