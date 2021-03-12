import rebound
import numpy as np
import csv

def setup_sim(pos, vel, elements, denom=100):
    sim = rebound.Simulation()
    sim.G = 0.000295981944048623 # AU, Days, Msol
    sim.integrator='ias15'
    sim.dt = elements[1,1]/denom

    # Set up initial conditions
    for i in range(pos.shape[0]):
        els = elements[i]
        x,y,z = pos[i]
        vx,vy,vz = vel[i]
        m = els[0]
        sim.add(m=m,x=x,y=y,z=z,vx=vx,vy=vy,vz=vz)

    sim.move_to_com()

    return sim

def compute_tts(pos, vel, elements, tmax, denom):
    """Adapted from the Rebound ttv example"""
    tts = np.zeros((7,int(tmax/elements[1,1])+1))
    mi = 0 # Star index
    planets = [1,2,3,4,5,6,7]
    for pi in planets:
        sim = setup_sim(pos,vel,elements,denom)
        p = sim.particles
        i = 0
        ntransits = tmax/elements[pi,1]
        while i<ntransits:
            # +x is observer direction
            y_old = p[pi].y - p[mi].y
            t_old = sim.t
            sim.integrate(sim.t+sim.dt)
            t_new = sim.t
            if y_old*(p[pi].y-p[mi].y)<0. and p[pi].x-p[mi].x>0:
                while abs(t_new-t_old)>1e-9:
                    if y_old*(p[pi].y-p[0].y)<0.:
                        t_new = sim.t
                    else:
                        t_old = sim.t
                    sim.integrate((t_new+t_old)/2.)
                tts[pi-1,i] = sim.t
                i+=1
                sim.integrate(sim.t+sim.dt)

    return tts[:,3000]

if __name__ == "__main__":
    #ntransits = 500
    tmax = 4533.0
    denom = 100
    elements = np.loadtxt("elements_noprior_students.txt", delimiter=',')
    pos = np.loadtxt("rebound_pos.txt", delimiter=',', unpack=True)
    vel = np.loadtxt("rebound_vel.txt", delimiter=',', unpack=True)
    reb_tts = compute_tts(pos, vel, elements, tmax, denom)

    with open("reb_tts.txt", "w", newline='') as f:
        wtr = csv.writer(f, delimiter=',')
        for i in range(7):
            wtr.writerow(reb_tts[i])
