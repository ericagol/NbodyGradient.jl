# Compares the run time of REBOUND with gradient
# versus NbodyGradient

import rebound
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from timeit import Timer
print(rebound.__build__)

def evaluate(order=0,N=0, nplanet = 0, integrator="ias15"):
    sim = rebound.Simulation()
    sim.integrator = integrator
    sim.add(m=1.)
    semi = 1.
    f0 = 0.0
    for i in range(nplanet):
      sim.add(primary=sim.particles[0],m=1e-5, a=semi, f=f0)
      semi *= 1.8
      f0 += 1.4
    
    vs = ["a","e","i","omega","Omega","f","m"]
    var = []
    if order>=1:
        for i in range(N):
            pi = 1
            shifti = 0
            if i>=7:
                pi = 2
                shifti = -7
            if i>=14:
                pi = 3
                shifti = -14
            if i>=21:
                pi = 4
                shifti = -21
            if i>=28:
                pi = 4
                shifti = -28
            if i>=35:
                pi = 5
                shifti = -35
            if i>=42:
                pi = 6
                shifti = -42
            if i>=49:
                pi = 7
                shifti = -49
            if i>=56:
                pi = 8
                shifti = -56
            if i>=63:
                pi = 9
                shifti = -63
            if i>=70:
                pi = 10
                shifti = -70
            var_d = sim.add_variation()
            var_d.vary(pi,vs[i+shifti])
            var.append(var_d)
    sim.move_to_com()
    sim.dt = np.pi/10.
    sim.integrate(5000.)
#    if order==0:
#        torb = 2.*np.pi
#        Noutputs = 100
#        times = np.linspace(500.+torb, 500.+2.*torb, Noutputs)
#        x = np.zeros([Noutputs,3])
#        for i,time in enumerate(times):
#           sim.integrate(time,exact_finish_time=1)
#           x[i,0] = times[i]
#           x[i,1] = sim.particles[1].x    
#           x[i,2] = sim.particles[1].y
#        np.savetxt('rebound_4body_xy.txt',x,delimiter=',')
    return 

def evaluateWithN(order,N,integrator="ias15",nplanet = 0):
    def _e():
        evaluate(order,N,integrator,nplanet)
        pass
    return _e

for i in range(10):
  Nmax = (i+1)*7+1
  repeat = 1
  t = Timer(evaluateWithN(0,0,i+1))
  var_0 = t.timeit(repeat)/2.
  t = Timer(evaluateWithN(1,Nmax-1,i+1))
  var_1 =  t.timeit(repeat)/2.

  print("Nplanet: ",i+1," No gradient: ",var_0, " gradient: ", var_1, " ratio: ", var_1/var_0)
