#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "universal.h"

void compute_a(double kc, State state, double *a)
{
  double r, vs;

  r = sqrt(state.x * state.x + state.y * state.y + state.z * state.z);
  vs = state.xd * state.xd + state.yd * state.yd + state.zd * state.zd;

  *a = 1.0 / (2.0 / r - vs / kc);
}

int main(int argc, char** argv)
{
  double kc;
  double a, e, beta, Q;
  State s, sp;
  double a0, ap;
  double dt;
  void kepler_step();

  /* H(t, x, p) = p^2/(2m) - mu/r */
  /* kepler constant = mu/m */
  kc = 0.0172*0.0172; 
  /* units are AU, day, solar mass */
  
  a = 0.4;  /* AU */
  e = 0.2;
  dt = 1.0; /* days */
  
  beta = kc/a;
  Q = a*(1.0 + e);   /* the apocentric distance */

  s.x = Q;
  s.y = 0.0;
  s.z = 0.0;
  s.xd = 0.0;
  s.yd = sqrt(2.0*kc/Q - beta);
  s.zd = 0.0;

  compute_a(kc, s, &a0);
  printf("initial state:\n");
  printf("%.16le %.16le %.16le\n", s.x, s.y, s.z);
  printf("%.16le %.16le %.16le\n", s.xd, s.yd, s.zd);

  kepler_step(kc, dt, &s, &sp);

  compute_a(kc, sp, &ap);
  printf("final state:\n");
  printf("%.16le %.16le %.16le\n", sp.x, sp.y, sp.z);
  printf("%.16le %.16le %.16le\n", sp.xd, sp.yd, sp.zd);

  printf("relative semimajor axis error: %.16le\n", (ap-a0)/a0);

  return(0);
}
    
