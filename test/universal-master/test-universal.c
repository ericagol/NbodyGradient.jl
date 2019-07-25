#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef _STATE_
#define _STATE_
typedef struct {
  double x, y, z, xd, yd, zd, xdd, ydd, zdd;
} State;
#endif

void compute_a(double gm, State state, double *a)
{
  double r, vs;

  r = sqrt(state.x * state.x + state.y * state.y + state.z * state.z);
  vs = state.xd * state.xd + state.yd * state.yd + state.zd * state.zd;

  *a = 1.0 / (2.0 / r - vs / gm);
}

int main(int argc, char** argv)
{
  double h_T, a, e, T;
  double re, rh;
  double k;
  double do_test();
  int i, j;
  int NX, NY;
  double value;
  double f;
  double n;

  NX = 400;
  NY = 240;

  k = 0.0172*0.0172;
  a = 0.4; 

  n = sqrt(k/(a*a*a));
  T = 2.0*M_PI/n;

  re = pow(1.e8, 1./((double) (NY+1)));
  rh = pow(1.e3, 1./((double) (NX+1)));

  f = 1.0/re;
  for(j=0; j<NY; j++) {
    h_T = 0.001;
    for(i=0; i<NX; i++) {
      h_T *= rh;
      e = 1.0 - f;
      value = (log10(fabs(do_test(k, h_T*T, a, e, T))));
      if(value < -16.0) {
	printf("%.6le ", -16.0);
      } else if (value > 0.0) {
	printf("%.6le ", 0.0);
      } else {
	printf("%.6le ", value);
      }
    }
    printf("\n");
    f /= re;
  }
}

void copy_state(State *s1, State *s2)
{
  s2->x = s1->x;
  s2->y = s1->y;
  s2->z = s1->z;
  s2->xd = s1->xd;
  s2->yd = s1->yd;
  s2->zd = s1->zd;
}

double do_test(double k, double h, double a, double e, double T)
{
  State s, s1;
  double beta, q;
  double gamma, hp;
  int orbit;
  double a0, ap;
  double time;
  void kepler_step();

  beta = k/a;
  q = a*(1.0 - e);
  gamma = (sqrt(5.0) - 1.0)/2.0;
  hp = h * gamma;

  time = 0.0;

  s.x = q;
  s.y = 0.0;
  s.z = 0.0;
  s.xd = 0.0;
  s.yd = sqrt(2.0*k/q - beta);
  s.zd = 0.0;

  while(time < T/2.0) {
    kepler_step(k, h, &s, &s1);
    time += h;
    copy_state(&s1, &s);
  }

  /* tweak phase */
  kepler_step(k, hp, &s, &s1);
  time += hp;
  copy_state(&s1, &s);

  compute_a(k, s, &a0);

  for(orbit=0; orbit<50; orbit++) {

    while(time > -T/2.0) {
      kepler_step(k, -h, &s, &s1);
      time -= h;
      copy_state(&s1, &s);
    }

    /* tweak phase */
    kepler_step(k, hp, &s, &s1);
    time += hp;
    copy_state(&s1, &s);

    while(time < T/2.0) {
      kepler_step(k, h, &s, &s1);
      time += h;
      copy_state(&s1, &s);
    }

    /* tweak phase */
    kepler_step(k, hp, &s, &s1);
    time += hp;
    copy_state(&s1, &s);

  }

  compute_a(k, s, &ap);

  return((ap-a0)/a0);
}
    
