#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef _STATE_
#define _STATE_
typedef struct {
  double x, y, z, xd, yd, zd, xdd, ydd, zdd;
} State;
#endif

#ifndef M_PI
#define M_PI           3.14159265358979323846
#endif

#define SUCCESS 1
#define FAILURE 0

