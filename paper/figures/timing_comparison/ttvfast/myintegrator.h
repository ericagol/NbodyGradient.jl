//Here are some other parameters used.

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

#define MAX(x,y) ((x) > (y)) ? (x) : (y)
#define MIN(x,y) ((x) < (y)) ? (x) : (y)
#define ABS(a) ((a) < 0 ? -(a) : (a))

typedef struct {
  double x, y, z;
} Vector;


typedef struct {
  double x, y, z, xd, yd, zd;
} PhaseState;




