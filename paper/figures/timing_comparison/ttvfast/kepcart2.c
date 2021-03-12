/*If you make use of this code, please cite Deck, Agol, Holman & Nesvorny, 2014 */
// This file has two routines which convert a GM & Cartesian state to orbital elements and VICE VERSA.

extern double machine_epsilon;
void keplerian(double gm, PhaseState state, double *a, double *e, double *i, double *longnode, double *argperi, double *meananom)
{
  double rxv_x, rxv_y, rxv_z, hs, h, parameter;
  double r, vs, rdotv, rdot, ecostrueanom, esintrueanom, cosnode, sinnode;
  double rcosu, rsinu, u, trueanom, eccanom;

  /* find direction of angular momentum vector */
  rxv_x = state.y * state.zd - state.z * state.yd;
  rxv_y = state.z * state.xd - state.x * state.zd;
  rxv_z = state.x * state.yd - state.y * state.xd;
  hs = rxv_x * rxv_x + rxv_y * rxv_y + rxv_z * rxv_z;
  h = sqrt(hs);

  r = sqrt(state.x * state.x + state.y * state.y + state.z * state.z);
  vs = state.xd * state.xd + state.yd * state.yd + state.zd * state.zd;
  /* v = sqrt(vs);  unnecessary */
  rdotv = state.x * state.xd + state.y * state.yd + state.z * state.zd;
  rdot = rdotv / r;
  parameter = hs / gm;

  *i = acos(rxv_z / h);

  if(rxv_x!=0.0 || rxv_y!=0.0) {
    *longnode = atan2(rxv_x, -rxv_y);
  } else {
    *longnode = 0.0;
  }

  *a = 1.0 / (2.0 / r - vs / gm);

  ecostrueanom = parameter / r - 1.0;
  esintrueanom = rdot * h / gm;
  *e = sqrt(ecostrueanom * ecostrueanom + esintrueanom * esintrueanom);

  if(esintrueanom!=0.0 || ecostrueanom!=0.0) {
    trueanom = atan2(esintrueanom, ecostrueanom);
  } else {
    trueanom = 0.0;
  }

  cosnode = cos(*longnode);
  sinnode = sin(*longnode);

  /* u is the argument of latitude */
  rcosu = state.x * cosnode + state.y * sinnode;
  rsinu = (state.y * cosnode - state.x * sinnode)/cos(*i);

  if(rsinu!=0.0 || rcosu!=0.0) {
    u = atan2(rsinu, rcosu);
  } else {
    u = 0.0;
  }

  *argperi = u - trueanom;

  eccanom = 2.0 * atan(sqrt((1.0 - *e)/(1.0 + *e)) * tan(trueanom/2.0));
  *meananom = eccanom - *e * sin(eccanom);

  return;
}


void cartesian(double gm, double a, double e, double i, double longnode, double argperi, double meananom, PhaseState *state)
{
  double meanmotion, cosE, sinE, foo;
  double x, y, z, xd, yd, zd;
  double xp, yp, zp, xdp, ydp, zdp;
  double cosw, sinw, cosi, sini, cosnode, sinnode;
  double E0, E1, E2, den;

  /* first compute eccentric anomaly */
  E0 = meananom; 
  do {
    E1 = meananom + e * sin(E0);
    E2 = meananom + e * sin(E1);

    den = E2 - 2.0*E1 + E0;
    if(fabs(den) > machine_epsilon) {
      E0 = E0 - (E1-E0)*(E1-E0)/den;
    }
    else {
      E0 = E2;
      E2 = E1;
    }
  } while(fabs(E0-E2) > machine_epsilon);

  cosE = cos(E0);
  sinE = sin(E0);

  /* compute unrotated positions and velocities */
  foo = sqrt(1.0 - e*e);
  meanmotion = sqrt(gm/(a*a*a));
  x = a * (cosE - e);
  y = foo * a * sinE;
  z = 0.0;
  xd = -a * meanmotion * sinE / (1.0 - e * cosE);
  yd = foo * a * meanmotion * cosE / (1.0 - e * cosE);
  zd = 0.0;

  /* rotate by argument of perihelion in orbit plane*/
  cosw = cos(argperi);
  sinw = sin(argperi);
  xp = x * cosw - y * sinw;
  yp = x * sinw + y * cosw;
  zp = z;
  xdp = xd * cosw - yd * sinw;
  ydp = xd * sinw + yd * cosw;
  zdp = zd;

  /* rotate by inclination about x axis */
  cosi = cos(i);
  sini = sin(i);
  x = xp;
  y = yp * cosi - zp * sini;
  z = yp * sini + zp * cosi;
  xd = xdp;
  yd = ydp * cosi - zdp * sini;
  zd = ydp * sini + zdp * cosi;

  /* rotate by longitude of node about z axis */
  cosnode = cos(longnode);
  sinnode = sin(longnode);
  state->x = x * cosnode - y * sinnode;
  state->y = x * sinnode + y * cosnode;
  state->z = z;
  state->xd = xd * cosnode - yd * sinnode;
  state->yd = xd * sinnode + yd * cosnode;
  state->zd = zd;

  return;
}

