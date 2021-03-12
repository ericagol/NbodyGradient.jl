/*If you make use of this code, please cite Deck, Agol, Holman & Nesvorny, 2014,  ApJ, 787, 132, arXiv:1403.1895 */

//This file holds all the auxiliary files for the integration, including the Kepler step, the kick step, transit time solver employing Newton's method, transit time finder employing the bisection method, the symplectic corrector sub routines, etc.

int kepler_step(double gm, double dt, PhaseState *s0, PhaseState *s,int planet)
{
  
  double r0, v0s, u, a, n, ecosE0, esinE0;
  double dM, x, sx, cx, f, fp, fpp, fppp, dx;
  double fdot, g, gdot;
  double sx2, cx2, x2;
  double xx, yy, xx1, yy1, omx, h;
  double k0x, k0y, k1x, k1y, k2x, k2y, k3y;
  double ecosE, esinE;
  int count;
  r0 = sqrt(s0->x*s0->x + s0->y*s0->y + s0->z*s0->z);

  v0s = s0->xd*s0->xd + s0->yd*s0->yd + s0->zd*s0->zd;
  u = s0->x*s0->xd + s0->y*s0->yd + s0->z*s0->zd;
  a = 1.0/(2.0/r0 - v0s/gm);

  if(a<0.0) {
    printf("hyperbolic orbit %lf\n", a);
    exit(-1);
  }

  n = sqrt(gm/(a*a*a));
  ecosE0 = 1.0 - r0/a;
  esinE0 = u/(n*a*a);
  
  dM = n*dt;

  x = 3.0*guess[planet][2]+guess[planet][0]-3.0*guess[planet][1];

  count = 0;
  do {
    x2 = x/2.0;
    sx2 = sin(x2); cx2 = cos(x2);
    sx = 2.0*sx2*cx2; cx = cx2*cx2 - sx2*sx2;
    f = x + 2.0*sx2*(sx2*esinE0 - cx2*ecosE0) - dM;
    ecosE = cx*ecosE0 - sx*esinE0;
    fp = 1.0 - ecosE;
    fpp = (sx*ecosE0 + cx*esinE0)/2.0;
    fppp = ecosE/6.0;
    dx = -f/fp;
    dx = -f/(fp + dx*fpp);
    dx = -f/(fp + dx*(fpp + dx*fppp));
    x += dx;
    count ++;
    
  } while(fabs(dx) > 1.0e-4 && count < MAX_ITER);
  
  

  if(fabs(f)> 1.0e-14){
    x = dM;
    count = 0;
    do {
      x2 = x/2.0;
      sx2 = sin(x2); cx2 = cos(x2);
      sx = 2.0*sx2*cx2; cx = cx2*cx2 - sx2*sx2;
      f = x + 2.0*sx2*(sx2*esinE0 - cx2*ecosE0) - dM;
      ecosE = cx*ecosE0 - sx*esinE0;
      fp = 1.0 - ecosE;
      fpp = (sx*ecosE0 + cx*esinE0)/2.0;
      fppp = ecosE/6.0;
      dx = -f/fp;
      dx = -f/(fp + dx*fpp);
      dx = -f/(fp + dx*(fpp + dx*fppp));
      x += dx;
      count++;
    } while(fabs(f) > 1.0e-14 && count < MAX_ITER);

  }

  if(count==MAX_ITER){
    printf("Kepler step not converging in MAX_ITER. Likely need a smaller dt\n");
    exit(-1);
  }

  guess[planet][0]=guess[planet][1];
  guess[planet][1]=guess[planet][2];
  guess[planet][2]=x;
  

  /* compute f and g */
  x2 = x/2.0;
  sx2 = sin(x2); cx2 = cos(x2);
  f = 1.0 - (a/r0)*2.0*sx2*sx2;
  sx = 2.0*sx2*cx2; cx = cx2*cx2 - sx2*sx2;
  g = (2.0*sx2*(esinE0*sx2 + cx2*r0/a))/n;
  fp = 1.0 - cx*ecosE0 + sx*esinE0;
  fdot = -(a/(r0*fp))*n*sx;
  gdot = (1.0 + g*fdot)/f;

  /* compute new position and velocity */
  s->x = f*s0->x + g*s0->xd;
  s->y = f*s0->y + g*s0->yd;
  s->z = f*s0->z + g*s0->zd;
  s->xd = fdot*s0->x + gdot*s0->xd;
  s->yd = fdot*s0->y + gdot*s0->yd;
  s->zd = fdot*s0->z + gdot*s0->zd;

}

double kepler_transit_locator(double gm, double dt,  PhaseState *s0, PhaseState *s)
{
    double y,y2,sy2,sy,test;
  double ecc,transitM,eprior;
  double  a, n,r0, ecosE0, esinE0,u,v0s;
  double  x, sx, cx, fp, fp2, dx;
  double fdot, g, gdot,f;
  double sx2, cx2, x2;
  double aOverR,dfdz,dgdz,dfdotdz,dgdotdz,dotproduct,dotproductderiv,rsquared,vsquared,xdotv;

  r0 = sqrt(s0->x*s0->x + s0->y*s0->y + s0->z*s0->z);
  v0s = s0->xd*s0->xd + s0->yd*s0->yd + s0->zd*s0->zd;
  u = s0->x*s0->xd + s0->y*s0->yd + s0->z*s0->zd;
  a = 1.0/(2.0/r0 - v0s/gm);

  n = sqrt(gm/(a*a*a));
  ecosE0 = 1.0 - r0/a;
  esinE0 = u/(n*a*a);

  aOverR= a/r0;
  rsquared = s0->x*s0->x + s0->y*s0->y;
  vsquared = s0->xd*s0->xd + s0->yd*s0->yd;
  xdotv =s0->x*s0->xd + s0->y*s0->yd;

  /*Initial Guess */
  x = n*dt/2.0;
  do{
    x2 = x/2.0;
    sx2 = sin(x2); cx2 = cos(x2);
    f = 1.0 - aOverR*2.0*sx2*sx2;
    sx = 2.0*sx2*cx2; cx = cx2*cx2 - sx2*sx2;
    g = (2.0*sx2*(esinE0*sx2 + cx2/aOverR))/n;
    fp = 1.0 - cx*ecosE0 + sx*esinE0;
    fdot = -(aOverR/fp)*n*sx;
    fp2 = sx*ecosE0 + cx*esinE0;
    gdot = 1.0-2.0*sx2*sx2/fp;

    dgdotdz = -sx/fp+2.0*sx2*sx2/fp/fp*fp2;
    dfdz = -aOverR*sx;
    dgdz= 1.0/n*(sx*esinE0-(ecosE0-1.0)*cx);
    dfdotdz= -n*aOverR/fp*(cx+sx/fp*fp2);

    dotproduct = f*fdot*(rsquared)+g*gdot*(vsquared)+(f*gdot+g*fdot)*(xdotv);

    dotproductderiv = dfdz*(gdot*xdotv+fdot*rsquared)+dfdotdz*(f*rsquared+g*xdotv)+dgdz*(fdot*xdotv+gdot*vsquared)+dgdotdz*(g*vsquared+f*xdotv);

    dx = -dotproduct/dotproductderiv;

    x += dx;

  }while(fabs(dx)> sqrt(TOLERANCE));
  /* Now update state */
  x2 = x/2.0;
  sx2 = sin(x2); cx2 = cos(x2);
  sx = 2.0*sx2*cx2; 
  
  cx = cx2*cx2 - sx2*sx2;
  f = 1.0 - (a/r0)*2.0*sx2*sx2;
  g = (2.0*sx2*(esinE0*sx2 + cx2/aOverR))/n;
  fp = 1.0 - cx*ecosE0 + sx*esinE0;
  fdot = -(aOverR/fp)*n*sx;
  gdot = (1.0 + g*fdot)/f;
  s->x = f*s0->x + g*s0->xd;
  s->y = f*s0->y + g*s0->yd;
  s->z = f*s0->z + g*s0->zd;
  s->xd = fdot*s0->x + gdot*s0->xd;
  s->yd = fdot*s0->y + gdot*s0->yd;
  s->zd = fdot*s0->z + gdot*s0->zd;
  /*Corresponding Delta Mean Anomaly */
  transitM = x+esinE0*2.0*sx2*sx2-sx*ecosE0;

  return(transitM/n);
}

double bisection(double gm,PhaseState *s0, PhaseState *s1, PhaseState *s)
{
  double g0,g1,g2;
  double  a, n,r0, ecosE0, esinE0,u,v0s;
  double  x, sx, cx, x2,sx2,cx2,fdot,gdot,f,g,fp,transitM,rsquared,vsquared,xdotv;
  double E0,esinE1,ecosE1,E1;
  double aOverR;
  double dot_product(double y, double aOverR, double esinE0, double ecosE0, double n,double rsquared, double vsquared, double xdotv);  
  int iter;
  double dg2;

  r0 = sqrt(s1->x*s1->x + s1->y*s1->y + s1->z*s1->z);
  v0s = s1->xd*s1->xd + s1->yd*s1->yd + s1->zd*s1->zd;
  u = s1->x*s1->xd + s1->y*s1->yd + s1->z*s1->zd;
  a = 1.0/(2.0/r0 - v0s/gm);

  n = sqrt(gm/(a*a*a));
  ecosE1 = 1.0 - r0/a;
  esinE1 = u/(n*a*a);

  E1 = atan2(esinE1,ecosE1);

  r0 = sqrt(s0->x*s0->x + s0->y*s0->y + s0->z*s0->z);
  v0s = s0->xd*s0->xd + s0->yd*s0->yd + s0->zd*s0->zd;
  u = s0->x*s0->xd + s0->y*s0->yd + s0->z*s0->zd;
  a = 1.0/(2.0/r0 - v0s/gm);
  rsquared = s0->x*s0->x+s0->y*s0->y;
  vsquared = s0->xd*s0->xd+s0->yd*s0->yd;
  xdotv = s0->x*s0->xd+s0->y*s0->yd;
  n = sqrt(gm/(a*a*a));
  ecosE0 = 1.0 - r0/a;
  esinE0 = u/(n*a*a);
  E0 = atan2(esinE0,ecosE0);
  aOverR= a/r0;
  
  /* Interval endpoints */
  g0 =0.0;
  g1  =(E1-E0);
  if(dot_product(g0,aOverR,esinE0,ecosE0,n,rsquared,vsquared,xdotv)*dot_product(g1,aOverR,esinE0,ecosE0,n,rsquared,vsquared,xdotv) < 0.0){
    iter = 0;
  }else{
    iter = MAX_ITER;
  }
  
  do{
    g2 = (g1+g0)/2.0;
    dg2 =  dot_product(g2,aOverR,esinE0,ecosE0,n,rsquared,vsquared,xdotv);  
    if(dg2*dot_product(g0,aOverR,esinE0,ecosE0,n,rsquared,vsquared,xdotv) > 0.0){
      g0 = g2;
    }
    else{
      g1 = g2;
    }
    
    iter++;
    
  } while( (fabs(dg2)> TOLERANCE) && (iter < MAX_ITER));

  if(iter ==MAX_ITER){
    bad_transit_flag = 1;
  }
  
  /* Now update state */
  x = g2;
  x2 = x/2.0;
  sx2 = sin(x2); cx2 = cos(x2);
  sx = 2.0*sx2*cx2; 
  
  cx = cx2*cx2 - sx2*sx2;
  f = 1.0 - (a/r0)*2.0*sx2*sx2;
  g = (2.0*sx2*(esinE0*sx2 + cx2/aOverR))/n;
  fp = 1.0 - cx*ecosE0 + sx*esinE0;
  fdot = -(aOverR/fp)*n*sx;
  gdot = 1.0-2.0*sx2*sx2/fp;
  s->x = f*s0->x + g*s0->xd;
  s->y = f*s0->y + g*s0->yd;
  s->z = f*s0->z + g*s0->zd;
  s->xd = fdot*s0->x + gdot*s0->xd;
  s->yd = fdot*s0->y + gdot*s0->yd;
  s->zd = fdot*s0->z + gdot*s0->zd;
  /*Corresponding Delta Mean Anomaly */
  transitM = (x+esinE0*2.0*sx2*sx2-sx*ecosE0);

  return(transitM/n);
}


double dot_product(double y, double aOverR, double esinE0, double ecosE0, double n,double rsquared, double vsquared, double xdotv)
{
  double y2,sy2,cy2,f,sy,cy,fdot,gdot,dotprod,g,fp;
  y2 = y/2.0   ;

  sy2 = sin(y2)                                 ;
  cy2 = cos(y2)                                 ;
  f = 1.0 - aOverR*2.0*sy2*sy2                  ;
  sy = 2.0*sy2*cy2                              ; 
  cy = cy2*cy2 - sy2*sy2                        ;
  g = (2.0*sy2*(esinE0*sy2 + cy2/aOverR))/n     ;
  fp = 1.0 - cy*ecosE0 + sy*esinE0              ;
  fdot = -(aOverR/fp)*n*sy                      ;
  gdot = 1.0-2.0*sy2*sy2/fp                     ;
  dotprod = f*fdot*(rsquared)+g*gdot*(vsquared)+(f*gdot+g*fdot)*(xdotv) ;

  return(dotprod);
}



void nbody_kicks(PhaseState p[], double dt)
{
  Vector FF[MAX_N_PLANETS], GG[MAX_N_PLANETS], acc_tp;
  Vector tmp[MAX_N_PLANETS], h[MAX_N_PLANETS], XX;
  double GMsun_over_r3[MAX_N_PLANETS], rp2, dx, dy, dz, r2, rij2, rij5;
  double q, q1, fq;
  double f0, fi, fij, fijm, fr;
  double sx, sy, sz, tx, ty, tz;
  double indirectx, indirecty, indirectz, over_GMsun;
  double constant;
  int i, j;
  double qb0, qb1, qb2, qb3;
  double c, c1;

  double a0, a11, a22, a33, a12, a13, a23, a21, a32, a31, b01, b02, b03, b12, b13, b23;
  double x1, x2, x3;

  sx = 0.0; sy = 0.0; sz = 0.0;
  for(i=0; i<n_planets; i++) {

    tmp[i].x = 0.0; tmp[i].y = 0.0; tmp[i].z = 0.0;
    h[i].x = p[i].x + sx; h[i].y = p[i].y + sy; h[i].z = p[i].z + sz;
    XX.x = sx; XX.y = sy; XX.z = sz; /* XX is cm up to particle i-1 */
    sx += p[i].x*m_eta[i]; sy += p[i].y*m_eta[i]; sz += p[i].z*m_eta[i];

    rp2 = p[i].x*p[i].x + p[i].y*p[i].y + p[i].z*p[i].z;
    r2 =  h[i].x*h[i].x + h[i].y*h[i].y + h[i].z*h[i].z;
    GMsun_over_r3[i] = GMsun/(r2*sqrt(r2));
    q = (XX.x*XX.x + XX.y*XX.y + XX.z*XX.z 
	 + 2*(XX.x*p[i].x + XX.y*p[i].y + XX.z*p[i].z))/rp2;
    q1 = 1.0 + q;
    fq = q*(3.0 + q*(3.0 + q))/(1.0 + q1*sqrt(q1));
    GG[i].x = (fq*p[i].x - XX.x)*GMsun_over_r3[i];
    GG[i].y = (fq*p[i].y - XX.y)*GMsun_over_r3[i];
    GG[i].z = (fq*p[i].z - XX.z)*GMsun_over_r3[i];


  }

  for(i=0; i<n_planets; i++) {
    FF[i].x = 0.0; FF[i].y = 0.0; FF[i].z = 0.0;
    for(j=i+1; j<n_planets; j++) {
      dx = h[i].x - h[j].x; dy = h[i].y - h[j].y; dz = h[i].z - h[j].z; 
      rij2 = dx*dx + dy*dy + dz*dz;
      fij = -1.0/(rij2*sqrt(rij2));
      tx = dx*fij; ty = dy*fij; tz = dz*fij;
      GG[i].x += GM[j]*tx; GG[i].y += GM[j]*ty; GG[i].z += GM[j]*tz;
      GG[j].x -= GM[i]*tx; GG[j].y -= GM[i]*ty; GG[j].z -= GM[i]*tz;
      fijm = fij*GM[i]*GM[j];
      tmp[j].x += dx*fijm; tmp[j].y += dy*fijm; tmp[j].z += dz*fijm;
      FF[i].x -= tmp[j].x; FF[i].y -= tmp[j].y; FF[i].z -= tmp[j].z;
    }
  }
  
  tx = 0.0; ty = 0.0; tz = 0.0;
  for(i=n_planets-1; i>=0; i--) {
    FF[i].x -= tx; FF[i].y -= ty; FF[i].z -= tz;
    f0 = GM[i]*GMsun_over_r3[i];
    tx += h[i].x*f0; ty += h[i].y*f0; tz += h[i].z*f0;
  }
    
  /* factor1 = 1/eta[i-1] factor2 = eta[i]/eta[i-1] */
  for(i=0; i<n_planets; i++) {
    p[i].xd += dt*(factor1[i]*FF[i].x + factor2[i]*GG[i].x);  
    p[i].yd += dt*(factor1[i]*FF[i].y + factor2[i]*GG[i].y);  
    p[i].zd += dt*(factor1[i]*FF[i].z + factor2[i]*GG[i].z);  
  }
  

}


double corr_Chambers,coeffb1,coeffb2,coeffa1,coeffa2,TOa1,TOa2,TOb1,TOb2,alpha,beta,btil1,btil2,atil1,atil2,ssq,corr_alpha,corr_beta,FOa1,FOa2,FOa3,FOa4,FOb1,FOb2,FOb3,FOb4,SOa1,SOa2,SOa3,SOa4,SOa5,SOa6,SOb1,SOb2,SOb3,SOb4,SOb5,SOb6;


void real_to_mapTO(PhaseState rp[], PhaseState p[])
{

  void Z(PhaseState p[], double a, double b);
  void copy_system(PhaseState p1[], PhaseState p2[]);

  copy_system(rp, p);

  Z(p, TOa2, TOb2);
  Z(p,  TOa1, TOb1);

}


void map_to_realTO(PhaseState p[], PhaseState rp[])
{

  void Z(PhaseState p[],  double a, double b);
  void copy_system(PhaseState p1[], PhaseState p2[]);

  copy_system(p,rp);

  Z(rp,  TOa1, -TOb1);
  Z(rp,  TOa2, -TOb2);
 }

void compute_corrector_coefficientsTO(double dt)
{
  corr_alpha = sqrt(7.0/40.0);
  corr_beta = 1./(48.0*corr_alpha);

  TOa1  = corr_alpha * (  -1.0 );
  TOa2  = corr_alpha * ( 1.0 );
  
  TOb1  = corr_beta * (  -0.5 );
  TOb2  = corr_beta * ( 0.5);
  TOa1 *= dt;
  TOa2 *= dt;
  
  TOb1 *= dt;
  TOb2 *= dt;
 }



void A(PhaseState p[],  double dt)
{
  PhaseState tmp;
  int planet;
  void copy_state(PhaseState *s1,  PhaseState *s2);

  for(planet=0; planet<n_planets; planet++) {
    kepler_step(kc[planet], dt, &p[planet], &tmp,planet);
    copy_state(&tmp, &p[planet]);
  }
}

void B(PhaseState p[], double dt)
{
  nbody_kicks(p, dt);  
}

void C(PhaseState p[], double a, double b)
{
  A(p, -a);
  B(p, b);
  A(p, a);
}

void Z(PhaseState p[], double a, double b)
{
  C(p, -a, -b);
  C(p, a, b);
}

void copy_state(PhaseState *s1,  PhaseState *s2)
{
  s2->x = s1->x;
  s2->y = s1->y;
  s2->z = s1->z;
  s2->xd = s1->xd;
  s2->yd = s1->yd;
  s2->zd = s1->zd;
}

void copy_system(PhaseState p1[], PhaseState p2[])
{
  int planet;

  for(planet=0; planet<n_planets; planet++) {
    copy_state(&p1[planet], &p2[planet]);
  }
}
