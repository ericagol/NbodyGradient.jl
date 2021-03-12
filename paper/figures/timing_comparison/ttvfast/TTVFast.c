/*If you make use of this code, please cite Deck, Agol, Holman & Nesvorny,
2014  ApJ, 787, 132, arXiv:1403.1895 */

// Main TTVFast file, which takes in the initial conditions, masses, G, timestep, t0, total amount of time, number of planets, number of RV measurements, size of Transit structure, and the RV and Transit structures. This is where the integration is formed & the transit times, rsky & vsky at transit, and RV at an observation time are calculated. Note that things called "helio" or "heliocentric" really should be "astrocentric". 

#define PI 3.14159265358979323846
#define TWOPI 6.283185307179586476925287
#define MAX_N_PLANETS 9
#include"myintegrator.h"
#include"transit.h"
#define TOLERANCE 1e-10
#define MAX_ITER 35
#define BAD_TRANSIT -1
int bad_transit_flag;
PhaseState rp[MAX_N_PLANETS];    /* real coordinates */
PhaseState p[MAX_N_PLANETS];     /* map coordinates */
PhaseState temporary;
PhaseState p_tmp[MAX_N_PLANETS];
PhaseState helioAhead[MAX_N_PLANETS];
PhaseState helioBehind[MAX_N_PLANETS];

PhaseState rp_tmp[MAX_N_PLANETS];
PhaseState p_RV[MAX_N_PLANETS];
PhaseState p_ahead[MAX_N_PLANETS];
PhaseState p_behind[MAX_N_PLANETS];
PhaseState rp_ahead[MAX_N_PLANETS];
PhaseState rp_behind[MAX_N_PLANETS];

double G;
double GMsun;
double GM[MAX_N_PLANETS];
double GJM[MAX_N_PLANETS];
double kc[MAX_N_PLANETS];
double kc_helio[MAX_N_PLANETS];
double m_eta[MAX_N_PLANETS];
double mm[MAX_N_PLANETS];
double Geta[MAX_N_PLANETS];
double guess[MAX_N_PLANETS][3];
double factor1[MAX_N_PLANETS], factor2[MAX_N_PLANETS];
double prev_dot[MAX_N_PLANETS],curr_dot[MAX_N_PLANETS],curr_z[MAX_N_PLANETS];
int count[MAX_N_PLANETS];

double tdot_result,dotA,dotB,TimeA,TimeB,RVTime;
int  tplanet;
double dt;
int n_planets;
double Time;
double machine_epsilon;
#include "ttv-map-jacobi.c"
#include "kepcart2.c"
#include "machine-epsilon.c"

void TTVFast(double *params,double dt, double Time, double total,int n_plan,CalcTransit *transit,CalcRV *RV_struct, int nRV, int n_events, int input_flag)
{
  n_planets=n_plan;
  int  planet;
  int i, j;
  j=0;
  void jacobi_heliocentric(PhaseState *jacobi, PhaseState *helio, double GMsun, double *GM);
  double dot0,dot1,dot2,rskyA,rskyB,vprojA,vprojB,rsky,vproj,velocity,new_dt;
  double compute_RV(PhaseState *ps);
  double compute_deriv(PhaseState ps,int planet);
  int RV_count = 0;
  int k=0;
  double deriv;
  double  dt2 = dt/2.0;

  if(RV_struct !=NULL){
    RV_count = nRV;
  }

  machine_epsilon = determine_machine_epsilon();
  if(input_flag ==0){
    read_jacobi_planet_elements(params);
  }
  if(input_flag ==1){
    read_helio_planet_elements(params);
  }
  if(input_flag ==2){
    read_helio_cartesian_params(params);
  }
  if(input_flag !=0 && input_flag !=1 && input_flag !=2){
    printf("Input flag must be 0,1, or 2. \n");
    exit(-1);
  }
  
  copy_system(p, rp);
  compute_corrector_coefficientsTO(dt);
  real_to_mapTO(rp, p);

  for(i = 0;i<n_planets;i++){
  prev_dot[i] = p[i].x*p[i].xd+p[i].y*p[i].yd;
  count[i]=0;

  }

  A(p, dt2);

  while(Time < total){
    copy_system(p, p_tmp);
    B(p,dt);
    A(p,dt);
    Time+=dt;
    /* Calculate RV if necessary */

    if(j <RV_count){
      RVTime = Time+dt2;
      
      if(RVTime>(RV_struct+j)->time && RVTime-dt<(RV_struct+j)->time){
	if(RVTime-(RV_struct+j)->time > (RV_struct+j)->time-(RVTime-dt)){
	  copy_system(p_tmp,p_RV);
	  new_dt= (RV_struct+j)->time-(RVTime-dt);
	  A(p_RV,new_dt);
	  velocity = compute_RV(p_RV);
	  (RV_struct+j)->RV =velocity;
	  
	}else{
	  copy_system(p,p_RV);
	  new_dt= (RV_struct+j)->time-RVTime;
	  A(p_RV,new_dt);
	  velocity = compute_RV(p_RV);
	  (RV_struct+j)->RV =velocity;
	}
	j++;
      }
    }
    
    /* now look for transits */
    
    for(i = 0;i<n_planets;i++){ /* for each planet */
      curr_dot[i] = p[i].x*p[i].xd+p[i].y*p[i].yd;
      curr_z[i] = p[i].z;
      tplanet=i;
      /* Check Transit Condition*/
      if(prev_dot[tplanet]<0 && curr_dot[tplanet]>0 && curr_z[tplanet]>0){
	bad_transit_flag = 0;
	copy_system(p,p_ahead);
	copy_system(p_tmp,p_behind);    
	A(p_ahead,-dt2);
	A(p_behind,-dt2);
	jacobi_heliocentric(p_behind,helioBehind,GMsun,GM);	  
	jacobi_heliocentric(p_ahead,helioAhead,GMsun,GM);
	
	/* Calculate Rsky.Vsky */
	dot2 = helioAhead[tplanet].x*helioAhead[tplanet].xd+helioAhead[tplanet].y*helioAhead[tplanet].yd;
	dot1 = helioBehind[tplanet].x*helioBehind[tplanet].xd+helioBehind[tplanet].y*helioBehind[tplanet].yd;
	
	if(dot1 < 0 && dot2 >0){
	  /* If Rsky.Vsky passed through zero*/
	  TimeA=Time;
	  TimeB = Time-dt;
	  dotA= TimeA+kepler_transit_locator(kc_helio[tplanet],-dt,&helioAhead[tplanet],&temporary);

	  deriv = compute_deriv(temporary,tplanet);

	  if(deriv < 0.0 || temporary.z <0.0 || dotA < TimeB-PI/mm[tplanet] || dotA > TimeA+PI/mm[tplanet]){ /* was the right root found?*/
	    dotA= TimeA+bisection(kc_helio[tplanet],&helioAhead[tplanet],&helioBehind[tplanet],&temporary);
	  }
	  
	  rskyA = sqrt(temporary.x*temporary.x + temporary.y*temporary.y);
	  vprojA = sqrt(temporary.xd*temporary.xd + temporary.yd*temporary.yd);
	  
	  dotB= TimeB+kepler_transit_locator(kc_helio[tplanet],dt,&helioBehind[tplanet],&temporary);

	  deriv = compute_deriv(temporary,tplanet);	  

	  if(deriv < 0.0 || temporary.z <0.0 || dotB < TimeB-PI/mm[tplanet] || dotB > TimeA+PI/mm[tplanet]){ /* was the right root found?*/
	    dotB= TimeB+bisection(kc_helio[tplanet],&helioBehind[tplanet],&helioAhead[tplanet],&temporary);
	  }
	  
	  rskyB = sqrt(temporary.x*temporary.x + temporary.y*temporary.y);
	  vprojB = sqrt(temporary.xd*temporary.xd + temporary.yd*temporary.yd);
	  
	  tdot_result = ((dotB-TimeB)*dotA+(TimeA-dotA)*dotB)/(TimeA-TimeB-dotA+dotB);
	  
	  rsky = ((dotB-TimeB)*rskyA+(TimeA-dotA)*rskyB)/(TimeA-TimeB-dotA+dotB);
	  vproj = ((dotB-TimeB)*vprojA+(TimeA-dotA)*vprojB)/(TimeA-TimeB-dotA+dotB);
	}else{
	  
	  copy_system(helioAhead,helioBehind);
	  copy_system(p,p_ahead);
	  B(p_ahead,dt);
	  A(p_ahead,dt2);
	  TimeA=Time+dt;
	  TimeB = Time;
	  jacobi_heliocentric(p_ahead,helioAhead,GMsun,GM);
	  dotB= TimeB+kepler_transit_locator(kc_helio[tplanet],dt,&helioBehind[tplanet],&temporary);
	  
	  deriv = compute_deriv(temporary,tplanet);	  
	  
	  if(deriv < 0.0 || temporary.z <0.0 || dotB < TimeB-PI/mm[tplanet] || dotB > TimeA+PI/mm[tplanet]){ /* was the right root found?*/
	    dotB= TimeB+bisection(kc_helio[tplanet],&helioBehind[tplanet],&helioAhead[tplanet],&temporary);
	  }
	  
	  rskyB = sqrt(temporary.x*temporary.x + temporary.y*temporary.y);
	  vprojB = sqrt(temporary.xd*temporary.xd + temporary.yd*temporary.yd);
	  
	  dotA= TimeA+kepler_transit_locator(kc_helio[tplanet],-dt,&helioAhead[tplanet],&temporary);
	  
	  deriv = compute_deriv(temporary,tplanet);	  
	  
	  if(deriv < 0.0 || temporary.z <0.0 || dotA < TimeB-PI/mm[tplanet] || dotA > TimeA+PI/mm[tplanet]){ /* was the right root found?*/
	    dotA= TimeA+bisection(kc_helio[tplanet],&helioAhead[tplanet],&helioBehind[tplanet],&temporary);
	  }
	  
	  rskyA = sqrt(temporary.x*temporary.x + temporary.y*temporary.y);
	  vprojA = sqrt(temporary.xd*temporary.xd + temporary.yd*temporary.yd);
	  
	  tdot_result = ((dotB-TimeB)*dotA+(TimeA-dotA)*dotB)/(TimeA-TimeB-dotA+dotB);
	  rsky =  ((dotB-TimeB)*rskyA+(TimeA-dotA)*rskyB)/(TimeA-TimeB-dotA+dotB);
	  vproj =  ((dotB-TimeB)*vprojA+(TimeA-dotA)*vprojB)/(TimeA-TimeB-dotA+dotB);
	}
	if(k< n_events){
	  
	  if(bad_transit_flag ==0){
	    (transit+k)->planet = tplanet;
	    (transit+k)->epoch = count[tplanet];
	    (transit+k)->time = tdot_result;
	    (transit+k)->rsky = rsky;
	    (transit+k)->vsky = vproj;
	  }else{
	    (transit+k)->planet = tplanet;
	    (transit+k)->epoch = count[tplanet];
	    (transit+k)->time = BAD_TRANSIT;
	    (transit+k)->rsky = BAD_TRANSIT;
	    (transit+k)->vsky = BAD_TRANSIT;
	  }
	  count[tplanet]++;
	  k++;
	}else{
	  printf("Not enough memory allocated for Transit structure: more events triggering as transits than expected. Possibily indicative of larger problem.\n");
	  exit(-1);
	}
      }
      
      prev_dot[i]=curr_dot[i];      
    }
  }
}




read_jacobi_planet_elements(params)
     double *params;
{
  int planet;
  double solar_mass, GMplanet;
  double Getatmp, Getatmp0;
  double period,e,incl,longnode,argperi,MeanAnom;
  double a;
  G = params[0];
  GMsun = params[1]*G;

  planet = 0;

  Getatmp = GMsun;

  while(planet < n_planets){
    GM[planet] = params[planet*7+2];
    GM[planet] *= G;
    
    Getatmp0 = Getatmp;
    Getatmp += GM[planet];
    factor1[planet] = 1.0/Getatmp0;
    factor2[planet] = Getatmp/Getatmp0;
    GJM[planet] = GM[planet]/factor2[planet];
    Geta[planet] = Getatmp;
    kc[planet] = GMsun*factor2[planet];
    kc_helio[planet] = GMsun+GM[planet];
    m_eta[planet] = GM[planet]/Geta[planet];
    period = params[planet*7+3];
    e = params[planet*7+4];
    incl = params[planet*7+5];
    longnode = params[planet*7+6];
    argperi = params[planet*7+7];
    MeanAnom = params[planet*7+8];
    mm[planet] = 2*PI/period;
    a = pow(mm[planet]*mm[planet]/kc[planet],-1.0/3.0);
    incl *= PI/180.0;
    longnode *= PI/180.0;
    argperi *= PI/180.0;
    MeanAnom *=PI/180.0;
    guess[planet][0] = mm[planet]*dt;
    guess[planet][1] = mm[planet]*dt;
    guess[planet][2] = mm[planet]*dt;
    cartesian(kc[planet],a,e,incl,longnode,argperi,MeanAnom,&p[planet]);
    planet += 1;

    if(planet > MAX_N_PLANETS) {
      printf("too many planets: %d\n", planet);
      exit(-1);
    }
  }
}



read_helio_planet_elements(params)
     double *params;
{
  int planet;
  double solar_mass, GMplanet;
  double Getatmp, Getatmp0;
  double period,e,incl,longnode,argperi,MeanAnom;
  double a;
  void heliocentric_jacobi(PhaseState *helio, PhaseState *jacobi, double GMsun, double *GM);
  PhaseState helio[MAX_N_PLANETS];
  G = params[0];
  GMsun = params[1]*G;
  
  planet = 0;

  Getatmp = GMsun;

  while(planet < n_planets){
    GM[planet] = params[planet*7+2];
    GM[planet] *= G;
    
    Getatmp0 = Getatmp;
    Getatmp += GM[planet];
    factor1[planet] = 1.0/Getatmp0;
    factor2[planet] = Getatmp/Getatmp0;
    GJM[planet] = GM[planet]/factor2[planet];
    Geta[planet] = Getatmp;
    kc[planet] = GMsun*factor2[planet];
    kc_helio[planet] = GMsun+GM[planet];
    m_eta[planet] = GM[planet]/Geta[planet];
    period = params[planet*7+3];
    e = params[planet*7+4];
    incl = params[planet*7+5];
    longnode = params[planet*7+6];
    argperi = params[planet*7+7];
    MeanAnom = params[planet*7+8];
    mm[planet] = 2*PI/period;
    a = pow(mm[planet]*mm[planet]/kc[planet],-1.0/3.0);
    incl *= PI/180.0;
    longnode *= PI/180.0;
    argperi *= PI/180.0;
    MeanAnom *=PI/180.0;
    guess[planet][0] = mm[planet]*dt;
    guess[planet][1] = mm[planet]*dt;
    guess[planet][2] = mm[planet]*dt;
    cartesian(kc_helio[planet],a,e,incl,longnode,argperi,MeanAnom,&helio[planet]);
    planet += 1;

    if(planet > MAX_N_PLANETS) {
      printf("too many planets: %d\n", planet);
      exit(-1);
    }
  }
  heliocentric_jacobi(helio,p,GMsun,GM);

}


read_helio_cartesian_params(params)
double *params;
{
  int planet;
  double solar_mass, GMplanet;
  double Getatmp, Getatmp0;
  double a,r,vs,u;
  void heliocentric_jacobi(PhaseState *helio, PhaseState *jacobi, double GMsun, double *GM);
  PhaseState helio[MAX_N_PLANETS];
  G = params[0];
  GMsun = params[1]*G;
  
  planet = 0;

  Getatmp = GMsun;

  while(planet < n_planets){
    GM[planet] = params[planet*7+2];
    GM[planet]*=G;
    Getatmp0 = Getatmp;
    Getatmp += GM[planet];
    factor1[planet] = 1.0/Getatmp0;
    factor2[planet] = Getatmp/Getatmp0;
    GJM[planet] = GM[planet]/factor2[planet];
    Geta[planet] = Getatmp;
    kc[planet] = GMsun*factor2[planet];
    kc_helio[planet] = GMsun+GM[planet];
    m_eta[planet] = GM[planet]/Geta[planet];
    helio[planet].x = params[planet*7+3];
    helio[planet].xd = params[planet*7+3+3];
    helio[planet].y = params[planet*7+4];
    helio[planet].yd = params[planet*7+4+3];
    helio[planet].z = params[planet*7+5];
    helio[planet].zd = params[planet*7+5+3];
    r = sqrt(helio[planet].x*helio[planet].x + helio[planet].y*helio[planet].y + helio[planet].z*helio[planet].z);
    vs = helio[planet].xd*helio[planet].xd + helio[planet].yd*helio[planet].yd + helio[planet].zd*helio[planet].zd;
    u = helio[planet].x*helio[planet].xd + helio[planet].y*helio[planet].yd + helio[planet].z*helio[planet].zd;
    a = 1.0/(2.0/r - vs/kc_helio[planet]);
    mm[planet] = sqrt(kc_helio[planet]/(a*a*a));
    guess[planet][0] = mm[planet]*dt;
    guess[planet][1] = mm[planet]*dt;
    guess[planet][2] = mm[planet]*dt;
    planet += 1;

    if(planet > MAX_N_PLANETS) {
      printf("too many planets: %d\n", planet);
      exit(-1);
    }
  }
  heliocentric_jacobi(helio,p,GMsun,GM);
}


void jacobi_heliocentric(PhaseState *jacobi, PhaseState *helio, double GMsun, double *GM)
{
  int i;
  double GetaC, GM_over_Geta;
  PhaseState s;
  s.x = 0.0; s.y = 0.0; s.z = 0.0;
  s.xd = 0.0; s.yd = 0.0; s.zd = 0.0;


  GetaC = GMsun;

  for(i=0; i<n_planets; i++){

    (helio+i)->x = (jacobi+i)->x + s.x;
    (helio+i)->y = (jacobi+i)->y + s.y;
    (helio+i)->z = (jacobi+i)->z + s.z;
    (helio+i)->xd = (jacobi+i)->xd + s.xd;
    (helio+i)->yd = (jacobi+i)->yd + s.yd;
    (helio+i)->zd = (jacobi+i)->zd + s.zd;

    GetaC += GM[i];
    GM_over_Geta = GM[i]/GetaC;

    s.x += GM_over_Geta * (jacobi+i)->x;
    s.y += GM_over_Geta * (jacobi+i)->y;
    s.z += GM_over_Geta * (jacobi+i)->z;
    s.xd += GM_over_Geta * (jacobi+i)->xd;
    s.yd += GM_over_Geta * (jacobi+i)->yd;
    s.zd += GM_over_Geta * (jacobi+i)->zd;
  }


    

}

double compute_deriv(PhaseState s, int planet)
{
  double deriv; /* Deriv of x*xd+y*yd = vsky^2+rsky.asky*/
  deriv = s.xd*s.xd+s.yd*s.yd-kc_helio[planet]/pow(s.x*s.x+s.y*s.y+s.z*s.z,3.0/2.0)*(s.x*s.x+s.y*s.y);

  return(deriv);
}

double compute_RV(PhaseState *ps)
{
  void jacobi_heliocentric(PhaseState *jacobiS, PhaseState *helioS, double GMsunS, double *GMS);
  PhaseState sun,helio[MAX_N_PLANETS];
  jacobi_heliocentric(ps,helio,GMsun,GM);

  int i;
  double mtotal=GMsun;
  sun.x = 0.0; sun.y = 0.0; sun.z = 0.0;
  sun.xd = 0.0; sun.yd = 0.0; sun.zd = 0.0;
  for(i=0;i<n_planets;i++){   
    sun.x +=-GM[i]*(helio+i)->x;
    sun.xd+=-GM[i]*(helio+i)->xd;
    sun.y +=-GM[i]*(helio+i)->y;
    sun.yd+=-GM[i]*(helio+i)->yd;
    sun.z +=-GM[i]*(helio+i)->z;
    sun.zd+=-GM[i]*(helio+i)->zd;
    mtotal+=GM[i];
  }
  sun.x /= mtotal;
  sun.y /= mtotal;
  sun.z /= mtotal;
  sun.xd /= mtotal;
  sun.yd /= mtotal;
  sun.zd /= mtotal;

  return(-sun.zd); /* in keeping with convention */
}


void heliocentric_jacobi(PhaseState *helio, PhaseState *jacobi, double GMsun, double *GM)
{
  int i;
  double mass, over_mass;
  PhaseState s;

  mass = 0.0;
  s.x = 0.0; s.y = 0.0; s.z = 0.0;
  s.xd = 0.0; s.yd = 0.0; s.zd = 0.0;

  mass = GMsun;
  over_mass = 1.0/GMsun;
  
  for(i=0; i<n_planets; i++){
    (jacobi+i)->x = (helio+i)->x - s.x * over_mass;
    (jacobi+i)->y = (helio+i)->y - s.y * over_mass;
    (jacobi+i)->z = (helio+i)->z - s.z * over_mass;
    (jacobi+i)->xd = (helio+i)->xd - s.xd * over_mass;
    (jacobi+i)->yd = (helio+i)->yd - s.yd * over_mass;
    (jacobi+i)->zd = (helio+i)->zd - s.zd * over_mass;
    
    s.x += GM[i] * (helio+i)->x;
    s.y += GM[i] * (helio+i)->y;
    s.z += GM[i] * (helio+i)->z;
    s.xd += GM[i] * (helio+i)->xd;
    s.yd += GM[i] * (helio+i)->yd;
    s.zd += GM[i] * (helio+i)->zd;
    
    mass += GM[i];
    over_mass = 1.0/mass;
  }
}



