/*If you make use of this code, please cite Deck, Agol, Holman & Nesvorny, 2014,  ApJ, 787, 132, arXiv:1403.1895 */

// This is the sample program showing how TTVFast is called.

#include"transit.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#define CALCULATE_RV 0

void TTVFast(double *params,double dt, double Time, double total,int n_planets, CalcTransit *transit, CalcRV *RV_struct,int n,int m, int input);

int main(int argc, char **argv)
{
  double DEFAULT;
  FILE  *setup_file;
  char ic_file_name[100];
  FILE *dynam_param_file;
  FILE *RV_file;
  FILE *Transit_output;
  FILE *RV_output;
  double Time, dt,total;
  int nplanets;
  int planet,i;
  int n_events;
  int input_flag;
  setup_file = fopen(argv[1], "r");
  fscanf(setup_file, "%s", ic_file_name);
  fscanf(setup_file, "%lf", &Time); /* T0*/
  fscanf(setup_file, "%lf", &dt); /* timestep */
  fscanf(setup_file, "%lf", &total); /* Final time*/
  fscanf(setup_file, "%d", &nplanets);
  fscanf(setup_file, "%d", &input_flag);
  fclose(setup_file);
  Transit_output = fopen(argv[2], "w");
  int RV_count = 0;
  double RV_time;
  CalcRV *RV_model;
  if(CALCULATE_RV){
    if(argc!=5){
      printf("Incorrect number of arguments: Setup File, Output File, RV Input file, RV output file - expecting RV data!\n");
      exit(-1);
    }
    RV_file = fopen(argv[3],"r");
    RV_output = fopen(argv[4],"w");
    while(fscanf(RV_file,"%lf",&RV_time) != EOF){
      RV_count ++;
    }
    fclose(RV_file);

    RV_model = (CalcRV*) calloc(RV_count,sizeof(CalcRV));
    RV_file = fopen(argv[3],"r");
    RV_count = 0;
    while(fscanf(RV_file,"%lf",&RV_time) != EOF){
      (RV_model+RV_count)->time = RV_time;
      RV_count++;
    }
    fclose(RV_file);

  }else{
    RV_model = NULL;
  }

  double p[2+nplanets*7];
  dynam_param_file = fopen(ic_file_name,"r");
  fscanf(dynam_param_file, "%lf %lf ",&p[0],&p[1]);
  planet=0;
  n_events=0;
  /*Read in planet params: */
  /*Planet Mass/Stellar Mass, Period, Eccentricity, Inclination, Longnode, Arg Peri, Mean Anomaly */
  while(planet < nplanets){
    fscanf(dynam_param_file, "%lf %lf %lf %lf %lf %lf %lf",&p[planet*7+2],&p[planet*7+3],&p[planet*7+4],&p[planet*7+5],&p[planet*7+6],&p[planet*7+7],&p[planet*7+8]);
    planet++;
  }
  fclose(dynam_param_file);
  // n_events = 5000; /*HARDWIRED, see README. you may need to change this depending on the number of planets and their orbital periods. */
  n_events = 16400;
  /* create structure to hold transit calculations*/
  CalcTransit *model;
  model = (CalcTransit*) calloc(n_events,sizeof(CalcTransit));
  DEFAULT = -2; /* value for transit information that is not determined by TTVFast*/
  for(i=0;i<n_events;i++){
    (model+i)->time = DEFAULT;
  }
  TTVFast(p,dt,Time,total,nplanets,model,RV_model,RV_count,n_events,input_flag);


for(i=0;i<n_events;i++){
  /*A time of -1 = BAD_TRANSIT (specified in TTVFast.c) indicates that the root was not bracketed in the bisection method. In this case, the time determined by bisection method would be incorrect, so a value of BAD_TRANSIT is returned so that the user knows not to use this time. A time of DEFAULT indicates that spot in the Transit structure was never filled */
  if((model+i)->time !=DEFAULT && (model+i)->time !=-1){
    fprintf(Transit_output,"%d %d %.15le %.15le %.15le\n",(model+i)->planet, (model+i)->epoch, (model+i)->time,(model+i)->rsky,(model+i)->vsky);
  }
 }
 fflush(Transit_output);
 fclose(Transit_output);

 if(CALCULATE_RV){
   for(i=0;i<RV_count;i++){
    fprintf(RV_output,"%.15le %.15le \n",(RV_model+i)->time, (RV_model+i)->RV);
   }

   fflush(RV_output);
   fclose(RV_output);
 }
 free(model);
 free(RV_model);

}
