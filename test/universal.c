/* written by Jack Wisdom in June, 2015  
   David M. Hernandez helped test and improve it */

int solve_universal_newton(double kc, double r0, double beta, double eta, double zeta, double h,
			   double *X, double *B, double *S2, double *C2)
{
  double b, x;
  double g1, g2, g3, g, arg;
  double s2, c2, xnew;
  int count;
  double err;

  b = sqrt(beta);
  xnew = *X;
  err = 1.e-9*fabs(xnew);

  count = 0;
  do {
    x = xnew;
    arg = b*x/2.0;
    s2 = sin(arg);
    c2 = cos(arg);
    g1 = 2.0*s2*c2/b;
    g2 = 2.0*s2*s2/beta;
    g3 = (x - g1)/beta;
    g = eta*g1 + zeta*g2;
    xnew = ((x*g - eta*g2 - zeta*g3) + h)/(r0 + g);
    if(count++ > 10) {
      return(FAILURE);
    }
  } while(fabs(x - xnew) > err);
  
  x = xnew;
  arg = b*x/2.0;
  s2 = sin(arg);
  c2 = cos(arg);
  /* g1 = 2.0*s2*c2/b; */
  /* g2 = 2.0*s2*s2/beta; */

  *X = x;
  *B = b;
  *S2 = s2;
  *C2 = c2;

  return(SUCCESS);
}

int solve_universal_laguerre(double kc, double r0, double beta, double eta, double zeta, double h, 
			     double *X, double *B, double *S2, double *C2)
{
  double b, x;
  double g1, g2, g3;
  double arg, s2, c2, xnew, dx;
  double f, fp, fpp, g0;
  int count;
  double err;

  b = sqrt(beta);
  xnew = *X;
  err = 1.e-9*fabs(xnew);

  count = 0;
  do {
    double c5=5.0, c16 = 16.0, c20 = 20.0;

    x = xnew;
    arg = b*x/2.0;
    s2 = sin(arg);
    c2 = cos(arg);
    g1 = 2.0*s2*c2/b;
    g2 = 2.0*s2*s2/beta;
    g3 = (x - g1)/beta;
    f = r0*x + eta*g2 + zeta*g3 - h;
    fp = r0 + eta*g1 + zeta*g2;
    g0 = 1.0 - beta*g2;
    fpp = eta*g0 + zeta*g1;
    dx = -c5*f/(fp + sqrt(fabs(c16*fp*fp - c20*f*fpp)));
    xnew = x + dx;
    if(count++ > 20) {
      return(FAILURE);
    }
  } while(fabs(dx) > err);

  /* do a final newton */
  /*
  x = xnew;
  arg = b*x/2.0;
  s2 = sin(arg);
  c2 = cos(arg);
  g1 = 2.0*s2*c2/b;
  g2 = 2.0*s2*s2/beta;
  g3 = (x - g1)/beta;
  g = eta*g1 + zeta*g2;
  xnew = (x*g - eta*g2 - zeta*g3 + h)/(r0 + g);
   */
  x = xnew;
  arg = b*x/2.0;
  s2 = sin(arg);
  c2 = cos(arg);

  *X = x;
  *B = b;
  *S2 = s2;
  *C2 = c2;

  return(SUCCESS);
}

int solve_universal_newton_bisection(double kc, double r0, double beta, double eta, double zeta, double h, 
				     double *X, double *B, double *S2, double *C2)
{
  double b, x;
  double g1, g2, g3;
  double arg, s2, c2;
  double f;
  double X_min, X_max, X_per_period, invperiod;
  int count;
  double xnew, g;
  double err;

  xnew = *X;

  err = 1.e-9*fabs(xnew);

  b = sqrt(beta);

  /* bisection limits due to Rein et al. */
  invperiod = b*beta/(2.*M_PI*kc);
  X_per_period = 2.*M_PI/b;
  X_min = X_per_period * floor(h*invperiod);
  X_max = X_min + X_per_period;
  xnew = (X_max + X_min)/2.;

  count = 0;
  do {
    x = xnew;
    arg = b*x/2.0;
    s2 = sin(arg);
    c2 = cos(arg);
    g1 = 2.0*s2*c2/b;
    g2 = 2.0*s2*s2/beta;
    g3 = (x - g1)/beta;
    f = r0*x + eta*g2 + zeta*g3 - h;
    if (f>=0.){
      X_max = x;
    }else{
      X_min = x;
    }
    xnew = (X_max + X_min)/2.;
    if(count++ > 100) return(FAILURE);

  } while(fabs(x - xnew) > err);

  /* do a final newton */
  x = xnew;
  arg = b*x/2.0;
  s2 = sin(arg);
  c2 = cos(arg);
  g1 = 2.0*s2*c2/b;
  g2 = 2.0*s2*s2/beta;
  /* compute_dgs(b, x, s2, c2, &g3, &g1beta, &g2beta, &g3beta); */
  g3 = (x - g1)/beta;
  g = eta*g1 + zeta*g2;
  xnew = (x*g - eta*g2 - zeta*g3 + h)/(r0 + g);

  x = xnew;
  
  arg = b*x/2.0;
  s2 = sin(arg);
  c2 = cos(arg);

  *X = x;
  *B = b;
  *S2 = s2;
  *C2 = c2;

  return(SUCCESS);
}

double sign(double x)
{
  if(x > 0.0) {
    return(1.0);
  } else if(x < 0.0) {
    return(-1.0);
  } else {
    return(0.0);
  }
}

double cubic1(double a, double b, double c)
{
  double Q, R;
  double theta, A, B, x1;
  
  Q = (a*a - 3.0*b)/9.0;
  R = (2.0*a*a*a - 9.0*a*b + 27.0*c)/54.0;
  if(R*R < Q*Q*Q) {
    theta = acos(R/sqrt(Q*Q*Q));
    printf("three cubic roots %.16le %.16le %.16le\n", 
	   -2.0*sqrt(Q)*cos(theta/3.0) - a/3.0,
	   -2.0*sqrt(Q)*cos((theta + 2.0*M_PI)/3.0) - a/3.0,
	   -2.0*sqrt(Q)*cos((theta - 2.0*M_PI)/3.0) - a/3.0);
    exit(-1);
  } else {
    A = -sign(R)*pow(fabs(R) + sqrt(R*R - Q*Q*Q), 1./3.);
    if(A == 0.0) {
      B = 0.0;
    } else {
      B = Q/A;
    }
    x1 = (A + B) - a/3.0;

    return(x1);
  }
}

int solve_universal_parabolic(double kc, double r0, double beta, double eta, double zeta, double h, 
			      double *X, double *B, double *S2, double *C2)
{
  double b, x;
  double s2, c2;

  b = 0.0;

  s2 = 0.0;
  c2 = 1.0;
  /*
    g1 = x;
    g2 = x*x/2.0;
    g3 = x*x*x/6.0;
    g = eta*g1 + zeta*g2;
   */
  /*   f = r0*x + eta*g2 + zeta*g3 - h; */
  /*   this is just a cubic equation */
  x = cubic1(3.0*eta/zeta, 6.0*r0/zeta, -6.0*h/zeta);

  s2 = 0.0;
  c2 = 1.0;
  /* g1 = 2.0*s2*c2/b; */
  /* g2 = 2.0*s2*s2/beta; */

  *X = x;
  *B = b;
  *S2 = s2;
  *C2 = c2;

  return(SUCCESS);
}

int solve_universal_hyperbolic_laguerre(double kc, double r0, double minus_beta, double eta, double zeta, double h, double v2,
					double *X, double *B, double *S2, double *C2)
{
  double b, x;
  double g0, g1, g2, g3;
  double arg, s2, c2, xnew;
  int count;
  double f;

  b = sqrt(minus_beta);
  xnew = *X;

  count = 0;
  do {
    double c5=5.0, c16 = 16.0, c20 = 20.0;
    double fp, fpp, dx;
    double den;

    x = xnew;
    arg = b*x/2.0;
    if(fabs(arg)>50.0) return(FAILURE);
    s2 = sinh(arg);
    c2 = cosh(arg);
    g1 = 2.0*s2*c2/b;
    g2 = 2.0*s2*s2/minus_beta;
    g3 = -(x - g1)/minus_beta;
    f = r0*x + eta*g2 + zeta*g3 - h;
    fp = r0 + eta*g1 + zeta*g2;
    g0 = 1.0 + minus_beta*g2;
    fpp = eta*g0 + zeta*g1;
    den = (fp + sqrt(fabs(c16*fp*fp - c20*f*fpp)));
    if(den == 0.0) return(FAILURE);
    dx = -c5*f/den;
    xnew = x + dx;
    if(count++ > 20) return(FAILURE);

  } while(fabs(x - xnew) > 1.e-9*fabs(xnew));
  
  /* do a final newton */
  /*
  x = xnew;
  arg = b*x/2.0;
  s2 = sinh(arg);
  c2 = cosh(arg);
  g1 = 2.0*s2*c2/b;
  g2 = 2.0*s2*s2/minus_beta;
  g3 = -(x - g1)/minus_beta;
  g = eta*g1 + zeta*g2;
  xnew = (x*g - eta*g2 - zeta*g3 + h)/(r0 + g);
   */

  x = xnew;
  arg = b*x/2.0;
  s2 = sinh(arg);
  c2 = cosh(arg);

  *X = x;
  *B = b;
  *S2 = s2;
  *C2 = c2;

  return(SUCCESS);
}

int solve_universal_hyperbolic_bisection(double kc, double r0, double minus_beta, double eta, double zeta, double h, double v2,
					 double *X, double *B, double *S2, double *C2)
{
  double b, x;
  double g1, g2, g3;
  double arg, s2, c2, xnew;
  int count;
  double h2, q, vq, X_min, X_max;
  double f;
  double g;

  b = sqrt(minus_beta);

  xnew = *X;

  /* bisection limits due to Rein et al. */
  h2 = r0*r0*v2-eta*eta;
  q = h2/kc/(1.+sqrt(1.+h2*minus_beta/(kc*kc)));
  vq = sqrt(h2)/q;
  X_min = 1./(vq+r0/h);

  X_max = 10.0*xnew;
  X_min = 0.99*X_min; /* kludge */

  if((X_min - xnew)*(X_max - xnew) > 0.0) {
    return(FAILURE);
  }

  {
    double fmin, fmax;
    x = X_min;
    arg = b*x/2.0;
    if(fabs(arg)>50.0) return(FAILURE);
    s2 = sinh(arg);
    c2 = cosh(arg);
    g1 = 2.0*s2*c2/b;
    g2 = 2.0*s2*s2/minus_beta;
    g3 = -(x - g1)/minus_beta;
    fmin = r0*x + eta*g2 + zeta*g3 - h;

    x = X_max;
    arg = b*x/2.0;
    if(fabs(arg)>50.0) {
      x = 50.0/(b/2.0);
      arg = 50.0;
    }
    s2 = sinh(arg);
    c2 = cosh(arg);
    g1 = 2.0*s2*c2/b;
    g2 = 2.0*s2*s2/minus_beta;
    g3 = -(x - g1)/minus_beta;
    fmax = r0*x + eta*g2 + zeta*g3 - h;

    if(fmin*fmax > 0.0) {
      return(FAILURE);
    }
  }


  count = 0;
  do {
    x = xnew;
    arg = b*x/2.0;
    if(fabs(arg)>50.0) return(FAILURE);
    s2 = sinh(arg);
    c2 = cosh(arg);
    g1 = 2.0*s2*c2/b;
    g2 = 2.0*s2*s2/minus_beta;
    g3 = -(x - g1)/minus_beta;
    f = r0*x + eta*g2 + zeta*g3 - h;
    if (f>=0.){
      X_max = x;
    }else{
      X_min = x;
    }
    xnew = (X_max + X_min)/2.;
    if(count++ > 100) return(FAILURE);

  } while(fabs(x - xnew) > 1.e-12*fabs(xnew));
  
  /* do a final newton */
  x = xnew;
  arg = b*x/2.0;
  if(fabs(arg)>50.0) return(FAILURE);
  s2 = sinh(arg);
  c2 = cosh(arg);
  g1 = 2.0*s2*c2/b;
  g2 = 2.0*s2*s2/minus_beta;
  g3 = -(x - g1)/minus_beta;
  g = eta*g1 + zeta*g2;
  xnew = (x*g - eta*g2 - zeta*g3 + h)/(r0 + g);

  x = xnew;
  arg = b*x/2.0;
  s2 = sinh(arg);
  c2 = cosh(arg);

  *X = x;
  *B = b;
  *S2 = s2;
  *C2 = c2;

  return(SUCCESS);
}

int solve_universal_hyperbolic_newton(double kc, double r0, double minus_beta, double eta, double zeta, double h, double v2,
				      double *X, double *B, double *S2, double *C2)
{
  double b, x;
  double g1, g2, g3;
  double arg, s2, c2, xnew;
  int count;
  double g;

  b = sqrt(minus_beta);
  xnew = *X;

  count = 0;
  do {
    x = xnew;
    arg = b*x/2.0;
    if(fabs(arg)>50.0) return(FAILURE);
    s2 = sinh(arg);
    c2 = cosh(arg);
    g1 = 2.0*s2*c2/b;
    g2 = 2.0*s2*s2/minus_beta;
    g3 = -(x - g1)/minus_beta;
    g = eta*g1 + zeta*g2;
    xnew = (x*g - eta*g2 - zeta*g3 + h)/(r0 + g);

    if(count++ > 10) return(FAILURE);

  } while(fabs(x - xnew) > 1.e-9*fabs(xnew));
  
  /* do a final newton */
  x = xnew;
  arg = b*x/2.0;
  s2 = sinh(arg);
  c2 = cosh(arg);
  g1 = 2.0*s2*c2/b;
  g2 = 2.0*s2*s2/minus_beta;
  g3 = -(x - g1)/minus_beta;
  g = eta*g1 + zeta*g2;
  xnew = (x*g - eta*g2 - zeta*g3 + h)/(r0 + g);

  x = xnew;
  arg = b*x/2.0;
  s2 = sinh(arg);
  c2 = cosh(arg);

  *X = x;
  *B = b;
  *S2 = s2;
  *C2 = c2;

  return(SUCCESS);
}

void kepler_step(double kc, double dt, State *s0, State *s)
{
  void kepler_step_depth();

  kepler_step_depth(kc, dt, s0, s, 0);
}

void kepler_step_depth(double kc, double dt, State *s0, State *s, int depth)
{
  int flag;
  State ss;
  int kepler_step_internal();

  if(depth > 10) {
    printf("kepler depth exceeded\n");
    exit(-1);
  }

  flag = kepler_step_internal(kc, dt, s0, s);

  if(flag == FAILURE) {
    kepler_step_depth(kc, dt/4.0, s0, &ss, depth+1);
    kepler_step_depth(kc, dt/4.0, &ss, s, depth+1);
    kepler_step_depth(kc, dt/4.0, s, &ss, depth+1);
    kepler_step_depth(kc, dt/4.0, &ss, s, depth+1);
  }
}

int kepler_step_internal(double kc, double dt, State *s0, State *s)
{
  double r0, r, v2, eta, beta, zeta;
  double b, c2, s2;
  double f, g, fdot, gdot;
  double G1, G2;
  double a, x;
  int flag;
  double c, ca, bsa;

  r0 = sqrt(s0->x*s0->x + s0->y*s0->y + s0->z*s0->z);
  v2 = s0->xd*s0->xd + s0->yd*s0->yd + s0->zd*s0->zd;
  eta = s0->x*s0->xd + s0->y*s0->yd + s0->z*s0->zd;
  beta = 2.0*kc/r0 - v2;
  zeta = kc - beta*r0;

  if(beta < 0.0) {
    double x0;

    a = kc/(-beta);

    /* following Danby */
    /*
    n = sqrt(kc/(a*a*a));
    ch = 1.0 + r0/a;
    sh = eta/sqrt(a*kc);
    e = sqrt(ch*ch - sh*sh);
    dm = n*dt;
    if(dm<0.0){
      x0 = -log((-2.0*dm + 1.8*e)/(ch - sh))/sqrt(-beta);
    }else{
      x0 = log((2.0*dm + 1.8*e)/(ch + sh))/sqrt(-beta);
    }
     */

    /* use a parabolic guess */
    x0 = cubic1(3.0*eta/zeta, 6.0*r0/zeta, -6.0*dt/zeta);

    x = x0;
    flag = solve_universal_hyperbolic_newton(kc, r0, -beta, eta, zeta, dt, v2, &x, &b, &s2, &c2);
    if(flag == FAILURE) {
      x = x0;
      flag = solve_universal_hyperbolic_laguerre(kc, r0, -beta, eta, zeta, dt, v2, &x, &b, &s2, &c2);
    }
    if(flag == FAILURE) {
      x = x0;
      flag = solve_universal_hyperbolic_bisection(kc, r0, -beta, eta, zeta, dt, v2, &x, &b, &s2, &c2);
    }
    if(flag == FAILURE) {
      return(FAILURE);
    }

    G1 = 2.0*s2*c2/b;
    c = 2.0*s2*s2;
    G2 = c/(-beta);
    ca = c*a;
    r = r0 + eta*G1 + zeta*G2;
    bsa = (a/r)*(b/r0)*2.0*s2*c2;

  } else if(beta > 0.0) {

    x = dt/r0;

    flag = solve_universal_newton(kc, r0, beta, eta, zeta, dt, &x, &b, &s2, &c2);
    if(flag == FAILURE) {
      x = dt/r0;
      flag = solve_universal_laguerre(kc, r0, beta, eta, zeta, dt, &x, &b, &s2, &c2);
    }
    if(flag == FAILURE) {
      x = dt/r0;
      flag = solve_universal_newton_bisection(kc, r0, beta, eta, zeta, dt, &x, &b, &s2, &c2);
    }
    if(flag == FAILURE) {
      return(FAILURE);
    }

    a = kc/beta;
    G1 = 2.0*s2*c2/b;
    c = 2.0*s2*s2;
    G2 = c/beta;
    ca = c*a;
    r = r0 + eta*G1 + zeta*G2;
    bsa = (a/r)*(b/r0)*2.0*s2*c2;

  } else {
    
    x = dt/r0;

    flag = solve_universal_parabolic(kc, r0, beta, eta, zeta, dt, &x, &b, &s2, &c2);
    if(flag == FAILURE) {
      exit(-1);
    }
    
    G1 = x;
    G2 = x*x/2.0;
    ca = kc*G2;
    r = r0 + eta*G1 + zeta*G2;
    bsa = kc*x/(r*r0);
    
  }

  r = r0 + eta*G1 + zeta*G2;

  f = 1.0 - (ca/r0);
  g = eta*G2 + r0*G1;
  fdot = -bsa;
  gdot = 1.0 - (ca/r);

  s->x = (f*s0->x + g*s0->xd);
  s->y = (f*s0->y + g*s0->yd);
  s->z = (f*s0->z + g*s0->zd);
  s->xd = (fdot*s0->x + gdot*s0->xd);
  s->yd = (fdot*s0->y + gdot*s0->yd);
  s->zd = (fdot*s0->z + gdot*s0->zd);

  return(SUCCESS);
}

