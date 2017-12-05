function ekepler(m,ecc)
eps=1e-12
#real*8 e,e0,eps,m,ms,pi2,f0,f1,f2,f3,d1,d2,d3
# This routine solves Kepler's equation for E as a function of (e,M)
# using the procedure outlined in Murray & Dermott:
pi2=2.0*pi
ms=mod(m,pi2)
d3=1e10
e0=ms+ecc*0.85*sin(ms)/abs(sin(ms))
while abs(d3) > eps
  f3=ecc*cos(e0)
  f2=ecc*sin(e0)
  f1=1.0-f3
  f0=e0-ms-f2
  d1=-f0/f1
  d2=-f0/(f1+0.5*d1*f2)
  d3=-f0/(f1+d2*0.5*(f2+d2*f3/3.))
  e0=e0+d3
end
ekep=e0+m-ms
return ekep
end

function kepler(m,ecc)
@assert(ecc >= 0.0)
@assert(ecc <= 1.0)
f=m
if ecc > 0
  ekep=ekepler(m,ecc)
#  println(m-ekep+ecc*sin(ekep))
  f=2.0*atan(sqrt((1.0+ecc)/(1.0-ecc))*tan(0.5*ekep))
end
return f
end
