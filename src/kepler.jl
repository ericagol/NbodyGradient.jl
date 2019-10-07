
function ekepler(m::T,ecc::T) where {T <: Real}
KEPLER_TOL = sqrt(eps(m))
if m != 0.0
#real*8 e,e0,eps,m,ms,pi2,f0,f1,f2,f3,d1,d2,d3
# This routine solves Kepler's equation for E as a function of (e,M)
# using the procedure outlined in Murray & Dermott:
  pi2=2.0*pi
  ms=mod(m,pi2)
  d3 =one(T)
  d31=2*d3
  d32=3*d3
  de0=ecc*0.85*sign(ms)
  iter = 0
  ITMAX = 50
#  while abs(d3) > KEPLER_TOL
  while true
    d32 = d31
    d31 = d3
    f3=ecc*cos(de0+ms)
    f2=ecc*sin(de0+ms)
    f1=1.0-f3
    f0=de0-f2
    d1=-f0/f1
    d2=-f0/(f1+0.5*d1*f2)
    d3=-f0/(f1+d2*0.5*(f2+d2*f3/3.))
    de0 += d3
    iter += 1
    if iter >= ITMAX || d3 == d31 || d3 == d32
      break
    end
  end
  if iter > 20
    println("iterations in ekepler: ",iter)
  end
  ekep=de0+m
else
  ekep = 0.0
end
return ekep::typeof(m)
end

function ekepler2(m::T,ecc::T) where {T <: Real}
KEPLER_TOL = sqrt(eps(m))
if m != 0.0
#real*8 e,e0,eps,m,ms,pi2,f0,f1,f2,f3,d1,d2,d3
# This routine solves Kepler's equation for E as a function of (e,M)
# using the procedure outlined in Murray & Dermott:
  pi2=2.0*pi
  ms=mod(m,pi2)
  d3 =one(T)
  d31=2*d3
  d32=3*d3
  e0=ms+ecc*0.85*sin(ms)/abs(sin(ms))
#  while abs(d3) > KEPLER_TOL
  ITMAX = 50
  while true
    d32 = d31
    d31 = d3
    f3=ecc*cos(e0)
    f2=ecc*sin(e0)
    f1=1.0-f3
    f0=e0-ms-f2
    d1=-f0/f1
    d2=-f0/(f1+0.5*d1*f2)
    d3=-f0/(f1+d2*0.5*(f2+d2*f3/3.))
    e0=e0+d3
    if iter >= ITMAX || d3 == d31 || d3 == d32
      break
    end
  end
  ekep=e0+m-ms
else
  ekep = 0.0
end
return ekep::typeof(m)
end

function kepler(m,ecc)
@assert(ecc >= 0.0)
@assert(ecc <= 1.0)
f=m
if ecc > 0
  ekep=ekepler(m,ecc)
#  println(m-ekep+ecc*sin(ekep))
#  f=2.0*atan(sqrt((1.0+ecc)/(1.0-ecc))*tan(0.5*ekep))
  f=2.0*atan2(sqrt(1.0+ecc)*sin(0.5*ekep),sqrt(1.0-ecc)*cos(0.5*ekep))
end
return f
end
