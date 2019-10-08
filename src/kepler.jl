
function ekepler(m::T,ecc::T) where {T <: Real}
KEPLER_TOL = sqrt(eps(m))
if m != zero(T)
#real*8 e,e0,eps,m,ms,pi2,f0,f1,f2,f3,d1,d2,d3
# This routine solves Kepler's equation for E as a function of (e,M)
# using the procedure outlined in Murray & Dermott:
  pi2=2pi
  ms=mod(m,pi2)
  d3 =one(T)
  de0=ecc*0.85*sign(ms)
  de1=2*de0
  de2=3*de0
  iter = 0
  ITMAX = 20
#  while abs(d3) > KEPLER_TOL
  while true
    de2 = de1
    de1 = de0
    f3=ecc*cos(de0+ms)
    f2=ecc*sin(de0+ms)
#    f1=1.0-f3
#    f0=de0-f2
#    d1=-f0/f1
#    d2=-f0/(f1+0.5*d1*f2)
#    d3=-f0/(f1+d2*0.5*(f2+d2*f3/3.))
#    de0 += d3
    de0 = (f2-de1*f3)/(1-f3)
    iter += 1
    if iter >= ITMAX || de0 == de1 || de0 == de2
      break
    end
  end
  if iter >= ITMAX && !(T == BigFloat && minimum([abs(de0-de1),abs(de0-de2)]) < eps(one(T)))
    println("iterations in ekepler: ",iter," de0: ",de0," de1-de0: ",de1-de0," de2-de0: ",de2-de0)
  end
  ekep=de0+m
else
  ekep = zero(T)
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
  e0=ms+ecc*0.85*sin(ms)/abs(sin(ms))
  e1=2*e0
  e2=3*e0
#  while abs(d3) > KEPLER_TOL
  ITMAX = 50
  while true
    e2 = e1
    e1 = e0
    f3=ecc*cos(e0)
    f2=ecc*sin(e0)
    f1=1.0-f3
    f0=e0-ms-f2
    d1=-f0/f1
    d2=-f0/(f1+0.5*d1*f2)
    d3=-f0/(f1+d2*0.5*(f2+d2*f3/3.))
    e0=e0+d3
    if iter >= ITMAX || e0 == e1 || e0 == e2
      break
    end
  end
  if iter >= ITMAX
    println("iterations in ekepler2: ",iter," de0: ",de0," de1-de0: ",de1-de0," de2-de0: ",de2-de0)
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
