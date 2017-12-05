# Runs a simple test of kepler_init.jl

const GNEWT = 39.4845/365.242^2
const third = 1.0/3.0

include("kepler_init.jl")

t0 = 2.4
mass = 1.0
period = 1.5
elements = ones(Float64,6)
while elements[3]^2+elements[4]^2 >= 1.0
  elements = [1.5,rand()*period,randn(),randn(),rand()*pi,rand()*pi]
end
elements_diff = zeros(Float64,6)
ecc = sqrt(elements[3]^2+elements[4]^2)
ntime = 100000
time = linspace(t0,t0+period,ntime)
xvec = zeros(Float64,3,ntime)
vvec = zeros(Float64,3,ntime)
vfvec = zeros(Float64,3,ntime)
#dt = 1e-8
dt = period/ntime
jac_init = zeros(Float64,7,7)
jac_tmp  = zeros(Float64,7,7)
delements = [1e-5,1e-5,1e-5,1e-5,1e-5,1e-5,1e-5]
elements_name = ["P","t0","ecosom","esinom","inc","Omega","Mass"]
cartesian_name = ["x","y","z","vx","vy","vz","mass"]
for i=1:ntime
  x,v = kepler_init(time[i],mass,elements)
  if i==1 
#    x_ekep,v_ekep,tp,dtpdecos,dtpdesin = kepler_init(time[i],mass,elements,jac_init)
    x_ekep,v_ekep = kepler_init(time[i],mass,elements,jac_init)
  # Check that these agree:
    println("x-x_ekep: ",maximum(abs.(x-x_ekep))," v-v_ekep: ",maximum(abs.(v-v_ekep)))
  # Now take some derivatives:
    jac_init_num = zeros(Float64,7,7)
    for j=1:7
      elements_diff .= elements
      if j <= 6
        elements_diff[j] -= delements[j]
      else
        mass -= delements[7]
      end
#      x_minus,v_minus,tp_minus,dtpdecos_minus,dtpdesin_minus= kepler_init(time[i],mass,elements_diff,jac_tmp)
      x_minus,v_minus= kepler_init(time[i],mass,elements_diff,jac_tmp)
      elements_diff .= elements
      if j <= 6
        elements_diff[j] += delements[j]
      else
        mass += 2.0*delements[7]
      end
#      x_plus,v_plus,tp_plus,dtpdecos_plus,dtpdesin_plus= kepler_init(time[i],mass,elements_diff,jac_tmp)
      x_plus,v_plus= kepler_init(time[i],mass,elements_diff,jac_tmp)
      if j == 7
        mass -= delements[7]
      end
#      if j==3
#        dtpdecos_num = .5*(tp_plus-tp_minus)/delements[j]
#        println("dtpdecos: ",dtpdecos," dtpdecos_num: ",dtpdecos_num," ",dtpdecos-dtpdecos_num)
#      end
#      if j==4
#        dtpdesin_num = .5*(tp_plus-tp_minus)/delements[j]
#        println("dtpdesin: ",dtpdesin," dtpdesin_num: ",dtpdesin_num," ",dtpdesin-dtpdesin_num)
#      end
      for k=1:3
        jac_init_num[  k,j] = .5*(x_plus[k]-x_minus[k])/delements[j]
        jac_init_num[3+k,j] = .5*(v_plus[k]-v_minus[k])/delements[j]
      end
    end
    jac_init_num[7,7]=1.0
    for j=1:7, k=1:7
      if abs(jac_init[k,j]-jac_init_num[k,j]) > 1e-8
        println(elements_name[j]," ",cartesian_name[k]," ",jac_init[k,j]," ",jac_init_num[k,j]," ",jac_init[k,j]-jac_init_num[k,j])
      end
    end
    println("Jacobians: ",maximum(abs.(jac_init-jac_init_num)))
  end
  xvec[:,i]=x
  vvec[:,i]=v
  # Compute finite difference velocity:
  xf,vf = kepler_init(time[i]+dt,mass,elements)
  vfvec[:,i]=(xf-x)/dt
end
period = elements[1]
omega=atan2(elements[4],elements[3])
semi = (GNEWT*mass*period^2/4/pi^2)^third
# Compute the focus:
focus = [semi*ecc,0.,0.]
xfocus = semi*ecc*(-cos(omega)-sin(omega))
zfocus = semi*ecc*sin(omega)
# Check that we have an ellipse (we're assuming that the motion is in the x-z plane; no longer true):
Atot = 0.0
dAdt = zeros(Float64,ntime)
dx = xvec[1,1]-xvec[1,ntime-1]
dy = xvec[2,1]-xvec[2,ntime-1]
dz = xvec[3,1]-xvec[3,ntime-1]
dAvec = 0.5*[xvec[2,1]*dz-xvec[3,1]*dy;xvec[3,1]*dx-xvec[1,1]*dz;xvec[1,1]*dy-xvec[2,1]*dx]
#println("dA: ",norm(dAvec)," ",dAvec/norm(dAvec))
#dA = 0.5*(xvec[3,1]*dx-xvec[1,1]*dz)
dAdt[1] = norm(dAvec)/(time[2]-time[1])
Atot += norm(dAvec)
for i=2:ntime
  dx = xvec[1,i]-xvec[1,i-1]
  dy = xvec[2,i]-xvec[2,i-1]
  dz = xvec[3,i]-xvec[3,i-1]
  dAvec = 0.5*[xvec[2,i]*dz-xvec[3,i]*dy;xvec[3,i]*dx-xvec[1,i]*dz;xvec[1,i]*dy-xvec[2,i]*dx]
#  println("dA: ",norm(dAvec)," ",dAvec/norm(dAvec))
  #dA = 0.5*(xvec[3,i]*dx-xvec[1,i]*dz)
  dAdt[i] = norm(dAvec)/(time[i]-time[i-1])
  Atot += norm(dAvec)
end  
println("Total area: ",Atot," ",pi*sqrt(1.-ecc^2)*semi^2," ratio: ",Atot/pi/semi^2/sqrt(1.-ecc^2))
using PyPlot
#plot(time,dAdt)
#axis([minimum(time),maximum(time),0.,1.5*maximum(dAdt)])
println("Scatter in dAdt: ",std(dAdt))

plot(time,vvec[1,:])
plot(time,vfvec[1,:],".")
plot(time,vvec[1,:]-vfvec[1,:],".")
plot(time,vvec[3,:])
plot(time,vfvec[3,:],".")
plot(time,vvec[3,:]-vfvec[3,:],".")
# Check that velocities match finite difference values
