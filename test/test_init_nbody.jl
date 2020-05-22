
const YEAR  = 365.242
const GNEWT = 39.4845/YEAR^2  # Units of MSUN*AU^3/YEAR^2
const NDIM  = 3
const third = 1.0/3.0

include("../src/init_nbody.jl")

#@testset "init_nbody" begin

elements = "elements.txt"
H = [3,1,1]
t0 = 7257.93115525-7300.0
init = ElementsIC(elements,H,t0) 
init.elements[:,3] .-= 7300.0
x,v,jac_init = init_nbody(init)
elements_big = big.(init.elements)
t0big = big(t0)
init_big = ElementsIC(elements_big,H,t0big)
xbig,vbig,jac_init_big = init_nbody(init_big)

#elements = readdlm("elements.txt",',')

#elements[2:3,1] /= 100.0
#n_body = 4
n_body = 3

#jac_init     = zeros(Float64,7*n_body,7*n_body)
#jac_init_big = zeros(BigFloat,7*n_body,7*n_body)
jac_init_num = zeros(BigFloat,7*n_body,7*n_body)
#x,v = init_nbody(elements,t0,n_body,jac_init)
#elements_big = big.(elements)
#t0big = big(t0)
#xbig,vbig = init_nbody(elements_big,t0big,n_body,jac_init_big)
elements0 = copy(init.elements)
#dq = big.([1e-10,1e-5,1e-6,1e-6,1e-6,1e-5,1e-5])
#dq = big.([1e-10,1e-8,1e-8,1e-8,1e-8,1e-8,1e-8])
dq = big.([1e-12,1e-10,1e-10,1e-10,1e-10,1e-10,1e-10])

#dq = big.([1e-15,1e-15,1e-15,1e-15,1e-15,1e-15,1e-15])
#t0big = big(t0)
# Now, compute derivatives numerically:
for j=1:n_body
  for k=1:7
    elementsbig = big.(elements0)
    dq0 = dq[k]; if j==1 && k==1 ; dq0 = big(1e-15); end
    elementsbig[j,k] += dq0
    initp = ElementsIC(elementsbig,H,t0big)
    xp,vp,_ = init_nbody(initp)
    elementsbig[j,k] -= 2dq0
    initm = ElementsIC(elementsbig,H,t0big)
    xm,vm,_ = init_nbody(initm)
    for l=1:n_body, p=1:3
      i1 = (l-1)*7+p
      if k == 1
        j1 = j*7
      else
        j1 = (j-1)*7+k-1
      end
      jac_init_num[i1,  j1] = (xp[p,l]-xm[p,l])/dq0*.5
      jac1 = jac_init[i1,j1]; jac2 = jac_init_num[i1,j1]
      if abs(jac1-jac2) > 1e-4*abs(jac1+jac2) && abs(jac1+jac2) > 1e-14
        println(l," ",p," ",j," ",k," ",jac_init_num[i1,j1]," ",jac_init[i1,j1]," ",jac_init_num[i1,j1]/jac_init[i1,j1])
      end
      jac_init_num[i1+3,j1] = (vp[p,l]-vm[p,l])/dq0*.5
      jac1 = jac_init[i1+3,j1]; jac2 = jac_init_num[i1+3,j1]
      if abs(jac1-jac2) > 1e-4*abs(jac1+jac2) && abs(jac1+jac2) > 1e-14
        println(l," ",p+3," ",j," ",k," ",jac_init_num[i1+3,j1]," ",jac_init[i1+3,j1]," ",jac_init_num[i1+3,j1]/jac_init[i1+3,j1])
      end
    end
  end
  jac_init_num[j*7,j*7]=1.0
end

#println("Maximum jac_init-jac_init_num: ",maximum(abs.(jac_init-jac_init_num)))
println("Maximum jac_init-jac_init_num: ",maximum(abs.(asinh.(jac_init)-asinh.(jac_init_num))))
println("Maximum jac_init-jac_init_big: ",maximum(abs.(asinh.(jac_init)-asinh.(jac_init_big))))
println("Maximum jac_init/jac_init_big-1: ",maximum(abs.(jac_init[jac_init_big .!= 0.0]./jac_init_big[jac_init_big .!= 0.0].-1)))
println("Maximum jac_init_big-jac_init_num: ",maximum(abs.(asinh.(jac_init_num)-asinh.(jac_init_big))))
#println(convert(Array{Float64,2},jac_init_big-jac_init_num))
#@test isapprox(jac_init_num,jac_init)
@test isapprox(jac_init_num,jac_init;norm=maxabs)
@test isapprox(jac_init,convert(Array{Float64,2},jac_init_big);norm=maxabs)
#end

for i=1:21, j=1:21
  if jac_init_big[i,j] != 0.0
    dij = abs(jac_init[i,j]/jac_init_big[i,j]-1)
    if dij > 1e-12
      jac_max = dij
      println(i," ",j," ",convert(Float64,jac_max)," ",jac_init[i,j]," ",convert(Float64,jac_init_big[i,j])," ",convert(Float64,jac_init[i,j]-jac_init_big[i,j]))
    end
  end
end
