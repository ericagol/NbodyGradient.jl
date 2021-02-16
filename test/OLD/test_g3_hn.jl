
include("../src/g3.jl")

using PyPlot

@testset "g3_hn" begin

gamma = logspace(-10.0,0.0,1000)
gbig = big.(gamma)

for beta in [-1.5,-0.75,-0.1,-0.01,-0.001,-0.0001,-0.00001,-0.000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,0.75,1.5]
  clf()
  betabig=big(beta)
  zbig = big(0.0)

  g3b = convert(Array{Float64,1},G3.(gbig,betabig;gc = zbig))
  h1b = convert(Array{Float64,1},H1.(gbig,betabig;gc = zbig))
  h2b = convert(Array{Float64,1},H2.(gbig,betabig;gc = zbig))
  h3b = convert(Array{Float64,1},H3.(gbig,betabig;gc = zbig))
  h5b = convert(Array{Float64,1},H5.(gbig,betabig;gc = zbig))
  h6b = convert(Array{Float64,1},H6.(gbig,betabig;gc = zbig))
  h7b = convert(Array{Float64,1},H7.(gbig,betabig;gc = zbig))
  h8b = convert(Array{Float64,1},H8.(gbig,betabig;gc = zbig))

  g3 = G3.(gamma,beta)
  h1 = H1.(gamma,beta) 
  h2 = H2.(gamma,beta) 
  h3 = H3.(gamma,beta) 
  h5 = H5.(gamma,beta) 
  h6 = H6.(gamma,beta) 
  h7 = H7.(gamma,beta) 
  h8 = H8.(gamma,beta) 
  semilogy(gamma,abs.(g3./g3b-1),label="G3")
  plot(gamma,abs.(h1./h1b-1),label="H1")
  plot(gamma,abs.(h2./h2b-1),label="H2")
  plot(gamma,abs.(h3./h3b-1),label="H3")
  plot(gamma,abs.(h5./h5b-1),label="H5")
  plot(gamma,abs.(h6./h6b-1),label="H6")
  plot(gamma,abs.(h7./h7b-1),label="H7")
  plot(gamma,abs.(h8./h8b-1),label="H8")
  @test isapprox(g3b,g3;norm=maxabs)
  @test isapprox(h1b,h1;norm=maxabs)
  @test isapprox(h2b,h2;norm=maxabs)
  @test isapprox(h3b,h3;norm=maxabs)
  @test isapprox(h5b,h5;norm=maxabs)
  @test isapprox(h6b,h6;norm=maxabs)
  @test isapprox(h7b,h7;norm=maxabs)
  @test isapprox(h8b,h8;norm=maxabs)
  println("beta: ",beta)
  axis([0.0,1.0,1e-16,0.1])
  legend(loc="upper left")
  read(STDIN,Char)
end
end
