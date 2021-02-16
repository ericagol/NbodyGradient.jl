# Converting cartesian coordinates to orbital elements

"""

A 3d point in cartesian space 
"""
struct Point{T<:AbstractFloat}
    x::T
    y::T
    z::T
end

# Constructors
Point(x::AbstractVector) = Point(x...)
Point(x::AbstractMatrix) = [Point(x[:,i]) for i in eachindex(x[1,:])]
Point(x::Real) = Point(float(x),float(x),float(x))
unpack(x::Point) = (x.x,x.y,x.z)

# Overloads for products
LinearAlgebra.dot(x::Point,v::Point) = x.x*v.x + x.y*v.y + x.z*v.z
LinearAlgebra.dot(x::Point) = x.x*x.x + x.y*x.y + x.z*x.z
LinearAlgebra.cross(x::Point,v::Point) = Point(x.y*v.z - x.z*v.y,-(x.x*v.z - x.z*v.x),x.x*v.y - x.y*v.x)

""" Calculate the relative positions from the A-Matrix. """
function get_relative_positions(x,v,ic::InitialConditions)
    n = ic.nbody
    X = zeros(3,n)
    V = zeros(3,n)
    
    X .= permutedims(ic.amat*x')
    V .= permutedims(ic.amat*v')

    return Point(X), Point(V)
end

""" Get relative masses(*G) from initial conditions. """
function get_relative_masses(ic::InitialConditions)
    N = length(ic.m)
    M = zeros(N-1)
    G = 39.4845/(365.242 * 365.242) # AU^3 Msol^-1 Day^-2
    for i in 1:N-1
        for j in 1:N
            M[i] += abs(ic.ϵ[i,j])*ic.m[j]
        end
    end
    return G .* M
end

# Position and velocity magnitudes
mag(x) = sqrt(dot(x))
mag(x,v) = sqrt(dot(x,v))
Rdotmag(R,V,h) = sqrt(V^2 - (h/R)^2)

function hvec(r,rdot) 
    hx,hy,hz = unpack(cross(r,rdot))
    hz >= 0.0 ? hy *= -1 : hx *= -1
    return Point(hx,hy,hz)
end

function calc_Ω(hx,hy,h,I)
    sinΩ = hx/(h*sin(I))
    cosΩ = hy/(h*sin(I))
    return atan(sinΩ,cosΩ) 
end

function calc_ϖ(x,R,Rdot,I,Ω,a,e,h)
    # Find ω + f
    wpf = 0.0
    if I != 0.0
        sinwpf = x.z/(R*sin(I))
        coswpf = ((x.x/R) + sin(Ω)*sinwpf*cos(I))/cos(Ω)
        wpf = atan(sinwpf,coswpf)
    end
    # Find f
    sinf = a*Rdot*(1.0 - e^2)/(h*e)
    cosf = (a*(1.0-e^2)/R - 1.0)/e
    f = atan(sinf,cosf)

    # ϖ = Ω + ω
    return Ω + wpf - f
end

function convert_to_elements(x,v,M)
    R = mag(x)
    V = mag(v)
    h = mag(hvec(x,v))
    hx,hy,hz = unpack(hvec(x,v))
    Rdot = sign(dot(x,v))*Rdotmag(R,V,h)
    
    Gmm = M 
    
    a = 1.0/((2.0/R) - (V*V)/(Gmm))
    e = sqrt(1.0 - (h*h/(Gmm*a)))
    I = acos(hz/h)
    
    # Make sure Ω is defined
    I != 0.0 ? Ω = calc_Ω(hx,hy,h,I) : Ω = 0.0
    
    ϖ = calc_ϖ(x,R,Rdot,I,Ω,a,e,h)
    P = (2.0*π)*sqrt(a*a*a/Gmm)

    return [P,0.0,e*cos(ϖ),e*sin(ϖ),I,Ω,a,e,ϖ]
end    

function get_orbital_elements(s::State{T},ic::InitialConditions{T}) where T<:AbstractFloat
    elems = Elements{T}[]
    μ = get_relative_masses(ic)
    X,V = get_relative_positions(s.x,s.v,ic)
    N = ic.nbody

    push!(elems,Elements(m=ic.m[1]))

    i = 1; b = 0
    while i < N

        # Check if new binary
        if first(ic.ϵ[i,:]) == zero(T)
            b+=1
        end

        new_elems = convert_to_elements(X[i+b],V[i+b],μ[i+b])
        push!(elems,Elements(ic.m[i+1],new_elems...))
        
        # Compensate for new binary in step
        if b > 0
            b -= 2
        elseif b < 0
            i += 1
        end
        
    i+=1
    end
    return elems
end
