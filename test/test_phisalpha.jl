# Tests the routine phisalpha jacobian.  This routine
# computes the force gradient correction after Dehnen & Hernandez (2017).

import NbodyGradient:phisalpha!

# Next, try computing three-body Keplerian Jacobian:

@testset "phisalpha" begin
n = 3
H = [3,1,1]
t0 = 7257.93115525
h  = 0.05
tmax = 600.0
dlnq = big(1e-15)

elements = readdlm("elements.txt", ',')

m = zeros(n)
x0 = zeros(3, n)
v0 = zeros(3, n)
alpha = 2.0

# Define which pairs will have impulse rather than -drift+Kepler:
pair = zeros(Bool, n, n)
for k = 1:n
    m[k] = elements[k,1]
end
m0 = copy(m)

init = ElementsIC(t0, H, elements)
ic_big = ElementsIC(big(t0), H, big.(elements))
x0, v0, _ = init_nbody(init)

# Tilt the orbits a bit:
x0[2,1] = 5e-1 * sqrt(x0[1,1]^2 + x0[3,1]^2)
x0[2,2] = -5e-1 * sqrt(x0[1,2]^2 + x0[3,2]^2)
x0[2,3] = -5e-1 * sqrt(x0[1,2]^2 + x0[3,2]^2)
v0[2,1] = 5e-1 * sqrt(v0[1,1]^2 + v0[3,1]^2)
v0[2,2] = -5e-1 * sqrt(v0[1,2]^2 + v0[3,2]^2)
v0[2,3] = -5e-1 * sqrt(v0[1,2]^2 + v0[3,2]^2)

# Save to state
s = State(init)
s.x .= x0; s.v .= v0

# Take a step:
ah18!(s,h,pair)
x0 .= s.x; v0 .= s.v

# Now, copy these to compute Jacobian (so that I don't step
# x0 & v0 forward in time):
x = copy(s.x); v = copy(s.v); m = copy(s.m)
# Compute jacobian exactly:
d = Derivatives(Float64, s.n)
phisalpha!(s,d,h,alpha,pair)
d.jac_phi .+= Matrix{Float64}(I, size(d.jac_phi))

# Now compute numerical derivatives, using BigFloat to avoid
# round-off errors:
jac_step_num = zeros(BigFloat, 7 * n, 7 * n)
# Save these so that I can compute derivatives numerically:
xsave = big.(x0)
vsave = big.(v0)
msave = big.(m0)
hbig = big(h)
abig = big(alpha)
sbig = deepcopy(State(ic_big))
sbig.x .= xsave; sbig.v .= vsave; sbig.m .= msave
# Carry out step using BigFloat for extra precision:
phisalpha!(sbig,hbig,abig,pair)
# Copy back to saves
xsave .= sbig.x; vsave .= sbig.v; msave .= sbig.m
# Compute numerical derivatives wrt time:
dqdt_num = zeros(BigFloat, 7 * n)
# Initial positions, velocities & masses:
hbig = big(h)
dq = dlnq * hbig
hbig += dq
sbig = deepcopy(State(ic_big))
sbig.x .= big.(x0); sbig.v .= big.(v0); sbig.m .= big.(m0)
phisalpha!(sbig,hbig,abig,pair)

# Now x & v are final positions & velocities after time step
for i in 1:n, k in 1:3
    dqdt_num[(i - 1) * 7 +  k] = (sbig.x[k,i] - xsave[k,i]) / dq
    dqdt_num[(i - 1) * 7 + 3 + k] = (sbig.v[k,i] - vsave[k,i]) / dq
end

hbig = big(h)
# Vary the initial parameters of planet j:
for j = 1:n
    # Vary the initial phase-space elements:
    for jj = 1:3
        # Initial positions, velocities & masses:
        sbig = deepcopy(State(ic_big))
        sbig.x .= big.(x0); sbig.v .= big.(v0); sbig.m .= big.(m0)
        dq = dlnq * sbig.x[jj,j]
        if sbig.x[jj,j] != 0.0
            sbig.x[jj,j] +=  dq
        else
            dq = dlnq
            sbig.x[jj,j] = dq
        end
        phisalpha!(sbig, hbig, abig, pair)
        # Now x & v are final positions & velocities after time step
        for i = 1:n
            for k = 1:3
                jac_step_num[(i - 1) * 7 +  k,(j - 1) * 7 + jj] = (sbig.x[k,i] - xsave[k,i]) / dq
                jac_step_num[(i - 1) * 7 + 3 + k,(j - 1) * 7 + jj] = (sbig.v[k,i] - vsave[k,i]) / dq
            end
        end

        sbig = deepcopy(State(ic_big))
        sbig.x .= big.(x0); sbig.v .= big.(v0); sbig.m .= big.(m0)
        dq = dlnq * sbig.v[jj,j]
        if sbig.v[jj,j] != 0.0
            sbig.v[jj,j] +=  dq
        else
            dq = dlnq
            sbig.v[jj,j] = dq
        end
        phisalpha!(sbig, hbig, abig, pair)

        for i = 1:n
            for k = 1:3
                jac_step_num[(i - 1) * 7 +  k,(j - 1) * 7 + 3 + jj] = (sbig.x[k,i] - xsave[k,i]) / dq
                jac_step_num[(i - 1) * 7 + 3 + k,(j - 1) * 7 + 3 + jj] = (sbig.v[k,i] - vsave[k,i]) / dq
            end
        end
    end
    # Now vary mass of planet:
    sbig = deepcopy(State(ic_big))
    sbig.x .= big.(x0); sbig.v .= big.(v0); sbig.m .= big.(m0)
    dq = sbig.m[j] * dlnq
    sbig.m[j] += dq
    phisalpha!(sbig, hbig, abig, pair)
    for i = 1:n
        for k = 1:3
            jac_step_num[(i - 1) * 7 +  k,j * 7] = (sbig.x[k,i] - xsave[k,i]) / dq
            jac_step_num[(i - 1) * 7 + 3 + k,j * 7] = (sbig.v[k,i] - vsave[k,i]) / dq
        end
        # Mass unchanged -> identity
        jac_step_num[7 * i,7 * i] = big(1.0)
    end
end

# Now, compare the results:
dqdt_num = convert(Array{Float64,1}, dqdt_num)
@test isapprox(d.jac_phi, jac_step_num;norm=maxabs)
@test isapprox(d.dqdt_phi, dqdt_num;norm=maxabs)
end