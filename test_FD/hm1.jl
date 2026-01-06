using Pkg
Pkg.activate(joinpath(@__DIR__))

import Plots
import LinearAlgebra
import DiffEqBase
import OrdinaryDiffEqBDF as BDF
import SciMLBase, SciMLOperators

a, b, n = 0, 1, 100                   # zmin, zmax, number of cells
n̂_min, n̂_max = -1, 1                  # Outward facing unit vectors
α = 1.0;                              # thermal diffusivity
β, γ = 5.0, 1.0;                      # advection coefficients
Δt = 0.0001;                            # timestep size
N_t = 10000;                            # number of timesteps to take
αₗ = 1.0;                             # Dirichlet BC coefficient
βₗ = 0.0;                             # Dirichlet BC coefficient
ϕ_bottom = 1;                         # Dirichlet BC coefficient
αᵣ = 1.0;                             # Robin BC coefficient
βᵣ = 1.0;                             # Robin BC coefficient
ϕ_top = 1;                            # Robin BC coefficient
FT = Float64;                         # float type
Δz = FT(b - a) / FT(n)
Δz² = Δz^2;
∇_op = [-1 / Δz, 1 / Δz];             # interior gradient operator
∇²_op = [1 / Δz², -2 / Δz², 1 / Δz²]; # interior Laplacian operator
zf = range(a, b, length=n + 1);       # coordinates on cell faces

# Initialize interior and boundary stencils:
∇ = LinearAlgebra.Tridiagonal(ones(FT, n) .* ∇_op[1],
    ones(FT, n + 1) .* ∇_op[2],
    zeros(FT, n));
∇² = LinearAlgebra.Tridiagonal(ones(FT, n) .* ∇²_op[1],
    ones(FT, n + 1) .* ∇²_op[2],
    ones(FT, n) .* ∇²_op[3]);


AT_b = zeros(FT, n + 1);

AT_b[n+1] = α * 2 * ϕ_top / (βᵣ * Δz);

T = zeros(FT, n + 1);
T .= [z * (1-z) for z in zf]; # initial condition
T[1] = ϕ_bottom; # set bottom BC

function rhs1!(dT, T, params, t)
    n = params.n
    i = 2:n # interior domain
    
    T[n+1] = (ϕ_top * Δz + βᵣ * T[n]) / (βᵣ + αᵣ * Δz); # set top BC
    dT[1] = 0.0 # BC at bottom
    dT[i] .= α .* (∇²*T)[i] .- T[i] .- (β .* T[i] .+ γ) .* (∇*T)[i]
    dT[n+1] = 0.0 # BC at top
    return dT
end;

function rhs2!(dT, T, params, t)
    n = params.n
    i = 2:n # interior domain
    dT[i] .= AT_b[i]
    return dT
end;

params = (; n)

tspan = (FT(0), N_t * FT(Δt))

prob = SciMLBase.SplitODEProblem(rhs1!,
    rhs2!,
    T,
    tspan,
    params)
alg = BDF.IMEXEuler()
println("Solving...")
sol = SciMLBase.solve(prob,
    alg,
    dt=Δt,
    saveat=range(FT(0), N_t * FT(Δt), length=N_t÷10),
    progress=true,
    progress_message=(dt, u, p, t) -> t);
println("Solved.")
T_end = sol.u[end]

p1 = Plots.plot(zf, sol.u[1], label="", markershape=:diamond)
for T in sol.u[2:end-1]
    Plots.plot!(p1, zf, T, label="", alpha=0.3)
end
Plots.plot!(p1, zf, sol.u[end], label="", markershape=:diamond)

Plots.plot(p1)

Plots.savefig(p1, joinpath(@__DIR__, "burger2D_solution.pdf"))