using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
# Solve a 1D Burger Equation
using LinearAlgebra
using DifferentialEquations
using QuadGK
using Roots
using Plots

# Define the Transformed 1D Burger equation
L = 80
A = Array{Float64}(undef, L, L, L)
B = Array{Float64}(undef, L, L)
C = Array{Float64}(undef, L, L)
G = Array{Float64}(undef, L)
Θ₀ = Array{Float64}(undef, L)

f = x -> tan(x) + x
a = 1.001*π/2
b = 0.999*π/2
eigenvals = [find_zero(f, (a + π * (i-1),b + π * i), Bisection()) for i in 1:L]
display(eigenvals)
α(i) = eigenvals[i]
ψ(i, x) = sin(α(i) * x)
Dψ(i, x) = cos(α(i) * x) * α(i)
Norms = [quadgk(x -> ψ(i, x)^2, 0, 1)[1] for i in 1:L]
ψ_norm(i, x) = ψ(i, x) / sqrt(Norms[i])
Dψ_norm(i, x) = Dψ(i, x) / sqrt(Norms[i])
xs = range(0, 1, length=200)
eigs = [ψ_norm(i, x) for x in xs, i in 1:10]
plt_basis = plot(xs, eigs, xlabel="x", ylabel="ψ(x)", title="Eigenfunctions ψ(x)")
savefig(plt_basis, "eigenfunctions.pdf")

for i in 1:L
    for j in 1:L
        for k in 1:L
            A[i, j, k] = 5 * (ψ_norm(i, 1) * ψ_norm(j, 1) * ψ_norm(k, 1) - quadgk(x -> Dψ_norm(i, x) * ψ_norm(j, x) * ψ_norm(k, x), 0, 1)[1])
        end
        B[i, j] = (α(i)^2 + 1) * (i == j ? 1 : 0) + 11 * (ψ_norm(i, 1) * ψ_norm(j, 1) - quadgk(x -> Dψ_norm(i, x) * ψ_norm(j, x), 0, 1)[1])
        C[i, j] = (α(i)^2 + 1) * (i == j ? 1 : 0)
    end
    G[i] = quadgk(x -> ψ_norm(i, x) + 6 * Dψ_norm(i, x), 0, 1)[1] - 6 * ψ_norm(i, 1)
    Θ₀[i] = quadgk(x -> (x * (1 - x) - 1) * ψ_norm(i, x), 0, 1)[1]
end
A = -A
B = -B
C = -C

function mul2!(z, A, x, y)
    L = size(A, 1)
    for i in 1:L
        for j in 1:L
            for k in 1:L
                z[i] += A[i, j, k] * x[j] * y[k]
            end
        end
    end
end

function mul1!(z, B, x)
    L = size(B, 1)
    for i in 1:L
        for j in 1:L
            z[i] += B[i, j] * x[j]
        end
    end
end

function burger!(dΘ, Θ, p, t)
    fill!(dΘ, 0.0)
    mul2!(dΘ, A, Θ, Θ)
    mul1!(dΘ, B, Θ)
    dΘ .+= G
end
function diffusion!(dΘ, Θ, p, t)
    fill!(dΘ, 0.0)
    mul1!(dΘ, C, Θ)
end
filter(t, x) = 1

tspan = (0.0, 1.0)
prob = ODEProblem(burger!, Θ₀, tspan)
sol = solve(prob, RadauIIA3(), reltol=1e-8, abstol=1e-8)
ts = range(tspan[1], stop=tspan[2], length=200)
xs = range(0, stop=1, length=100)
T = [[abs(sol(t)[i]) for t in ts] for i in 1:L]
us = [sum(sol(t)[i] * ψ_norm(i, x) for i in 1:L) + filter(t, x) for x in xs, t in ts]
u_slices = us[:, 1:10:end]

plt = plot(ts, T[1:10], yscale=:log10, xlabel="t", ylabel="Θᵢ(t)", title="Coefficient Θᵢ over time")
savefig(plt, "First coefficient.pdf")

plt_slices = plot(xs, u_slices, xlabel="x", ylabel="u(t,x)", title="1D Burger Equation Solution Slices")
savefig(plt_slices, "burger_equation_solution_slices.pdf")

hplt = heatmap(ts, xs, us, xlabel="t", ylabel="x", title="1D Burger Equation Solution", colorbar_title="u(t,x)")
savefig(hplt, "burger_equation_solution.pdf")