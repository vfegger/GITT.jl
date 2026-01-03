using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
# Solve a 1D Burger Equation
using LinearAlgebra
using DifferentialEquations
using QuadGK
using Roots
using Plots
using ProgressMeter
using JLD2

N_basis = ARGS[1] !== nothing ? parse(Int, ARGS[1]) : error("Please provide number of basis functions as first argument.")

# Define the Transformed 1D Burger equation
L = N_basis + 5
A = Array{Float64}(undef, L, L, L)
B = Array{Float64}(undef, L, L)
C = Array{Float64}(undef, L)
Θ₀ = Array{Float64}(undef, L)

f = x -> tan(x) + x
a = 1.001 * π / 2
b = 0.999 * π / 2
eigenvals = [find_zero(f, (a + π * (i - 1), b + π * i), Bisection()) for i in 1:L]
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

filter(t, x) = 1

@showprogress desc = "Computing A..." Threads.@threads for i in 1:L
    for j in 1:L
        for k in 1:L
            A[i, j, k] = 5 * (ψ_norm(i, 1) * ψ_norm(j, 1) * ψ_norm(k, 1) - quadgk(x -> Dψ_norm(i, x) * ψ_norm(j, x) * ψ_norm(k, x), 0, 1; rtol=1e-3, atol=1e-8)[1])
        end
    end
end
@showprogress desc = "Computing B..." Threads.@threads for i in 1:L
    for j in 1:L
        B[i, j] = (α(i)^2 + 1) * (i == j ? 1 : 0) + 11 * (ψ_norm(i, 1) * ψ_norm(j, 1) - quadgk(x -> Dψ_norm(i, x) * ψ_norm(j, x), 0, 1; rtol=1e-3, atol=1e-8)[1])
    end
end
@showprogress desc = "Computing C..." Threads.@threads for i in 1:L
    C[i] = quadgk(x -> ψ_norm(i, x) + 6 * Dψ_norm(i, x), 0, 1; rtol=1e-3, atol=1e-8)[1] - 6 * ψ_norm(i, 1)
end
@showprogress desc = "Computing Θ₀..." Threads.@threads for i in 1:L
    Θ₀[i] = quadgk(x -> (x * (1 - x) - 1) * ψ_norm(i, x), 0, 1; rtol=1e-3, atol=1e-8)[1]
end

function mul2!(z, A, x, y; α=1.0, β=0.0)
    L = size(z, 1)
    Threads.@threads for i in 1:L
        aux = 0.0
        for j in 1:L
            for k in 1:L
                @inbounds aux += A[i, j, k] * x[j] * y[k]
            end
        end
        @inbounds z[i] = α * aux + β * z[i]
    end
end

function mul1!(z, B, x; α=1.0, β=0.0)
    L = size(z, 1)
    Threads.@threads for i in 1:L
        aux = 0.0
        for j in 1:L
            @inbounds aux += B[i, j] * x[j]
        end
        @inbounds z[i] = α * aux + β * z[i]
    end
end

function burger!(dΘ, Θ, p, t)
    fill!(dΘ, 0.0)
    mul2!(dΘ, A, Θ, Θ; α=-1.0)
    mul1!(dΘ, B, Θ; α=-1.0)
    dΘ .-= C
end

tspan = (0.0, 1.0)
prob = ODEProblem(burger!, Θ₀, tspan)

function ProgressCallback!(integrator)
    print("\rTime: $(round(integrator.t, digits=4)) / $(tspan[2])         ")
    u_modified!(integrator, false)
    return nothing
end
condition1(u, t, integrator) = true
cb = DiscreteCallback(condition1, ProgressCallback!)
sol = solve(prob, RadauIIA3(), reltol=1e-8, abstol=1e-8, callback=cb)
print("\nSolution completed.\n")

results_path = joinpath(@__DIR__, "..", "results", "hm1", "N_$N_basis")
mkpath(results_path)
save_object(joinpath(results_path, "burger2D.jld2"), sol)

ts = range(tspan[1], stop=tspan[2], length=200)
xs = range(0, stop=1, length=100)
θ = [[abs(sol(t)[i]) for t in ts] for i in 1:L]
us_ref = [sum(sol(t)[i] * ψ_norm(i, x) for i in 1:L) + filter(t, x) for x in xs, t in ts]
us = [sum(sol(t)[i] * ψ_norm(i, x) for i in 1:N_basis) + filter(t, x) for x in xs, t in ts]
error_matrix = abs.(us_ref - us)
u_slices = us[:, 1:10:end]

plt = plot(ts, θ[1:min(10, N_basis)], yscale=:log10, xlabel="t", ylabel="Θᵢ(t)", title="Coefficient Θᵢ over time")
savefig(plt, joinpath(results_path, "Coefficients.pdf"))

plt_error = heatmap(ts, xs, error_matrix, xlabel="t", ylabel="x", title="Absolute Error |u_ref - u|", colorbar_title="Error")
savefig(plt_error, joinpath(results_path, "Absolute_Error.pdf"))

plt_slices = plot(xs, u_slices, xlabel="x", ylabel="u(t,x)", title="1D Burger Equation Solution Slices")
savefig(plt_slices, joinpath(results_path, "burger_equation_solution_slices.pdf"))

hplt = heatmap(ts, xs, us, xlabel="t", ylabel="x", title="1D Burger Equation Solution", colorbar_title="u(t,x)")
savefig(hplt, joinpath(results_path, "burger_equation_solution.pdf"))