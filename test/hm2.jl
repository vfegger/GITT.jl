using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
# Solve a 1D Burger Equation
using LinearAlgebra
using DifferentialEquations
using QuadGK
using Roots
using Plots
using HCubature

# Problem:
# DₜT + Dₓ(u T)= Dₓ(k DₓT) + Dᵧ(k DᵧT)
# where u(T) = 5 * T + 1
# k(y) = ln(10 + y)
# with BCs:
# T(t = 0, x, y) = x * (1 - x) * y * (1 - y)
# T(t, x = 0, y) = 1
# T(t, x = 1, y) + DₓT(t, x = 1, y) = 1
# DᵧT(t, x, y = 0) = 0
# DᵧT(t, x, y = 1) = exp(-x)

# First, we need to find an homogeneous problem through filters:
# In x:
# filterₓ(t, x, y) = 1
# New BCs:
# T(t, x = 0, y) = 0
# T(t, x = 1, y) + DₓT(t, x = 1, y) = 0
# DᵧT(t, x, y = 0) = 0
# DᵧT(t, x, y = 1) = exp(-x)
# In y:
# filterᵧ(t, x, y) = Σᵢ exp(-x) * y^2 / 2 * ψᵢ(x) where ψᵢ(x) is the solution of: Dₓ(kₑ Dₓψᵢ) = -λᵢ ψ with BCs: ψᵢ(x = 0) = 0 and ψᵢ + Dₓψᵢ(x = 1) = 0
# New BCs:
# T(t, x = 0, y) = 0
# T(t, x = 1, y) + DₓT(t, x = 1, y) = 0
# DᵧT(t, x, y = 0) = 0
# DᵧT(t, x, y = 1) = 0
# Finally, the new PDE to solve is:
# DₜT + Dₓ(u(T + filterₓ + filterᵧ) * (T + filterₓ + filterᵧ)) = Dₓ(k Dₓ(T + filterₓ + filterᵧ)) + Dᵧ(k Dᵧ(T + filterₓ + filterᵧ))
# with homogeneous BCs.
# Substituting u and k and expanding the derivatives, we get:
# DₜT + 5 * Dₓ(T²) + 5 * Dₓ(T * (filterₓ + filterᵧ)) + Dₓ(filterₓ²) + Dₓ(filterᵧ²) + 2 * Dₓ(filterₓ * filterᵧ) + Dₓ(filterₓ) + Dₓ(filterᵧ) = Dₓ(ln(10 + y) * DₓT) + Dᵧ(ln(10 + y) * DᵧT) + Dₓ(ln(10 + y) * Dₓ(filterₓ + filterᵧ)) + Dᵧ(ln(10 + y) * Dᵧ(filterₓ + filterᵧ))
# Rearranging terms, we have:
# DₜT + 5 * Dₓ(T²) + 5 * Dₓ(T * (filterₓ + filterᵧ)) = Dₓ(ln(10 + y) * DₓT) + Dᵧ(ln(10 + y) * DᵧT) + S(t, x, y)
# where S(t, x, y) = Dₓ(ln(10 + y) * Dₓ(filterₓ + filterᵧ)) + Dᵧ(ln(10 + y) * Dᵧ(filterₓ + filterᵧ)) - Dₓ(filterₓ²) - Dₓ(filterᵧ²) - 2 * Dₓ(filterₓ * filterᵧ) - Dₓ(filterₓ) - Dₓ(filterᵧ)
# Now, we can transform this problem using the eigenfunctions ψᵢ and ψⱼ found previously.
# T = Σᵢ Σⱼ Θᵢⱼ(t) * ψᵢ(x) * ψⱼ(y)
# DₜT = Σᵢ Σⱼ DₜΘᵢⱼ(t) * ψᵢ(x) * ψⱼ(y)
# Dₓ(T²) = Aₘₙᵢⱼₖₗ * Θᵢⱼ(t) * Θₖₗ(t) where Aₘₙᵢⱼₖₗ = ∫ Dₓ(ψᵢ(x) * ψⱼ(y) * ψ_k(x) * ψ_l(y)) * ψ_m(x) * ψ_n(y) dx dy = ∫ Dₓ(ψᵢ(x) * ψ_k(x)) * ψ_m(x) dx * ∫ ψⱼ(y) * ψ_l(y) * ψ_n(y) dy
# Dₓ(T * (filterₓ + filterᵧ)) = (Σᵢ Σⱼ Θᵢⱼ(t) * Dₓψᵢ(x) * ψⱼ(y)) * (filterₓ + filterᵧ) + (Σᵢ Σⱼ Θᵢⱼ(t) * ψᵢ(x) * ψⱼ(y)) * Dₓ(filterₓ + filterᵧ)
# Dₓ(ln(10 + y) * DₓT) = ln(10 + y) * Σᵢ Σⱼ Θᵢⱼ(t) * Dₓ²ψᵢ(x) * ψⱼ(y) = - ln(10 + y) * Σᵢ Σⱼ λᵢ Θᵢⱼ(t) * ψᵢ(x) * ψⱼ(y)
# Dᵧ(ln(10 + y) * DᵧT) = Dᵧ(ln(10 + y)) * Σᵢ Σⱼ Θᵢⱼ(t) * Dᵧψⱼ(y) * ψᵢ(x) + ln(10 + y) * Σᵢ Σⱼ Θᵢⱼ(t) * Dᵧ²ψⱼ(y) * ψᵢ(x) = Dᵧ(ln(10 + y)) * Σᵢ Σⱼ Θᵢⱼ(t) * Dᵧψⱼ(y) * ψᵢ(x) - ln(10 + y) * Σᵢ Σⱼ λⱼ Θᵢⱼ(t) * ψⱼ(y) * ψᵢ(x)
# S(t, x, y) = To be computed numerically
# Plugging these expressions into the transformed PDE and projecting onto the eigenfunctions will yield a system of ODEs for the coefficients Θᵢⱼ(t).

# Define the Transformed 2D Burger equation
Lᵢ = 2
Lⱼ = 2
A = Array{Float64}(undef, Lᵢ, Lᵢ, Lᵢ, Lⱼ, Lⱼ, Lⱼ)
B = Array{Float64}(undef, Lᵢ, Lᵢ, Lⱼ, Lⱼ)
C = Array{Float64}(undef, Lᵢ, Lⱼ)
Θ₀ = Array{Float64}(undef, Lᵢ, Lⱼ)

fᵢ = λ -> tan(λ) + λ
fⱼ = λ -> sin(λ)
Iᵢ = (1.001 * π / 2, 0.999 * π / 2)
Iⱼ = (-1.000 * π / 2, -1.000 * π / 2)
λᵢ = [find_zero(fᵢ, (Iᵢ[1] + π * (i - 1), Iᵢ[2] + π * i), Bisection()) for i in 1:Lᵢ]
λⱼ = [find_zero(fⱼ, (Iⱼ[1] + π * (j - 1), Iⱼ[2] + π * j), Bisection()) for j in 1:Lⱼ]
for i in 1:Lᵢ
    println("λᵢ[$i] = $(λᵢ[i]), λⱼ[$i] = $(λⱼ[i])")
end
αᵢ(i) = λᵢ[i]
αⱼ(j) = λⱼ[j]

ψᵢ_NN(i, x) = sin(αᵢ(i) * x)
ψⱼ_NN(j, x) = sin(αⱼ(j) * x)
Dψᵢ_NN(i, x) = cos(αᵢ(i) * x) * αᵢ(i)
Dψⱼ_NN(j, x) = cos(αⱼ(j) * x) * αⱼ(j)
D₂ψᵢ_NN(i, x) = -αᵢ(i)^2 * sin(αᵢ(i) * x)
D₂ψⱼ_NN(j, x) = -αⱼ(j)^2 * sin(αⱼ(j) * x)

Nᵢ = [quadgk(x -> ψᵢ_NN(i, x)^2, 0, 1)[1] for i in 1:Lᵢ]
Nⱼ = [quadgk(x -> ψⱼ_NN(j, x)^2, 0, 1)[1] for j in 1:Lⱼ]
ψᵢ(i, x) = ψᵢ_NN(i, x) / sqrt(Nᵢ[i])
ψⱼ(j, x) = ψⱼ_NN(j, x) / sqrt(Nⱼ[j])
Dψᵢ(i, x) = cos(αᵢ(i) * x) * αᵢ(i) / sqrt(Nᵢ[i])
Dψⱼ(j, x) = cos(αⱼ(j) * x) * αⱼ(j) / sqrt(Nⱼ[j])
D₂ψᵢ(i, x) = -αᵢ(i)^2 * sin(αᵢ(i) * x) / sqrt(Nᵢ[i])
D₂ψⱼ(j, x) = -αⱼ(j)^2 * sin(αⱼ(j) * x) / sqrt(Nⱼ[j])

k(y) = log(10 + y)
Dk(y) = 1 / (10 + y)
filterₓ(x, y) = 1
filterᵧ(x, y) = sum(exp(-x) * y^2 / 2 * ψᵢ(i, x) for i in 1:Lᵢ)

display("Calculating A, B, C, Θ₀ matrices...")
for (i₀, i₁, i₂, j₀, j₁, j₂) in Iterators.product(1:Lᵢ, 1:Lᵢ, 1:Lᵢ, 1:Lⱼ, 1:Lⱼ, 1:Lⱼ)
    A[i₀, i₁, i₂, j₀, j₁, j₂] = hcubature(
        (x) -> 10 * ψᵢ(i₀, x[1]) * ψᵢ(i₁, x[1]) * Dψᵢ(i₂, x[1]) * ψⱼ(j₀, x[2]) * ψⱼ(j₁, x[2]) * ψⱼ(j₂, x[2]), (0, 0), (1, 1)
    )[1]
end
display("A matrix computed.")
for (i₀, i₁, j₀, j₁) in Iterators.product(1:Lᵢ, 1:Lᵢ, 1:Lⱼ, 1:Lⱼ)
    aux = 0.0
    for k in 1:Lᵢ
        aux += hcubature(
            (x) -> 5 * exp(-x[1]) * x[2]^2 * ψᵢ(k, x[1]) * ψᵢ(i₀, x[1]) * Dψᵢ(i₁, x[1]) * ψⱼ(j₀, x[2]) * ψⱼ(j₁, x[2]), (0, 0), (1, 1)
        )[1]
    end
    aux += hcubature(
        (x) -> 11 * ψᵢ(i₀, x[1]) * Dψᵢ(i₁, x[1]) * ψⱼ(j₀, x[2]) * ψⱼ(j₁, x[2]), (0, 0), (1, 1)
    )[1]
    for k in 1:Lᵢ
        aux += hcubature(
            (x) -> 5 * exp(-x[1]) * x[2]^2 * (Dψᵢ(k, x[1]) - ψᵢ(k, x[1])) * ψᵢ(i₀, x[1]) * ψᵢ(i₁, x[1]) * ψⱼ(j₀, x[2]) * ψⱼ(j₁, x[2]), (0, 0), (1, 1)
        )[1]
    end
    aux += hcubature(
        (x) -> k(x[2]) * (D₂ψᵢ(i₁, x[1]) * ψⱼ(j₁, x[2]) + ψᵢ(i₁, x[1]) * D₂ψⱼ(j₁, x[2])) * ψᵢ(i₀, x[1]) * ψⱼ(j₀, x[2]), (0, 0), (1, 1)
    )[1]
    aux += hcubature(
        (x) -> Dk(x[2]) * ψᵢ(i₀, x[1]) * ψᵢ(i₁, x[1]) * ψⱼ(j₀, x[2]) * Dψⱼ(j₁, x[2]), (0, 0), (1, 1)
    )[1]
    B[i₀, i₁, j₀, j₁] = aux
end
display("B matrix computed.")
for (i₀, j₀) in Iterators.product(1:Lᵢ, 1:Lⱼ)
    aux = 0.0
    for k₁ in 1:Lᵢ
        for k₂ in 1:Lᵢ
            aux += hcubature(
                (x) -> (5 / 2) * exp(-2 * x[1]) * ψᵢ(k₁, x[1]) * ψᵢ(k₂, x[1]) * ψᵢ(i₀, x[1]) * ψⱼ(j₀, x[2]), (0, 0), (1, 1)
            )[1]
        end
    end
    for k₁ in 1:Lᵢ
        aux += hcubature(
            (x) -> (11 / 2) * x[2]^2 * exp(-x[1]) * ψᵢ(k₁, x[1]) * ψᵢ(i₀, x[1]) * ψⱼ(j₀, x[2]), (0, 0), (1, 1)
        )[1]
    end
    for k₁ in 1:Lᵢ
        aux += hcubature(
            (x) -> (1 / 2) * k(x[2]) * x[2]^2 * exp(-x[1]) * (D₂ψᵢ(k₁, x[1]) - 2 * Dψᵢ(k₁, x[1]) + ψᵢ(k₁, x[1])) * ψᵢ(i₀, x[1]) * ψⱼ(j₀, x[2]), (0, 0), (1, 1)
        )[1]
    end
    for k₁ in 1:Lᵢ
        aux += hcubature(
            (x) -> (k(x[2]) + Dk(x[2]) * x[2]) * exp(-x[1]) * ψᵢ(k₁, x[1]) * ψᵢ(i₀, x[1]) * ψⱼ(j₀, x[2]), (0, 0), (1, 1)
        )[1]
    end
    C[i₀, j₀] = aux
end
display("C matrix computed.")
for (i₀, j₀) in Iterators.product(1:Lᵢ, 1:Lⱼ)
    aux = 0.0
    aux += hcubature(
        (x) -> x[1] * (1 - x[1]) * x[2] * (1 - x[2]) * ψᵢ(i₀, x[1]) * ψⱼ(j₀, x[2]), (0, 0), (1, 1)
    )[1]
    aux += hcubature(
        (x) -> - 1 * ψᵢ(i₀, x[1]) * ψⱼ(j₀, x[2]), (0, 0), (1, 1)
    )[1]
    for k₁ in 1:Lᵢ
        aux += hcubature(
            (x) -> exp(-x[1]) * x[2]^2 / 2 * ψᵢ(k₁, x[1]) * ψᵢ(i₀, x[1]) * ψⱼ(j₀, x[2]), (0, 0), (1, 1)
        )[1]
    end
    Θ₀[i₀, j₀] = aux
end
display("Initial conditions computed.")
display(A)
display(B)
display(C)
display(Θ₀)

function muln!(z, A, xs...; α=1.0, β=0.0)
    L = size(A, 1)
    N = length(xs)
    M = length.(xs)
    for i in 1:L
        aux = 0.0
        for idxs in Iterators.product((1:m for m in M)...)
            temp = α * A[i, idxs...]
            for j in 1:N
                temp *= xs[j][idxs[j]]
            end
            aux += temp
        end
        z[i] = β * z[i] + aux
    end
end
function mul2D!(z, A, x, y; α=1.0, β=0.0)
    Lᵢ = size(z, 1)
    Lⱼ = size(z, 2)
    for (i₀, j₀) in Iterators.product(1:Lᵢ, 1:Lⱼ)
        aux = 0.0
        for (i₁, i₂, j₁, j₂) in Iterators.product(1:Lᵢ, 1:Lᵢ, 1:Lⱼ, 1:Lⱼ)
            aux += α * A[i₀, i₁, i₂, j₀, j₁, j₂] * x[i₁, j₁] * y[i₂, j₂]
        end
    end
    z[i₀, j₀] = β * z[i₀, j₀] + aux
end
function mul1!(z, B, x; α=1.0, β=0.0)
    Lᵢ = size(z, 1)
    Lⱼ = size(z, 2)
    for (i₀, j₀) in Iterators.product(1:Lᵢ, 1:Lⱼ)
        aux = 0.0
        for (i₁, j₁) in Iterators.product(1:Lᵢ, 1:Lⱼ)
            aux += α * B[i₀, i₁, j₀, j₁] * x[i₁, j₁]
        end
        z[i₀, j₀] = β * z[i₀, j₀] + aux
    end
end

function burger2D!(dΘ, Θ, p, t)
    fill!(dΘ, 0.0)
    mul2!(dΘ, A, Θ, Θ; α=-1.0, β=1.0)
    mul1!(dΘ, B, Θ; α=-1.0, β=1.0)
    dΘ .+= G
end
filter(t, x, y) = filterₓ(x, y) + filterᵧ(x, y)

tspan = (0.0, 1.0)
prob = ODEProblem(burger2D!, Θ₀, tspan)
sol = solve(prob, Rodas4P(), reltol=1e-8, abstol=1e-8)
ts = range(tspan[1], stop=tspan[2], length=200)
xs = range(0, stop=1, length=100)
ys = range(0, stop=1, length=100)
T = [[abs(sol(t)[i, j]) for t in ts] for i in 1:Lᵢ, j in 1:Lⱼ]
us = [sum(sol(t)[i, j] * ψᵢ(i, x) * ψⱼ(j, y) for i in 1:Lᵢ, j in 1:Lⱼ) + filter(t, x, y) for x in xs, y in ys, t in ts]

plt = plot(ts, T[1:10], yscale=:log10, xlabel="t", ylabel="Θ₁(t)", title="Coefficient Θ₁ over time")
savefig(plt, "First coefficient.pdf")
hplt = heatmap(ts, xs, us, xlabel="t", ylabel="x", title="1D Burger Equation Solution", colorbar_title="u(t,x)")
savefig(hplt, "burger_equation_solution.pdf")