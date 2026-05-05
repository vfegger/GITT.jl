using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using DifferentialEquations
using Plots
using GaussQuadrature

# Equation form:
# ρ * c * ∂u/∂t = ∇·(k * ∇u) + g(t, x, u)

# DifferentialEquations.jl Form:
# du/dt = (1 / (ρ * c)) * ∇·(k * ∇u) + g(t, x, u) / (ρ * c)

# Filtered Equation:
# θ = u - f(x)
# ρ * c * ∂θ/∂t = ∇·(k * ∇θ) + g(t, x, u) - ∇·(k * ∇f(x))

# Pre-transformed Equation:
# ρᵣ * cᵣ * ∂θ/∂t = ∇·(kᵣ * ∇θ) + g(t, x, u) - ∇·(k * ∇f(x)) - (ρ * c - ρᵣ * cᵣ) * ∂θ/∂t + ∇·((k - kᵣ) * ∇θ)
# ρᵣ = cst, cᵣ = cst, kᵣ = cst
# gᵣ(t, x, θ) = g(t, x, θ + f(x)) - ∇·(k * ∇f(x)) - (ρ * c - ρᵣ * cᵣ) * ∂θ/∂t + ∇·((k - kᵣ) * ∇θ)

# Transformed Equation:
# ρᵣ * cᵣ * ∂θ/∂t = ∇·(kᵣ * ∇θ) + gᵣ(t, x, u)
# θ = Σ ϕ(t) * ψ(x)
# ϕ(t) = ∫ θ * ψ(x) dx
# ψ(x) -> ∇·(kᵣ * ∇ψ(x)) = λ * ψ(x)
# ∫ ρᵣ * cᵣ * ∂θ/∂t * ψ(x) dx = ∫ ∇·(kᵣ * ∇θ) * ψ(x) dx + ∫ gᵣ(t, x, u) * ψ(x) dx
# ρᵣ * cᵣ * dϕ/dt = - λ * ϕ + ∫ gᵣ(t, x, θ) * ψ(x) dx

# Dimensions:
x₀ = 0.0
x₁ = 1.0
N = 20
Lₓ = 10

X_unscaled, W_unscaled = legendre(Lₓ)
X = X_unscaled .* ((x₁ - x₀) / 2) .+ (x₀ + x₁) / 2
W = (W_unscaled .* ((x₁ - x₀) / 2))'

display("X: $X, W: $W")

ρ_paraffin_solidus = 874.0
c_paraffin_solidus = 2400.0
k_paraffin_solidus = 0.24
T_paraffin_solidus = 321.0
ρ_paraffin_liquidus = 874.0
c_paraffin_liquidus = 2400.0
k_paraffin_liquidus = 0.24
T_paraffin_liquidus = 335.0
L_paraffin = 195.0

ρ_SGN = 1815.0
c_SGN = 1533.0
k_SGN = 0.60

ρ_wall = 7833.0
c_wall = 465.0
k_wall = 54.0

h_ext = 1000.0
T_ext = 277.0
T_in = 300.0

sigmoid(x, x₀, x₁) = 1 / (1 + exp(-(x - (x₀ + x₁) / 2) / ((x₁ - x₀) / 2)))

k_paraffin(u) = sigmoid(u, T_paraffin_solidus, T_paraffin_liquidus) * k_paraffin_liquidus + (1 - sigmoid(u, T_paraffin_solidus, T_paraffin_liquidus)) * k_paraffin_solidus
∇k_paraffin(x, u) = (sigmoid(u, T_paraffin_solidus, T_paraffin_liquidus) * (1 - sigmoid(u, T_paraffin_solidus, T_paraffin_liquidus)) * k_paraffin_liquidus - sigmoid(u, T_paraffin_solidus, T_paraffin_liquidus) * (1 - sigmoid(u, T_paraffin_solidus, T_paraffin_liquidus)) * k_paraffin_solidus) * (1 / ((T_paraffin_liquidus - T_paraffin_solidus) / 2))

ρ(x,u) = ρ_paraffin_solidus
c(x,u) = c_paraffin_solidus
k(x,u) = k_paraffin(u)
∇k(x,u) = ∇k_paraffin(x, u)
∇k₁(x,u) = 0.0
∇k₂(x,u) = 0.0
g(t,x,T) = 0.0

f(x) = T_ext
∇f(x) = 0.0
∇f₁(x) = 0.0
∇f₂(x) = 0.0
∇²f(x) = 0.0
u₀(x) = T_in

unity(x) = 1.0
Ωₓ = W * unity.(X)
display("Ωₓ: $Ωₓ")

ρᵣ = W * ρ.(X, 0.0) / Ωₓ
cᵣ = W * c.(X, 0.0) / Ωₓ
kᵣ = W * k.(X, 0.0) / Ωₓ 

ψ = [x -> sin(i * π * (x - x₀) / (x₁ - x₀)) for i in 1:N]
∇ψ = [x -> i * π / (x₁ - x₀) * cos(i * π * (x - x₀) / (x₁ - x₀)) for i in 1:N]
∇²ψ = [x -> -(i * π / (x₁ - x₀))^2 * sin(i * π * (x - x₀) / (x₁ - x₀)) for i in 1:N]
λ = [(W * map(x -> kᵣ * (i * π / (x₁ - x₀))^2 * ψ[i](x)^2, X)) / (W * map(x -> ψ[i](x)^2, X)) for i in 1:N]

display("ρᵣ: $ρᵣ, cᵣ: $cᵣ, kᵣ: $kᵣ, λ: $λ")

# How to deal with ∂θ/∂t in gᵣ?
# Iterative approach
# How to deal with ∇θ in gᵣ?
# θ = Σ ϕ(t) * ψ(x) -> ∇θ = Σ ϕ(t) * ∇ψ(x)

# To handle integrations, we need to define x points and weights for numerical quadrature. For simplicity, we can use a fixed number of points and weights, such as those from Gaussian quadrature.

Ψ = [ψ[i](x) for i in 1:N, x in X]
∇Ψ = [∇ψ[i](x) for i in 1:N, x in X]
∇²Ψ = [∇²ψ[i](x) for i in 1:N, x in X]
F = f.(X)
∇F = ∇f.(X)
∇F₁ = ∇f₁.(X)
∇F₂ = ∇f₂.(X)
∇²F = ∇²f.(X)


function gᵣ(t, X, Θ, ∇Θ, ∇²Θ, F, ∇F, ∇²F)
    acc = zeros(length(X))
    # acc .+= g.(t, X, θ .+ F)
    # acc .+= - ∇k.(X, Θ) .* ∇F .- k.(X, Θ) .* ∇²F
    # acc .+= - (ρ(t, x, θ) * c(t, x, θ) - ρᵣ * cᵣ) * ∂θ/∂t 
    acc .+= + ∇k.(X, Θ) .* ∇Θ .+ (k.(X, Θ) .- kᵣ) .* ∇²Θ
    return acc
end

function diff_eq!(dϕ, ϕ, p, t)
    Θ = Ψ' * ϕ 
    ∇Θ = ∇Ψ' * ϕ
    ∇²Θ = ∇²Ψ' * ϕ
    source = W * gᵣ(t, X, Θ, ∇Θ, ∇²Θ, F, ∇F, ∇²F)
    dϕ .= (1 / (ρᵣ * cᵣ)) .* ((-λ .* ϕ) .+ source)
end

dϕ = zeros(N)
ϕ₀ = [W * map(x -> (u₀(x) - f(x)) * ψ[i](x), X) for i in 1:N]
p = nothing
tspan = (0.0, 1.0e6)
prob = ODEProblem(diff_eq!, ϕ₀, tspan, p)

start_time = time()
sol = solve(prob, QNDF(autodiff=AutoFiniteDiff()))
end_time = time()
elapsed_time = end_time - start_time
display("Elapsed time: $elapsed_time seconds")

# Plot results
x_axis = range(tspan[1], tspan[2], length=length(sol))
y_axis = Array(sol)'

plot(sol, title="Transformed Equation Solution", xlabel="Time", ylabel="Θ(t)")
savefig("transformed_solution.pdf")

# Plot Log of the absolute value of the solution
plot(x_axis, abs.(y_axis), title="Log of Transformed Equation Solution", xlabel="Time", ylabel="log(Θ(t))", yscale=:log10)
savefig("log_transformed_solution.pdf")

exit()