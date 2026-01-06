using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
# Solve a 2D Burger Equation
using LinearAlgebra
using DifferentialEquations
using QuadGK
using Roots
using Plots
using HCubature
using ProgressMeter
using JLD2

status = :Symbolic
type = :Normal

N_basis = ARGS[1] !== nothing ? parse(Int, ARGS[1]) : error("Please provide number of basis functions as first argument.")

# Standard Form: Tₜ + ∇⋅(u * T) - ∇⋅(k ∇T) - S = 0
# Classic Integration from ψᵢⱼ(x, y) = ϕᵢ(x) * φⱼ(y)
# Integral:
#   ∫ Tₜ * ψᵢⱼ(x, y) + ∫ ∇⋅(u * T) * ψᵢⱼ(x, y) - ∫ ∇⋅(k ∇T) * ψᵢⱼ(x, y) - ∫ S * ψᵢⱼ(x, y) = 0 
#   ∫ Tₜ * ψᵢⱼ(x, y) + ∮ [(u * T * ψᵢⱼ) ⋅ n] - ∫ (u * T) ⋅ ∇ψᵢⱼ - ∮ [k ∇T ψᵢⱼ - k T ∇ψᵢⱼ] - ∫ T ∇⋅(k_e ∇ψᵢⱼ) - ∫ T ∇⋅((k-k_e) ∇ψᵢⱼ) - ∫ S * ψᵢⱼ(x, y) = 0

# T = θ + F
# F = 1 + Σ exp(-x) * y^2 / 2 * ψᵢ(x)

# Terms:
#   + ∫ θₜ * ψᵢⱼ
#   + ∮ [(u(θ) * θ * ψᵢⱼ) ⋅ n] 
#   - ∫ (u(θ) * θ) ⋅ ∇ψᵢⱼ 
#   + ∮ [(v * ψᵢⱼ) ⋅ n]
#   - ∫ v ⋅ ∇ψᵢⱼ
#   - ∮ [k ∇θ ψᵢⱼ - k θ ∇ψᵢⱼ]
#   - ∫ θ ∇⋅(k_e ∇ψᵢⱼ) 
#   - ∫ θ ∇⋅((k-k_e) ∇ψᵢⱼ) 
#   - ∫ ∇⋅(k ∇F) * ψᵢⱼ(x, y)
#   - ∫ S * ψᵢⱼ

# Define necessary functions:
# u(Θ) = [5 * θ + 10 * F + 1, 0]
# v = [5 * F^2 + F, 0]
# k(y) = log(10 + y)
# k_eq = 1
# S = 0

# θ = Σ Θᵢⱼ(t) * ψᵢⱼ(x, y)

Lᵢ = 200
Lⱼ = 200
L = (N_basis + 5 > Lᵢ * Lⱼ) ? (Lᵢ * Lⱼ) : N_basis + 5


fᵢ = λ -> tan(λ) + λ
fⱼ = λ -> sin(λ)
Iᵢ = (1.001 * π / 2, 0.999 * π / 2)
Iⱼ = (-1.000 * π / 2, -1.000 * π / 2)
λᵢ = [find_zero(fᵢ, (Iᵢ[1] + π * (i - 1), Iᵢ[2] + π * i), Bisection()) for i in 1:Lᵢ]
λⱼ = [find_zero(fⱼ, (Iⱼ[1] + π * (j - 1), Iⱼ[2] + π * j), Bisection()) for j in 1:Lⱼ]
energy = [(CartesianIndex(i, j), λᵢ[i]^2 + λⱼ[j]^2) for i in 1:Lᵢ for j in 1:Lⱼ]
sort!(energy, by=x -> x[2])
indexes = first.(energy)[1:L]

i_maximum = 1
j_maximum = 1
for i in 1:L
    i₀, j₀ = Tuple(indexes[i])
    global i_maximum = (λᵢ[i₀] > λᵢ[i_maximum]) ? i₀ : i_maximum
    global j_maximum = (λⱼ[j₀] > λⱼ[j_maximum]) ? j₀ : j_maximum
    println("λᵢ[$i₀] = $(λᵢ[i₀]), λⱼ[$j₀] = $(λⱼ[j₀])")
end
μᵢ(i) = λᵢ[i]
μⱼ(j) = λⱼ[j]

ψᵢ_NN(i, x) = sin(μᵢ(i) * x)
ψⱼ_NN(j, x) = cos(μⱼ(j) * x)
Dψᵢ_NN(i, x) = cos(μᵢ(i) * x) * μᵢ(i)
Dψⱼ_NN(j, x) = -sin(μⱼ(j) * x) * μⱼ(j)

Nᵢ = [quadgk(x -> ψᵢ_NN(i, x)^2, 0, 1)[1] for i in 1:Lᵢ]
Nⱼ = [quadgk(x -> ψⱼ_NN(j, x)^2, 0, 1)[1] for j in 1:Lⱼ]
if any(iszero, Nᵢ) || any(iszero, Nⱼ)
    error("Normalization factor is zero for some basis function.")
end
ψᵢ(i, x) = ψᵢ_NN(i, x) / sqrt(Nᵢ[i])
ψⱼ(j, x) = ψⱼ_NN(j, x) / sqrt(Nⱼ[j])
Dψᵢ(i, x) = Dψᵢ_NN(i, x) / sqrt(Nᵢ[i])
Dψⱼ(j, x) = Dψⱼ_NN(j, x) / sqrt(Nⱼ[j])
D₂ψᵢ(i, x) = -μᵢ(i)^2 * ψᵢ_NN(i, x) / sqrt(Nᵢ[i])
D₂ψⱼ(j, x) = -μⱼ(j)^2 * ψⱼ_NN(j, x) / sqrt(Nⱼ[j])

L_test = min(10, Lᵢ, Lⱼ)
plt_ψᵢ = plot(title="Basis Functions ψᵢ(x)", xlabel="x", ylabel="ψᵢ(x)")
for i in 1:L_test
    xs = range(0.0, stop=1.0, length=200)
    ys = [ψᵢ(i, x) for x in xs]
    plot!(plt_ψᵢ, xs, ys, label="i=$i")
end
savefig(plt_ψᵢ, "hm2_psi_i.pdf")
plt_ψⱼ = plot(title="Basis Functions ψⱼ(y)", xlabel="y", ylabel="ψⱼ(y)")
for j in 1:L_test
    ys = range(0.0, stop=1.0, length=200)
    zs = [ψⱼ(j, y) for y in ys]
    plot!(plt_ψⱼ, ys, zs, label="j=$j")
end
savefig(plt_ψⱼ, "hm2_psi_j.pdf")

X₀(x) = x * (1 - x)
Y₀(y) = y * (1 - y)
T₀(x, y) = X₀(x) * Y₀(y)
# α(x) * T(x, y) + k(x, y) * β(x) * ∇T ⋅ n = φ(x)
αᵣ(x, y) = 1.0
αₗ(x, y) = 1.0
αₙ(x, y) = 0.0
αₛ(x, y) = 0.0

βᵣ(x, y) = 1.0 / k(y)
βₗ(x, y) = 0.0
βₙ(x, y) = 1.0 / k(y)
βₛ(x, y) = 1.0 / k(y)

φᵣ(x, y) = 1.0
φₗ(x, y) = 1.0
φₙ(x, y) = exp(-x)
φₛ(x, y) = 0.0

u_1 = 5
u_0 = 1
k(y) = log(10 + y)
Dk(y) = 1 / (10 + y)
kₑ = 1
Dkₑ = 0
γ = [quadgk(x -> φₙ(x, 1) * ψᵢ(i, x), 0, 1)[1] for i in 1:i_maximum]
Cst = 1
G(x) = sum(γ[i] * ψᵢ(i, x) for i in 1:i_maximum)
DG(x) = sum(γ[i] * Dψᵢ(i, x) for i in 1:i_maximum)
D2G(x) = sum(γ[i] * D₂ψᵢ(i, x) for i in 1:i_maximum)
H(y) = y^2 / 2
DH(y) = y
D2H(y) = 1
F(x, y) = Cst + G(x) * H(y)

ϕᵣ(x, y) = 0.0
ϕₗ(x, y) = 0.0
ϕₙ(x, y) = 0.0
ϕₛ(x, y) = 0.0

#=
Ix = Integral(x in 0 .. 1)
Iy = Integral(y in 0 .. 1)
Ixy = Integral((x, y) in (0 .. 1, 0 .. 1))
u(θ) = [u_1 * θ + 2 * u_1 * F(x, y)) + u0, 0]
v = [u_1 * F(x, y)^2 + F(x, y), 0]
∇ ⋅ v = (5 * F(x, y) + 1) * ∂F/∂x

# First term: + ∫ θₜ * ψᵢⱼ = Dₜθᵢⱼ(t)
# Second term: + ∮ [(u(θ) * θ * ψᵢⱼ) ⋅ n] 
# Third term: - ∫ (u(θ) * θ) ⋅ ∇ψᵢⱼ
# Fourth term: - ∫ (∇ ⋅ v) ψᵢⱼ
# Fifth term: - ∫ ∇⋅(k ∇F) ψᵢⱼ = - ∫ (∇k ⋅ ∇F) ψᵢⱼ - ∫ k ΔF ψᵢⱼ
# Sixth term: - ∮ [k ∇θ ψᵢⱼ - k θ ∇ψᵢⱼ] ⋅ n 
# Seventh term: - ∫ θ ∇⋅(k_e ∇ψᵢⱼ) = (λᵢ^2 + λⱼ^2) ∫ θ(x, y) * ψᵢ(i, x) * ψⱼ(j, y)
# Eighth term: - ∫ θ ∇⋅((k-k_e) ∇ψᵢⱼ) = - ∫ θ ∇(k-k_e) ⋅∇ψᵢⱼ - ∫ θ * (k-k_e) * Δψᵢⱼ 
=#

# ODE Form: Dₜθᵢⱼ(t) + Σᵢ₁ Σⱼ₁ Σᵢ₂ Σⱼ₂ Aᵢ₀ⱼ₀ᵢ₁ⱼ₁ᵢ₂ⱼ₂ * θᵢ₁ⱼ₁(t) * θᵢ₂ⱼ₂(t) + Σᵢ₁ Σⱼ₁ Bᵢ₀ⱼ₀ᵢ₁ⱼ₁ * θᵢ₁ⱼ₁(t) + Cᵢ₀ⱼ₀ = 0
display("Preparing matrices...")
A = Array{Float64}(undef, L, L, L)
B = Array{Float64}(undef, L, L)
C = Array{Float64}(undef, L)
Θ₀ = Array{Float64}(undef, L)
if status == :Symbolic
    @showprogress desc = "Computing A..." Threads.@threads for index in CartesianIndices((L, L, L))
        k₀, k₁, k₂ = Tuple(index)
        i₀, j₀ = Tuple(indexes[k₀])
        i₁, j₁ = Tuple(indexes[k₁])
        i₂, j₂ = Tuple(indexes[k₂])
        aux_r = ψᵢ(i₀, 1) * ψᵢ(i₁, 1) * ψᵢ(i₂, 1) * quadgk(y -> ψⱼ(j₀, y) * ψⱼ(j₁, y) * ψⱼ(j₂, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux_l = ψᵢ(i₀, 0) * ψᵢ(i₁, 0) * ψᵢ(i₂, 0) * quadgk(y -> ψⱼ(j₀, y) * ψⱼ(j₁, y) * ψⱼ(j₂, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux1 = u_1 * (aux_r - aux_l)
        aux2_x = quadgk(x -> Dψᵢ(i₀, x) * ψᵢ(i₁, x) * ψᵢ(i₂, x), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux2_y = quadgk(y -> ψⱼ(j₀, y) * ψⱼ(j₁, y) * ψⱼ(j₂, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux2 = -u_1 * aux2_x * aux2_y
        A[index] = aux1 + aux2
    end
else
    @showprogress desc = "Computing A..." Threads.@threads for index in CartesianIndices((L, L, L))
        k₀, k₁, k₂ = Tuple(index)
        i₀, j₀ = Tuple(indexes[k₀])
        i₁, j₁ = Tuple(indexes[k₁])
        i₂, j₂ = Tuple(indexes[k₂])
        aux1_n = 0.0
        aux1_s = 0.0
        aux1_r = quadgk(y -> u_1 * ψᵢ(i₁, 1) * ψⱼ(j₁, y) * ψᵢ(i₂, 1) * ψⱼ(j₂, y) * ψᵢ(i₀, 1) * ψⱼ(j₀, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux1_l = quadgk(y -> u_1 * ψᵢ(i₁, 0) * ψⱼ(j₁, y) * ψᵢ(i₂, 0) * ψⱼ(j₂, y) * ψᵢ(i₀, 0) * ψⱼ(j₀, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux1 = aux1_n - aux1_s + aux1_r - aux1_l
        aux2 = hcubature((x) -> -u_1 * ψᵢ(i₁, x[1]) * ψⱼ(j₁, x[2]) * ψᵢ(i₂, x[1]) * ψⱼ(j₂, x[2]) * Dψᵢ(i₀, x[1]) * ψⱼ(j₀, x[2]), (0, 0), (1, 1); rtol=1e-3, atol=1e-8)[1]
        A[index] = aux1 + aux2
    end
end
if status == :Symbolic
    @showprogress desc = "Computing B..." Threads.@threads for index in CartesianIndices((L, L))
        k₀, k₁ = Tuple(index)
        i₀, j₀ = Tuple(indexes[k₀])
        i₁, j₁ = Tuple(indexes[k₁])
        aux1_r = quadgk(y -> (2 * u_1 * F(1, y) + u_0) * ψᵢ(i₁, 1) * ψⱼ(j₁, y) * ψᵢ(i₀, 1) * ψⱼ(j₀, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux1_l = quadgk(y -> (2 * u_1 * F(0, y) + u_0) * ψᵢ(i₁, 0) * ψⱼ(j₁, y) * ψᵢ(i₀, 0) * ψⱼ(j₀, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux1 = aux1_r - aux1_l

        aux2_cx = quadgk(x -> Dψᵢ(i₀, x) * ψᵢ(i₁, x), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux2_cy = quadgk(y -> ψⱼ(j₀, y) * ψⱼ(j₁, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux2_c = Cst * aux2_cx * aux2_cy
        aux2_gx = quadgk(x -> G(x) * Dψᵢ(i₀, x) * ψᵢ(i₁, x), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux2_hy = quadgk(y -> H(y) * ψⱼ(j₀, y) * ψⱼ(j₁, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux2_gh = aux2_gx * aux2_hy
        aux2_F =  (aux2_c + aux2_gh)
        aux2_x =  quadgk(x -> Dψᵢ(i₀, x) * ψᵢ(i₁, x), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux2_y = quadgk(y -> ψⱼ(j₀, y) * ψⱼ(j₁, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux2_u0 = aux2_x * aux2_y
        aux2 = -(2 * u_1 * aux2_F + u_0 * aux2_u0)
        # F(x,y) = C + G(x) * H(y) where C = 1, G(x) = Σ (exp(-x), ψᵢ(x)) * ψᵢ(x), H(y) = (1/2) * y^2
        aux3 = (μᵢ(i₀)^2 + μⱼ(j₀)^2) * (i₀ == i₁ && j₀ == j₁ ? 1.0 : 0.0)
        
        aux4_x = quadgk(x -> ψᵢ(i₀, x) * ψᵢ(i₁, x), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux4_y = quadgk(y -> (Dk(y) - Dkₑ) * Dψⱼ(j₀, y) * ψⱼ(j₁, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux4 = -aux4_x * aux4_y
        
        aux5_x = quadgk(x -> ψᵢ(i₀, x) * ψᵢ(i₁, x), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux5_y = quadgk(y -> (k(y) - kₑ) * ψⱼ(j₀, y) * ψⱼ(j₁, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux5 = (μᵢ(i₀)^2 + μⱼ(j₀)^2) * aux5_x * aux5_y
        B[index] = aux1 + aux2 + aux3 + aux4 + aux5
    end
else
    @showprogress desc = "Computing B..." Threads.@threads for index in CartesianIndices((L, L))
        k₀, k₁ = Tuple(index)
        i₀, j₀ = Tuple(indexes[k₀])
        i₁, j₁ = Tuple(indexes[k₁])
        aux1_r = quadgk(y -> (2 * u_1 * F(1, y) + u_0)* ψᵢ(i₁, 1) * ψⱼ(j₁, y) * ψᵢ(i₀, 1) * ψⱼ(j₀, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux1_l = quadgk(y -> (2 * u_1 * F(0, y) + u_0) * ψᵢ(i₁, 0) * ψⱼ(j₁, y) * ψᵢ(i₀, 0) * ψⱼ(j₀, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux1 = aux1_r - aux1_l
        aux2 = hcubature((x) -> -(2 * u_1 * F(x[1], x[2]) + u_0)* ψᵢ(i₁, x[1]) * ψⱼ(j₁, x[2]) * Dψᵢ(i₀, x[1]) * ψⱼ(j₀, x[2]), (0, 0), (1, 1); rtol=1e-3, atol=1e-8, initdiv=100)[1]
        aux3 = (μᵢ(i₀)^2 + μⱼ(j₀)^2) * (i₀ == i₁ && j₀ == j₁ ? 1.0 : 0.0)
        aux4 = hcubature((x) -> -(Dk(x[2]) - Dkₑ) * ψᵢ(i₀, x[1]) * Dψⱼ(j₀, x[2]) * ψᵢ(i₁, x[1]) * ψⱼ(j₁, x[2]), (0, 0), (1, 1); rtol=1e-3, atol=1e-8, initdiv=100)[1]
        aux5 = (μᵢ(i₀)^2 + μⱼ(j₀)^2) * hcubature((x) -> (k(x[2]) - kₑ) * ψᵢ(i₀, x[1]) * ψⱼ(j₀, x[2]) * ψᵢ(i₁, x[1]) * ψⱼ(j₁, x[2]), (0, 0), (1, 1); rtol=1e-3, atol=1e-8, initdiv=100)[1]
        B[index] = aux1 + aux2 + aux3 + aux4 + aux5
    end
end
if status == :Symbolic
    @showprogress desc = "Computing C..." Threads.@threads for index in CartesianIndices((L,))
        k₀ = Tuple(index)[1]
        i₀, j₀ = Tuple(indexes[k₀])
        # (2*u1*F + u0) * Fx * ψᵢ * ψⱼ
        aux1_cx = quadgk(x -> DG(x) * ψᵢ(i₀, x), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux1_cy = quadgk(y -> H(y) * ψⱼ(j₀, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux1_c = Cst * aux1_cx * aux1_cy
        aux1_gx = quadgk(x -> G(x) * DG(x) * ψᵢ(i₀, x), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux1_hy = quadgk(y -> H(y)^2 * ψⱼ(j₀, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux1_gh = aux1_gx * aux1_hy
        aux1_F = aux1_c + aux1_gh
        aux1_x = quadgk(x -> DG(x) * ψᵢ(i₀, x), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux1_y = quadgk(y -> H(y) * ψⱼ(j₀, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux1_u0 = aux1_x * aux1_y
        aux1 =  2 * u_1 * aux1_F + u_0 * aux1_u0
        
        # -(∇k ⋅ ∇F)
        aux2_x = quadgk(x ->  G(x) * ψᵢ(i₀, x), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux2_y = quadgk(y -> (Dk(y) - Dkₑ) * DH(y) * ψⱼ(j₀, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux2 = -aux2_x * aux2_y
        # -k ΔF
        aux3_Fxx_x = quadgk(x ->  D2G(x) * ψᵢ(i₀, x), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux3_Fxx_y = quadgk(y -> k(y) * H(y) * ψⱼ(j₀, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux3_Fxx = aux3_Fxx_x * aux3_Fxx_y
        aux3_Fyy_x = quadgk(x ->  G(x) * ψᵢ(i₀, x), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux3_Fyy_y = quadgk(y -> k(y) * D2H(y) * ψⱼ(j₀, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux3_Fyy = aux3_Fyy_x * aux3_Fyy_y
        aux3 = -(aux3_Fxx + aux3_Fyy)

        C[index] = aux1 + aux2 + aux3
    end
else
    @showprogress desc = "Computing C..." Threads.@threads for index in CartesianIndices((L,))
        k₀ = Tuple(index)[1]
        i₀, j₀ = Tuple(indexes[k₀])
        aux1_r = quadgk(y -> u_1 * F(1, y)^2 * ψᵢ(i₀, 1) * ψⱼ(j₀, y), 0, 1)[1]
        aux1_l = quadgk(y -> u_1 * F(0, y)^2 * ψᵢ(i₀, 0) * ψⱼ(j₀, y), 0, 1)[1]
        aux1 = aux1_r - aux1_l
        aux2_r = quadgk(y -> u_0 * F(1, y) * ψᵢ(i₀, 1) * ψⱼ(j₀, y), 0, 1)[1]
        aux2_l = quadgk(y -> u_0 * F(0, y) * ψᵢ(i₀, 0) * ψⱼ(j₀, y), 0, 1)[1]
        aux2 = aux2_r - aux2_l
        aux3 = hcubature((x) -> -u_1 * F(x[1], x[2])^2 * Dψᵢ(i₀, x[1]) * ψⱼ(j₀, x[2]), (0, 0), (1, 1); rtol=1e-3, atol=1e-8, initdiv=100)[1]
        aux4 = hcubature((x) -> -u_0 * F(x[1], x[2]) * Dψᵢ(i₀, x[1]) * ψⱼ(j₀, x[2]), (0, 0), (1, 1); rtol=1e-3, atol=1e-8, initdiv=100)[1]
        aux5_r = quadgk(y -> ϕᵣ(1, y) * (ψᵢ(i₀, 1) * ψⱼ(j₀, y) - k(y) * Dψᵢ(i₀, 1) * ψⱼ(j₀, y)) / (αᵣ(1, y) + βᵣ(1, y)), 0, 1)[1]
        aux5_l = quadgk(y -> ϕₗ(0, y) * (ψᵢ(i₀, 0) * ψⱼ(j₀, y) - k(y) * Dψᵢ(i₀, 0) * ψⱼ(j₀, y)) / (αₗ(0, y) + βₗ(0, y)), 0, 1)[1]
        aux5_n = quadgk(x -> ϕₙ(x, 1) * (ψᵢ(i₀, x) * ψⱼ(j₀, 1) - k(1) * ψᵢ(i₀, x) * Dψⱼ(j₀, 1)) / (ₙ(x, 1) + βₙ(x, 1)), 0, 1)[1]
        aux5_s = quadgk(y -> ϕₛ(y, 0) * (ψᵢ(i₀, 0) * ψⱼ(j₀, y) - k(0) * ψᵢ(i₀, 0) * Dψⱼ(j₀, y)) / (αₛ(y, 0) + βₛ(y, 0)), 0, 1)[1]
        aux5 = -1 * (aux5_r - aux5_l + aux5_n - aux5_s)
        aux6 = hcubature((x) -> Dk(x[2]) * G(x[1]) * DH(x[2]) * ψᵢ(i₀, x[1]) * ψⱼ(j₀, x[2]), (0, 0), (1, 1); rtol=1e-3, atol=1e-8, initdiv=100)[1]
        aux7 = hcubature((x) -> k(x[2]) * G(x[1]) * D2H(x[2]) * ψᵢ(i₀, x[1]) * ψⱼ(j₀, x[2]), (0, 0), (1, 1); rtol=1e-3, atol=1e-8, initdiv=100)[1]
        C[index] = aux1 + aux2 + aux3 + aux4 + aux5 + aux6 + aux7
    end
end
if status == :Symbolic
    @showprogress desc = "Computing initial conditions Θ₀..." Threads.@threads for index in CartesianIndices((L,))
        k₀ = Tuple(index)[1]
        i₀, j₀ = Tuple(indexes[k₀])
        aux_Tx = quadgk(x -> X₀(x) * ψᵢ(i₀, x), 0, 1)[1]
        aux_Ty = quadgk(y -> Y₀(y) * ψⱼ(j₀, y), 0, 1)[1]
        aux_T = aux_Tx * aux_Ty
        # F(x,y) = Cst + G(x) * H(y)
        aux_Fcx = quadgk(x -> ψᵢ(i₀, x), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux_Fcy = quadgk(y -> ψⱼ(j₀, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux_Fc = Cst * aux_Fcx * aux_Fcy
        aux_Fghx = quadgk(x -> G(x) * ψᵢ(i₀, x), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux_Fghy = quadgk(y -> H(y) * ψⱼ(j₀, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux_Fgh = aux_Fghx * aux_Fghy
        aux_F = aux_Fc + aux_Fgh
        aux = aux_T - aux_F
        Θ₀[index] = aux
    end
else
    @showprogress desc = "Computing initial conditions Θ₀..." Threads.@threads for index in CartesianIndices((L,))
        k₀ = Tuple(index)[1]
        i₀, j₀ = Tuple(indexes[k₀])
        aux = hcubature(
            (x) -> (T₀(x[1], x[2]) - F(x[1], x[2])) * ψᵢ(i₀, x[1]) * ψⱼ(j₀, x[2]), (0, 0), (1, 1)
        )[1]
        Θ₀[index] = aux
    end
end
@showprogress desc = "Testing validity of the numbers calculated..." for index in CartesianIndices((L,))
    k₀ = Tuple(index)[1]
    if isnan(Θ₀[k₀]) || isnan(C[k₀]) || isinf(Θ₀[k₀]) || isinf(C[k₀])
        error("NaN detected in Θ₀ or C at index $k₀")
    end
    for index2 in CartesianIndices((L,))
        k₁ = Tuple(index2)[1]
        if isnan(B[k₀, k₁]) || isinf(B[k₀, k₁])
            error("NaN detected in B at index ($k₀, $k₁)")
        end
    end
    for index2 in CartesianIndices((L, L))
        k₁, k₂ = Tuple(index2)
        if isnan(A[k₀, k₁, k₂]) || isinf(A[k₀, k₁, k₂])
            error("NaN detected in A at index ($k₀, $k₁, $k₂)")
        end
    end
end
display(Θ₀)

# Define the Transformed 2D Burger equation
function mul2!(z, A, x, y; α=1.0, β=0.0)
    L = size(z, 1)
    Threads.@threads for index in CartesianIndices((L,))
        i₀ = Tuple(index)[1]
        aux = 0.0
        for index2 in CartesianIndices((L, L))
            i₁, i₂ = Tuple(index2)
            @inbounds aux += α * A[i₀, i₁, i₂] * x[i₁] * y[i₂]
        end
        @inbounds z[i₀] = β * z[i₀] + aux
    end
end
function mul1!(z, B, x; α=1.0, β=0.0)
    L = size(z, 1)
    Threads.@threads for index in CartesianIndices((L,))
        i₀ = Tuple(index)[1]
        aux = 0.0
        for index2 in CartesianIndices((L,))
            i₁ = Tuple(index2)[1]
            @inbounds aux += α * B[i₀, i₁] * x[i₁]
        end
        @inbounds z[i₀] = β * z[i₀] + aux
    end
end

function burger2D!(dΘ, Θ, p, t)
    fill!(dΘ, 0.0)
    mul2!(dΘ, A, Θ, Θ; α=-1.0, β=1.0)
    mul1!(dΘ, B, Θ; α=-1.0, β=1.0)
    dΘ .-= C
end

display("Solving the 2D Burger equation...")
tspan = (0.0, 1.0)
prob = ODEProblem(burger2D!, Θ₀, tspan)
#p = ProgressThresh(1.0; desc="Solving ODE...", dt=0.5)
function ProgressCallback!(integrator)
    print("\rTime: $(round(integrator.t, digits=4)) / $(tspan[2])         ")
    u_modified!(integrator, false)
    return nothing
end
condition1(u, t, integrator) = true
cb = DiscreteCallback(condition1, ProgressCallback!)
sol = solve(prob, Rodas4P(), reltol=1e-8, abstol=1e-8, callback=cb)
print("\nSolution completed.\n")

results_path = joinpath(@__DIR__, "..", "results", "hm2", "N_$N_basis")
mkpath(results_path)
save_object(joinpath(results_path, "burger2D.jld2"), sol)

ts = range(tspan[1], stop=tspan[2], length=100)
xs = range(0, stop=1, length=150)
ys = range(0, stop=1, length=100)
Ts = [sol(t)[k] for t in ts, k in 1:L]
us = [sum(sol(t)[k] * ψᵢ(indexes[k][1], x) * ψⱼ(indexes[k][2], y) for k in 1:N_basis) + F(x, y) for x in xs, y in ys, t in ts]
max_u = maximum(us)
min_u = minimum(us)

heatmap(xs, ys, [F(x, y) for y in ys, x in xs], title="Filter F(x,y)", xlabel="x", ylabel="y", colorbar_title="F(x,y)")
savefig(joinpath(results_path, "hm2_Fxy.pdf"))
heatmap(xs, ys, [T₀(x, y) for y in ys, x in xs], title="Initial Condition T₀(x,y)", xlabel="x", ylabel="y", colorbar_title="T₀(x,y)")
savefig(joinpath(results_path, "hm2_T0xy.pdf"))
heatmap(xs, ys, [T₀(x, y) - F(x, y) for y in ys, x in xs], title="Initial Condition θ₀(x,y)", xlabel="x", ylabel="y", colorbar_title="θ₀(x,y)")
savefig(joinpath(results_path, "hm2_theta0xy.pdf"))

plt_Ts = plot(ts, Ts, xlabel="t", ylabel="Θ(t)", title="Evolution of Θ(t) over time", legend=false)
savefig(plt_Ts, joinpath(results_path, "hm2_Theta_t.pdf"))
mkpath(joinpath(results_path, "T_series"))
for t in 1:length(ts)
    hplt = heatmap(xs, ys, us[:, :, t]', xlabel="x", ylabel="y", title="t = $(round(ts[t], digits=3))", colorbar_title="u(t,x,y)", clims=(min_u, max_u))
    savefig(hplt, joinpath(results_path, "T_series", "burger_equation_t_$(lpad(t, 4, '0')).pdf"))
end