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

# Filter the problem to have homogeneous BCs:
# In x:
# filterₓ(t, x, y) = 1
# In y:
# filterᵧ(t, x, y) = Σᵢ exp(-x) * y^2 / 2 * ψᵢ(x) 

# Standard Form: Tₜ + ∇⋅(u * T) = ∇⋅(k ∇T) + S
# Classic Integration from ψᵢⱼ(x, y) = ϕᵢⱼ(x) * φᵢⱼ(y)
# Integral:
#   ∫ Tₜ * ψᵢⱼ(x, y) + ∫ ∇⋅(u * T) * ψᵢⱼ(x, y) = ∫ ∇⋅(k ∇T) * ψᵢⱼ(x, y) + ∫ S * ψᵢⱼ(x, y)
#   Θᵢⱼ'(t) + ∮ [(u * T * ψᵢⱼ) ⋅ n] - ∫ (u * T) ⋅ ∇ψᵢⱼ = ∮ [k ∇T ψᵢⱼ - k T ∇ψᵢⱼ] + ∫ T ∇⋅(k_e ∇ψᵢⱼ) + ∫ T ∇⋅((k-k_e) ∇ψᵢⱼ) + ∫ S * ψᵢⱼ(x, y)

# T = θ + F
# F = 1 + Σ exp(-x) * y^2 / 2 * ψᵢ(x)

#   ∫ θₜ * ψᵢⱼ(x, y) + ∫ ∇⋅(u * θ) * ψᵢⱼ(x, y) = ∫ ∇⋅(k ∇θ) * ψᵢⱼ(x, y) + ∫ [S - ∇⋅(u * F) + ∇⋅(k ∇F)] * ψᵢⱼ

#   + ∫ θₜ * ψᵢⱼ
#   + ∮ [(u(θ) * θ * ψᵢⱼ) ⋅ n] 
#   - ∫ (u(θ) * θ) ⋅ ∇ψᵢⱼ 
#   + ∮ [(v * ψᵢⱼ) ⋅ n]
#   - ∫ v ⋅ ∇ψᵢⱼ
#   - ∮ [k ∇θ ψᵢⱼ - k θ ∇ψᵢⱼ]
#   - ∫ θ ∇⋅(k_e ∇ψᵢⱼ) 
#   - ∫ F ∇⋅(k_e ∇ψᵢⱼ)
#   - ∫ θ ∇⋅((k-k_e) ∇ψᵢⱼ) 
#   - ∫ F ∇⋅((k-k_e) ∇ψᵢⱼ) 
#   - ∫ [S + ∇⋅(k ∇F)] * ψᵢⱼ

# Define necessary functions:
# u(Θ) = [5 * θ + 10 * F + 1, 0]
# v = [5 * F^2 + F, 0]
# k(y) = log(10 + y)
# k_eq = 1
# S = 0

# θ = Σ Θᵢⱼ(t) * ψᵢⱼ(x, y)
# ∇ψᵢⱼ = [Dₓψᵢ(x) * ψⱼ(y), ψᵢ(x) * Dᵧψⱼ(y)]

Lᵢ = 100
Lⱼ = 100
L = 200

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
αᵢ(i) = λᵢ[i]
αⱼ(j) = λⱼ[j]

ψᵢ_NN(i, x) = sin(αᵢ(i) * x)
ψⱼ_NN(j, x) = cos(αⱼ(j) * x)
Dψᵢ_NN(i, x) = cos(αᵢ(i) * x) * αᵢ(i)
Dψⱼ_NN(j, x) = -sin(αⱼ(j) * x) * αⱼ(j)
Nᵢ = [quadgk(x -> ψᵢ_NN(i, x)^2, 0, 1)[1] for i in 1:Lᵢ]
Nⱼ = [quadgk(x -> ψⱼ_NN(j, x)^2, 0, 1)[1] for j in 1:Lⱼ]
ψᵢ(i, x) = ψᵢ_NN(i, x) / sqrt(Nᵢ[i])
ψⱼ(j, x) = ψⱼ_NN(j, x) / sqrt(Nⱼ[j])
Dψᵢ(i, x) = cos(αᵢ(i) * x) * αᵢ(i) / sqrt(Nᵢ[i])
Dψⱼ(j, x) = -sin(αⱼ(j) * x) * αⱼ(j) / sqrt(Nⱼ[j])
D₂ψᵢ(i, x) = -αᵢ(i)^2 * sin(αᵢ(i) * x) / sqrt(Nᵢ[i])
D₂ψⱼ(j, x) = -αⱼ(j)^2 * cos(αⱼ(j) * x) / sqrt(Nⱼ[j])

L_test = 10
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
u_1 = 5
u_0 = 1

k(y) = log(10 + y)
Dk(y) = 1 / (10 + y)
kₑ = 1
Dkₑ = 0
γ = [quadgk(x -> exp(-x) * ψᵢ(i, x), 0, 1)[1] for i in 1:i_maximum]
G(x) = sum(γ[i] * ψᵢ(i, x) for i in 1:i_maximum)
H(y) = y^2 / 2
F(x, y) = 1 + G(x) * H(y)

xspan = (0.0, 1.0)
yspan = (0.0, 1.0)
xs = range(xspan[1], stop=xspan[2], length=250)
ys = range(yspan[1], stop=yspan[2], length=200)
heatmap(xs, ys, [F(x, y) for y in ys, x in xs], title="Initial Condition F(x,y)", xlabel="x", ylabel="y", colorbar_title="F(x,y)")
savefig("hm2_Fxy.pdf")

#=
Ix = Integral(x in 0 .. 1)
Iy = Integral(y in 0 .. 1)
Ixy = Integral((x, y) in (0 .. 1, 0 .. 1))
# First term: + ∫ θₜ * ψᵢⱼ = Dₜθᵢⱼ(t)
term1 = Dₜθᵢⱼ(t)
# Second term: + ∮ [(u(θ) * θ * ψᵢⱼ) ⋅ n] 
# + ∮ [(u(θ) * θ * ψᵢⱼ) ⋅ n] = ∫_0^1 (5 * θ^2(1, y) + 10 * F(1, y)* θ(1, y) + θ(1, y)) * ψᵢ(1) * ψⱼ(y) dy - ∫_0^1 (5 * θ^2(1, y) + 10 * F(1, y)* θ(1, y) + θ(1, y)) * ψᵢ(0) * ψⱼ(y) dy
term2_r = Iy(5 * ψᵢ(i₁, 1) * ψⱼ(j₁, y) * ψᵢ(i₂, 1) * ψⱼ(j₂, y) * ψᵢ(i, 1) * ψⱼ(j, y)) * θᵢ₁ⱼ₁(t) * θᵢ₂ⱼ₂(t) +
          Iy(10 * F(1, y) * ψᵢ(i₁, 1) * ψⱼ(j₁, y) * ψᵢ(i, 1) * ψⱼ(j, y)) * θᵢ₁ⱼ₁(t) +
          Iy(ψᵢ(i₁, 1) * ψⱼ(j₁, y) * ψᵢ(i, 1) * ψⱼ(j, y)) * θᵢ₁ⱼ₁(t)
term2_l = Iy(5 * ψᵢ(i₁, 0) * ψⱼ(j₁, y) * ψᵢ(i₂, 0) * ψⱼ(j₂, y) * ψᵢ(i, 0) * ψⱼ(j, y)) * θᵢ₁ⱼ₁(t) * θᵢ₂ⱼ₂(t) +
          Iy(10 * F(0, y) * ψᵢ(i₁, 0) * ψⱼ(j₁, y) * ψᵢ(i, 0) * ψⱼ(j, y)) * θᵢ₁ⱼ₁(t) +
          Iy(ψᵢ(i₁, 0) * ψⱼ(j₁, y) * ψᵢ(i, 0) * ψⱼ(j, y)) * θᵢ₁ⱼ₁(t)
term2 = term2_r - term2_l
# Third term: - ∫ (u(θ) * θ) ⋅ ∇ψᵢⱼ
term3 = Ixy(-5 * ψᵢ(i₁, x) * ψⱼ(j₁, y) * ψᵢ(i₂, x) * ψⱼ(j₂, y) * Dψᵢ(i, x) * ψⱼ(j, y)) * θᵢ₁ⱼ₁(t) * θᵢ₂ⱼ₂(t) +
        Ixy(-10 * F(x, y) * ψᵢ(i₁, x) * ψⱼ(j₁, y) * Dψᵢ(i, x) * ψⱼ(j, y)) * θᵢ₁ⱼ₁(t) +
        Ixy(-ψᵢ(i₁, x) * ψⱼ(j₁, y) * Dψᵢ(i, x) * ψⱼ(j, y)) * θᵢ₁ⱼ₁(t)
# Fourth term: + ∮ [(v * ψᵢⱼ) ⋅ n]
term4_r = Iy(5 * F(1, y)^2 * ψᵢ(i, 1) * ψⱼ(j, y)) +
          Iy(F(1, y) * ψᵢ(i, 1) * ψⱼ(j, y))
term4_l = Iy(5 * F(0, y)^2 * ψᵢ(i, 0) * ψⱼ(j, y)) +
          Iy(F(0, y) * ψᵢ(i, 0) * ψⱼ(j, y))
term4 = term4_r - term4_l
# Fifth term: - ∫ v ⋅ ∇ψᵢⱼ
term5 = Ixy(-5 * F(x, y)^2 * Dψᵢ(i, x) * ψⱼ(j, y)) +
        Ixy(-F(x, y) * Dψᵢ(i, x) * ψⱼ(j, y))
# Sixth term: - ∮ [k ∇T ψᵢⱼ - k T ∇ψᵢⱼ] ⋅ n => (Boundary Terms should not be zero as they are reffered to T)
term6_r = Iy(-k(y) * ψᵢ(i, 1) * ψⱼ(j, y))
term6_l = Iy(-k(y) * Dψᵢ(i, 0) * ψⱼ(j, y))
term6_n = Ix(-k(1) * ψᵢ(i, x) * ψⱼ(j, 1) * exp(-x))
term6_s = 0
term6 = term6_r - term6_l + term6_n - term6_s
# Seventh term: - ∫ θ ∇⋅(k_e ∇ψᵢⱼ) = ∫ (λᵢ^2 + λⱼ^2) * θ(x, y) * ψᵢ(i, x) * ψⱼ(j, y)
term7 = (λᵢ^2 + λⱼ^2) * δ(i, i₁) * δ(j, j₁) * θᵢ₁ⱼ₁(t)
# Eighth term: - ∫ F ∇⋅(k_e ∇ψᵢⱼ) = ∫ λᵢ + λⱼ * F(x, y) * ψᵢ(i, x) * ψⱼ(j, y)
term8 = (λᵢ^2 + λⱼ^2) * Ixy(F(x, y) * ψᵢ(i, x) * ψⱼ(j, y))
# Ninth term: - ∫ θ ∇⋅((k-k_e) ∇ψᵢⱼ) = - ∫ θ(x, y) * (Dk(y) - Dkₑ(y)) * Dᵧψⱼ(j, y) * ψᵢ(i, x) + (k(y) - kₑ) * (D₂ψⱼ(j, y) * ψᵢ(i, x) + ψⱼ(j, y) * D₂ψᵢ(i, x))
term9 = Ixy(-(Dk - Dkₑ) * Dᵧψⱼ(j, y) * ψᵢ(i, x) * ψᵢ(i₁, x) * ψⱼ(j₁, y)) * θᵢ₁ⱼ₁(t) +
        (λⱼ^2 + λᵢ^2) * Ixy((k - kₑ) * ψⱼ(j, y) * ψᵢ(i, x) * ψᵢ(i₁, x) * ψⱼ(j₁, y)) * θᵢ₁ⱼ₁(t)
# Tenth term: - ∫ F ∇⋅((k-k_e) ∇ψᵢⱼ) = - ∫ F(x, y) * (Dk(y) - Dkₑ(y)) * Dᵧψⱼ(j, y) * ψᵢ(i, x) + (k(y) - kₑ) * (D₂ψⱼ(j, y) * ψᵢ(i, x) + ψⱼ(j, y) * D₂ψᵢ(i, x))
=#

# ODE Form: Dₜθᵢⱼ(t) + Σᵢ₁ Σⱼ₁ Σᵢ₂ Σⱼ₂ Aᵢⱼᵢ₁ⱼ₁ᵢ₂ⱼ₂ * θᵢ₁ⱼ₁(t) * θᵢ₂ⱼ₂(t) + Σᵢ₁ Σⱼ₁ Bᵢⱼᵢ₁ⱼ₁ * θᵢ₁ⱼ₁(t) + Cᵢⱼ = 0
display("Preparing matrices...")
A = Array{Float64}(undef, L, L, L)
B = Array{Float64}(undef, L, L)
C = Array{Float64}(undef, L)
Θ₀ = Array{Float64}(undef, L)
if status == :Symbolic
    function bint(i₀, i₁, i₂, j₀, j₁, j₂)
        d = (αⱼ(j₀) - αⱼ(j₁) - αⱼ(j₂)) * (αⱼ(j₀) + αⱼ(j₁) - αⱼ(j₂)) * (αⱼ(j₀) - αⱼ(j₁) + αⱼ(j₂)) * (αⱼ(j₀) + αⱼ(j₁) + αⱼ(j₂))
        if αⱼ(j₀) != 0 && αⱼ(j₁) != 0 && αⱼ(j₂) != 0 && d != 0
            nr = u_1 * ψᵢ(i₁, 1) * ψᵢ(i₂, 1) * ψᵢ(i₀, 1)
            nl = u_1 * ψᵢ(i₁, 0) * ψᵢ(i₂, 0) * ψᵢ(i₀, 0)
            aux = 0.0
            y = 1
            aux -= αⱼ(j₀)^2 * Dψⱼ(j₀, y) * ψⱼ(j₁, y) * ψⱼ(j₂, y)
            aux -= αⱼ(j₁)^2 * ψⱼ(j₀, y) * Dψⱼ(j₁, y) * ψⱼ(j₂, y)
            aux -= αⱼ(j₂)^2 * ψⱼ(j₀, y) * ψⱼ(j₁, y) * Dψⱼ(j₂, y)
            aux += αⱼ(j₁)^2 * Dψⱼ(j₀, y) * ψⱼ(j₁, y) * ψⱼ(j₂, y)
            aux += αⱼ(j₂)^2 * Dψⱼ(j₀, y) * ψⱼ(j₁, y) * ψⱼ(j₂, y)
            aux += αⱼ(j₀)^2 * ψⱼ(j₀, y) * Dψⱼ(j₁, y) * ψⱼ(j₂, y)
            aux += αⱼ(j₂)^2 * ψⱼ(j₀, y) * Dψⱼ(j₁, y) * ψⱼ(j₂, y)
            aux += αⱼ(j₀)^2 * ψⱼ(j₀, y) * ψⱼ(j₁, y) * Dψⱼ(j₂, y)
            aux += αⱼ(j₁)^2 * ψⱼ(j₀, y) * ψⱼ(j₁, y) * Dψⱼ(j₂, y)
            aux += 2 * Dψⱼ(j₀, y) * Dψⱼ(j₁, y) * Dψⱼ(j₂, y)
            return aux * (nr - nl) / d
        else
            aux_r = u_1 * ψᵢ(i₀, 1) * ψᵢ(i₁, 1) * ψᵢ(i₂, 1) * quadgk(y -> ψⱼ(j₀, y) * ψⱼ(j₁, y) * ψⱼ(j₂, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
            aux_l = u_1 * ψᵢ(i₀, 0) * ψᵢ(i₁, 0) * ψᵢ(i₂, 0) * quadgk(y -> ψⱼ(j₀, y) * ψⱼ(j₁, y) * ψⱼ(j₂, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
            return aux_r - aux_l
        end
    end
    function rint(i₀, i₁, i₂, j₀, j₁, j₂)
        d_x = (αᵢ(i₀) - αᵢ(i₁) - αᵢ(i₂)) * (αᵢ(i₀) + αᵢ(i₁) - αᵢ(i₂)) * (αᵢ(i₀) - αᵢ(i₁) + αᵢ(i₂)) * (αᵢ(i₀) + αᵢ(i₁) + αᵢ(i₂))
        d_y = (αⱼ(j₀) - αⱼ(j₁) - αⱼ(j₂)) * (αⱼ(j₀) + αⱼ(j₁) - αⱼ(j₂)) * (αⱼ(j₀) - αⱼ(j₁) + αⱼ(j₂)) * (αⱼ(j₀) + αⱼ(j₁) + αⱼ(j₂))
        if αᵢ(i₀) != 0 && αᵢ(i₁) != 0 && αᵢ(i₂) != 0 && αⱼ(j₀) != 0 && αⱼ(j₁) != 0 && αⱼ(j₂) != 0 && d_x != 0 && d_y != 0
            aux_x = 0.0
            x = 1
            aux_x -= αᵢ(i₀)^2 * D₂ψᵢ(i₀, x) * ψᵢ(i₁, x) * ψᵢ(i₂, x)
            aux_x -= αᵢ(i₁)^2 * Dψᵢ(i₀, x) * Dψᵢ(i₁, x) * ψᵢ(i₂, x)
            aux_x -= αᵢ(i₂)^2 * Dψᵢ(i₀, x) * ψᵢ(i₁, x) * Dψᵢ(i₂, x)
            aux_x += αᵢ(i₀)^2 * Dψᵢ(i₀, x) * Dψᵢ(i₁, x) * ψᵢ(i₂, x)
            aux_x += αᵢ(i₀)^2 * Dψᵢ(i₀, x) * ψᵢ(i₁, x) * Dψᵢ(i₂, x)
            aux_x += αᵢ(i₁)^2 * Dψᵢ(i₀, x) * ψᵢ(i₁, x) * Dψᵢ(i₂, x)
            aux_x += αᵢ(i₂)^2 * Dψᵢ(i₀, x) * Dψᵢ(i₁, x) * ψᵢ(i₂, x)
            aux_x += αᵢ(i₁)^2 * D₂ψᵢ(i₀, x) * ψᵢ(i₁, x) * ψᵢ(i₂, x)
            aux_x += αᵢ(i₂)^2 * D₂ψᵢ(i₀, x) * ψᵢ(i₁, x) * ψᵢ(i₂, x)
            aux_x += 2 * D₂ψᵢ(i₀, x) * Dψᵢ(i₁, x) * Dψᵢ(i₂, x)

            aux_y = 0.0
            y = 1
            aux_y -= αⱼ(j₀)^2 * Dψⱼ(j₀, y) * ψⱼ(j₁, y) * ψⱼ(j₂, y)
            aux_y -= αⱼ(j₁)^2 * ψⱼ(j₀, y) * Dψⱼ(j₁, y) * ψⱼ(j₂, y)
            aux_y -= αⱼ(j₂)^2 * ψⱼ(j₀, y) * ψⱼ(j₁, y) * Dψⱼ(j₂, y)
            aux_y += αⱼ(j₁)^2 * Dψⱼ(j₀, y) * ψⱼ(j₁, y) * ψⱼ(j₂, y)
            aux_y += αⱼ(j₂)^2 * Dψⱼ(j₀, y) * ψⱼ(j₁, y) * ψⱼ(j₂, y)
            aux_y += αⱼ(j₀)^2 * ψⱼ(j₀, y) * Dψⱼ(j₁, y) * ψⱼ(j₂, y)
            aux_y += αⱼ(j₂)^2 * ψⱼ(j₀, y) * Dψⱼ(j₁, y) * ψⱼ(j₂, y)
            aux_y += αⱼ(j₀)^2 * ψⱼ(j₀, y) * ψⱼ(j₁, y) * Dψⱼ(j₂, y)
            aux_y += αⱼ(j₁)^2 * ψⱼ(j₀, y) * ψⱼ(j₁, y) * Dψⱼ(j₂, y)
            aux_y += 2 * Dψⱼ(j₀, y) * Dψⱼ(j₁, y) * Dψⱼ(j₂, y)
            d_y = (αⱼ(j₀) - αⱼ(j₁) - αⱼ(j₂)) * (αⱼ(j₀) + αⱼ(j₁) - αⱼ(j₂)) * (αⱼ(j₀) - αⱼ(j₁) + αⱼ(j₂)) * (αⱼ(j₀) + αⱼ(j₁) + αⱼ(j₂))
            return -u_1 * aux_y * aux_x / (d_x * d_y)
        else
            return -u_1 * quadgk((x) -> Dψᵢ(i₀, x) * ψᵢ(i₁, x) * ψᵢ(i₂, x), 0, 1; rtol=1e-3, atol=1e-8)[1] * quadgk(y -> ψⱼ(j₀, y) * ψⱼ(j₁, y) * ψⱼ(j₂, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        end
    end
    @showprogress desc = "Computing A..." Threads.@threads for index in CartesianIndices((L, L, L))
        k₀, k₁, k₂ = Tuple(index)
        i₀, j₀ = Tuple(indexes[k₀])
        i₁, j₁ = Tuple(indexes[k₁])
        i₂, j₂ = Tuple(indexes[k₂])
        aux1 = bint(i₀, i₁, i₂, j₀, j₁, j₂)
        aux2 = rint(i₀, i₁, i₂, j₀, j₁, j₂)
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
        aux1_r = quadgk(y -> 2 * u_1 * F(1, y) * ψᵢ(i₁, 1) * ψⱼ(j₁, y) * ψᵢ(i₀, 1) * ψⱼ(j₀, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux1_l = quadgk(y -> 2 * u_1 * F(0, y) * ψᵢ(i₁, 0) * ψⱼ(j₁, y) * ψᵢ(i₀, 0) * ψⱼ(j₀, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux1 = aux1_r - aux1_l
        aux2_r = quadgk(y -> u_0 * ψᵢ(i₁, 1) * ψⱼ(j₁, y) * ψᵢ(i₀, 1) * ψⱼ(j₀, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux2_l = quadgk(y -> u_0 * ψᵢ(i₁, 0) * ψⱼ(j₁, y) * ψᵢ(i₀, 0) * ψⱼ(j₀, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux2 = aux2_r - aux2_l
        # F(x,y) = C + G(x) * H(y) where C = 1, G(x) = Σ (exp(-x), ψᵢ(x)) * ψᵢ(x), H(y) = (1/2) * y^2
        aux3_cx = quadgk(x -> Dψᵢ(i₀, x) * ψᵢ(i₁, x), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux3_cy = quadgk(y -> ψⱼ(j₀, y) * ψⱼ(j₁, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux3_c = u_0 * aux3_cx * aux3_cy
        aux_gx = quadgk(x -> G(x) * Dψᵢ(i₀, x) * ψᵢ(i₁, x), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux_hy = quadgk(y -> H(y) * ψⱼ(j₀, y) * ψⱼ(j₁, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux3_gh = aux_gx * aux_hy
        aux3 = -2 * u_1 * (aux3_c + aux3_gh)
        aux4_x = quadgk(x -> ψᵢ(i₁, x) * Dψᵢ(i₀, x), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux4_y = quadgk(y -> ψⱼ(j₁, y) * ψⱼ(j₀, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux4 = -u_0 * aux4_x * aux4_y
        aux5 = (αᵢ(i₀)^2 + αⱼ(j₀)^2) * (i₀ == i₁ && j₀ == j₁ ? 1.0 : 0.0)
        aux6_x = quadgk(x -> ψᵢ(i₀, x) * ψᵢ(i₁, x), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux6_y = quadgk(y -> (Dk(y) - Dkₑ) * Dψⱼ(j₀, y) * ψⱼ(j₁, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux6 = -1 * aux6_x * aux6_y
        aux7_x = quadgk(x -> ψᵢ(i₀, x) * ψᵢ(i₁, x), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux7_y = quadgk(y -> (k(y) - kₑ) * ψⱼ(j₀, y) * ψⱼ(j₁, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux7 = (αᵢ(i₀)^2 + αⱼ(j₀)^2) * aux7_x * aux7_y
        B[index] = aux1 + aux2 + aux3 + aux4 + aux5 + aux6 + aux7
    end
else
    @showprogress desc = "Computing B..." Threads.@threads for index in CartesianIndices((L, L))
        k₀, k₁ = Tuple(index)
        i₀, j₀ = Tuple(indexes[k₀])
        i₁, j₁ = Tuple(indexes[k₁])
        aux1_r = quadgk(y -> 2 * u_1 * F(1, y) * ψᵢ(i₁, 1) * ψⱼ(j₁, y) * ψᵢ(i₀, 1) * ψⱼ(j₀, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux1_l = quadgk(y -> 2 * u_1 * F(0, y) * ψᵢ(i₁, 0) * ψⱼ(j₁, y) * ψᵢ(i₀, 0) * ψⱼ(j₀, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux1 = aux1_r - aux1_l
        aux2_r = quadgk(y -> u_0 * ψᵢ(i₁, 1) * ψⱼ(j₁, y) * ψᵢ(i₀, 1) * ψⱼ(j₀, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux2_l = quadgk(y -> u_0 * ψᵢ(i₁, 0) * ψⱼ(j₁, y) * ψᵢ(i₀, 0) * ψⱼ(j₀, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux2 = aux2_r - aux2_l
        aux3 = hcubature((x) -> -2 * u_1 * F(x[1], x[2]) * ψᵢ(i₁, x[1]) * ψⱼ(j₁, x[2]) * Dψᵢ(i₀, x[1]) * ψⱼ(j₀, x[2]), (0, 0), (1, 1); rtol=1e-3, atol=1e-8, initdiv=100)[1]
        aux4 = hcubature((x) -> -u_0 * ψᵢ(i₁, x[1]) * ψⱼ(j₁, x[2]) * Dψᵢ(i₀, x[1]) * ψⱼ(j₀, x[2]), (0, 0), (1, 1); rtol=1e-3, atol=1e-8, initdiv=100)[1]
        aux5 = (αᵢ(i₀)^2 + αⱼ(j₀)^2) * (i₀ == i₁ && j₀ == j₁ ? 1.0 : 0.0)
        aux6 = hcubature((x) -> -(Dk(x[2]) - Dkₑ) * ψᵢ(i₀, x[1]) * Dψⱼ(j₀, x[2]) * ψᵢ(i₁, x[1]) * ψⱼ(j₁, x[2]), (0, 0), (1, 1); rtol=1e-3, atol=1e-8, initdiv=100)[1]
        aux7 = (αᵢ(i₀)^2 + αⱼ(j₀)^2) * hcubature((x) -> (k(x[2]) - kₑ) * ψᵢ(i₀, x[1]) * ψⱼ(j₀, x[2]) * ψᵢ(i₁, x[1]) * ψⱼ(j₁, x[2]), (0, 0), (1, 1); rtol=1e-3, atol=1e-8, initdiv=100)[1]
        B[index] = aux1 + aux2 + aux3 + aux4 + aux5 + aux6 + aux7
    end
end
if status == :Symbolic
    @showprogress desc = "Computing C..." Threads.@threads for index in CartesianIndices((L,))
        k₀ = Tuple(index)[1]
        i₀, j₀ = Tuple(indexes[k₀])
        aux1_r = quadgk(y -> u_1 * F(1, y)^2 * ψᵢ(i₀, 1) * ψⱼ(j₀, y), 0, 1)[1]
        aux1_l = quadgk(y -> u_1 * F(0, y)^2 * ψᵢ(i₀, 0) * ψⱼ(j₀, y), 0, 1)[1]
        aux1 = aux1_r - aux1_l
        aux2_r = quadgk(y -> u_0 * F(1, y) * ψᵢ(i₀, 1) * ψⱼ(j₀, y), 0, 1)[1]
        aux2_l = quadgk(y -> u_0 * F(0, y) * ψᵢ(i₀, 0) * ψⱼ(j₀, y), 0, 1)[1]
        aux2 = aux2_r - aux2_l
        # F(x,y)^2 = C^2 + 2 * C * G(x) * H(y) + G^2(x) * H^2(y) where C = 1, G(x) = exp(-x), H(y) = (1/2) * y^2 * Σ ψᵢ(x)
        aux3_cx = quadgk(x -> Dψᵢ(i₀, x), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux3_cy = quadgk(y -> ψⱼ(j₀, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux3_c = -u_1 * 1^2 * aux3_cx * aux3_cy
        aux3_2cghx = quadgk(x -> G(x) * Dψᵢ(i₀, x), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux3_2cghy = quadgk(y -> H(y) * ψⱼ(j₀, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux3_2cgh = -u_1 * 2 * 1 * aux3_2cghx * aux3_2cghy
        aux3_ghghx = quadgk(x -> G(x)^2 * Dψᵢ(i₀, x), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux3_ghghy = quadgk(y -> H(y)^2 * ψⱼ(j₀, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux3_ghgh = -u_1 * aux3_ghghx * aux3_ghghy
        aux3 = aux3_c + aux3_2cgh + aux3_ghgh
        # F(x,y) = C + G(x) * H(y) where C = 1, G(x) = exp(-x), H(y) = (1/2) * y^2 * Σ ψᵢ(x)
        aux4_cx = quadgk(x -> Dψᵢ(i₀, x), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux4_cy = quadgk(y -> ψⱼ(j₀, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux4_c = -u_0 * 1 * aux4_cx * aux4_cy
        aux4_ghx = quadgk(x -> G(x) * Dψᵢ(i₀, x), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux4_ghy = quadgk(y -> H(y) * ψⱼ(j₀, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux4_gh = -u_0 * aux4_ghx * aux4_ghy
        aux4 = aux4_c + aux4_gh
        # TODO: Boundary Check
        aux5_r = quadgk(y -> k(y) * ψᵢ(i₀, 1) * ψⱼ(j₀, y), 0, 1)[1]
        aux5_l = quadgk(y -> k(y) * Dψᵢ(i₀, 0) * ψⱼ(j₀, y), 0, 1)[1]
        aux5_n = quadgk(x -> k(1) * ψᵢ(i₀, x) * ψⱼ(j₀, 1) * exp(-x), 0, 1)[1]
        aux5_s = 0.0
        aux5 = -1 * (aux5_r - aux5_l + aux5_n - aux5_s)
        # F(x,y) = C + G(x) * H(y) where C = 1, G(x) = Σ (exp(-x), ψᵢ(x)) ψᵢ(x), H(y) = y^2 / 2
        aux6_cx = quadgk(x -> ψᵢ(i₀, x), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux6_cy = quadgk(y -> (Dk(y) - Dkₑ) * Dψⱼ(j₀, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux6_c = 1 * aux6_cx * aux6_cy
        aux6_ghx = quadgk(x -> G(x) * ψᵢ(i₀, x), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux6_ghy = quadgk(y -> (Dk(y) - Dkₑ) * H(y) * Dψⱼ(j₀, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux6_gh = aux6_ghx * aux6_ghy
        aux6 = -1 * (aux6_c + aux6_gh)
        # F(x,y) = C + G(x) * H(y) where C = 1, G(x) = Σ (exp(-x), ψᵢ(x)) ψᵢ(x), H(y) = y^2 / 2
        aux7_cx = quadgk(x -> ψᵢ(i₀, x), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux7_cy = quadgk(y -> (k(y) - kₑ) * ψⱼ(j₀, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux7_c = 1 * aux7_cx * aux7_cy
        aux7_ghx = quadgk(x -> G(x) * ψᵢ(i₀, x), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux7_ghy = quadgk(y -> (k(y) - kₑ) * H(y) * ψⱼ(j₀, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux7_gh = aux7_ghx * aux7_ghy
        aux7 = (αᵢ(i₀)^2 + αⱼ(j₀)^2) * (aux7_c + aux7_gh)
        C[index] = aux1 + aux2 + aux3 + aux4 + aux5 + aux6 + aux7
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
        aux5_r = quadgk(y -> -k(y) * ψᵢ(i₀, 1) * ψⱼ(j₀, y), 0, 1)[1]
        aux5_l = quadgk(y -> -k(y) * Dψᵢ(i₀, 0) * ψⱼ(j₀, y), 0, 1)[1]
        aux5_n = quadgk(x -> -k(1) * ψᵢ(i₀, x) * ψⱼ(j₀, 1) * exp(-x), 0, 1)[1]
        aux5_s = 0.0
        aux5 = aux5_r - aux5_l + aux5_n - aux5_s
        aux6 = (αᵢ(i₀)^2 + αⱼ(j₀)^2) * hcubature((x) -> F(x[1], x[2]) * ψᵢ(i₀, x[1]) * ψⱼ(j₀, x[2]), (0, 0), (1, 1); rtol=1e-3, atol=1e-8, initdiv=100)[1]
        C[index] = aux1 + aux2 + aux3 + aux4 + aux5 + aux6
    end
end
if status == :Symbolic
    @showprogress desc = "Computing initial conditions Θ₀..." Threads.@threads for index in CartesianIndices((L,))
        k₀ = Tuple(index)[1]
        i₀, j₀ = Tuple(indexes[k₀])
        aux_Tx = quadgk(x -> (x * (1 - x)) * ψᵢ(i₀, x), 0, 1)[1]
        aux_Ty = quadgk(y -> (y * (1 - y)) * ψⱼ(j₀, y), 0, 1)[1]
        aux_T = aux_Tx * aux_Ty
        # F(x,y) = C + G(x) * H(y) where C = 1, G(x) = exp(-x), H(y) = (1/2) * y^2 * Σ ψᵢ(x)
        aux_Fcx = quadgk(x -> ψᵢ(i₀, x), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux_Fcy = quadgk(y -> ψⱼ(j₀, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux_Fc = 1 * aux_Fcx * aux_Fcy
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
        aux1 = hcubature(
            (x) -> (x[1] * (1 - x[1]) * x[2] * (1 - x[2]) - F(x[1], x[2])) * ψᵢ(i₀, x[1]) * ψⱼ(j₀, x[2]), (0, 0), (1, 1)
        )[1]
        Θ₀[index] = aux1
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
    Lᵢ = size(z, 1)
    Lⱼ = size(z, 2)
    Threads.@threads for index in CartesianIndices((L,))
        i₀ = Tuple(index)[1]
        aux = 0.0
        for index2 in CartesianIndices((L, L))
            i₁, i₂ = Tuple(index2)
            aux += α * A[i₀, i₁, i₂] * x[i₁] * y[i₂]
        end
        z[i₀] = β * z[i₀] + aux
    end
end
function mul1!(z, B, x; α=1.0, β=0.0)
    Lᵢ = size(z, 1)
    Lⱼ = size(z, 2)
    for index in CartesianIndices((L,))
        i₀ = Tuple(index)[1]
        aux = 0.0
        for index2 in CartesianIndices((L,))
            i₁ = Tuple(index2)[1]
            aux += α * B[i₀, i₁] * x[i₁]
        end
        z[i₀] = β * z[i₀] + aux
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
    print("\rTime: $(round(integrator.t, digits=4)) / $(tspan[2])\t\t")
    u_modified!(integrator, false)
    return nothing
end
condition1(u, t, integrator) = true
cb = DiscreteCallback(condition1, ProgressCallback!)
sol = solve(prob, Rodas4P(), reltol=1e-8, abstol=1e-8, callback=cb)
print("\nSolution completed.\n")

save_object("burger2D.jld2", sol)
ts = range(tspan[1], stop=tspan[2], length=100)
xs = range(0, stop=1, length=150)
ys = range(0, stop=1, length=100)
Ts = [sol(t)[k] for t in ts, k in 1:L]
us = [sum(sol(t)[k] * ψᵢ(indexes[k][1], x) * ψⱼ(indexes[k][2], y) for k in 1:L) + F(x, y) for x in xs, y in ys, t in ts]

plt_Ts = plot(ts, Ts, xlabel="t", ylabel="Θ(t)", title="Evolution of Θ(t) over time", legend=false)
savefig(plt_Ts, "hm2_Theta_t.pdf")
mkpath("T_series")
for t in 1:length(ts)
    hplt = heatmap(xs, ys, us[:, :, t]', xlabel="x", ylabel="y", title="t = $(round(ts[t], digits=3))", colorbar_title="u(t,x,y)")
    savefig(hplt, "T_series/burger_equation_t_$(lpad(t, 4, '0')).pdf")
end