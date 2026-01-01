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

# Standard Form: Tₜ + ∇⋅(u * T) - ∇⋅(k ∇T) - S = 0
# Classic Integration from ψᵢⱼ(x, y) = ϕᵢ(x) * φⱼ(y)
# Integral:
#   ∫ Tₜ * ψᵢⱼ(x, y) + ∫ ∇⋅(u * T) * ψᵢⱼ(x, y) - ∫ ∇⋅(k ∇T) * ψᵢⱼ(x, y) - ∫ S * ψᵢⱼ(x, y) = 0 
#   ∫ Tₜ * ψᵢⱼ(x, y) + ∮ [(u * T * ψᵢⱼ) ⋅ n] - ∫ (u * T) ⋅ ∇ψᵢⱼ - ∮ [k ∇T ψᵢⱼ - k T ∇ψᵢⱼ] - ∫ T ∇⋅(k_e ∇ψᵢⱼ) - ∫ T ∇⋅((k-k_e) ∇ψᵢⱼ) - ∫ S * ψᵢⱼ(x, y) = 0

# T = θ + F
# F = exp(-x) * y^2 / 2 + Σ (a * x + b) * φⱼ(y) 

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
L = 100

if type == :Diffusion
    fᵢ = λ -> sin(λ)
    fⱼ = λ -> sin(λ)
    Iᵢ = (1.000 * π / 2, 1.000 * π / 2)
    Iⱼ = (1.000 * π / 2, 1.000 * π / 2)
else
    fᵢ = λ -> tan(λ) + λ
    fⱼ = λ -> sin(λ)
    Iᵢ = (1.001 * π / 2, 0.999 * π / 2)
    Iⱼ = (-1.000 * π / 2, -1.000 * π / 2)
end
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

if type == :Diffusion
    ψᵢ_NN(i, x) = sin(μᵢ(i) * x)
    ψⱼ_NN(j, x) = sin(μⱼ(j) * x)
    Dψᵢ_NN(i, x) = cos(μᵢ(i) * x) * μᵢ(i)
    Dψⱼ_NN(j, x) = cos(μⱼ(j) * x) * μⱼ(j)
else
    ψᵢ_NN(i, x) = sin(μᵢ(i) * x)
    ψⱼ_NN(j, x) = cos(μⱼ(j) * x)
    Dψᵢ_NN(i, x) = cos(μᵢ(i) * x) * μᵢ(i)
    Dψⱼ_NN(j, x) = -sin(μⱼ(j) * x) * μⱼ(j)
end
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

if type == :Diffusion
    X₀(x) = x * (1 - x)
    Y₀(y) = y * (1 - y)
    T₀(x, y) = X₀(x) * Y₀(y)

    # α(x) * T(x, y) + k(x, y) * β(x) * ∇T ⋅ n = φ(x)
    αᵣ(x, y) = 1.0
    αₗ(x, y) = 1.0
    αₙ(x, y) = 1.0
    αₛ(x, y) = 1.0

    βᵣ(x, y) = 0.0
    βₗ(x, y) = 0.0
    βₙ(x, y) = 0.0
    βₛ(x, y) = 0.0

    φᵣ(x, y) = 1.0
    φₗ(x, y) = 1.0
    φₙ(x, y) = 1.0
    φₛ(x, y) = 1.0

    u_1 = 0
    u_0 = 0
    k(y) = 1
    Dk(y) = 0
    kₑ = 1
    Dkₑ = 0

    Gx(x) = 0.0
    DGx(x) = 0.0
    D2Gx(x) = 0.0
    Gy(y) = 0.0
    DGy(y) = 0.0
    D2Gy(y) = 0.0
    γ_0 = [0.0 for j in 1:j_maximum]
    γ_1 = [0.0 for j in 1:j_maximum]
    H0(y) = sum(γ_0[i] * ψⱼ(j, y) for j in 1:j_maximum)
    DH0(y) = sum(γ_0[i] * Dψⱼ(j, y) for j in 1:j_maximum)
    D2H0(y) = sum(γ_0[i] * D₂ψⱼ(j, y) for j in 1:j_maximum)
    H1(y) = sum(((γ_1[i] - γ_0[i]) / 2.0) * ψⱼ(j, y) for j in 1:j_maximum)
    DH1(y) = sum(((γ_1[i] - γ_0[i]) / 2.0) * Dψⱼ(j, y) for j in 1:j_maximum)
    D2H1(y) = sum(((γ_1[i] - γ_0[i]) / 2.0) * D₂ψⱼ(j, y) for j in 1:j_maximum)
    F(x, y) = Gx(x) * Gy(y) + H0(y) + x * H1(y)

    ϕᵣ(x, y) = 0.0
    ϕₗ(x, y) = 0.0
    ϕₙ(x, y) = 0.0
    ϕₛ(x, y) = 0.0
else
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

    u_1 = 0
    u_0 = 0
    k(y) = 1# log(10 + y)
    Dk(y) = 0# 1 / (10 + y)
    kₑ = 1
    Dkₑ = 0

    Gx(x) = exp(-x)
    DGx(x) = -exp(-x)
    D2Gx(x) = exp(-x)
    Gy(y) = y^2 / 2.0
    DGy(y) = y
    D2Gy(y) = 1.0
    γ_0 = [quadgk(y -> (1 - Gy(y)) * ψⱼ(j, y), 0, 1)[1] for j in 1:j_maximum]
    γ_1 = [quadgk(y -> 1 * ψⱼ(j, y), 0, 1)[1] for j in 1:j_maximum]
    H0(y) = sum(γ_0[j] * ψⱼ(j, y) for j in 1:j_maximum)
    DH0(y) = sum(γ_0[j] * Dψⱼ(j, y) for j in 1:j_maximum)
    D2H0(y) = sum(γ_0[j] * D₂ψⱼ(j, y) for j in 1:j_maximum)
    H1(y) = sum(((γ_1[j] - γ_0[j]) / 2.0) * ψⱼ(j, y) for j in 1:j_maximum)
    DH1(y) = sum(((γ_1[j] - γ_0[j]) / 2.0) * Dψⱼ(j, y) for j in 1:j_maximum)
    D2H1(y) = sum(((γ_1[j] - γ_0[j]) / 2.0) * D₂ψⱼ(j, y) for j in 1:j_maximum)
    F(x, y) = Gx(x) * Gy(y) + H0(y) + x * H1(y)

    ϕᵣ(x, y) = 0.0
    ϕₗ(x, y) = 0.0
    ϕₙ(x, y) = 0.0
    ϕₛ(x, y) = 0.0
end

xspan = (0.0, 1.0)
yspan = (0.0, 1.0)
xs = range(xspan[1], stop=xspan[2], length=250)
ys = range(yspan[1], stop=yspan[2], length=200)
heatmap(xs, ys, [F(x, y) for y in ys, x in xs], title="Filter F(x,y)", xlabel="x", ylabel="y", colorbar_title="F(x,y)")
savefig("hm2_Fxy.pdf")
heatmap(xs, ys, [T₀(x, y) for y in ys, x in xs], title="Initial Condition T₀(x,y)", xlabel="x", ylabel="y", colorbar_title="T₀(x,y)")
savefig("hm2_T0xy.pdf")
heatmap(xs, ys, [T₀(x, y) - F(x, y) for y in ys, x in xs], title="Initial Condition θ₀(x,y)", xlabel="x", ylabel="y", colorbar_title="θ₀(x,y)")
savefig("hm2_theta0xy.pdf")

# Test F behavior at boundaries
plt_Fn = plot(title="F(x,1) and ∂F/∂y(x,1)", xlabel="x", ylabel="F and ∂F/∂y")
plot!(plt_Fn, xs, [φₙ(x, 1) - αₙ(x, 1) * F(x, 1) - k(1) * βₙ(x, 1) * (Gx(x) * DGy(1) + DH0(1) + x * DH1(1)) for x in xs], label="F(x,1) BC")
plt_Fs = plot(title="F(x,0) and ∂F/∂y(x,0)", xlabel="x", ylabel="F and ∂F/∂y")
plot!(plt_Fs, xs, [φₛ(x, 0) - αₛ(x, 0) * F(x, 0) - k(0) * βₛ(x, 0) * (Gx(x) * DGy(0) + DH0(0) + x * DH1(0)) for x in xs], label="F(x,0) BC")
plt_Fr = plot(title="F(1,y) and ∂F/∂x(1,y)", xlabel="y", ylabel="F and ∂F/∂x")
plot!(plt_Fr, ys, [φᵣ(1, y) - αᵣ(1, y) * F(1, y) - k(y) * βᵣ(1, y) * (DGx(1) * Gy(y) + 1 * DH1(y)) for y in ys], label="F(1,y) BC")
plt_Fl = plot(title="F(0,y) and ∂F/∂x(0,y)", xlabel="y", ylabel="F and ∂F/∂x")
plot!(plt_Fl, ys, [φₗ(0, y) - αₗ(0, y) * F(0, y) - k(y) * βₗ(0, y) * (DGx(0) * Gy(y) + 0 * DH1(y)) for y in ys], label="F(0,y) BC")
savefig(plt_Fn, "hm2_Fn_bc.pdf")
savefig(plt_Fs, "hm2_Fs_bc.pdf")
savefig(plt_Fr, "hm2_Fr_bc.pdf")
savefig(plt_Fl, "hm2_Fl_bc.pdf")

#=
Ix = Integral(x in 0 .. 1)
Iy = Integral(y in 0 .. 1)
Ixy = Integral((x, y) in (0 .. 1, 0 .. 1))
# First term: + ∫ θₜ * ψᵢⱼ = Dₜθᵢⱼ(t)
# Second term: + ∮ [(u(θ) * θ * ψᵢⱼ) ⋅ n] 
# Third term: - ∫ (u(θ) * θ) ⋅ ∇ψᵢⱼ
# Fourth term: + ∮ [(v * ψᵢⱼ) ⋅ n]
# Fifth term: - ∫ v ⋅ ∇ψᵢⱼ
# Sixth term: - ∮ [k ∇T ψᵢⱼ - k T ∇ψᵢⱼ] ⋅ n = 
# Seventh term: - ∫ θ ∇⋅(k_e ∇ψᵢⱼ) = ∫ (λᵢ^2 + λⱼ^2) * θ(x, y) * ψᵢ(i, x) * ψⱼ(j, y)
# Eighth term: - ∫ F ∇⋅(k_e ∇ψᵢⱼ) = ∫ (λᵢ^2 + λⱼ^2) * F(x, y) * ψᵢ(i, x) * ψⱼ(j, y)
# Ninth term: - ∫ θ ∇⋅((k-k_e) ∇ψᵢⱼ) 
# Tenth term: - ∫ F ∇⋅((k-k_e) ∇ψᵢⱼ)
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
        aux_r = quadgk(y -> ψᵢ(i₀, 1) * ψᵢ(i₁, 1) * ψᵢ(i₂, 1) * ψⱼ(j₀, y) * ψⱼ(j₁, y) * ψⱼ(j₂, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux_l = quadgk(y -> ψᵢ(i₀, 0) * ψᵢ(i₁, 0) * ψᵢ(i₂, 0) * ψⱼ(j₀, y) * ψⱼ(j₁, y) * ψⱼ(j₂, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux1 = u_1 * (aux_r - aux_l)
        aux2_x = quadgk((x) -> Dψᵢ(i₀, x) * ψᵢ(i₁, x) * ψᵢ(i₂, x), 0, 1; rtol=1e-3, atol=1e-8)[1]
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
        aux1_r = quadgk(y -> 2 * u_1 * F(1, y) * ψᵢ(i₁, 1) * ψⱼ(j₁, y) * ψᵢ(i₀, 1) * ψⱼ(j₀, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux1_l = quadgk(y -> 2 * u_1 * F(0, y) * ψᵢ(i₁, 0) * ψⱼ(j₁, y) * ψᵢ(i₀, 0) * ψⱼ(j₀, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux1 = aux1_r - aux1_l
        aux2_r = quadgk(y -> u_0 * ψᵢ(i₁, 1) * ψⱼ(j₁, y) * ψᵢ(i₀, 1) * ψⱼ(j₀, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux2_l = quadgk(y -> u_0 * ψᵢ(i₁, 0) * ψⱼ(j₁, y) * ψᵢ(i₀, 0) * ψⱼ(j₀, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux2 = aux2_r - aux2_l
        # F(x,y) = Gx(x) * Gy(y) + H0(y) + x * H1(y)
        aux3_gx = quadgk(x -> Gx(x) * Dψᵢ(i₀, x) * ψᵢ(i₁, x), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux3_gy = quadgk(y -> Gy(y) * ψⱼ(j₀, y) * ψⱼ(j₁, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux3_c = aux3_gx * aux3_gy
        aux3_h0x = quadgk(x -> 1 * Dψᵢ(i₀, x) * ψᵢ(i₁, x), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux3_h0y = quadgk(y -> H0(y) * ψⱼ(j₀, y) * ψⱼ(j₁, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux3_h0 = aux3_h0x * aux3_h0y
        aux3_h1x = quadgk(x -> x * Dψᵢ(i₀, x) * ψᵢ(i₁, x), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux3_h1y = quadgk(y -> H1(y) * ψⱼ(j₀, y) * ψⱼ(j₁, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux3_h1 = aux3_h1x * aux3_h1y
        aux3 = -2 * u_1 * (aux3_c + aux3_h0 + aux3_h1)
        aux4_x = quadgk(x -> ψᵢ(i₁, x) * Dψᵢ(i₀, x), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux4_y = quadgk(y -> ψⱼ(j₁, y) * ψⱼ(j₀, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux4 = -u_0 * aux4_x * aux4_y
        aux5 = (μᵢ(i₀)^2 + μⱼ(j₀)^2) * (i₀ == i₁ && j₀ == j₁ ? 1.0 : 0.0)
        aux6_x = quadgk(x -> ψᵢ(i₀, x) * ψᵢ(i₁, x), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux6_y = quadgk(y -> (Dk(y) - Dkₑ) * Dψⱼ(j₀, y) * ψⱼ(j₁, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux6 = -1 * aux6_x * aux6_y
        aux7_x = quadgk(x -> ψᵢ(i₀, x) * ψᵢ(i₁, x), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux7_y = quadgk(y -> (k(y) - kₑ) * ψⱼ(j₀, y) * ψⱼ(j₁, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux7 = (μᵢ(i₀)^2 + μⱼ(j₀)^2) * aux7_x * aux7_y
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
        aux5 = (μᵢ(i₀)^2 + μⱼ(j₀)^2) * (i₀ == i₁ && j₀ == j₁ ? 1.0 : 0.0)
        aux6 = hcubature((x) -> -(Dk(x[2]) - Dkₑ) * ψᵢ(i₀, x[1]) * Dψⱼ(j₀, x[2]) * ψᵢ(i₁, x[1]) * ψⱼ(j₁, x[2]), (0, 0), (1, 1); rtol=1e-3, atol=1e-8, initdiv=100)[1]
        aux7 = (μᵢ(i₀)^2 + μⱼ(j₀)^2) * hcubature((x) -> (k(x[2]) - kₑ) * ψᵢ(i₀, x[1]) * ψⱼ(j₀, x[2]) * ψᵢ(i₁, x[1]) * ψⱼ(j₁, x[2]), (0, 0), (1, 1); rtol=1e-3, atol=1e-8, initdiv=100)[1]
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
        # F(x,y)^2 = Gx(x)^2 * Gy(y)^2 + 2 * Gx(x) * Gy(y) * H0(y) + 2 * x * Gx(x) * Gy(y) * H1(y) + H0(y)^2 + 2 * x * H0(y) * H1(y) + x^2 * H1(y)^2
        aux3_gxgx = quadgk(x -> Gx(x)^2 * Dψᵢ(i₀, x), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux3_gygy = quadgk(y -> Gy(y)^2 * ψⱼ(j₀, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux3_gg = aux3_gxgx * aux3_gygy
        aux3_2gH0x = quadgk(x -> 2 * Gx(x) * Dψᵢ(i₀, x), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux3_2gH0y = quadgk(y -> Gy(y) * H0(y) * ψⱼ(j₀, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux3_2gH0 = aux3_2gH0x * aux3_2gH0y
        aux3_2gH1x = quadgk(x -> 2 * x * Gx(x) * Dψᵢ(i₀, x), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux3_2gH1y = quadgk(y -> Gy(y) * H1(y) * ψⱼ(j₀, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux3_2gH1 = aux3_2gH1x * aux3_2gH1y
        aux3_H0H0x = quadgk(x -> 1 * Dψᵢ(i₀, x), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux3_H0H0y = quadgk(y -> H0(y)^2 * ψⱼ(j₀, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux3_H0H0 = aux3_H0H0x * aux3_H0H0y
        aux3_2H0H1x = quadgk(x -> 2 * x * Dψᵢ(i₀, x), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux3_2H0H1y = quadgk(y -> H0(y) * H1(y) * ψⱼ(j₀, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux3_2H0H1 = aux3_2H0H1x * aux3_2H0H1y
        aux3_H1H1x = quadgk(x -> x^2 * Dψᵢ(i₀, x), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux3_H1H1y = quadgk(y -> H1(y)^2 * ψⱼ(j₀, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux3_H1H1 = aux3_H1H1x * aux3_H1H1y
        aux3 = -u_1 * (aux3_gg + aux3_2gH0 + aux3_2gH1 + aux3_H0H0 + aux3_2H0H1 + aux3_H1H1)
        # F(x,y) = Gx(x) * Gy(y) + H0(y) + x * H1(y)
        aux4_gx = quadgk(x -> Gx(x) * Dψᵢ(i₀, x), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux4_gy = quadgk(y -> Gy(y) * ψⱼ(j₀, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux4_g = aux4_gx * aux4_gy
        aux4_H0x = quadgk(x -> 1 * Dψᵢ(i₀, x), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux4_H0y = quadgk(y -> H0(y) * ψⱼ(j₀, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux4_H0 = aux4_H0x * aux4_H0y
        aux4_H1x = quadgk(x -> x * Dψᵢ(i₀, x), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux4_H1y = quadgk(y -> H1(y) * ψⱼ(j₀, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux4_H1 = aux4_H1x * aux4_H1y
        aux4 = -u_0 * (aux4_g + aux4_H0 + aux4_H1)
        # Diffusion boundary terms:
        aux5_r = quadgk(y -> ϕᵣ(1, y) * (ψᵢ(i₀, 1) * ψⱼ(j₀, y) - k(y) * Dψᵢ(i₀, 1) * ψⱼ(j₀, y)) / (αᵣ(1, y) + βᵣ(1, y)), 0, 1)[1]
        aux5_l = quadgk(y -> ϕₗ(0, y) * (ψᵢ(i₀, 0) * ψⱼ(j₀, y) - k(y) * Dψᵢ(i₀, 0) * ψⱼ(j₀, y)) / (αₗ(0, y) + βₗ(0, y)), 0, 1)[1]
        aux5_n = quadgk(x -> ϕₙ(x, 1) * (ψᵢ(i₀, x) * ψⱼ(j₀, 1) - k(1) * ψᵢ(i₀, x) * Dψⱼ(j₀, 1)) / (αₙ(x, 1) + βₙ(x, 1)), 0, 1)[1]
        aux5_s = quadgk(x -> ϕₛ(x, 0) * (ψᵢ(i₀, x) * ψⱼ(j₀, 0) - k(0) * ψᵢ(i₀, x) * Dψⱼ(j₀, 0)) / (αₛ(x, 0) + βₛ(x, 0)), 0, 1)[1]
        aux5 = -1 * (aux5_r - aux5_l + aux5_n - aux5_s)
        # F(x,y) = Gx(x) * Gy(y) + H0(y) + x * H1(y) | ∇ ⋅ (k ∇F) = Dk(y) * ∂F/∂y + k(y) * (∂²F/∂x² + ∂²F/∂y²)
        # Dk(y) * ∂F/∂y = Dk(y) * (G(x) * Gy'(y) + H0'(y) + x * H1'(y))
        aux6_gx = quadgk(x -> Gx(x) * ψᵢ(i₀, x), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux6_gy = quadgk(y -> Dk(y) * DGy(y) * ψⱼ(j₀, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux6_gh = aux6_gx * aux6_gy
        aux6_h0x = quadgk(x -> 1 * ψᵢ(i₀, x), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux6_h0y = quadgk(y -> Dk(y) * DH0(y) * ψⱼ(j₀, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux6_h0 = aux6_h0x * aux6_h0y
        aux6_h1x = quadgk(x -> x * ψᵢ(i₀, x), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux6_h1y = quadgk(y -> Dk(y) * DH1(y) * ψⱼ(j₀, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux6_h1 = aux6_h1x * aux6_h1y
        aux6 = -1 * (aux6_gh + aux6_h0 + aux6_h1)
        # k(y) * ∂²F/∂x² = k(y) * (Gx''(x) * Gy(y))
        aux7_gx = quadgk(x -> D2Gx(x) * ψᵢ(i₀, x), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux7_gy = quadgk(y -> k(y) * Gy(y) * ψⱼ(j₀, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux7_g = aux7_gx * aux7_gy
        aux7 = -1 * aux7_g
        # k(y) * ∂²F/∂y² = k(y) * (Gx(x) * Gy''(y) + H0''(y) + x * H1''(y))
        aux8_gx = quadgk(x -> Gx(x) * ψᵢ(i₀, x), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux8_gy = quadgk(y -> k(y) * D2Gy(y) * ψⱼ(j₀, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux8_gh = aux8_gx * aux8_gy
        aux8_h0x = quadgk(x -> 1 * ψᵢ(i₀, x), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux8_h0y = quadgk(y -> k(y) * D2H0(y) * ψⱼ(j₀, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux8_h0 = aux8_h0x * aux8_h0y
        aux8_h1x = quadgk(x -> x * ψᵢ(i₀, x), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux8_h1y = quadgk(y -> k(y) * D2H1(y) * ψⱼ(j₀, y), 0, 1; rtol=1e-3, atol=1e-8)[1]
        aux8_h1 = aux8_h1x * aux8_h1y
        aux8 = -1 * (aux8_gh + aux8_h0 + aux8_h1)
        C[index] = aux1 + aux2 + aux3 + aux4 + aux5 + aux6 + aux7 + aux8
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
        aux6 = hcubature((x) -> Dk(x[2]) * (Gx(x[1]) * DGy(x[2]) + DH0(x[2]) + x[1] * DH1(x[2])) * ψᵢ(i₀, x[1]) * ψⱼ(j₀, x[2]), (0, 0), (1, 1); rtol=1e-3, atol=1e-8, initdiv=100)[1]
        aux7 = hcubature((x) -> k(x[2]) * (Gx(x[1]) * D2Gy(x[2]) + D2H0(x[2]) + x[1] * D2H1(x[2]) + D2Gx(x[1]) * Gy(x[2])) * ψᵢ(i₀, x[1]) * ψⱼ(j₀, x[2]), (0, 0), (1, 1); rtol=1e-3, atol=1e-8, initdiv=100)[1]
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
        # F(x,y) = Gx(x) * Gy(y) + H0(y) + x * H1(y)
        aux_Fgx = quadgk(x -> Gx(x) * ψᵢ(i₀, x), 0, 1)[1]
        aux_Fgy = quadgk(y -> Gy(y) * ψⱼ(j₀, y), 0, 1)[1]
        aux_Fg = aux_Fgx * aux_Fgy
        aux_Fh0x = quadgk(x -> 1 * ψᵢ(i₀, x), 0, 1)[1]
        aux_Fh0y = quadgk(y -> H0(y) * ψⱼ(j₀, y), 0, 1)[1]
        aux_Fh0 = aux_Fh0x * aux_Fh0y
        aux_Fh1x = quadgk(x -> x * ψᵢ(i₀, x), 0, 1)[1]
        aux_Fh1y = quadgk(y -> H1(y) * ψⱼ(j₀, y), 0, 1)[1]
        aux_Fh1 = aux_Fh1x * aux_Fh1y
        aux_F = aux_Fg + aux_Fh0 + aux_Fh1
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
display(C)
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
    print("\rTime: $(round(integrator.t, digits=4)) / $(tspan[2])         ")
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
max_u = maximum(us)
min_u = minimum(us)

plt_Ts = plot(ts, Ts, xlabel="t", ylabel="Θ(t)", title="Evolution of Θ(t) over time", legend=false)
savefig(plt_Ts, "hm2_Theta_t.pdf")
mkpath("T_series")
for t in 1:length(ts)
    hplt = heatmap(xs, ys, us[:, :, t]', xlabel="x", ylabel="y", title="t = $(round(ts[t], digits=3))", colorbar_title="u(t,x,y)", clims=(min_u, max_u))
    savefig(hplt, "T_series/burger_equation_t_$(lpad(t, 4, '0')).pdf")
end