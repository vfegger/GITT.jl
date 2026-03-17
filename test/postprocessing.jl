using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
include("../src/GITT.jl")

using Plots
using SpecialFunctions
using QuadGK
using Roots
using LaTeXStrings

default(
    fontfamily = "Computer Modern",
    size = (700, 450),   # larger canvas
    titlefontsize = 22,
    guidefontsize = 20,
    tickfontsize = 16,
    legendfontsize = 16,
    linewidth = 2.5
)

# Create graph
x = range(0, stop=1, length=100)
z0 = 0.0
zL = 1.0
r0 = 0.0
rN = 1.0

k = 54.0
hinf = 1000.0

# Find the first 5 eigenvalues
ν(n) = n * π / (zL - z0)
Z(x, n) = cos(ν(n) * x)

# Find the first 5 eigenvalues of k * μ * besselj1(μ * rN) = hinf * besselj0(μ * rN)
eq(μ) = k * μ * besselj1(μ * rN) - hinf * besselj0(μ * rN)
μ_arr = range(0.1, 50.0, step=0.1)
val_arr = eq.(μ_arr)
x_arr = val_arr[1:end-1] .* val_arr[2:end]
indexes = findall(x -> x < 0, x_arr)


eigvals = [find_zero(μ -> eq(μ), (μ_arr[i], μ_arr[i+1]), Bisection()) for i in indexes]
μ(m) = eigvals[m]
R(r, m) = besselj0(μ(m) * r)

# Plot the first 5 eigenfunctions
plot(x, [Z.(x, n) for n in 0:4], label=[L"n=0" L"n=1" L"n=2" L"n=3" L"n=4"], title=L"Eigenfunctions $Z(x, n)$", xlabel=L"x", ylabel=L"Z", legend=:topright)
savefig("eigenfunctions_Z.pdf") 

plot(x, [R.(x, m) for m in 1:5], label=[L"m=1" L"m=2" L"m=3" L"m=4" L"m=5"], title=L"Eigenfunctions $R(r, m)$", xlabel=L"r", ylabel=L"R", legend=:topright)
savefig("eigenfunctions_R.pdf")