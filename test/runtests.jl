using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
include("../src/GITT.jl")
using Symbolics
using IntervalSets
using DomainSets
using Test
using .GITT
using DifferentialEquations
using Plots
using QuadGK

@testset "Diffusion Test" begin
    Ω = DomainSets.ClosedInterval(0.0, 1.0)
    T = DomainSets.ClosedInterval(0.0, 1.0)
    @variables a t x u(..)
    A = At(x ∈ Point(a))
    Dₜ = Differential(t)
    Dₓ = Differential(x)
    ic = InitialCondition(sin(π * x))
    α = 1.0
    β = 0.0
    φ = 0.0
    bc = BoundaryCondition(α, β, φ)
    @test typeof(ic) == InitialCondition
    ic_test = build_function(ic.at, x)
    ic_evaluated = eval(ic_test)
    @test ic_evaluated(0.5) ≈ sin(π * 0.5)
    @test typeof(bc) == BoundaryCondition
    α_test = build_function(A(bc.α), a)
    β_test = build_function(A(bc.β), a)
    φ_test = build_function(A(bc.φ), a)
    α_evaluated = eval(α_test)
    β_evaluated = eval(β_test)
    φ_evaluated = eval(φ_test)
    @test α_evaluated(0.0) ≈ α
    @test β_evaluated(0.0) ≈ β
    @test φ_evaluated(0.0) ≈ φ

    terms = Dict{Symbol,Symbolics.Num}(
        :Temporal => Dₜ(u(t, x)),
        :Diffusion => -Dₓ(Dₓ(u(t, x))),
    )
    addition_rules = Dict(
        :Temporal => @rule(Dₜ(u(~t, ~x) + ~f) => Dₜ(u(~t, ~x)) + Dₜ(~f)),
        :Diffusion => @rule(-Dₓ(Dₓ(u(~t, ~x) + ~f)) => -Dₓ(Dₓ(u(~t, ~x))) - Dₓ(Dₓ(~f))),
    )
    pde = PDE(Ω, T, t, x, u, terms; ic=ic, bc=bc)
    #GITT.filter!(pde; addition_rules=addition_rules)
    eq = pde()
    display(eq)
    @test typeof(eq) <: Symbolics.Num
    @test isequal(eq, Dₜ(u(t, x)) - Dₓ(Dₓ(u(t, x))))

    transformed_eq = Transform(pde)
end

exit()

@testset "GITT.jl" begin
    @variables t x u(..)
    ic = InitialCondition_1D(u(t, x) ~ 1, t, x, u)
    bc_left = BoundaryCondition_1D(0.0, u(t, x) ~ 1, t, x, u)
    bc_right = BoundaryCondition_1D(1.0, u(t, x) ~ 2, t, x, u)
    pde = PDE_1T2X(t, x, u, 2, 1, 5, 0, 0, ic, bc_left, bc_right)
    transformed_problem, associated_eigenproblem, config = Transform(pde)
    display(transformed_problem)
    display(associated_eigenproblem)
    display(config)
    # Get solution for associated eigenproblem
    N = 5
    eigenvalues(n) = (π * n)^2
    eigenfunctions(n, x) = sin(π * n * x)
    result = Solve(transformed_problem, config, eigenvalues, eigenfunctions)
    v = Array{Float64}(undef, length(result.t), length(eigenvalues))
    for (i, u) in enumerate(result.u)
        v[i, :] = u
    end
    savefig(plot(result.t, v), "result_plot.pdf")
    Θ = Recover(pde, result, eigenfunctions, range(0, 1, length=100))
    savefig(surface(result.t, range(0, 1, length=100), Θ'), "result_surface.pdf")
    @test 0 == 0
end

exit()

@testset "GITT.jl (Array Form)" begin
    L = 5
    @variables t x u(..)
    ic = InitialCondition_1D(u(t, x) ~ 1, t, x, u)
    bc_left = BoundaryCondition_1D(0.0, u(t, x) ~ 1, t, x, u)
    bc_right = BoundaryCondition_1D(1.0, u(t, x) ~ 2, t, x, u)
    pde = PDE_1T2X(t, x, u, 2, 1, 5, 0, 0, ic, bc_left, bc_right)
    Transform_array(pde, L)
    error("Debug")
    @test 0 == 0
end

@testset "Burger Equation 1D" begin
    @variables t x u(t, x)
    Dₓ = Differential(x)
    ic = InitialCondition_1D(u ~ x * (1 - x))
    bc_left = BoundaryCondition_1D(0.0, u ~ 1)
    bc_right = BoundaryCondition_1D(1.0, u + Dₓ(u) ~ 1)
    pde = PDE_1T2X(t, x, u,
        1, 5 * u + 1, 1, 1, 0,
        ic, bc_left, bc_right)

    N = 5
    # Get true eigenvalues and eigenfunctions
    eigenvalues = [(π * n)^2 for n ∈ 1:N]
    eigenfunctions = [x -> sin(π * n * x) for n ∈ 1:N]
    transformed_sys = Transform(pde, eigenvalues, eigenfunctions)
    result = Solve(transformed_sys)
    v = Array{Float64}(undef, length(result.t), length(eigenvalues))
    for (i, u) in enumerate(result.u)
        v[i, :] = u
    end
    savefig(plot(result.t, v), "result_plot.pdf")
    Θ = Recover(pde, result, eigenfunctions, range(0, 1, length=100))
    savefig(surface(result.t, range(0, 1, length=100), Θ'), "result_surface.pdf")
    @test 0 == 0
end