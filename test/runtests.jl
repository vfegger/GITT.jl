using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
include("../src/GITT.jl")
using Symbolics
using IntervalSets
using Test
using .GITT
using DifferentialEquations
using Plots

@testset "Summation Operator" begin
    @variables x a b f(..)
    S = Symbolics.Summation()
    S_f = S(x, f(x), a, b)
    rw = @rule(f(~x) => (~x)^2)
    expr0 = Symbolics.unwrap(S_f)
    expr1 = SymbolicUtils.Postwalk(rw)(expr0)
    expr2 = Symbolics.wrap(expr1)
    S_f_applied = expr2
    pre_compute = pre_build(S_f_applied)
    f_expr = build_function(pre_compute, a, b)
    f_evaluated = eval(f_expr)
    f_analytic(a, b) = (b * (b + 1) * (2b + 1) - (a - 1) * a * (2a - 1)) / 6
    @test f_evaluated(1, 3) == f_analytic(1, 3)
    @test f_evaluated(4, 6) == f_analytic(4, 6)
end

@testset "Numerical Integral Operator" begin
    @variables x a b f(..)
    I = NIntegral()
    I_f = I(x, f(x), a, b)
    rw = @rule(f(~x) => (~x)^2)
    expr0 = Symbolics.unwrap(I_f)
    expr1 = SymbolicUtils.Postwalk(rw)(expr0)
    expr2 = Symbolics.wrap(expr1)
    I_f_applied = expr2
    pre_compute = pre_build(I_f_applied)
    f_expr = build_function(pre_compute, a, b)
    f_evaluated = eval(f_expr)
    f_analytic(a, b) = (b^3 - a^3) / 3
    @test f_evaluated(0, 1) ≈ f_analytic(0, 1)
    @test f_evaluated(3, 5) ≈ f_analytic(3, 5)
end

@testset "At Operator" begin
    @variables a x f(..)
    At_x = At()
    expr = At_x(x, f(x), a)
    rw = @rule(f(~x) => (~x)^2)
    expr_applied = Symbolics.wrap(SymbolicUtils.Postwalk(rw)(Symbolics.value(expr)))
    pre_compute = pre_build(expr_applied)
    f_expr = build_function(pre_compute, a)
    f_evaluated = eval(f_expr)
    @test f_evaluated(2.0) ≈ 4.0
    @test f_evaluated(3.0) ≈ 9.0
    @test f_evaluated(-3.0) ≈ 9.0
end

@testset "Composition Test" begin
    @variables x y a b c d f(..)
    I = NIntegral()
    A = At()
    S = Symbolics.Summation()
    @testset "At of NIntegral" begin
        I_f = I(x, f(x, y), a, b)
        At_I_f = A(y, I_f, c)
        rw = @rule(f(~x, ~y) => (~x)^2 + (~y))
        expr0 = Symbolics.unwrap(At_I_f)
        expr1 = SymbolicUtils.Postwalk(rw)(expr0)
        expr2 = Symbolics.wrap(expr1)
        At_I_f_applied = expr2
        pre_compute = pre_build(At_I_f_applied)
        f_expr = build_function(pre_compute, a, b, c)
        f_evaluated = eval(f_expr)
        f_analytic(a, b, c) = (b^3 - a^3) / 3 + (b - a) * c
        @test f_evaluated(0.0, 4.0, 2.0) ≈ f_analytic(0.0, 4.0, 2.0)
        @test f_evaluated(2.0, 4.0, 2.0) ≈ f_analytic(2.0, 4.0, 2.0)
    end
    @testset "Summation of At" begin
        At_f = A(x, f(x, y), a)
        S_At_f = S(y, At_f, b, c)
        rw = @rule(f(~x, ~y) => (~x)^2 + (~y))
        expr0 = Symbolics.unwrap(S_At_f)
        expr1 = SymbolicUtils.Postwalk(rw)(expr0)
        expr2 = Symbolics.wrap(expr1)
        S_At_f_applied = expr2
        pre_compute = pre_build(S_At_f_applied)
        f_expr = build_function(pre_compute, a, b, c)
        f_evaluated = eval(f_expr)
        f_analytic(a, b, c) = a^2 * (c - b + 1) + ((c * (c + 1)) / 2 - (b * (b - 1)) / 2)
        @test f_evaluated(1, 2, 3) ≈ f_analytic(1, 2, 3)
        @test f_evaluated(2, 3, 4) ≈ f_analytic(2, 3, 4)
    end
    @testset "NIntegral of Summation" begin
        S_f = S(x, f(x), 1, 3)
        I_S_f = I(x, S_f, a, b)
        rw = @rule(f(~x) => (~x)^2)
        expr0 = Symbolics.unwrap(I_S_f)
        expr1 = SymbolicUtils.Postwalk(rw)(expr0)
        expr2 = Symbolics.wrap(expr1)
        I_S_f_applied = expr2
        pre_compute = pre_build(I_S_f_applied)
        f_expr = build_function(pre_compute, a, b)
        #display(f_expr)
        f_evaluated = eval(f_expr)
        f_analytic(a, b) = (b - a) * (1^2 + 2^2 + 3^2)
        @test f_evaluated(0, 1) ≈ f_analytic(0, 1)
        @test f_evaluated(1, 2) ≈ f_analytic(1, 2)
    end
    @testset "Summation of NIntegral" begin
        I_f = I(x, f(x), a, b)
        S_I_f = S(x, I_f, 1, 3)
        rw = @rule(f(~x) => (~x)^2)
        expr0 = Symbolics.unwrap(S_I_f)
        expr1 = SymbolicUtils.Postwalk(rw)(expr0)
        expr2 = Symbolics.wrap(expr1)
        S_I_f_applied = expr2
        pre_compute = pre_build(S_I_f_applied)
        f_expr = build_function(pre_compute, a, b)
        #display(f_expr)
        f_evaluated = eval(f_expr)
        f_analytic(a, b) = ((b^3 - a^3) / 3) * 3
        @test f_evaluated(0, 1) ≈ f_analytic(0, 1)
        @test f_evaluated(1, 2) ≈ f_analytic(1, 2)
    end
    @testset "At of Summation" begin
        S_f = S(x, f(x), 1, 3)
        At_S_f = A(x, S_f, a)
        rw = @rule(f(~x) => (~x)^2)
        expr0 = Symbolics.unwrap(At_S_f)
        expr1 = SymbolicUtils.Postwalk(rw)(expr0)
        expr2 = Symbolics.wrap(expr1)
        At_S_f_applied = expr2
        pre_compute = pre_build(At_S_f_applied)
        f_expr = build_function(pre_compute, a)
        #display(f_expr)
        f_evaluated = eval(f_expr)
        f_analytic(a) = 1^2 + 2^2 + 3^2
        @test f_evaluated(0) ≈ f_analytic(0)
        @test f_evaluated(5) ≈ f_analytic(5)
    end
    @testset "NIntegral of At" begin
        At_f = A(x, f(x), a)
        I_At_f = I(x, At_f, 0, 2)
        rw = @rule(f(~x) => (~x)^2)
        expr0 = Symbolics.unwrap(I_At_f)
        expr1 = SymbolicUtils.Postwalk(rw)(expr0)
        expr2 = Symbolics.wrap(expr1)
        I_At_f_applied = expr2
        pre_compute = pre_build(I_At_f_applied)
        f_expr = build_function(pre_compute, a)
        #display(f_expr)
        f_evaluated = eval(f_expr)
        f_analytic(a) = (2 - 0) * a^2
        @test f_evaluated(1) ≈ f_analytic(1)
        @test f_evaluated(3) ≈ f_analytic(3)
    end
end

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