using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Symbolics
using IntervalSets
using DomainSets
using Test

include("../src/expand_integrals.jl")

@testset "Summation Operator" begin
    @variables x a b f(..)
    S = Symbolics.Summation(x in ClosedInterval(a, b))
    S_f = S(f(x))
    rw = @rule(f(~x) => (~x)^2)
    expr0 = Symbolics.unwrap(S_f)
    expr1 = SymbolicUtils.Postwalk(rw)(expr0)
    expr2 = Symbolics.wrap(expr1)
    @test typeof(expr2) <: Symbolics.Num
    @test isequal(expr2, S(x^2))
    S_f_applied = expr2
    f_expr = build_function(S_f_applied, a, b)
    f_evaluated = eval(f_expr)
    f_analytic(a, b) = (b * (b + 1) * (2b + 1) - (a - 1) * a * (2a - 1)) / 6
    @test f_evaluated(1, 3) == f_analytic(1, 3)
    @test f_evaluated(4, 6) == f_analytic(4, 6)
end

@testset "Integral Operator" begin
    @testset "Symbolic Expansion" begin
        @variables x a b f(..)
        I = Integral(x in ClosedInterval(a, b))
        I_f = I(f(x))
        rw = @rule(f(~x) => (~x)^2)
        expr0 = Symbolics.unwrap(I_f)
        expr1 = SymbolicUtils.Postwalk(rw)(expr0)
        expr2 = Symbolics.wrap(expr1)
        @test typeof(expr2) <: Symbolics.Num
        @test isequal(expr2, I(x^2))
        I_f_applied = expand_integrals(expr2)
        f_expr = build_function(I_f_applied, a, b)
        f_evaluated = eval(f_expr)
        f_analytic(a, b) = (b^3 - a^3) / 3
        @test f_evaluated(0, 1) ≈ f_analytic(0, 1)
        @test f_evaluated(3, 5) ≈ f_analytic(3, 5)
    end
    @testset "Numerical Calculation" begin
        @variables x a b f(..)
        I = Integral(x in ClosedInterval(a, b))
        I_f = I(f(x))
        rw = @rule(f(~x) => (~x)^2)
        expr0 = Symbolics.unwrap(I_f)
        expr1 = SymbolicUtils.Postwalk(rw)(expr0)
        expr2 = Symbolics.wrap(expr1)
        @test typeof(expr2) <: Symbolics.Num
        @test isequal(expr2, I(x^2))
        I_f_applied = expr2
        f_expr = build_function(I_f_applied, a, b)
        f_evaluated = eval(f_expr)
        f_evaluated(0, 1) # test
        f_analytic(a, b) = (b^3 - a^3) / 3
        @test f_evaluated(0, 1) ≈ f_analytic(0, 1)
        @test f_evaluated(3, 5) ≈ f_analytic(3, 5)
    end
end

@testset "At Operator" begin
    @variables a x f(..)
    A = At(x ∈ Point(a))
    expr = A(f(x))
    rw = @rule(f(~x) => (~x)^2)
    expr0 = Symbolics.unwrap(expr)
    expr1 = SymbolicUtils.Postwalk(rw)(expr0)
    expr2 = Symbolics.wrap(expr1)
    @test typeof(expr2) <: Symbolics.Num
    @test isequal(expr2, A(x^2))
    A_applied = Symbolics.wrap(SymbolicUtils.Postwalk(rw)(Symbolics.value(expr)))
    f_expr = build_function(A_applied, a)
    f_evaluated = eval(f_expr)
    @test f_evaluated(2.0) ≈ 4.0
    @test f_evaluated(3.0) ≈ 9.0
    @test f_evaluated(-3.0) ≈ 9.0
end

@testset "Composition Test" begin
    @variables x y a b c d f(..)
    Ix = Integral(x ∈ ClosedInterval(a, b))
    Iy = Integral(y ∈ ClosedInterval(c, d))
    Sx = Summation(x ∈ ClosedInterval(a, b))
    Sy = Summation(y ∈ ClosedInterval(c, d))
    Ax = At(x ∈ Point(a))
    Ay = At(y ∈ Point(c))
    @testset "At of Integral" begin
        I_f = Ix(f(x, y))
        At_I_f = Ay(I_f)
        rw = @rule(f(~x, ~y) => (~x)^2 + (~y))
        expr0 = Symbolics.unwrap(At_I_f)
        expr1 = SymbolicUtils.Postwalk(rw)(expr0)
        expr2 = Symbolics.wrap(expr1)
        At_I_f_applied = expr2
        f_expr = build_function(At_I_f_applied, a, b, c)
        f_evaluated = eval(f_expr)
        f_analytic(a, b, c) = (b^3 - a^3) / 3 + (b - a) * c
        @test f_evaluated(0.0, 4.0, 2.0) ≈ f_analytic(0.0, 4.0, 2.0)
        @test f_evaluated(2.0, 4.0, 2.0) ≈ f_analytic(2.0, 4.0, 2.0)
    end
    @testset "Summation of At" begin
        At_f = Ax(f(x, y))
        S_At_f = Sy(At_f)
        rw = @rule(f(~x, ~y) => (~x)^2 + (~y))
        expr0 = Symbolics.unwrap(S_At_f)
        expr1 = SymbolicUtils.Postwalk(rw)(expr0)
        expr2 = Symbolics.wrap(expr1)
        S_At_f_applied = expr2
        f_expr = build_function(S_At_f_applied, a, c, d)
        f_evaluated = eval(f_expr)
        f_analytic(a, c, d) = begin
            aux = 0
            for y in c:d
                aux += a^2 + y
            end
            aux
        end
        @test f_evaluated(1, 2, 3) ≈ f_analytic(1, 2, 3)
        @test f_evaluated(2, 3, 4) ≈ f_analytic(2, 3, 4)
    end
    @testset "Integral of Summation" begin
        S_f = Sy(f(y))
        I_S_f = Ix(S_f)
        rw = @rule(f(~x) => (~x)^2)
        expr0 = Symbolics.unwrap(I_S_f)
        expr1 = SymbolicUtils.Postwalk(rw)(expr0)
        expr2 = Symbolics.wrap(expr1)
        I_S_f_applied = expr2
        f_expr = build_function(I_S_f_applied, a, b, c, d)
        f_evaluated = eval(f_expr)
        f_analytic(a, b, c, d) = begin
            aux = 0
            for y in c:d
                aux += y^2
            end
            aux *= (b - a)
            aux
        end
        @test f_evaluated(0, 1, 2, 3) ≈ f_analytic(0, 1, 2, 3)
        @test f_evaluated(1, 2, 3, 4) ≈ f_analytic(1, 2, 3, 4)
    end
    @testset "Summation of Integral" begin
        I_f = Ix(f(x))
        S_I_f = Sy(I_f)
        rw = @rule(f(~x) => (~x)^2)
        expr0 = Symbolics.unwrap(S_I_f)
        expr1 = SymbolicUtils.Postwalk(rw)(expr0)
        expr2 = Symbolics.wrap(expr1)
        S_I_f_applied = expr2
        f_expr = build_function(S_I_f_applied, a, b, c, d)
        f_evaluated = eval(f_expr)
        f_analytic(a, b, c, d) = begin
            aux = 0
            for y in c:d
                aux += (b^3 - a^3) / 3
            end
            aux
        end
        @test f_evaluated(0, 1, 2, 3) ≈ f_analytic(0, 1, 2, 3)
        @test f_evaluated(1, 2, 3, 4) ≈ f_analytic(1, 2, 3, 4)
    end
    @testset "At of Summation" begin
        S_f = Sx(f(x))
        At_S_f = Ay(S_f)
        rw = @rule(f(~x) => (~x)^2)
        expr0 = Symbolics.unwrap(At_S_f)
        expr1 = SymbolicUtils.Postwalk(rw)(expr0)
        expr2 = Symbolics.wrap(expr1)
        At_S_f_applied = expr2
        f_expr = build_function(At_S_f_applied, a, b, c)
        f_evaluated = eval(f_expr)
        f_analytic(a, b, c) = begin
            aux = 0
            for x in a:b
                aux += x^2
            end
            aux
        end
        @test f_evaluated(0, 1, 2) ≈ f_analytic(0, 1, 2)
        @test f_evaluated(5, 6, 7) ≈ f_analytic(5, 6, 7)
    end
    @testset "Integral of At" begin
        At_f = Ax(f(x))
        I_At_f = Iy(At_f)
        rw = @rule(f(~x) => (~x)^2)
        expr0 = Symbolics.unwrap(I_At_f)
        expr1 = SymbolicUtils.Postwalk(rw)(expr0)
        expr2 = Symbolics.wrap(expr1)
        I_At_f_applied = expr2
        f_expr = build_function(I_At_f_applied, a, c, d)
        f_evaluated = eval(f_expr)
        f_analytic(a, c, d) = (d - c) * a^2
        @test f_evaluated(1, 2, 3) ≈ f_analytic(1, 2, 3)
        @test f_evaluated(3, 4, 5) ≈ f_analytic(3, 4, 5)
    end
end