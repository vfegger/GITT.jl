module GITT

import IntervalSets
import DomainSets
import SymbolicUtils
import SymbolicUtils.Code: function_to_expr, toexpr, search_variables!
using Symbolics
import DifferentialEquations
import QuadGK: quadgk

export expand_integrals

export InitialCondition, BoundaryCondition
export PDE, filter!, Transform, Solve, Recover

export quadgk

include("symbolics_structs.jl")
include("extend_domain.jl")
include("expand_integrals.jl")

# Main features:
# Transform partial equation to system of ODEs via spectral method
# Solve system of ODEs
# Recover solution of original PDE
# Support for initial and boundary conditions
# Support for filtering of non-homogeneous boundary conditions and source terms

@register_symbolic Tag(s::Symbol, x)
@register_symbolic δ(x, y)

_dict_engineer = Dict([
    :w => +1,
    :v => +1,
    :k => -1,
    :d => +1,
    :g => -1
])
_dict_math = Dict([
    :w => +1,
    :v => +1,
    :k => +1,
    :d => +1,
    :g => +1
])

struct InitialCondition
    at::Symbolics.Num
    function InitialCondition(at::Number)
        return new(Symbolics.wrap(at))
    end
    function InitialCondition(at::Symbolics.Num)
        return new(at)
    end
    function InitialCondition()
        return new(Symbolics.wrap(0.0))
    end
end

struct BoundaryCondition
    α::Symbolics.Num
    β::Symbolics.Num
    φ::Symbolics.Num
    function BoundaryCondition(α::Union{Number,Symbolics.Num}, β::Union{Number,Symbolics.Num}, φ::Union{Number,Symbolics.Num})
        return new(isa(α, Number) ? Symbolics.wrap(α) : α, isa(β, Number) ? Symbolics.wrap(β) : β, isa(φ, Number) ? Symbolics.wrap(φ) : φ)
    end
    function BoundaryCondition()
        return new(Symbolics.wrap(1.0), Symbolics.wrap(0.0), Symbolics.wrap(0.0))
    end
end

mutable struct PDE
    Ω::DomainSets.Domain # Spatial domain
    T::DomainSets.Domain # Temporal domain

    var_t::Symbolics.Num
    var_x::Symbolics.Num
    op_u::Symbolics.CallAndWrap{Num}

    terms::Dict{Symbol,Symbolics.Num}

    ic::InitialCondition
    bc::BoundaryCondition

    filter::Symbolics.Num

    function PDE(Ω::DomainSets.Domain, T::DomainSets.Domain, t::Symbolics.Num, x::Symbolics.Num, u::Symbolics.CallAndWrap{Num}, terms::Dict{Symbol,Symbolics.Num}; ic=InitialCondition(), bc::BoundaryCondition=BoundaryCondition())
        return new(Ω, T, t, x, u, terms, ic, bc, Symbolics.wrap(0.0))
    end
end

function filter!(pde::PDE; addition_rules::Dict{Symbol,<:SymbolicUtils.Rule})
    # Filtering based in a 1D space domain with no dependency in u
    # Should work for linear terms only
    l, r = DomainSets.leftendpoint(pde.Ω), DomainSets.rightendpoint(pde.Ω)
    t = pde.var_t
    x = pde.var_x
    @variables a(t) b(t)
    if pde.bc.α == 0.0
        γ = (a * x + b) * x
    else
        γ = a * x + b
    end
    bc_left = expand_derivatives(At(x ∈ DomainSets.Point(l))(pde.bc.α * γ + pde.bc.β * Differential(x)(γ) - pde.bc.φ), l) ~ 0
    bc_right = expand_derivatives(At(x ∈ DomainSets.Point(r))(pde.bc.α * γ + pde.bc.β * Differential(x)(γ) - pde.bc.φ), r) ~ 0

    display(bc_left)
    display(bc_right)
    result = Symbolics.symbolic_linear_solve([bc_left, bc_right], [a, b])
    subs = [a => result[1], b => result[2]]
    pde.filter = Symbolics.substitute(γ, subs)

    display(pde.filter)

    for (_, term) in pde.terms
        Symbolics.substitute(term, pde.op_u(pde.var_t, pde.var_x) => pde.op_u(pde.var_t, pde.var_x) + pde.filter)
    end
    # Check if all terms have conversion rules
    for (key, _) in pde.terms
        if !haskey(addition_rules, key)
            error("No conversion rule provided for term: $key")
        end
    end

    # Expand terms with provided rules
    for (key, rule) in addition_rules
        pde.terms[key] = Symbolics.wrap(SymbolicUtils.Postwalk(rule)(Symbolics.value(pde.terms[key])))
    end
end

function (pde::PDE)()
    return sum(values(pde.terms))
end

function apply(expr, rules)
    result = expr
    for (name, rw) in rules
        println("Applying rule: $name")
        result_new = Num(0)
        try
            result_new = Symbolics.wrap(rw(Symbolics.value(result)))
        catch err
            println("Error applying rule: $name")
            dump(open(joinpath("dump", "apply.txt"), "w"), result)
            rethrow(err)
        end
        if string(result) != string(result_new)
            print("\t")
            show(stdout, MIME"text/plain"(), result_new)
            println()
            result = result_new
        else
            println("\tRule was not applied.")
        end
    end
    println()
    return Symbolics.wrap(result)
end

function Transform(pde::PDE)
    # Transform terms in system of differential equations
    t = pde.var_t
    x = pde.var_x
    u = pde.op_u
    @variables n::Integer N::Integer m::Integer θ(..) Ψ(..) λ(..)
    Ω = pde.Ω
    ∂Ωs, ∂Ω_normals = normed_boundary(Ω)
    display(∂Ωs)
    display(∂Ω_normals)
    Iₓ = Symbolics.Integral(DomainSets.in(x, Ω))
    Sₙ = Symbolics.Summation(n ∈ DomainSets.ClosedInterval(1, N))
    Dₜ = Symbolics.Differential(t)
    Dₓ = Symbolics.Differential(x)
    Dₓ₂ = Symbolics.Differential(x, 2)

    eq = pde()
    display(eq)
    dump(open(joinpath("dump", "eq.txt"), "w"), eq)

    initial_condition = pde.ic.at

    # Transform Rules
    rule_IS = @acrule(Iₓ(+(~~xs)) => +(Iₓ.(~~xs)...))
    rule_IxDt2DtIx = @acrule(Iₓ(Dₜ(~f) * ~g::(e -> !any(depend_on.(t, e)))) => Dₜ(Iₓ(~f * ~g)))
    rule_IxDDxy2IxDDyx_0 = @acrule(Iₓ(*(~!a, Dₓ₂(u(~t, ~x)), Ψ(~n, ~x))) => Iₓ(*(~a, Dₓ₂(Ψ(~n, ~x)), u(~t, ~x))))
    rule_IxDDxy2IxDDyx_1 = @acrule(Iₓ(*(~!a0, Dₓ(*(~!a1, Dₓ(u(~t, ~x)))), Ψ(~n, ~x))) => Iₓ(*(~a0, Dₓ(*(~a1, Dₓ(Ψ(~n, ~x)))), u(~t, ~x))))
    rule_DDψ2λψ_0 = @acrule(Dₓ₂(Ψ(~n, ~x)) => λ(~n) * Ψ(~n, ~x))
    rule_DDψ2λψ_1 = @acrule(Dₓ(*(~!a, Dₓ(u(~t, ~x)))) => λ(~n) * Ψ(~n, ~x))
    rule_TransformF = @acrule(Iₓ(*(~!a::(e -> !any(depend_on.(x, e))), u(~t, ~x), Ψ(~n, ~x), ~~as::(e -> !any(depend_on.(x, e))))) => *(~a, θ(~n, ~t), ~~as...))
    rule_TransformI = @acrule(u(~t, ~x) => Sₙ(*(θ(n, ~t), Ψ(n, ~x))))
    rule_SAssociativity = @acrule(*(Sₙ(~f), ~~gs::(e -> !any(depend_on.(x, e)))) => Sₙ(*(~f, ~~gs...)))
    rule_IS2SI = @acrule(Iₓ(Sₙ(~f)) => Sₙ(Iₓ(~f)))

    rules_AT = [
        ("IS", SymbolicUtils.Postwalk(rule_IS)),
        ("IxDt2DtIx", SymbolicUtils.Postwalk(rule_IxDt2DtIx)),
        ("IxDDxy2IxDDyx_0", SymbolicUtils.Postwalk(rule_IxDDxy2IxDDyx_0)),
        ("IxDDxy2IxDDyx_1", SymbolicUtils.Postwalk(rule_IxDDxy2IxDDyx_1)),
        ("DDψ2λψ_0", SymbolicUtils.Postwalk(rule_DDψ2λψ_0)),
        ("DDψ2λψ_1", SymbolicUtils.Postwalk(rule_DDψ2λψ_1)),
        ("TransformF", SymbolicUtils.Postwalk(rule_TransformF)),
        ("TransformI", SymbolicUtils.Postwalk(rule_TransformI)),
        ("SAssociativity", SymbolicUtils.Postwalk(rule_SAssociativity)),
        ("SumIntegral", SymbolicUtils.Postwalk(rule_IS2SI)),
    ]

    transformed_form = apply(expand(Iₓ(eq * Ψ(n, x))), rules_AT)
    dump(open(joinpath("dump", "transformed_form.txt"), "w"), transformed_form)
    transformed_initial_condition = apply(expand(Iₓ(initial_condition * Ψ(n, x))), rules_AT)

    return transformed_form, transformed_initial_condition
end

function Solve(equation, initial_condition, eigenfunctions, eigenvalues)
    @variables n::Integer t x p θ(..) Θ(..)[1:length(eigenfunctions)] Ψ(..) λ(..)
    Dₜ = Symbolics.Differential(t)
    system = Symbolics.Num[]
    ics = Symbolics.Num[]
    for (i, (eigfun, eigval)) in enumerate(zip(eigenfunctions, eigenvalues))
        rule_eigenfun_i = @acrule(Ψ(~n, ~x) => eigfun)
        rule_eigenval_i = @acrule(λ(~n) => eigval)
        rule_function_i = @acrule(θ(~n, ~t) => Θ(t)[i])
        rules = Dict("Eigenfun" => SymbolicUtils.Postwalk(rule_eigenfun_i), "Eigenval" => SymbolicUtils.Postwalk(rule_eigenval_i), "Function" => SymbolicUtils.Postwalk(rule_function_i))
        eq = apply(equation, rules)
        ic = apply(initial_condition, rules)
        eq_lhs = Symbolics.coeff(eq, Dₜ(Θ(t)[i]))
        eq_rhs = Symbolics.expand((eq - eq_lhs * Dₜ(Θ(t)[i])) / eq_lhs)

        push!(system, eq_rhs)
        push!(ics, ic)

        @assert typeof(eq) <: Symbolics.Num "Transformed equation is not a symbolic expression. Got: $(typeof(eq))"
        @assert typeof(ic) <: Symbolics.Num "Transformed initial condition is not a symbolic expression. Got: $(typeof(ic))"
    end
    @assert typeof(system) <: Vector{Symbolics.Num} "System of equations is not a vector of symbolic expressions. Got: $(typeof(system))"
    @assert typeof(ics) <: Vector{Symbolics.Num} "Initial conditions is not a vector of symbolic expressions. Got: $(typeof(ics))"

    # Algebraic Solver of known integrals and numerical solver for unknown integrals
    system = expand_integrals.(system)
    ics = expand_integrals.(ics)

    @assert typeof(system) <: Vector{Symbolics.Num} "System of equations after expanding integrals is not a vector of symbolic expressions. Got: $(typeof(system))"
    @assert typeof(ics) <: Vector{Symbolics.Num} "Initial conditions after expanding integrals is not a vector of symbolic expressions. Got: $(typeof(ics))"

    display(system)
    display(ics)
    eq_expr = first(build_function(system, Θ(t), p, t; expression=Val(false)))
    ic_expr = first(build_function(ics; expression=Val(false)))

    display(eq_expr)
    display(ic_expr)

    tspan = (0.0, 1.0)
    prob = DifferentialEquations.ODEProblem(eq_expr, ic_expr(), tspan)
    result = DifferentialEquations.solve(prob, DifferentialEquations.Rosenbrock23())

    return result
end

function Recover(result, eigenfunctions, x_values)
    @variables n::Integer x
    Θ = Array{Float64}(undef, length(result.t), length(x_values))
    eigenfunctions = [build_function(eigfun, x; expression=Val(false)) for eigfun in eigenfunctions]
    for (i, t) in enumerate(result.t)
        for (j, x) in enumerate(x_values)
            Θ[i, j] = sum(result.u[i][n] * eigenfunctions[n](x) for n in 1:length(eigenfunctions))
        end
    end
    return Θ
end

function depend_on(var, expr)
    # 1. Unwrap Symbolics.Num → inner expression
    if expr isa Symbolics.Num
        return depend_on(var, Symbolics.unwrap(expr))
    end

    # 2. If it's exactly the same object, it depends
    if expr === Symbolics.unwrap(var)
        return true
    end

    # 3. Plain number → no dependence
    if expr isa Number
        return false
    end

    # 4. Arrays / tuples → depends if any element depends
    if expr isa AbstractArray || expr isa Tuple
        return any(e -> depend_on(var, e), expr)
    end

    # 5. SymbolicUtils term: check arguments recursively
    if SymbolicUtils.iscall(expr)
        return any(arg -> depend_on(var, arg), SymbolicUtils.arguments(expr))
    end

    # 6. Fallback: if it's some other kind of object, assume no dependence
    return false
end

#=
struct InitialCondition_1D
    at::Number
    eq::Symbolics.Equation

    var_t::Symbolics.Num
    var_x::Symbolics.Num
    op_u::Symbolics.CallAndWrap{Num}

    InitialCondition_1D(eq::Symbolics.Equation, var_t::Symbolics.Num, var_x::Symbolics.Num, op_u::Symbolics.CallAndWrap{Num}) = new(0.0, eq, var_t, var_x, op_u)
    InitialCondition_1D() = begin
        @variables t x u(..)
        return new(0.0, u(t, x) ~ 1, t, x, u)
    end
end

struct BoundaryCondition_1D
    at::Number
    eq::Symbolics.Equation

    var_t::Symbolics.Num
    var_x::Symbolics.Num
    op_u::Symbolics.CallAndWrap{Num}

    BoundaryCondition_1D(at::Number, eq::Symbolics.Equation, var_t::Symbolics.Num, var_x::Symbolics.Num, op_u::Symbolics.CallAndWrap{Num}) = new(at, eq, var_t, var_x, op_u)
    BoundaryCondition_1D(at::Number) = begin
        @variables t x u(..)
        return new(at, u(t, x) ~ 0, t, x, u)
    end
    BoundaryCondition_1D(at::Number, α::Symbolics.Num, β::Symbolics.Num, φ::Symbolics.Num, var_t::Symbolics.Num, var_x::Symbolics.Num, op_u::Symbolics.CallAndWrap{Num}) = begin
        eq = α * op_u(var_t, var_x) + β * Differential(var_x)(op_u(var_t, var_x)) ~ φ
        return new(at, eq, var_t, var_x, op_u)
    end
end

function get_op(var)
    return Symbolics.operation(Symbolics.value(var))
end

function at(x, expr, t::Number)
    rule_at = @rule(x => t)
    return SymbolicUtils.Postwalk(rule_at)(expr)
end

function isparallel(var1, var2)
    # Test if both are linearly dependent
    # Test if equal to zero
    if isequal(var1, 0) || isequal(var2, 0)
        return false
    end
    # If not zero, test if their ratio is constant
    ratio = simplify(expand_derivatives(var1 / var2))
    return isempty(Symbolics.get_variables(ratio))
end

struct PDE_1T2X
    t # Time variable
    x # Spatial variable
    u # Function

    Ω # Spatial domain

    w # coefficient of ∂(w*u)/∂t
    v # coefficient of ∂(v*u)/∂x
    k # coefficient of ∂(k*∂u/∂x)/∂x
    d # coefficient of u
    g # source term

    filter # Recovery filter for non-homogeneous boundary conditions

    initial_condition_α
    initial_condition_φ
    boundary_condition_left_α
    boundary_condition_left_β
    boundary_condition_left_φ
    boundary_condition_right_α
    boundary_condition_right_β
    boundary_condition_right_φ

    type::Symbol

    function PDE_1T2X(t::Symbolics.Num, x::Symbolics.Num, u::Symbolics.CallAndWrap{Num}, w, v, k, d, g, ic::InitialCondition_1D=InitialCondition_1D(), bc_left::BoundaryCondition_1D=BoundaryCondition_1D(0.0), bc_right::BoundaryCondition_1D=BoundaryCondition_1D(1.0))
        Dₜ = Differential(t)
        Dₓ = Differential(x)

        # Set default initial and boundary conditions if not provided
        initial_condition_α = Symbolics.coeff(ic.eq.lhs - ic.eq.rhs, ic.op_u(ic.var_t, ic.var_x))
        initial_condition_φ = initial_condition_α * u(t, x) - ic.eq.lhs + ic.eq.rhs

        Ω = IntervalSets.ClosedInterval(bc_left.at, bc_right.at)
        boundary_condition_left_α = Symbolics.coeff(bc_left.eq.lhs - bc_left.eq.rhs, bc_left.op_u(bc_left.var_t, bc_left.var_x))
        boundary_condition_left_β = Symbolics.coeff(bc_left.eq.lhs - bc_left.eq.rhs, Differential(bc_left.var_x)(bc_left.op_u(bc_left.var_t, bc_left.var_x)))
        boundary_condition_left_φ = boundary_condition_left_α * u(t, x) + boundary_condition_left_β * Dₓ(u(t, x)) - bc_left.eq.lhs + bc_left.eq.rhs
        boundary_condition_right_α = Symbolics.coeff(bc_right.eq.lhs - bc_right.eq.rhs, bc_right.op_u(bc_left.var_t, bc_left.var_x))
        boundary_condition_right_β = Symbolics.coeff(bc_right.eq.lhs - bc_right.eq.rhs, Differential(bc_right.var_x)(bc_right.op_u(bc_right.var_t, bc_right.var_x)))
        boundary_condition_right_φ = boundary_condition_right_α * u(t, x) + boundary_condition_right_β * Dₓ(u(t, x)) - bc_right.eq.lhs + bc_right.eq.rhs

        # Filtering of the variables in the boundary conditions
        # If boundary condtions are not zero and do not depend on x, create explicit filter
        if !isequal(boundary_condition_left_φ, 0) && !depend_on(x, boundary_condition_left_φ) && !isequal(boundary_condition_right_φ, 0) && !depend_on(x, boundary_condition_right_φ)
            @variables a(t) b(t)
            if isequal(boundary_condition_left_α, 0) && isequal(boundary_condition_right_α, 0)
                γ = (a * x + b) * x
            else
                γ = a * x + b
            end
            bc_left = at(x, expand_derivatives(boundary_condition_left_α * γ + boundary_condition_left_β * Differential(x)(γ) - boundary_condition_left_φ), bc_left.at) ~ 0
            bc_right = at(x, expand_derivatives(boundary_condition_right_α * γ + boundary_condition_right_β * Differential(x)(γ) - boundary_condition_right_φ), bc_right.at) ~ 0
            result = Symbolics.symbolic_linear_solve([bc_left, bc_right], [a, b])
            subs = [a => result[1], b => result[2]]
            filter = Symbolics.substitute(γ, subs)

            # Linear adjustment of initial and boundary conditions
            initial_condition_φ = expand_derivatives(initial_condition_φ - initial_condition_α * filter)
            boundary_condition_left_φ = expand_derivatives(boundary_condition_left_α * filter + boundary_condition_left_β * Differential(x)(filter))
            boundary_condition_right_φ = expand_derivatives(boundary_condition_right_α * filter + boundary_condition_right_β * Differential(x)(filter))
        else
            filter = 0
        end

        return new(t, x, u, Ω, w, v, k, d, g, filter, initial_condition_α, initial_condition_φ, boundary_condition_left_α, boundary_condition_left_β, boundary_condition_left_φ, boundary_condition_right_α, boundary_condition_right_β, boundary_condition_right_φ, :Math)
    end
end

function (pde::PDE_1T2X)()
    Dₜ = Differential(pde.t)
    Dₓ = Differential(pde.x)

    # Filter adjustments for non-homogeneous boundary conditions
    dict = if pde.type == :Engineer
        _dict_engineer
    elseif pde.type == :Math
        _dict_math
    else
        error("Unknown PDE type: $(pde.type). Supported types are :Engineer and :Math.")
    end
    eq = dict[:w] * Dₜ(pde.w * pde.u(pde.t, pde.x)) + dict[:v] * Dₓ(pde.v * pde.u(pde.t, pde.x)) + dict[:k] * Dₓ(pde.k * Dₓ(pde.u(pde.t, pde.x))) + dict[:d] * pde.d * pde.u(pde.t, pde.x) + dict[:g] * pde.g
    filtered_eq = Symbolics.wrap(Symbolics.expand(Symbolics.expand_derivatives(Symbolics.substitute(Symbolics.value(eq), pde.u(pde.t, pde.x) => pde.u(pde.t, pde.x) + pde.filter))))
    return filtered_eq
end

function Eigenproblem(var_n, var_x, op_λ, op_Ψ, p, q, w, Ω, bc_left_α, bc_left_β, bc_right_α, bc_right_β)
    # Define the eigenproblem
    Dₓ = Differential(var_x)
    term = Dₓ(p * Dₓ(op_Ψ(var_n, var_x))) + (op_λ(var_n) * w - q) * op_Ψ(var_n, var_x)

    # Define homogeneous boundary conditions
    bc_left_expr = at(var_x, Symbolics.value(bc_left_α * op_Ψ(var_n, var_x) + bc_left_β * Dₓ(op_Ψ(var_n, var_x))), Ω.left) ~ 0
    bc_right_expr = at(var_x, Symbolics.value(bc_right_α * op_Ψ(var_n, var_x) + bc_right_β * Dₓ(op_Ψ(var_n, var_x))), Ω.right) ~ 0

    return term, bc_left_expr, bc_right_expr
end

function Transform(pde::PDE_1T2X)
    # Transform terms in system of differential equations
    t = pde.t
    x = pde.x
    u = pde.u
    @variables n::Integer N::Integer m::Integer Θ(..) Ψ(..) λ(..)
    Ω = IntervalSets.ClosedInterval(0.0, 1.0)
    ∂Ω = IntervalSets.ClosedInterval(0.0, 1.0)
    Iₓ = Symbolics.Integral(x ∈ Ω)
    CIₓ = Symbolics.BoundaryIntegral(x ∈ ∂Ω)
    Sₙ = Symbolics.Summation(n ∈ 1:N)
    Aₗ = Symbolics.At(x ∈ Point(infimum(Ω)))
    Aᵣ = Symbolics.At(x ∈ Point(supremum(Ω)))
    Dₜ = Symbolics.Differential(t)
    Dₓ = Symbolics.Differential(x)
    D₂ₓ = Symbolics.Differential(x, 2)

    eq = pde()
    w_eq = Symbolics.coeff(eq, Dₜ(u(t, x)))
    v_eq = Symbolics.coeff(eq, Dₓ(u(t, x)))
    k_eq = Symbolics.coeff(eq, D₂ₓ(u(t, x)))
    d_eq = Symbolics.coeff(eq, u(t, x))
    g_eq = Symbolics.wrap(Symbolics.simplify(Symbolics.value(eq - w_eq * Dₜ(u(t, x)) - v_eq * Dₓ(u(t, x)) - k_eq * D₂ₓ(u(t, x)) - d_eq * u(t, x)), expand=true))
    coeffs = (w_eq, v_eq, k_eq, d_eq, g_eq)
    display(coeffs)

    initial_condition = pde.initial_condition_α * u(t, x) + pde.initial_condition_φ

    # Canonical form of the PDE
    canonical_terms = Dict{Symbol,Symbolics.Num}(
        :w => Dₜ(w_eq * u(t, x)),
        :v => Dₓ(Symbolics.expand_derivatives(v_eq - Dₓ(k_eq)) * u(t, x)),
        :k => Dₓ(k_eq * Dₓ(u(t, x))),
        :d => Symbolics.expand_derivatives(d_eq - Dₜ(w_eq) - Dₓ(v_eq - Dₓ(k_eq))) * u(t, x),
        :g => g_eq
    )
    canonical_form() = begin
        expr = zero(Symbolics.Num)
        for (_, term) in canonical_terms
            expr += term
        end
        return expr
    end
    eq = canonical_form()
    coeffs = (w_eq, Symbolics.expand_derivatives(v_eq - Dₓ(k_eq)), k_eq, Symbolics.expand_derivatives(d_eq - Dₜ(w_eq) - Dₓ(v_eq - Dₓ(k_eq))), g_eq)
    display(coeffs)
    display(typeof(eq))

    @variables kₑ dₑ wₑ
    eig_eq, eig_bc_left_expr, eig_bc_right_expr = Eigenproblem(n, x, λ, Ψ, kₑ, dₑ, wₑ, Ω, pde.boundary_condition_left_α, pde.boundary_condition_left_β, pde.boundary_condition_right_α, pde.boundary_condition_right_β)


    display(eig_eq)
    display(eig_bc_left_expr)
    display(eig_bc_right_expr)

    # Transform Rules
    rule_Distribution = @acrule(Iₓ(*(+(~~xs), Ψ(~~y))) => +(map(xi -> Iₓ(*(xi, Ψ(~~y...))), ~~xs)...))
    # Temporal part
    rule_TimeDerivative = @acrule(Iₓ(*(Dₜ(*(~!a, u(~~x))), Ψ(~~y))) => Dₜ(Iₓ(*(~a, u(~~x...), Ψ(~~y...)))))
    # Advection part
    rule_Advection = @acrule(Iₓ(*(Dₓ(*(~!a, u(~~x))), Ψ(~~y))) => -Iₓ(*(~a, u(~~x...), Dₓ(Ψ(~~y...)))) + CIₓ(*(~a, u(~~x...), Ψ(~~y...))))
    # Diffusion part
    rule_Diffusion = @acrule(Iₓ(*(Dₓ(*(~!a, Dₓ(u(~~x)))), Ψ(~~y))) => Iₓ(*(Dₓ(*(~a, Dₓ(Ψ(~~y...)))), u(~~x...))) + CIₓ(*(Dₓ(*(~a, u(~~x...))), Ψ(~~y...)) - *(Dₓ(*(~a, Ψ(~~y...))), u(~~x...))))
    # Boundary Conditions part
    rule_DiffusionBoundaryCondition = @acrule(CIₓ(*(Dₓ(*(~!a, u(~~x))), Ψ(~~y)) + *(-1, Dₓ(*(~!a, Ψ(~~y))), u(~~x))) =>
        ifelse(pde.boundary_condition_left_β == 0, A(x, -Dₓ(Ψ(~~y...) * pde.boundary_condition_left_φ / pde.boundary_condition_left_α), Ω.left), A(x, -Ψ(~~y...) * pde.boundary_condition_left_φ / (pde.boundary_condition_left_β * ~a), Ω.left))
        +
        ifelse(pde.boundary_condition_right_β == 0, A(x, Dₓ(Ψ(~~y...) * pde.boundary_condition_right_φ / pde.boundary_condition_right_α), Ω.right), A(x, Ψ(~~y...) * pde.boundary_condition_right_φ / (pde.boundary_condition_right_β * ~a), Ω.right)))

    rule_AdvectionBoundaryCondition = @acrule(CIₓ(*(~!a, u(~~x), Ψ(~~y))) => A(x, *(-1, ~a, u(~~x...), Ψ(~~y...)), Ω.left) + A(x, *(~a, u(~~x...), Ψ(~~y...)), Ω.right))
    # Eigenproblem contribution part
    rule_Equivalent = @acrule(Iₓ(*(u(~~x), Dₓ(*(~!a, Dₓ(Ψ(~~y)))))) => Iₓ(*(u(~~x...), Dₓ(*(~a - kₑ, Dₓ(Ψ(~~y...)))))) + Iₓ(*(u(~~x...), Dₓ(*(kₑ, Dₓ(Ψ(~~y...)))))))
    rule_EquivalentEigenproblem = @acrule(Iₓ(*(u(~~x), Dₓ(*(~!a::(e -> isparallel(e, coeffs[3])), Dₓ(Ψ(~~y)))))) => Tag(:IntegrationSum, expand(Iₓ(*(kₑ / coeffs[3], u(~~x...), eig_eq - Dₓ(*(~a, Dₓ(Ψ(~~y...)))))))))
    # Integration Distribution
    rule_IntegrationSum = @acrule(Tag(:IntegrationSum, Iₓ(+(~~xs))) => +(map(xi -> Iₓ(xi), ~~xs)...))
    rule_NoIntegrationSum = @acrule(Tag(:IntegrationSum, ~x) => ~x)
    # Zero boundary
    rule_BCZero = @acrule(CIₓ(0) => 0)
    # Main Transform Rule
    rule_TransformI = @acrule(u(~t, ~x) => S(m, *(Θ(m, ~t), Ψ(m, ~x)), 1, N))
    # Summation Rules
    rule_Summation = @acrule(*(S(~i, ~x, ~a, ~b), ~~xs) => S(~i, *(~x, ~~xs...), ~a, ~b))
    rule_SummationIntegration = @rule(Iₓ(S(~i, ~x, ~a, ~b)) => S(~i, Iₓ(~x), ~a, ~b))
    # Summation Integration properties
    rule_LinearIndependent = @acrule(Iₓ(*(~!a::(e -> isparallel(e, coeffs[1])), Ψ(~i, ~y), Ψ(~j, ~y), ~~xs::(e -> !any(depend_on.(x, e))))) => *(~a * δ(~i, ~j) / coeffs[1], ~~xs...))
    rule_LinearDependent = @acrule(S(~i, *(~!c, Iₓ(*(~d::(e -> !depend_on(x, e)), ~~xs))), ~a, ~b) => S(~i, *(~c, ~d, Iₓ(*(~~xs...))), ~a, ~b))
    # Delta Collapse
    rule_DeltaLeft = @acrule(S(~i, *(~!a, δ(~j, ~i), Θ(~i, ~y)), ~b, ~c) => *(~a, Θ(~j, ~y)))
    rule_DeltaRight = @acrule(S(~i, *(~!a, δ(~i, ~j), Θ(~i, ~y)), ~b, ~c) => *(~a, Θ(~j, ~y)))
    # Power Summation Expand
    rule_PowerSummationExpand = @acrule(S(~i, *(~!a, S(~j, ~x, ~d, ~e), ~~xs), ~b, ~c) => S(~j, S(~i, *(~a, ~x, ~~xs...), ~b, ~c), ~d, ~e))

    # Invert At Summation
    rule_InvertAtSummation = @acrule(A(~x, S(~i, ~y, ~a, ~b), ~x_val) => S(~i, A(~x, ~y, ~x_val), ~a, ~b))

    rules_BT = [
        ("Distribution", SymbolicUtils.Postwalk(rule_Distribution)),
        ("TimeDerivative", SymbolicUtils.Postwalk(rule_TimeDerivative)),
        ("Advection", SymbolicUtils.Postwalk(rule_Advection)),
        ("Diffusion", SymbolicUtils.Postwalk(rule_Diffusion)),
        ("DiffusionBoundaryCondition", SymbolicUtils.Postwalk(rule_DiffusionBoundaryCondition)),
        ("AdvectionBoundaryCondition", SymbolicUtils.Postwalk(rule_AdvectionBoundaryCondition)),
        ("BCZero", SymbolicUtils.Postwalk(rule_BCZero)),
        ("Equivalent", SymbolicUtils.Postwalk(rule_Equivalent)),
        ("EquivalentEigenproblem", SymbolicUtils.Postwalk(rule_EquivalentEigenproblem)),
        ("IntegrationSum", SymbolicUtils.Postwalk(rule_IntegrationSum)),
        ("NoIntegrationSum", SymbolicUtils.Postwalk(rule_NoIntegrationSum))
    ]
    rules_AT = [
        ("TransformI", SymbolicUtils.Postwalk(rule_TransformI)),
        ("Summation", SymbolicUtils.Fixpoint(SymbolicUtils.Postwalk(rule_Summation))),
        ("SummationIntegration", SymbolicUtils.Postwalk(rule_SummationIntegration)),
        ("LinearIndependent", SymbolicUtils.Postwalk(rule_LinearIndependent)),
        ("LinearDependent", SymbolicUtils.Fixpoint(SymbolicUtils.Postwalk(rule_LinearDependent))),
        ("DeltaLeft", SymbolicUtils.Postwalk(rule_DeltaLeft)),
        ("DeltaRight", SymbolicUtils.Postwalk(rule_DeltaRight)),
        ("InvertAtSummation", SymbolicUtils.Postwalk(rule_InvertAtSummation))
    ]

    function apply(expr, rules)
        result = expr
        for (name, rw) in rules
            display("Applying rule: $name")
            result_new = Symbolics.wrap(rw(Symbolics.value(result)))
            result = result_new
            display(result)
        end
        return Symbolics.wrap(result)
    end
    pre_form = apply(Iₓ(eq * Ψ(n, x)), rules_BT)
    pre_initial_condition = apply(Iₓ(initial_condition * Ψ(n, x)), rules_BT)
    display(pre_form)
    display(pre_initial_condition)
    transformed_form = apply(pre_form, rules_AT)
    transformed_initial_condition = apply(pre_initial_condition, rules_AT)

    # For now, there is two things to do:
    # 1. Solve powers of u in the transformed problem as products of summations appears and they generate tensors of the degree of the power.

    # Prepare the numerical integration for terms that weren't symbolically solved
    rules_generation = [
        ("NumericalIntegral", SymbolicUtils.Postwalk(@rule(Iₓ(~x) => NI(x, ~x, Ω.left, Ω.right))))
    ]

    numerical_transformed_form = apply(transformed_form, rules_generation)
    numerical_transformed_initial_condition = apply(transformed_initial_condition, rules_generation)

    problem = tuple(numerical_transformed_form, numerical_transformed_initial_condition)
    eigenproblem = (eig_eq, eig_bc_left_expr, eig_bc_right_expr)
    config = Dict(
        :var_n => n,
        :var_N => N,
        :var_t => t,
        :var_x => x,
        :op_Θ => Θ,
        :op_Ψ => Ψ,
        :op_λ => λ,
        :Ω => Ω
    )

    return problem, eigenproblem, config
end

function Transform_array(pde::PDE_1T2X, N::Integer)
    # Transform terms in system of differential equation
    t = pde.t
    x = pde.x
    u = pde.u
    @variables Θ[1:N] Ψ[1:N] λ[1:N]
    Ω = IntervalSets.ClosedInterval(0.0, 1.0)
    ∂Ω = IntervalSets.ClosedInterval(0.0, 1.0)
    Iₓ = Symbolics.Integral(x ∈ Ω)
    CIₓ = Symbolics.BoundaryIntegral(x ∈ ∂Ω)
    NI = NIntegral()
    S = Symbolics.Summation()
    A = At()
    Dₜ = Symbolics.Differential(t)
    Dₓ = Symbolics.Differential(x)
    D₂ₓ = Symbolics.Differential(x, 2)

    eq = pde()
    w_eq = Symbolics.coeff(eq, Dₜ(u(t, x)))
    v_eq = Symbolics.coeff(eq, Dₓ(u(t, x)))
    k_eq = Symbolics.coeff(eq, D₂ₓ(u(t, x)))
    d_eq = Symbolics.coeff(eq, u(t, x))
    g_eq = Symbolics.wrap(Symbolics.simplify(Symbolics.value(eq - w_eq * Dₜ(u(t, x)) - v_eq * Dₓ(u(t, x)) - k_eq * D₂ₓ(u(t, x)) - d_eq * u(t, x)), expand=true))
    coeffs = (w_eq, v_eq, k_eq, d_eq, g_eq)
    display(coeffs)

    initial_condition = pde.initial_condition_α * u(t, x) + pde.initial_condition_φ

    # Canonical form of the PDE
    canonical_terms = Dict{Symbol,Symbolics.Num}(
        :w => Dₜ(w_eq * u(t, x)),
        :v => Dₓ(Symbolics.expand_derivatives(v_eq - Dₓ(k_eq)) * u(t, x)),
        :k => Dₓ(k_eq * Dₓ(u(t, x))),
        :d => Symbolics.expand_derivatives(d_eq - Dₜ(w_eq) - Dₓ(v_eq - Dₓ(k_eq))) * u(t, x),
        :g => g_eq
    )
    canonical_form() = begin
        expr = zero(Symbolics.Num)
        for (_, term) in canonical_terms
            expr += term
        end
        return expr
    end
    eq = canonical_form()
    coeffs = (w_eq, Symbolics.expand_derivatives(v_eq - Dₓ(k_eq)), k_eq, Symbolics.expand_derivatives(d_eq - Dₜ(w_eq) - Dₓ(v_eq - Dₓ(k_eq))), g_eq)
    display(coeffs)
    display(typeof(eq))

    #eig_eq, eig_bc_left_expr, eig_bc_right_expr = Eigenproblem(n, x, λ, Ψ, coeffs[3], coeffs[4], coeffs[1], Ω, pde.boundary_condition_left_α, pde.boundary_condition_left_β, pde.boundary_condition_right_α, pde.boundary_condition_right_β)

    #display(eig_eq)
    #display(eig_bc_left_expr)
    #display(eig_bc_right_expr)

    # Residual form (for each direction)
    eqs = collect(Symbolics.@arrayop (i,) Iₓ(eq * Ψ[i]))

    # Transform Rules
    # Transformation u => Θ' * Ψ

    apply(expr, rw) = Symbolics.wrap.(rw.(Symbolics.value.(expr)))
    eqs = apply(eqs, SymbolicUtils.Postwalk(rule_transform))
    display(eqs)
end

# integral2quadgk(expr) = first(quadgk(eval(Symbolics.build_function(expr, x; expression=Val(false))), Ω.left, Ω.right))

function Solve(problem, config, eigenvalues, eigenfunctions, L::Integer)
    # Assertions over eigenvalues and eigenfunctions
    # Normalize eigenfunctions
    norms = Vector{Float64}(undef, L)
    for i ∈ 1:L
        norms[i] = sqrt(quadgk(x -> eigenfunctions(i, x)^2, 0.0, 1.0)[1])
    end

    # Unpack problem
    eq, ic = problem
    # Build transformed system of ODEs
    n = config[:var_n]
    N = config[:var_N]
    t = config[:var_t]
    x = config[:var_x]
    Θ = config[:op_Θ]
    Ψ = config[:op_Ψ]
    λ = config[:op_λ]
    Ω = config[:Ω]
    Dₜ = Symbolics.Differential(t)
    eqs = Vector{Symbolics.Num}(undef, L)
    ics = Vector{Symbolics.Num}(undef, L)
    eq_lhs = Symbolics.coeff(eq, Dₜ(Θ(n, t)))
    ic_lhs = Symbolics.coeff(ic, Θ(n, t))
    eq_rhs = (eq - eq_lhs * Dₜ(Θ(n, t))) / eq_lhs
    ic_rhs = (ic - ic_lhs * Θ(n, t)) / ic_lhs

    @variables p Θ_array[1:L]

    #TODO: Implement rule to substitute the eigenfunctions 
    Ind = Indexer()
    rw_instancing(i::Integer) = SymbolicUtils.Chain([
        SymbolicUtils.Postwalk(@rule(n => i)),
        SymbolicUtils.Postwalk(@rule(N => L)),
        SymbolicUtils.Postwalk(@rule(Θ(~n, ~t) => Ind(Θ_array, ~n))),
        SymbolicUtils.Postwalk(@rule(λ(~n) => eigenvalues(~n))),
        SymbolicUtils.Postwalk(@rule(Ψ(~n, ~x) => eigenfunctions(~n, x) / norms[~n])),
    ])

    #TODO: Create toexpr of NIntegral that evaluates the integral numerically
    for i ∈ 1:L
        rw = rw_instancing(i)
        intance_eq = rw(Symbolics.value(eq_rhs))
        intance_ic = rw(Symbolics.value(ic_rhs))
        eqs[i] = Symbolics.wrap(expand_derivatives(intance_eq))
        ics[i] = Symbolics.wrap(expand_derivatives(intance_ic))
    end
    for i in 1:L
        display("Equation $i:")
        display(eqs[i])
        display(typeof(eqs[i]))
        eqs[i] = pre_build(eqs[i])
        ics[i] = pre_build(ics[i])
    end

    # Test singular behavior
    first_eq = eqs[1]
    first_f_expr = build_function(first_eq, Θ_array, p, t; expression=Val(false))
    display(first_f_expr)

    #TODO: eqs and ics might have variables that are not in Θ_array, p or t, need to substitute them with numerical values
    # Solve transformed system of ODEs
    f_expr = first(build_function(eqs, Θ_array, p, t; expression=Val(false)))
    ic_expr = first(build_function(ics; expression=Val(false)))

    tspan = (0.0, 1.0)
    prob = DifferentialEquations.ODEProblem(f_expr, ic_expr(), tspan)
    result = DifferentialEquations.solve(prob, DifferentialEquations.Rosenbrock23())
    return result
end

function Recover(pde, result, eigenfunctions, x_values)
    Θ = Array{Float64}(undef, length(result.t), length(x_values))
    filter = eval(build_function(pde.filter, pde.t, pde.x; expression=Val(false)))
    for (i, t) in enumerate(result.t)
        for (j, x) in enumerate(x_values)
            Θ[i, j] = sum(result.u[i][n] * eigenfunctions(n, x) for n in 1:length(result.u)) + filter(t, x)
        end
    end
    return Θ
end

=#
end # module GITT
