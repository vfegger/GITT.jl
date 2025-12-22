module GITT

using Symbolics
import IntervalSets
import SymbolicUtils
import SymbolicUtils.Code: toexpr, search_variables!
import DifferentialEquations
import QuadGK: quadgk

export PDE_1T2X, Eigenproblem, Transform, Solve, InitialCondition_1D, BoundaryCondition_1D, Recover

export Transform_array

export NIntegral, At, pre_build, quadgk

#TODO: Progressive and Implicit Filtering of the variables in the boundary

#TODO: Implement Delta function to change variables throughout a given expression

struct _Indexer
    array
    index
    _Indexer(array, index) = new(array, index)
end
Base.nameof(::_Indexer) = :_Indexer

function (I::_Indexer)()
    return SymbolicUtils.Term{Symbolics.VartypeT}(I, []; type=SymbolicUtils.symtype(I.index), shape=SymbolicUtils.shape(I.index))
end
SymbolicUtils.promote_symtype(I::_Indexer) = I.index

struct Indexer
    Indexer() = new()
end
Base.nameof(::Indexer) = :Indexer

function (I::Indexer)(array, index)
    num_cond = (array isa Num) || (index isa Num)
    array_var = (array isa Num) ? Symbolics.value(array) : array
    index_var = (index isa Num) ? Symbolics.value(index) : index
    return (num_cond) ?
           Symbolics.Num(SymbolicUtils.Term{Symbolics.VartypeT}(I, [array_var, index_var]; type=SymbolicUtils.symtype(index), shape=SymbolicUtils.shape(index))) :
           SymbolicUtils.Term{Symbolics.VartypeT}(I, [array_var, index_var]; type=SymbolicUtils.symtype(index), shape=SymbolicUtils.shape(index))
end
SymbolicUtils.promote_symtype(::Indexer, array, index) = index

struct _Summation
    x
    f
    _Summation(x, f) = new(x, f)
end
Base.nameof(::_Summation) = :_Summation

function (S::_Summation)(lower, upper)
    num_cond = (lower isa Num) || (upper isa Num)
    return (num_cond) ?
           Symbolics.Num(SymbolicUtils.Term{Symbolics.VartypeT}(S, [lower, upper]; type=SymbolicUtils.symtype(S.f), shape=SymbolicUtils.shape(S.f))) :
           SymbolicUtils.Term{Symbolics.VartypeT}(S, [lower, upper]; type=SymbolicUtils.symtype(S.f), shape=SymbolicUtils.shape(S.f))
end
SymbolicUtils.promote_symtype(S::_Summation, lower, upper) = S.f

struct _NIntegral
    x
    f
    _NIntegral(x, f) = new(x, f)
end
Base.nameof(::_NIntegral) = :_NIntegral

function (I::_NIntegral)(lower, upper)
    num_cond = (lower isa Num) || (upper isa Num)
    return (num_cond) ?
           Symbolics.Num(SymbolicUtils.Term{Symbolics.VartypeT}(I, [lower, upper]; type=SymbolicUtils.symtype(I.f), shape=SymbolicUtils.shape(I.f))) :
           SymbolicUtils.Term{Symbolics.VartypeT}(I, [lower, upper]; type=SymbolicUtils.symtype(I.f), shape=SymbolicUtils.shape(I.f))
end
SymbolicUtils.promote_symtype(I::_NIntegral, lower, upper) = I.f

struct NIntegral
    NIntegral() = new()
end
Base.nameof(::NIntegral) = :NIntegral

function (I::NIntegral)(x, f::Union{Rational,AbstractIrrational,AbstractFloat,Integer}, a, b)
    return wrap((b - a) * f)
end
function (I::NIntegral)(x, f, lower, upper)
    num_cond = (x isa Num) || (f isa Num) || (lower isa Num) || (upper isa Num)
    x_var = (x isa Num) ? Symbolics.value(x) : x
    f_var = (f isa Num) ? Symbolics.value(f) : f
    lower_var = (lower isa Num) ? Symbolics.value(lower) : lower
    upper_var = (upper isa Num) ? Symbolics.value(upper) : upper
    return (num_cond) ?
           Symbolics.Num(SymbolicUtils.Term{Symbolics.VartypeT}(I, [x_var, f_var, lower_var, upper_var]; type=SymbolicUtils.symtype(f_var), shape=SymbolicUtils.shape(f_var))) :
           SymbolicUtils.Term{Symbolics.VartypeT}(I, [x_var, f_var, lower_var, upper_var]; type=SymbolicUtils.symtype(f_var), shape=SymbolicUtils.shape(f_var))
end
SymbolicUtils.promote_symtype(::NIntegral, x, f, a, b) = f

function Base.show(io::IO, ::NIntegral)
    print(io, "NumIntegral")
end
function Base.:(==)(::NIntegral, ::NIntegral)
    return true
end

struct _At
    x
    f
    _At(x, f) = new(x, f)
end
Base.nameof(::_At) = :_At

function (A::_At)(x_val)
    return (x_val isa Num) ?
           Symbolics.Num(SymbolicUtils.Term{Symbolics.VartypeT}(A, [x_val]; type=SymbolicUtils.symtype(A.f), shape=SymbolicUtils.shape(A.f))) :
           SymbolicUtils.Term{Symbolics.VartypeT}(A, [x_val]; type=SymbolicUtils.symtype(A.f), shape=SymbolicUtils.shape(A.f))
end
SymbolicUtils.promote_symtype(::_At, T) = T

struct At
    At() = new()
end
Base.nameof(::At) = :At

function (A::At)(x, f::Union{Rational,AbstractIrrational,AbstractFloat,Integer}, x_val)
    return f
end

function (A::At)(x, f, x_val)
    num_cond = (x isa Num) || (f isa Num) || (x_val isa Num)
    x_var = (x isa Num) ? Symbolics.value(x) : x
    f_var = (f isa Num) ? Symbolics.value(f) : f
    x_val_var = (x_val isa Num) ? Symbolics.value(x_val) : x_val
    return (num_cond) ?
           Symbolics.Num(SymbolicUtils.Term{Symbolics.VartypeT}(A, [x_var, f_var, x_val_var]; type=SymbolicUtils.symtype(f_var), shape=SymbolicUtils.shape(f_var))) :
           SymbolicUtils.Term{Symbolics.VartypeT}(A, [x_var, f_var, x_val_var]; type=SymbolicUtils.symtype(f_var), shape=SymbolicUtils.shape(f_var))
end
SymbolicUtils.promote_symtype(::At, x, f, x_val) = f

function Base.show(io::IO, A::At)
    print(io, "At")
end
function Base.:(==)(::At, ::At)
    return true
end

function search_variables!(buffer, expr::_Indexer; is_atomic::F=default_is_atomic, recurse::G=iscall) where {F,G}
    push!(buffer, expr.index)
    push!(buffer, expr.array)
end

function search_variables!(buffer, expr::Union{_Summation,_NIntegral,_At}; is_atomic::F=default_is_atomic, recurse::G=iscall) where {F,G}
    buffer_integrand = Set{eltype(buffer)}()
    search_variables!(buffer_integrand, expr.f; is_atomic, recurse)
    for v in buffer_integrand
        if !(v in buffer) && (v !== expr.x)
            push!(buffer, v)
        end
    end
end

function toexpr(ex::_Indexer, st)
    A_sym = ex.array
    I_sym = ex.index
    A_expr = toexpr(A_sym, st)
    I_expr = toexpr(I_sym, st)
    return quote
        () -> getindex($(A_expr), $(I_expr))
    end
end

function toexpr(ex::_Summation, st)
    x_sym = ex.x
    f_sym = ex.f
    vars_sym = SymbolicUtils.search_variables(f_sym)
    other_sym = filter(v -> v !== x_sym, vars_sym)
    all_vars_sym = [x_sym; other_sym...]
    f_expr = build_function(f_sym, all_vars_sym...; expression=Val(false))

    a = gensym(:a)
    b = gensym(:b)
    s = gensym(:s)
    g = gensym(:g)
    i = gensym(:i)

    if isempty(other_sym)
        return quote
            let $g = $f_expr
                ($a, $b) -> begin
                    $s = zero(typeof($g($a)))
                    for $i in $a:$b
                        $s += $g($i)
                    end
                    return $s
                end
            end
        end
    else
        other_expr = [toexpr(o, st) for o in other_sym]
        return quote
            let $g = $f_expr
                ($a, $b) -> begin
                    $s = zero(typeof($g($a, $(other_expr...))))
                    for $i in $a:$b
                        $s += $g($i, $(other_expr...))
                    end
                    return $s
                end
            end
        end
    end
end

function toexpr(ex::_NIntegral, st)
    x_sym = ex.x
    f_sym = ex.f
    vars_sym = SymbolicUtils.search_variables(f_sym)
    other_sym = filter(v -> v !== x_sym, vars_sym)
    all_vars_sym = [x_sym; other_sym...]
    f_expr = build_function(f_sym, all_vars_sym...; expression=Val(false))

    a = gensym(:a)
    b = gensym(:b)
    x = gensym(:x)
    g = gensym(:g)
    other_expr = [toexpr(o, st) for o in other_sym]

    qgk = quadgk

    if isempty(other_sym)
        return quote
            let $g = $f_expr, qgk = $qgk
                ($a, $b) -> first(qgk($x -> $g($x), $a, $b))
            end
        end
    else
        other_expr = [toexpr(o, st) for o in other_sym]
        return quote
            let $g = $f_expr, qgk = $qgk
                ($a, $b) -> first(qgk($x -> $g($x, $(other_expr...)), $a, $b))
            end
        end
    end
end

function toexpr(ex::_At, st)
    x_sym = ex.x
    f_sym = ex.f
    vars_sym = SymbolicUtils.search_variables(f_sym)
    other_sym = filter(v_sym -> v_sym !== x_sym, vars_sym)
    all_vars_sym = [x_sym; other_sym...]
    f_expr = build_function(f_sym, all_vars_sym...; expression=Val(false))

    t = gensym(:t)

    if isempty(other_sym)
        return quote
            let g = $f_expr
                $t -> g($t)
            end
        end
    else
        other_expr = [toexpr(o, st) for o in other_sym]
        return quote
            let g = $f_expr
                $t -> g($t, $(other_expr...))
            end
        end
    end
end

# This function changes the symbolics callable structures like NIntegral and At into their rigid struct counterparts for calculation purposes
function pre_build(ex::Num)
    res = Symbolics.wrap(pre_build(Symbolics.value(ex)))
    return res
end
function pre_build(ex)
    if !istree(ex)
        return ex
    else
        if SymbolicUtils.operation(ex) isa Symbolics.Summation
            args = map(pre_build, SymbolicUtils.arguments(ex))
            f = _Summation(args[1], args[2])
            x = [args[3], args[4]]
            return SymbolicUtils.Term{Symbolics.VartypeT}(f, x; type=SymbolicUtils.symtype(args[2]), shape=SymbolicUtils.shape(args[2]))
        elseif SymbolicUtils.operation(ex) isa NIntegral
            args = map(pre_build, SymbolicUtils.arguments(ex))
            f = _NIntegral(args[1], args[2])
            x = [args[3], args[4]]
            return SymbolicUtils.Term{Symbolics.VartypeT}(f, x; type=SymbolicUtils.symtype(args[2]), shape=SymbolicUtils.shape(args[2]))
        elseif SymbolicUtils.operation(ex) isa At
            args = map(pre_build, SymbolicUtils.arguments(ex))
            f = _At(args[1], args[2])
            x = [args[3]]
            return SymbolicUtils.Term{Symbolics.VartypeT}(f, x; type=SymbolicUtils.symtype(args[2]), shape=SymbolicUtils.shape(args[2]))
        elseif SymbolicUtils.operation(ex) isa Indexer
            args = map(pre_build, SymbolicUtils.arguments(ex))
            f = _Indexer(args[1], args[2])
            return SymbolicUtils.Term{Symbolics.VartypeT}(f, []; type=SymbolicUtils.symtype(args[2]), shape=SymbolicUtils.shape(args[2]))
        else
            op = SymbolicUtils.operation(ex)
            args = map(pre_build, SymbolicUtils.arguments(ex))
            return SymbolicUtils.Term{Symbolics.VartypeT}(op, args; type=SymbolicUtils.symtype(ex), shape=SymbolicUtils.shape(ex))
        end
    end
end

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
    rule_Equivalent = @acrule(Iₓ(*(u(~~x), Dₓ(*(~!a, Dₓ(Ψ(~~y)))))) => Iₓ(*(u(~~x...), Dₓ(*(~a - kₑ, Dₓ(Ψ(~~y...)))))) + Iₓ(*(u(~~x... ), Dₓ(*(kₑ, Dₓ(Ψ(~~y...)))))))
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

end # module GITT
