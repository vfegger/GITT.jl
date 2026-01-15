# Structures to extend the usage of Symbolics.jl and SymbolicUtils.jl
import Symbolics
import SymbolicUtils

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
