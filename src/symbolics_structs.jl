# Structures to extend the usage of Symbolics.jl and SymbolicUtils.jl
import Symbolics
import SymbolicUtils
import SymbolicIntegration

function expand_integrals(expr)
    if !istree(expr)
        return expr
    else
        op = SymbolicUtils.operation(expr)
        args = map(expand_integrals, SymbolicUtils.arguments(expr))
        if op isa Integral
            pair = op.domain
            a = Symbolics.infimum(pair.domain)
            b = Symbolics.supremum(pair.domain)
            x = pair.variables
            f = args[1]
            # Try integrate symbolically
            si = SymbolicIntegration.integrate(f, x)
            if si !== nothing
                sup = Symbolics.substitute(si, Dict(x => b))
                inf = Symbolics.substitute(si, Dict(x => a))
                return sup - inf
            else
                return SymbolicUtils.Term{Symbolics.VartypeT}(op, args; type=Symbolics.symtype(expr), shape=Symbolics.shape(expr))
            end
        else
            return SymbolicUtils.Term{Symbolics.VartypeT}(op, args; type=Symbolics.symtype(expr), shape=Symbolics.shape(expr))
        end
    end
end

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

function search_variables!(buffer, expr::_Indexer; is_atomic::F=default_is_atomic, recurse::G=iscall) where {F,G}
    push!(buffer, expr.index)
    push!(buffer, expr.array)
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
