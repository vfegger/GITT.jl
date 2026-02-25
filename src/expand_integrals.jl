import Symbolics
import SymbolicUtils
import SymbolicUtils.Code: function_to_expr, toexpr
import SymbolicIntegration
import QuadGK: quadgk

function SymbolicUtils.Code.function_to_expr(op::Integral, O, st)
    pair = op.domain
    var = pair.variables
    domain = pair.domain
    a_sym = Symbolics.infimum(domain)
    b_sym = Symbolics.supremum(domain)

    x = gensym(:x)

    args = SymbolicUtils.arguments(O)
    all_vars = SymbolicUtils.search_variables(args[1])

    aux = haskey(st.rewrites, var) ? st.rewrites[var] : nothing
    st.rewrites[var] = x
    body_expr = toexpr(args[1], st)
    if aux !== nothing
        st.rewrites[var] = aux
    else
        pop!(st.rewrites, var)
    end

    closure_vars = [(haskey(st.rewrites, e)) ? st.rewrites[e] : e for e in setdiff(all_vars, [var])]
    integrand_expr = isempty(closure_vars) ? build_function(body_expr, x) : build_function(body_expr, x, closure_vars...)

    integral_expr = isempty(closure_vars) ? (:($(quadgk)($x -> $integrand_expr($x), ($(toexpr(a_sym))), ($(toexpr(b_sym))))[1])) : (:($(quadgk)($x -> $integrand_expr($x, $(closure_vars...)...), ($(toexpr(a_sym))), ($(toexpr(b_sym))))[1]))

    return integral_expr
end

function expand_integrals(expr)
    unwrap_expr = Symbolics.unwrap(expr)
    if !istree(unwrap_expr)
        return expr
    else
        op = SymbolicUtils.operation(unwrap_expr)
        unwrap_args = SymbolicUtils.arguments(unwrap_expr)
        args = map(Symbolics.wrap ∘ expand_integrals, unwrap_args)
        if op isa Symbolics.Integral
            pair = op.domain
            a = Symbolics.infimum(pair.domain)
            b = Symbolics.supremum(pair.domain)
            x = Symbolics.wrap(pair.variables)
            f = args[1]
            # Try integrate symbolically
            si = SymbolicIntegration.integrate(f, x)
            if si !== nothing
                sup = Symbolics.substitute(si, Dict(x => b))
                inf = Symbolics.substitute(si, Dict(x => a))
                return sup - inf
            else
                return Symbolics.wrap(SymbolicUtils.Term{Symbolics.VartypeT}(op, unwrap_args; type=Symbolics.symtype(expr), shape=Symbolics.shape(expr)))
            end
        else
            return Symbolics.wrap(SymbolicUtils.Term{Symbolics.VartypeT}(op, unwrap_args; type=Symbolics.symtype(expr), shape=Symbolics.shape(expr)))
        end
    end
end