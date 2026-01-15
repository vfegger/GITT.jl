using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using JLD2
using Plots
using Roots
using QuadGK
using Printf
using DifferentialEquations

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
base_dir = joinpath(@__DIR__, "..", "results", "hm1")
Ns       = 10:10:200
out_dir  = joinpath(@__DIR__, "..", "results", "hm1_postprocessing")
mkpath(out_dir)

# Spatial grid
# Time grid
xs = range(0, 1; length=300) |> collect
ts = range(0, 1; length=250) |> collect

# Eigenfunction (replace with your exact definition)
L = maximum(Ns) + 5
f = x -> tan(x) + x
a = 1.0001 * π / 2
b = 0.9999 * π / 2
eigenvals = [find_zero(f, (a + π * (i - 1), b + π * i), Bisection()) for i in 1:L]
α(i) = eigenvals[i]
ψ_NN(i, x) = sin(α(i) * x)
Norms = [quadgk(x -> ψ_NN(i, x)^2, 0, 1)[1] for i in 1:L]
ψ(i, x) = ψ_NN(i, x) / sqrt(Norms[i])

Tf = 1.0                   # filter

# --------------------------------------------------
# Helper functions
# --------------------------------------------------

nearest_index(v, x) = argmin(abs.(v .- x))

# Reconstruct T(x,t)
function reconstruct_T(sol, t, xs, ψ, Tf, N, N_extra = 0)
    θ̄ = sol(t)

    T = similar(xs)
    @inbounds for (ix, x) in enumerate(xs)
        s = 0.0
        N_total = N + N_extra
        for i in 1:N_total
            s += θ̄[i] * ψ(i, x)
        end
        T[ix] = Tf + s
    end
    return T
end

# --------------------------------------------------
# Load solutions
# --------------------------------------------------
loaded = Dict{Int, Any}()

for N in Ns
    path = joinpath(base_dir, "N_$(N)", "burger2D.jld2")
    isfile(path) || continue

    file = JLD2.load(path)
    sol = file["single_stored_object"]

    loaded[N] = (sol=sol, ts=ts)
    println("Loaded N = $N")
end

Nref = maximum(keys(loaded))

# --------------------------------------------------
# FIGURE 1 — modal coefficients vs time
# --------------------------------------------------
begin
    sol = loaded[Nref].sol
    ts  = loaded[Nref].ts
    N   = length(loaded[Nref].ts)

    plt = plot(xlabel="t", ylabel="Θᵢ(t)",
               title="Modal coefficients (N = $Nref)")

    for i in 1:min(6, N)
        plot!(plt, ts, [sol(t)[i] for t in ts], label="i = $i")
    end

    savefig(plt, joinpath(out_dir, "coeffs_vs_time_N$(Nref).pdf"))
end

# --------------------------------------------------
# FIGURE 2 — reconstructed profiles
# --------------------------------------------------
times = [0.0, 0.01, 0.05, 0.1, 0.3, 0.6, 1.0]

begin
    sol = loaded[Nref].sol
    ts  = loaded[Nref].ts

    plt = plot(xlabel="x", ylabel="T(x,t)",
               title="Reconstructed solution (N = $Nref)")

    for τ in times
        T  = reconstruct_T(sol, τ, xs, ψ, Tf, Nref)
        plot!(plt, xs, T, label=@sprintf("t = %.3f", τ))
    end

    savefig(plt, joinpath(out_dir, "profiles_N$(Nref).pdf"))
end

# --------------------------------------------------
# FIGURE 3 — error analysis
# --------------------------------------------------
begin
    plt = plot(xlabel="t", ylabel="Error",
               title="Error between N and Nref")
    plt_max = plot(xlabel="N", ylabel="Error",
               title="Error")
    max_max_error = []
    for N in Ns
        sol_N = loaded[N].sol

        max_errors = Float64[]
        for τ in times
            T_ref = reconstruct_T(sol_N, τ, xs, ψ, Tf, N, 5)
            T_N   = reconstruct_T(sol_N, τ, xs, ψ, Tf, N)

            error = maximum(abs.(T_ref .- T_N))
            push!(max_errors, error)
        end

        plot!(plt, times, max_errors, xlabel="t", ylabel="Error", label="N = $N", marker=:o, yscale=:log10)
        push!(max_max_error, maximum(max_errors))
    end
    plot!(plt_max, Ns, max_max_error, xlabel="N", ylabel="Max Error", label="Max Error vs N", marker=:o, yscale=:log10)

    savefig(plt, joinpath(out_dir, "error_analysis.pdf"))
    savefig(plt_max, joinpath(out_dir, "max_error_analysis.pdf"))
end

# --------------------------------------------------
# FIGURE 4 - Modal Coefficients Amplitude Decay
# --------------------------------------------------
begin
    sol = loaded[Nref].sol
    ts  = loaded[Nref].ts
    sol_t = Array{Float64,2}(undef, Nref, length(ts))
    for (j, t) in enumerate(ts)
        θ̄ = sol(t)
        for i in 1:Nref
            sol_t[i, j] = abs(θ̄[i])
        end
    end
    max_sol = maximum(sol_t, dims=2)[:]
    plt = plot(1:Nref, max_sol, label="", xlabel="Mode i", ylabel="|Θᵢ(t)|",
               title="Modal Coefficients Amplitude Decay (N = $Nref)",
               yscale=:log10)

    savefig(plt, joinpath(out_dir, "modal_amplitude_decay_N$(Nref).pdf"))
end

println("Postprocessing complete. Figures saved in $out_dir")
