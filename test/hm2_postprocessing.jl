using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using JLD2
using Plots
using Roots
using QuadGK
using Printf
using DifferentialEquations
using ProgressMeter


# --------------------------------------------------
# CONFIG
# --------------------------------------------------
base_dir = joinpath(@__DIR__, "..", "results", "hm2")
Ns = 10:10:200
out_dir = joinpath(@__DIR__, "..", "results", "hm2_postprocessing")
mkpath(out_dir)

# Spatial grid
# Time grid
xs = range(0, 1; length=100) |> collect
ys = range(0, 1; length=100) |> collect
ts = range(0, 1; length=50) |> collect

# Eigenfunction (replace with your exact definition)
Lᵢ = 200
Lⱼ = 200
L = maximum(Ns) + 5

fᵢ = λ -> tan(λ) + λ
fⱼ = λ -> sin(λ)
Iᵢ = (1.0001 * π / 2, 0.9999 * π / 2)
Iⱼ = (-1.000 * π / 2, -1.000 * π / 2)
λᵢ = [find_zero(fᵢ, (Iᵢ[1] + π * (i - 1), Iᵢ[2] + π * i), Bisection()) for i in 1:Lᵢ]
λⱼ = [find_zero(fⱼ, (Iⱼ[1] + π * (j - 1), Iⱼ[2] + π * j), Bisection()) for j in 1:Lⱼ]
energy = [(CartesianIndex(i, j), λᵢ[i]^2 + λⱼ[j]^2) for i in 1:Lᵢ for j in 1:Lⱼ]
sort!(energy, by=x -> x[2])
indexes = first.(energy)[1:L]

is_maximum = Dict{Integer,Integer}()
js_maximum = Dict{Integer,Integer}()
for n in Ns
    i_maximum = 1
    j_maximum = 1
    for i in 1:n+5
        i₀, j₀ = Tuple(indexes[i])
        i_maximum = (λᵢ[i₀] > λᵢ[i_maximum]) ? i₀ : i_maximum
        j_maximum = (λⱼ[j₀] > λⱼ[j_maximum]) ? j₀ : j_maximum
    end
    is_maximum[n] = i_maximum
    js_maximum[n] = j_maximum
end
μᵢ(i) = λᵢ[i]
μⱼ(j) = λⱼ[j]

ψᵢ_NN(i, x) = sin(μᵢ(i) * x)
ψⱼ_NN(j, x) = cos(μⱼ(j) * x)
Nᵢ = [quadgk(x -> ψᵢ_NN(i, x)^2, 0, 1)[1] for i in 1:Lᵢ]
Nⱼ = [quadgk(x -> ψⱼ_NN(j, x)^2, 0, 1)[1] for j in 1:Lⱼ]

ψᵢ(i, x) = ψᵢ_NN(i, x) / sqrt(Nᵢ[i])
ψⱼ(j, x) = ψⱼ_NN(j, x) / sqrt(Nⱼ[j])
ψᵢ_array = [ψᵢ(i, x) for i in 1:Lᵢ, x in xs]
ψⱼ_array = [ψⱼ(j, y) for j in 1:Lⱼ, y in ys]

Cst = 1
φₙ(x, y) = exp(-x)
γ = [quadgk(x -> φₙ(x, 1) * ψᵢ(i, x), 0, 1)[1] for i in 1:maximum(values(is_maximum))]
G(x, n) = sum(γ[i] * ψᵢ(i, x) for i in 1:is_maximum[n])
H(y) = y^2 / 2
F(x, y, n) = Cst + G(x, n) * H(y)                         # Filter function
F_array = [F(x, y, n) for x in xs, y in ys, n in Ns]

if any(isnan.(F_array)) || any(isinf.(F_array))
    error("Missing values detected in precomputed arrays.")
end

#--------------------------------------------------------------------------
# Helper functions
#--------------------------------------------------------------------------
function reconstruct_T(sol, t, xs, ys, N, N_extra=0)
    θ̄ = sol(t)
    T = Array{Float64}(undef, length(xs), length(ys))
    @inbounds for (ix, x) in enumerate(xs)
        for (iy, y) in enumerate(ys)
            s = 0.0
            N_total = N + N_extra
            for n in 1:N_total
                i, j = Tuple(indexes[n])
                s += θ̄[n] * ψᵢ_array[i, ix] * ψⱼ_array[j, iy]
            end
            T[ix, iy] = s
        end
    end
    in = findfirst(==(N), Ns)
    if in === nothing
        error("N = $N not found in Ns")
    end
    T .+= F_array[:, :, in]
    return T
end

#--------------------------------------------------------------------------
# Load solutions
#--------------------------------------------------------------------------
loaded = Dict{Int,Any}()

for N in Ns
    path = joinpath(base_dir, "N_$(N)", "burger2D.jld2")
    isfile(path) || continue

    file = JLD2.load(path)
    sol = file["single_stored_object"]

    loaded[N] = (sol=sol, ts=ts)
    println("Loaded N = $N")
end

Nref = maximum(keys(loaded))

# Find max and min of T for plotting limits
T_min = 0.0
T_max = 1.225
display(@sprintf("T_min = %.5f, T_max = %.5f", T_min, T_max))
#--------------------------------------------------------------------------
# FIGURE 1 — modal coefficients vs time
#--------------------------------------------------------------------------
begin
    sol = loaded[Nref].sol
    ts = loaded[Nref].ts
    N = Nref

    plt = plot(xlabel="t", ylabel="Θᵢ(t)",
        title="Modal coefficients (N = $N)")

    for i in 1:min(6, N)
        plot!(plt, ts, [sol(t)[i] for t in ts], label="i = $i")
    end

    savefig(plt, joinpath(out_dir, "coeffs_vs_time_N$(N).pdf"))
end
begin
    sol = loaded[Nref].sol
    ts = loaded[Nref].ts
    N = Nref
    sol_t = Array{Float64}(undef, N, length(ts))
    for (j, t) in enumerate(ts)
        θ̄ = sol(t)
        for i in 1:N
            sol_t[i, j] = abs(θ̄[i])
        end
    end
    max_sol = maximum(sol_t, dims=2)[:]

    plt = plot(1:Nref, max_sol, label="", xlabel="Mode i", ylabel="|Θᵢ(t)|",
        title="Modal Coefficients Amplitude Decay (N = $N)",
        yscale=:log10)


    savefig(plt, joinpath(out_dir, "modal_amplitude_decay_N$(N).pdf"))
end
exit()
#--------------------------------------------------------------------------
# FIGURE 2 — Reconstructed profiles at Different Times
#--------------------------------------------------------------------------
begin
    sol = loaded[Nref].sol
    ts = loaded[Nref].ts
    N = Nref

    times_to_plot = [0.0, 0.01, 0.05, 0.1, 0.3, 0.6, 1.0]
    for t in times_to_plot
        T_reconstructed = reconstruct_T(sol, t, xs, ys, N)

        plt = heatmap(xs, ys, T_reconstructed',
            xlabel="x", ylabel="y",
            title=@sprintf("Reconstructed T(x,y) at t = %.2f (N = %d)", t, Nref),
            colorbar_title="T", clim=(T_min, T_max))

        savefig(plt, joinpath(out_dir, @sprintf("T_reconstructed_t%.2f_N%d.pdf", t, Nref)))
    end
end

#--------------------------------------------------------------------------
# FIGURE 3 — Error analysis
#--------------------------------------------------------------------------

plt = plot(xlabel="t", ylabel="Error",
    title="Error between N and Nref")
plt_max = plot(xlabel="N", ylabel="Error",
    title="Error")
max_max_error = []
@showprogress desc = "Loading errors from each N..." for N in Ns
    sol_N = loaded[N].sol
    max_errors = Float64[]
    for τ in ts
        T_ref = reconstruct_T(sol_N, τ, xs, ys, N, 5)
        if any(isnan.(T_ref)) || any(isinf.(T_ref))
            Base.error("Missing values detected in reconstructed T_ref for N = $N at time τ = $τ.")
        end
        T_N = reconstruct_T(sol_N, τ, xs, ys, N)
        if any(isnan.(T_N)) || any(isinf.(T_N))
            Base.error("Missing values detected in reconstructed T_N for N = $N at time τ = $τ.")
        end
        err = maximum(abs.(T_ref .- T_N))
        push!(max_errors, err)
    end
    plot!(plt, ts, max_errors, label="N = $N")
    push!(max_max_error, maximum(max_errors))
end
savefig(plt, joinpath(out_dir, "error_vs_time.pdf"))
plot!(plt_max, Ns, max_max_error, marker=:o, label="Max Error vs N", yscale=:log10)
savefig(plt_max, joinpath(out_dir, "max_error_vs_N.pdf"))

println("Postprocessing complete. Figures saved in $out_dir")