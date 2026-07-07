using Plots, FFTW, LinearAlgebra, Statistics, Printf
using Parameters, Distributions, Roots, Optim, Random, Test, Distributed, SharedArrays
using JLD2, DelimitedFiles

FFTW.set_num_threads(1)

include("trap_periodic.jl")
include("dealias_product_direct.jl")
include("taylor_KdV_solver.jl")
include("RK2_files/Exp_Integrator_RK2_Dealiasing_H23.jl")
include("compute_h2.jl")
include("compute_h3.jl")
include("Sampling/main.jl")

# ── Sample initial condition from main.jl ─────────────────────────────────────
params = ParamSet(
    min_samps_accept = 10,
    max_samps_accept = 20,
)

xdata, accept_rate = main(params)
println("Samples accepted: ", size(xdata, 2))

if size(xdata, 2) == 0
    error("No samples accepted — check rej_const in max_fg")
end



K    = params.nmodes
C2   = 1/120
C3   = 1
a    = 0
tfin = 5
h    = 1e-3
P    = 3

n_samps = size(xdata, 2)

save_every = round(Int, 0.02 / h)  # = 20

maxu_timeseries = Vector{Any}(undef, n_samps)
results = Vector{Any}(undef, n_samps)

Threads.@threads for j in 1:n_samps
    local samp, real_parts, imag_parts, u0, E0_samp
    local t_T, uk_T, Energy_T, M_T, H_T, H2_T, H3_T, U_phys_T

    samp = xdata[:, j]
    real_parts = samp[1:K]
    imag_parts = samp[K+1:end]
    u0 = zeros(ComplexF64, K+1)
    u0[1] = 0.0


    ### HERE: This should be a minus sign I believe.
    u0[2:K+1] = real_parts .+ im .* imag_parts




    E0_samp = 2π * sum(abs2.(u0[2:end]))
    u0 ./= sqrt(E0_samp / params.E0)

    t_T, uk_T, Energy_T, M_T, H_T, H2_T, H3_T, U_phys_T = Taylor_KdV(C2, C3, K, a, u0, h, tfin, P)

    # subsample every 0.02
    idx = 1:save_every:size(uk_T, 2)

   maxu_timeseries[j] = vec(maximum(U_phys_T, dims=1))[idx]
results[j] = (
    collect(t_T)[idx],
    uk_T[:, idx],
    H2_T[1],
    H3_T[1],
    H_T[1],
    H2_T,
    H3_T,
    U_phys_T[:, idx]   # full spatial field subsampled every 0.02
) 
    println("Sample $j max wave amplitude: ", maximum(maxu_timeseries[j]))
end

# ── Find IC with largest umax(tb)/umax(ta) ratio ─────────────────────────────
ratios = zeros(n_samps)
for j in 1:n_samps
    maxu = maxu_timeseries[j]
    tb_idx = argmax(maxu)
    ta_idx = argmin(maxu[1:tb_idx])
    ratios[j] = maxu[tb_idx] / maxu[ta_idx]
end
best_ratio_j = argmax(ratios)
println("Best IC: sample $best_ratio_j with ratio $(ratios[best_ratio_j])")

# ── Save ──────────────────────────────────────────────────────────────────────
max_amps_save = [maximum(maxu_timeseries[j]) for j in 1:n_samps]
H2_init  = [results[j][3] for j in 1:n_samps]
H3_init  = [results[j][4] for j in 1:n_samps]
H_init   = [results[j][5] for j in 1:n_samps]
H2_save  = [results[j][6] for j in 1:n_samps]
H3_save  = [results[j][7] for j in 1:n_samps]

# Save full uhat timeseries only for best IC
uk_best  = results[best_ratio_j][2]
t_best   = collect(results[best_ratio_j][1])
maxu_best = maxu_timeseries[best_ratio_j]

maxu = maxu_timeseries[best_ratio_j]
tb_idx_best = argmax(maxu)
ta_idx_best = argmin(maxu[1:tb_idx_best])
t_b_best = t_best[tb_idx_best]
t_a_best = t_best[ta_idx_best]
println("t_a = $t_a_best, t_b = $t_b_best")



# ── Find IC that minimizes H * A * |B| at t=0 ────────────────────────────────
# A = C2*H2, B = C3*H3, minimize H * (C2*H2) * |C3*H3|
HAB_vals = [H_init[j] * (C2 * H2_init[j]) * abs(C3 * H3_init[j]) for j in 1:n_samps]
best_HAB_j = argmin(HAB_vals)
println("Best HAB IC: sample $best_HAB_j with H*A*|B| = $(HAB_vals[best_HAB_j])")

# ── Compute max(t in [0,tfin]) umax(t)/umax(0) for each sample ───────────────
growth_ratios = zeros(n_samps)
for j in 1:n_samps
    maxu = maxu_timeseries[j]
    growth_ratios[j] = maximum(maxu) / maxu[1]
end
println("Best growth ratio IC: sample $(argmax(growth_ratios)) with ratio $(maximum(growth_ratios))")

uk_best_HAB  = results[best_HAB_j][2]
t_best_HAB   = collect(results[best_HAB_j][1])
maxu_best_HAB = maxu_timeseries[best_HAB_j]
U_phys_save = [results[j][8] for j in 1:n_samps]


@save "results_new3.jld2" max_amps_save H2_init H3_init H_init H2_save H3_save maxu_timeseries uk_best t_best maxu_best best_ratio_j ratios U_phys_save t_a_best t_b_best HAB_vals best_HAB_j growth_ratios uk_best_HAB t_best_HAB maxu_best_HAB n_samps tfin K P
println("Save done!")



# Call with:
# cd "Desktop/Summer Research 2026/Week 4"
#julia --threads 8 finding_best_IC.jl
