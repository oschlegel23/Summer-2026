cd(@__DIR__)
using Plots, Statistics, JLD2
using FFTW
using LaTeXStrings
using Plots.PlotMeasures  # add this at the top with your other usings

@load "results_new3.jld2" max_amps_save H2_init H3_init H_init U_phys_save H2_save H3_save maxu_timeseries uk_best t_best maxu_best best_ratio_j ratios U_phys_save t_a_best t_b_best n_samps tfin K P

# ── Compute SD from full spatial field across all samples ─────────────────────
all_displacements = vcat([vec(U_phys_save[j]) for j in 1:n_samps]...)
mu_disp = mean(all_displacements)
sigma_disp = std(all_displacements)
println("μ = $(round(mu_disp, digits=4)), σ = $(round(sigma_disp, digits=4))")

for mult in 1.0:0.1:5.0
    threshold = mu_disp + mult * sigma_disp
    n_exceed = sum(max_amps_save .>= threshold)
    if 2 <= n_exceed <= 3
        println("mult=$(round(mult, digits=1)), threshold=$(round(threshold, digits=4)), n_exceed=$n_exceed")
    end
end

# ── Histograms of H, H2, H3 at t=0 ──────────────────────────────────────────
pH_hist = histogram(H_init, bins=20, normalize=:probability, bar_edges=false,
                    xlabel=L"H(0)", ylabel=L"P(H(0))", 
                    title=L"Distribution of $H(0)$", label=false, color=:steelblue, alpha=0.7,
                    xlims=(-maximum(abs.(H_init)), maximum(abs.(H_init))),
                    framestyle=:origin, left_margin=5mm)

pH2_hist = histogram(H2_init, bins=15, normalize=:probability, bar_edges=false,
                     xlabel=L"H_2(0)", ylabel=L"P(H_2(0))",
                     title=L"Distribution of $H_2(0)$", label=false, color=:green, alpha=0.7,
                     xlims=(-maximum(abs.(H2_init)), maximum(abs.(H2_init))),
                     framestyle=:origin, left_margin=5mm)

pH3_hist = histogram(H3_init, bins=30, normalize=:probability, bar_edges=false,
                     xlabel=L"H_3(0)", ylabel=L"P(H_3(0))",
                     title=L"Distribution of $H_3(0)$", label=false, color=:crimson, alpha=0.7,
                     xlims=(-maximum(abs.(H3_init)), maximum(abs.(H3_init))),
                     framestyle=:origin, left_margin=5mm)

display(pH_hist)
display(pH2_hist)
display(pH3_hist)

# ── Best ratio IC: max(u) over time ──────────────────────────────────────────

existing_ticks = 0:1:tfin
all_ticks = sort(vcat(collect(existing_ticks), [t_a_best, t_b_best]))
tick_labels = [t in [t_a_best, t_b_best] ? (t == t_a_best ? "t_a" : "t_b") : string(round(Int, t)) for t in all_ticks]

best_mult = 3
threshold = mu_disp + best_mult * sigma_disp

pBest = plot(t_best, maxu_best, xlabel="t", ylabel="max(u)",
             title="Best Ratio IC (sample $best_ratio_j, ratio=$(round(ratios[best_ratio_j], digits=3)))",
             label=false, color=:steelblue,
             xticks=(all_ticks, tick_labels))
hline!(pBest, [threshold], color=:crimson, linestyle=:dash,
       label="μ + $(best_mult)σ = $(round(threshold, digits=3))")
display(pBest)




#--------------------------------------------------------------------------------------------------------------------------------

# Find indices for t_a, (t_b-t_a)/2, t_b in t_best
t_mid1 = t_a_best + (t_b_best - t_a_best)/3
t_mid2 = t_a_best + 2*(t_b_best - t_a_best)/3

times_plot = [t_a_best, t_mid1, t_mid2, t_b_best]
idxs_plot  = [argmin(abs.(t_best .- t)) for t in times_plot]
labels_t = [L"t_a", L"t_a + (t_b-t_a)/3", L"t_a + 2(t_b-t_a)/3", L"t_b"]

N = 4*(2K+1)
x = -π .+ 2π * (0:N-1) ./ N

# compute shared axis limits first
all_u = []
all_spec = []
for idx in idxs_plot
    uk_t = uk_best[:, idx]
    uk_full = zeros(ComplexF64, N)
    uk_full[1:K+1] = uk_t
    for k in 1:K; uk_full[N-k+1] = conj(uk_t[k+1]); end
    push!(all_u, real(ifft(uk_full) * N))
    push!(all_spec, abs2.(uk_t[2:end]))
end

u_min = minimum(minimum.(all_u))
u_max = maximum(maximum.(all_u))
s_min = minimum(minimum.(all_spec))
s_max = maximum(maximum.(all_spec))

wave_plots = []
spec_plots = []

for (i, idx) in enumerate(idxs_plot)
    uk_t = uk_best[:, idx]
    uk_full = zeros(ComplexF64, N)
    uk_full[1:K+1] = uk_t
    for k in 1:K; uk_full[N-k+1] = conj(uk_t[k+1]); end
    u_phys = real(ifft(uk_full) * N)
    kvals = 1:K
    spectrum = abs2.(uk_t[2:end])

pW = plot(x, u_phys, xlabel=L"x", ylabel=L"u(x,t)",
      title=L"t = %$(labels_t[i]) = %$(round(times_plot[i], digits=2))",
      label=false, color=:steelblue, ylims=(u_min, u_max),
      left_margin=10mm)


hline!(pW, [threshold], color=:crimson, linestyle=:dash,
       label=L"3\sigma \text{ threshold}")

kvals = collect(1:K)
spectrum = abs2.(uk_t[2:end])

# linear fit
k_mean = mean(kvals)
s_mean = mean(spectrum)
slope_s = sum((kvals .- k_mean) .* (spectrum .- s_mean)) / sum((kvals .- k_mean).^2)
intercept_s = s_mean - slope_s * k_mean
fit_line = slope_s .* kvals .+ intercept_s

pSpec = plot(kvals, spectrum, xlabel=L"k", ylabel=L"|\hat{u}_k|^2",
             title=L"Spectrum at %$(labels_t[i])",
             label=false, color=:crimson,
             marker=:circle, markersize=3,
             ylims=(s_min, s_max),
             left_margin=10mm)
plot!(pSpec, kvals, fit_line, color=:black, linestyle=:dash,
      label=L"slope = %$(round(slope_s, digits=4))")

    push!(wave_plots, pW)
    push!(spec_plots, pSpec)
end

wave_spec = plot(wave_plots[1], spec_plots[1],
                 wave_plots[2], spec_plots[2],
                 wave_plots[3], spec_plots[3],
                 wave_plots[4], spec_plots[4],
                 layout=(4,2), size=(900, 1200))
display(wave_spec)






























# x1 = (1/120).*H2_init #for each IC
# y1 = (1).*H3_init # for each IC


# min_val = min(minimum(x1), minimum(y1))
# max_val = max(maximum(x1), maximum(y1))

# # 2. Apply the identical range to both axes
# pScatter = scatter(x1, y1, 
#              xlabel="x = C2H2", 
#              ylabel="y = C3H3",
#              title="Scatter of ICs", 
#              color=:steelblue,
#              xlims=(min_val, max_val), 
#              ylims=(min_val, max_val))

# display(pScatter)


# x2 = H2_init #for each IC
# y2 = H3_init # for each IC


# min_val = min(minimum(x2), minimum(y2))
# max_val = max(maximum(x2), maximum(y2))

# # 2. Apply the identical range to both axes
# pScatter = scatter(x2, y2, 
#              xlabel="x = H2", 
#              ylabel="y = H3",
#              title="Scatter of ICs", 
#              color=:steelblue,
#              xlims=(min_val, max_val), 
#              ylims=(min_val, max_val))

# display(pScatter)



# x2 = H2_init #for each IC
# y2 = H3_init # for each IC

# min_val = min(minimum(x2), minimum(y2))
# max_val = max(maximum(x2), maximum(y2))

# # 1. Create a color array matching the length of your data
# point_colors = fill(:steelblue, length(x2))

# # 2. Set the 43rd item to red
# point_colors[43] = :red

# # 3. Apply the identical range to both axes and pass the color array
# pScatter = scatter(x2, y2, 
#              xlabel="x = H2", 
#              ylabel="y = H3",
#              title="Scatter of ICs", 
#              color=point_colors,          # Use the array here
#              xlims=(min_val, max_val), 
#              ylims=(min_val, max_val))

# display(pScatter)
