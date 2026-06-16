using FFTW
using Plots
using LinearAlgebra
include("trap_periodic.jl")
include("dealias_product_direct.jl")
include("Exp_Integrator_RK2_Dealiasing.jl")
include("Exp_Integrator_RK2.jl")


# assign parameters 
C2 = 1/120
C3 = 1
K = 9
a = 0
tfin = 1
#N = 4(2*31+1)
h = C2^-1 .* K^-3  .* .01      #in terms of K now

kpos=0:K

u0 = randn(ComplexF64, K+1) .* exp.(-kpos.^2 ./ 4) ## make random fourier coefficients for IC
u0[1] = real(u0[1])  # ADD THIS LINE

# normalize energy to 1
E0 = 2*pi*sum(abs2.(u0[2:end]))
u0 ./= sqrt(E0)
#u0 .*= 10 



# Call function
t, uk, Energy, M, H = Exp_Integrator_RK2(C2, C3, K, a, u0, h, tfin);
t1, uk1, Energy1, M1, H1, H2, H3 = Exp_Integrator_RK2_Dealiasing(C2, C3, K, a, u0, h, tfin);







# version 1 plots:---------------------------------------------------------------------------------------------------------
# Plots of Conserved Quantities
# Make sure t and the diagnostic arrays have the same length
tplot = t[1:length(Energy)]

p1 = plot(tplot, Energy, xlabel="t", ylabel="E", title="Energy no Dealiasing",  ylims=(Energy[1] - 1e-6, Energy[1] + 1e-6), legend=false);
p2 = plot(tplot, M,      xlabel="t", ylabel="M",   title="Momentum no Dealiasing",  ylims=(M[1] - 1e-6, M[1] + 1e-6),  legend=false);
p3 = plot(tplot, H,      xlabel="t", ylabel="H",      title="Hamiltonian no Dealiasing",  ylims=(H[1] - 1e-6, H[1] + 1e-6), legend=false);

conservation = plot(p1, p2, p3, layout=(3,1), size=(700, 700));
display(conservation)







# ## H Convergence test
# h_ref =  C2^-1 .* N^-3
# hs = [8*h_ref, 4*h_ref, 2*h_ref, sqrt(2)*h_ref, h_ref, h_ref/sqrt(2), h_ref/2, h_ref/4, h_ref/8]
# idx_ref = 5  

# errors = zeros(length(hs))

# for i in 1:length(hs)

#     _, _, _, _, H_i = Exp_Integrator_RK2(C2, C3, K, a, u0, hs[i], tfin);
#     errors[i] = abs((H_i[end] - H_i[1])) / abs(H_i[1]);
    
# end



# # plot convergence
# convergence = plot(hs, errors, xscale=:log10, yscale=:log10, marker=:circle,
#      xlabel="h", ylabel="|H(t_fin) - H(0)| / |H(0)|",title="Error no Dealiasing", label="error");
# plot!(convergence, hs, hs.^2, linestyle=:dash, label="h^2");
# scatter!(convergence, [h_ref], [errors[idx_ref]],
#          marker=:star5, markersize=12, color=:red,
#          label="h = 1/(C2·N³)");
# display(convergence)
#-----------------------------------------------------------------------------------------------------------------------------


# version 2 plots:---------------------------------------------------------------------------------------------------------
# Plots of Conserved Quantities
# Make sure t and the diagnostic arrays have the same length
tplot1 = t1[1:length(Energy1)]

p11 = plot(tplot1, Energy1, xlabel="t", ylabel="E", title="Energy Dealiasing",  ylims=(Energy1[1] - 1e-6, Energy1[1] + 1e-6), 
legend=false);
p21 = plot(tplot1, M1,      xlabel="t", ylabel="M",   title="Momentum Dealiasing",  ylims=(M1[1] - 1e-6, M1[1] + 1e-6),  legend=false);
p31 = plot(tplot1, H1,      xlabel="t", ylabel="H",      title="Hamiltonian Dealiasing",  ylims=(H1[1] - 1e-6, H1[1] + 1e-6), 
legend=false);

conservation1 = plot(p11, p21, p31, layout=(3,1), size=(700, 700));
display(conservation1)

# if using larger amplitude, get rid of ylims on plots. 





# ## H Convergence test
h_ref =  C2^-1 .* N^-3
hs = [8*h_ref, 4*h_ref, 2*h_ref, sqrt(2)*h_ref, h_ref, h_ref/sqrt(2), h_ref/2, h_ref/4, h_ref/8]
idx_ref = 5  # now correctly points to h_ref

errors = zeros(length(hs))


for i in 1:length(hs)

    _, _, _, _, H_i1, _, _ = Exp_Integrator_RK2_Dealiasing(C2, C3, K, a, u0, hs[i], tfin);
    errors[i] = abs((H_i1[end] - H_i1[1])) / abs(H_i1[1]);

end




# plot convergence
convergence1 = plot(hs, errors, xscale=:log10, yscale=:log10, marker=:circle,
     xlabel="h", ylabel="|H(t_fin) - H(0)| / |H(0)|",  title="Error Dealiasing", label="error");
plot!(convergence1, hs, hs.^2, linestyle=:dash, label="h^2");
scatter!(convergence1, [h_ref], [errors[idx_ref]],
         marker=:star5, markersize=12, color=:red,
         label="h = 1/(C2·N³)");
display(convergence1)




# plot H2 and H3
p = plot(t[1:end-1], H2, label="H2", xlabel="t", ylabel="H", title="Hamiltonian Components")
plot!(t[1:end-1], H3, label="H3")
display(p)







# zoom in on hamiltonian drift
idx = Int(round(0.9 * length(tplot1))):length(tplot1)
idx2 = idx[2:end]

p_normal = plot(tplot1[idx], H1[idx], xlabel="t", ylabel="H", title="H zoomed (normal)", legend=false)

p_loglin = plot(tplot1[idx2], abs.(H1[idx2] .- H1[idx[1]]), xlabel="t", ylabel="|ΔH|", 
                title="H drift (log-lin)", yscale=:log10, legend=false)

p_loglog = plot(tplot1[idx2], abs.(H1[idx2] .- H1[idx[1]]), xlabel="t", ylabel="|ΔH|", 
                title="H drift (log-log)", xscale=:log10, yscale=:log10, legend=false)

p = plot(p_normal, p_loglin, p_loglog, layout=(1,3), size=(1200, 400))
display(p)

p_sq = plot((tplot1[idx2] .- tplot1[idx[1]]).^2, abs.(H1[idx2] .- H1[idx[1]]), 
              xlabel="t²", ylabel="|ΔH|", title="H drift vs t²", legend=false)
display(p_sq)






# test for local error
scales = [1.0, 0.5, 0.25, 0.1, 0.05]
hs = []
local_errors = []

for s in scales
    h_test = C2^-1 * K^-3 * s
    _, _, _, _, H_test, _, _ = Exp_Integrator_RK2_Dealiasing(C2, C3, K, a, u0, h_test, tfin)
    push!(hs, h_test)
    push!(local_errors, abs(H_test[2] - H_test[1]))  # error after just one step
end

hs = Float64.(hs)
local_errors = Float64.(local_errors)

p = plot(hs, local_errors, xscale=:log10, yscale=:log10, xlabel="h", ylabel="local error", 
         title="Local error vs h", marker=:circle, label="measured", legend=:topleft)
plot!(hs, hs.^3 .* (local_errors[1] / hs[1]^3), linestyle=:dash, label="h³ reference")
display(p)
