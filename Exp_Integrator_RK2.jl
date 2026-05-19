using FFTW
using Plots
include("trap_periodic.jl")


function Exp_Integrator_RK2(C2, C3, kvec, a, u0, h, tfin)
    N = length(kvec)
    nsteps = Int(round((tfin - a) / h))
    
    # Precompute linear operator
    alpha = -im .* C2 .* kvec.^3
    mu = exp.(-alpha .* h)  # exponential term
    mu2 = exp.(-alpha.*(h/2))
    
    # Initialize Fourier modes over time
    uk = zeros(ComplexF64, N, nsteps+1)
    uk[:, 1] = fft(u0)./N  # IC in physical space, convert to Fourier space
    
    t = range(a, length=nsteps+1, step=h)

    Energy = zeros(nsteps)
    M = zeros(nsteps)
    H = zeros(nsteps)

    x = -π .+ 2π * (0:N-1)./N   # grid with N points, periodic

    
    for n in 1:nsteps
        # U from uk at time n
        u_phys = real(ifft(uk[:, n]) * N)
        
        # Compute u_x:
        # Derivative: i k * uk, then ifft
        ux_phys = real(ifft(im .* kvec .* uk[:, n]) * N)
        
        # Get vk at current time
        vk = fft(u_phys .* ux_phys) / N




        # Midpoint estimate for u_k
        uk_mid = exp.(-alpha .* (h/2)) .* (uk[:,n] - C3.*(h/2).*vk)
        

        # Compute v_k at Midpoint
        u_phys_mid = real(ifft(uk_mid)*N)
        ux_phys_mid = real(ifft(im.*kvec.*uk_mid)*N)
        vk_mid = fft(u_phys_mid .* ux_phys_mid)/N




        
        # Update Fourier coefficients uk for next step
        uk[:, n+1] = mu .* uk[:, n] - C3 * h .* mu2 .* vk_mid

        Energy[n] = 0.5 * trap_periodic(u_phys.^2, x)
        M[n] = trap_periodic(u_phys, x)
        H[n] = (C2/2) * trap_periodic(ux_phys.^2, x) - (C3/6) * trap_periodic(u_phys.^3, x)   
    end


    
    return t, uk,Energy,M,H

end





# assign parameters 
C2 = 1/120
C3 = 1
N = 64
a = 0
tfin = 1
h = C2^-1 .* N^-3

kvec = fftfreq(N,N)
x = -π .+ 2π * (0:N-1)./N 
u0 = sin.(x)


# Call function
t, uk, Energy, M, H = Exp_Integrator_RK2(C2, C3, kvec, a, u0, h, tfin);



# Plots of Conserved Quantities
# Make sure t and the diagnostic arrays have the same length
tplot = t[1:length(Energy)]

p1 = plot(tplot, Energy, xlabel="t", ylabel="E", title="Energy",  ylims=(1.5, 1.7),   legend=false)
p2 = plot(tplot, M,      xlabel="t", ylabel="M",   title="Momentum",   ylims=(-1, 1),    legend=false)
p3 = plot(tplot, H,      xlabel="t", ylabel="H",      title="Hamiltonian",  ylims=(0, 0.05), legend=false)

conservation = plot(p1, p2, p3, layout=(3,1), size=(700, 700))
display(conservation)





## H Convergence test
h_ref =  C2^-1 .* N^-3
hs = [8* h_ref, 4*h_ref, 2*h_ref, sqrt(2)*h_ref, h_ref/sqrt(2), h_ref/2, h_ref/4, h_ref/8]

errors = zeros(length(hs))

for i in 1:length(hs)

    _, _, _, _, H_i = Exp_Integrator_RK2(C2, C3, kvec, a, u0, hs[i], tfin);
    errors[i] = abs(H_i[end] - H_i[1]);

end

idx_ref = 5

convergence = plot(hs, errors, xscale=:log10, yscale=:log10, marker=:circle,
     xlabel="h", ylabel="|H(t_fin) - H(0)|", label="error")
plot!(convergence, hs, hs.^2, linestyle=:dash, label="h^2")
scatter!(convergence, [h_ref], [errors[idx_ref]],
         marker=:star5, markersize=12, color=:red,
         label="h = 1/(C2·N³)")
display(convergence)