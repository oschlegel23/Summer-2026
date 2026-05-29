using FFTW
using Plots
include("trap_periodic.jl")
include("dealias_product_direct.jl")


function Exp_Integrator_RK2_Dealiasing(C2, C3, K, a, u0, h, tfin)
    nsteps = Int(round((tfin - a) / h))
    kpos= 0:K
    
    # Precompute linear operator
    alpha = -im .* C2 .* kpos.^3
    mu = exp.(-alpha .* h)  # exponential term
    mu2 = exp.(-alpha.*(h/2))
    
    # Initialize Fourier modes over time
    uk = zeros(ComplexF64, K+1, nsteps+1)
    uk[:, 1] = u0  # IC in Fourier space
    
    t = range(a, length=nsteps+1, step=h)

    Energy = zeros(nsteps)
    M = zeros(nsteps)
    H = zeros(nsteps)
    H2 = zeros(nsteps)  
    H3 = zeros(nsteps) 

    # need for fft an ifft
    N = 4*(2K+1)
    x = -π .+ 2π * (0:N-1)./N
    kvec_full = fftfreq(N, N)
    U_phys = zeros(Float64, N, nsteps+1)

    for n in 1:nsteps
        # U from uk at time n

        #need to refill full vector to use ifft
        u_pos = uk[:,n]
        uk_full = zeros(ComplexF64, N)
        uk_full[1:K+1] = u_pos
        for k in 1:K
            uk_full[N-k+1] = conj(u_pos[k+1])
        end
        u_phys = real(ifft(uk_full) * N)
        U_phys[:, n] = u_phys
        
        # Compute u_x:
        # Derivative: i k * uk, then ifft
        ux_phys = real(ifft(im .* kvec_full .* uk_full) * N)
        
        # Get vk at current time
        ux_pos = im .* kpos .* u_pos
        vk_pos = dealias_product_direct(u_pos, ux_pos)


        # Midpoint estimate for u_k
        uk_mid = exp.(-alpha .* (h/2)) .* (u_pos - C3.*(h/2).*vk_pos)
    

        # Compute v_k at Midpoint

        ux_mid_pos = im .* kpos .* uk_mid
        vk_mid_pos = dealias_product_direct(uk_mid, ux_mid_pos)

        
        # Update Fourier coefficients uk for next step
        uk[:, n+1] = mu .* u_pos - C3 * h .* mu2 .* vk_mid_pos


        Energy[n] = 2*pi * sum(abs2.(u_pos[2:end]))
        M[n] = 2*pi *  real(u_pos[1])
        H2[n] = (C2/2) * trap_periodic(ux_phys.^2, x)
        H3[n] = -(C3/6) * trap_periodic(u_phys.^3, x)
        H[n] = H2[n] + H3[n]
    end 
    
    return t, uk,Energy,M,H, H2,H3, U_phys

end






